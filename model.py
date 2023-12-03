import torch
from torch import nn
from torch.autograd import Variable




class PositionalEncoding(nn.Module):

    def __init__(self, e_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, e_dim).float()
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = 10000.0 ** (torch.arange(0., e_dim, 2.) / e_dim)

        # Calculate sin for even-numbered digits, and calculate cos for odd-numbered digits.
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False).cuda()
        return self.dropout(x)


class Social_Transformer(nn.Module):
    '''Aggregate spatial relative position features'''

    def __init__(self, dim, mlp_dim, depth, heads, dropout):
        super(Social_Transformer, self).__init__()
        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )
        # self.token=nn.Parameter(torch.randn(1,dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)

    def forward(self, obs, seq_start_end):
        #s_rel = obs_to_srel(obs, seq_start_end)
        s_rel = obs.permute(1, 0, 2).contiguous() #seq,agent_count,dim
        # s_in = self.pos_to_dim(s_rel).permute(1, 0, 2).contiguous()
        s_out = []
        for se in seq_start_end:
            cur_sin = s_rel[:, se[0]:se[1], :]
            cur_sin = cur_sin.repeat(cur_sin.size(1), 1, 1)
            for i in range(se[1] - se[0]):
                cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)] = cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0),i:i + 1] - cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)] #n*seq,agent_count,dim n=0 第0个人相对其他人的位置seq,agent_count,dim
            cur_sin = self.pos_to_dim(cur_sin)
            group_out = self.transformer(cur_sin)
            cur_out = []
            for i in range(se[1] - se[0]):
                cur_out.append(group_out[i * s_rel.size(0):(i + 1) * s_rel.size(0), i:i + 1])
            s_out.append(torch.concat(cur_out, dim=1))
        s_out = torch.concat(s_out, dim=1).permute(1, 0, 2).contiguous()
        return s_out




class Generator(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, noise_dim, traj_len, dropout):
        super(Generator, self).__init__()
        self.cat_pos_to_dim = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU()
        )
        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )#social
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # Time series feature extraction


        self.space_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # Spatial sequence feature extraction
        self.pos_encoder = PositionalEncoding(dim)
        self.noise_dim = noise_dim

        self.hidden_to_dim = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU()
        )
        self.dim_to_pos = nn.Sequential(
            nn.Linear(dim+noise_dim+int(dim/4), dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

        self.col_to_dim = nn.Sequential(
            nn.Linear(1, int(dim/4)),
            nn.ReLU()
        )
        self.len = traj_len


    def forward(self, traj, rel_traj, target,col_label, seq_start_end):
        # Aggregation time features
        traj_cat = torch.concat([rel_traj, target], dim=-1)
        pos_emb = self.cat_pos_to_dim(traj_cat)  # n,len,dim
        t_in = self.pos_encoder(pos_emb)
        mask = nn.Transformer.generate_square_subsequent_mask(traj.size(1)).to(traj)
        t_out = self.time_encoder(t_in, mask=mask) #Time feature output
        s_out = []
        s_rel = traj.permute(1, 0, 2).contiguous()
        for se in seq_start_end:
            cur_sin = s_rel[:, se[0]:se[1], :]
            cur_sin = cur_sin.repeat(cur_sin.size(1), 1, 1)
            for i in range(se[1] - se[0]):
                cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)] = \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0),i:i + 1] - \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)]  # n*seq,agent_count,dim n=0 The position of the 0th person relative to other people seq, agent_count, dim
            t_info=t_out.permute(1, 0, 2)[:, se[0]:se[1], :].repeat(cur_sin.size(1), 1, 1)#Temporal feature repetition
            cur_sin=self.pos_to_dim(cur_sin)
            cur_sin = torch.concat((cur_sin,t_info),dim=-1).contiguous() #space time splicing
            cur_sin=self.hidden_to_dim(cur_sin)
            group_out = self.space_encoder(cur_sin)
            cur_out = []
            for i in range(se[1] - se[0]):
                cur_out.append(group_out[i * s_rel.size(0):(i + 1) * s_rel.size(0), i:i + 1])
            s_out.append(torch.concat(cur_out, dim=1))

        s_out = torch.concat(s_out, dim=1).permute(1, 0, 2).contiguous()
        #noise
        noise = torch.randn(traj.size(0), traj.size(1), self.noise_dim, requires_grad=False).to(traj)
        #col label
        colinfo = self.col_to_dim(col_label)
        # Sequence basis combined with social characteristics
        f_in=torch.concat([s_out,noise,colinfo],dim=-1)
        out = self.dim_to_pos(f_in) #Each step predicts the next relative position of the step, so the last position is not taken (the loss cannot be verified)
        return out

    def pred(self, traj, rel_traj, target, seq_start_end,length):
        speed=torch.zeros(traj.size(0),1,2)
        for i in range(length):
            if(i!=0):
                traj=torch.concat([traj,traj[:,-1,:]+speed],dim=1)
                rel_traj=torch.concat([rel_traj,speed],dim=1)
            speed=self.forward(traj, rel_traj, target, seq_start_end)[:,-1:,]
        return torch.concat([traj,traj[:,-1,:]+speed],dim=1)


class Discrimiter(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dropout):
        super(Discrimiter, self).__init__()
        self.cat_pos_to_dim = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU()
        )

        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )
        self.time_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # Time series feature extraction

        self.space_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads, batch_first=True,
                                       dropout=dropout), num_layers=depth)  # Spatial sequence feature extraction
        self.pos_encoder = PositionalEncoding(dim)
        self.hidden_to_dim = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )
        self.speed_to_dim= nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ) #velocity

        self.final= nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )


    def forward(self, traj, rel_traj, target,speed, seq_start_end):
        traj_cat = torch.concat([rel_traj, target], dim=-1)
        pos_emb = self.cat_pos_to_dim(traj_cat)  # n,len,dim
        t_in = self.pos_encoder(pos_emb)
        mask = nn.Transformer.generate_square_subsequent_mask(traj.size(1)).to(traj)
        t_out = self.time_encoder(t_in, mask=mask)  # Temporal feature output
        s_out = []
        s_rel = traj.permute(1, 0, 2).contiguous()
        for se in seq_start_end:
            cur_sin = s_rel[:, se[0]:se[1], :]
            cur_sin = cur_sin.repeat(cur_sin.size(1), 1, 1)
            for i in range(se[1] - se[0]):
                cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)] = \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0), i:i + 1] - \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(
                        0)]  # n*seq,agent_count,dim n=0 The position of the 0th person relative to other people seq, agent_count, dim
            t_info = t_out.permute(1, 0, 2)[:, se[0]:se[1], :].repeat(cur_sin.size(1), 1, 1)  # Temporal feature repetition
            cur_sin = self.pos_to_dim(cur_sin)
            cur_sin = torch.concat((cur_sin, t_info), dim=-1).contiguous()  # space time splicing
            cur_sin = self.hidden_to_dim(cur_sin)
            group_out = self.space_encoder(cur_sin)
            cur_out = []
            for i in range(se[1] - se[0]):
                cur_out.append(group_out[i * s_rel.size(0):(i + 1) * s_rel.size(0), i:i + 1])
            s_out.append(torch.concat(cur_out, dim=1))

        s_out = torch.concat(s_out, dim=1).permute(1, 0, 2).contiguous()
        speed_dim=self.speed_to_dim(speed)
        f_in=torch.concat([s_out,speed_dim],dim=-1)
        out= self.final(f_in)
        return out




class ColClassifier(nn.Module):
    def __init__(self, dim, mlp_dim, depth, heads, dropout):
        super(ColClassifier, self).__init__()
        self.posandspeed_to_dim = nn.Sequential(
            nn.Linear(4, dim),
            nn.ReLU()
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=mlp_dim, nhead=heads,
                                       batch_first=True, dropout=dropout), num_layers=depth)  # spatial aggregation

        self.pos_to_dim = nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )


    def forward(self, obs, speed_pre,speed_current, seq_start_end):
        s_rel = obs.permute(1, 0, 2).contiguous()  # seq,agent_count,dim
        s_out = []
        for se in seq_start_end:
            cur_sin = s_rel[:, se[0]:se[1], :]
            cur_sin = cur_sin.repeat(cur_sin.size(1), 1, 1)
            for i in range(se[1] - se[0]):
                cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)] = \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0),i:i + 1] - \
                    cur_sin[i * s_rel.size(0):(i + 1) * s_rel.size(0)]  # n*seq,agent_count,dim n=0 The position of the 0th person relative to other people seq, agent_count, dim
            cur_pre_speed=speed_pre.permute(1, 0, 2)[:, se[0]:se[1], :].repeat(cur_sin.size(1), 1, 1)
            cur_sin = torch.concat((cur_sin,cur_pre_speed),dim=-1).contiguous()
            cur_sin=self.posandspeed_to_dim(cur_sin)
            group_out = self.encoder(cur_sin)
            cur_out = []
            for i in range(se[1] - se[0]):
                cur_out.append(group_out[i * s_rel.size(0):(i + 1) * s_rel.size(0), i:i + 1])
            s_out.append(torch.concat(cur_out, dim=1))
        s_out = torch.concat(s_out, dim=1).permute(1, 0, 2).contiguous()
        cur_speed_dim=self.pos_to_dim(speed_current)
        s_out=torch.concat([s_out,cur_speed_dim], dim=-1)
        out = self.final(s_out)
        return out