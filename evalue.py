import numpy as np
from utils import collision_label
from  evaluefunction import *
from model import Generator
from config import parse_args
import torch

# Generate trajectory
def generateTrajByDataset(dsname,G,args):
    # Load model
    G.eval()
    # Load data
    datainfo = getDatasetPersonInfo(dsname)
    agent_list=[]
    for info in datainfo:
        id,start,end,init,dest,pos=info# id, start frame, end frame, initial position, destination, trajectory
        dict={}
        for i in range(len(id)):
            dict[id[i]]=init[i] # Dictionary of id and trajectory

        current = min(start)+1 # Current frame ID to generate

        while current<max(end):
            current += 1  # Current frame ID to generate
            current_id = []  # IDs to be fed into the generator for the current frame
            traj, last = [], []  # Trajectories and targets of agents to generate speeds for
            model_traj = []  # Trajectories to feed into the model
            traj_rel = []  # Relative trajectories

            for i in range(len(id)):  # Iterate over each person
                if start[i] + 1 < current and end[i] >= current:  # Check if agent needs to generate info at current frame
                    current_id.append(id[i])  # id
                    traj.append(dict[id[i]])  # Trajectory
                    last.append(dest[i])  # Destination
            if len(current_id)==0:
                continue

            minseqlen= min([len(t) for t in traj ])  # Shortest trajectory in the current group
            for agent_traj in traj:
                agent_traj=np.array(agent_traj)
                model_traj.append(agent_traj[-minseqlen:,:])
                rel=agent_traj[1:]-agent_traj[:-1]
                if len(rel)<minseqlen:
                    rel=np.concatenate((np.zeros((1,2)),rel),axis=0)
                traj_rel.append(rel[-minseqlen:,:])
            last=np.expand_dims(np.stack(last,axis=0),axis=1)
            model_traj = np.stack(model_traj, axis=0)  # Trajectory
            traj_rel = np.stack(traj_rel, axis=0)  # Relative trajectory

            target=last-model_traj

            target = target / (np.expand_dims(np.linalg.norm(target, axis=-1), -1))  # Direction
            target[np.isnan(target)] = 0  # Direction


            real_traj=[]
            for i in range(len(id)):  # Iterate over each person
                if start[i] + 1 < current and end[i] >= current:  # Check if agent needs to generate info at current frame
                    endpos = int(current - start[i])
                    startpos = int(endpos - minseqlen)
                    real_traj.append(torch.from_numpy(pos[i][startpos:endpos + 1]).to(args.device))

            try:
                real_traj=torch.stack(real_traj,dim=0)
            except:
                real_traj=real_traj

            model_traj=torch.from_numpy(model_traj).to(dtype=torch.float32).to(args.device)
            traj_rel=torch.from_numpy(traj_rel).to(dtype=torch.float32).to(args.device)
            target = torch.from_numpy(target).to(dtype=torch.float32).to(args.device)
            seq=torch.Tensor([[0,len(current_id)]]).to(dtype=torch.int32).to(args.device)
            col_label=collision_label(real_traj,seq)
            pre_col = torch.zeros([col_label.size(0), 1, 1]).to(args.device)
            col_label = torch.concat([pre_col, col_label], dim=1)
            pred_speed=G(model_traj,traj_rel,target,col_label,seq)[:,-1,:].cpu()*0.4

            for i in range(len(current_id)):
                pred_pos=dict[current_id[i]][-1]+pred_speed[i].detach().numpy()
                pred_pos=np.expand_dims(pred_pos, axis=0)
                dict[current_id[i]]=np.concatenate([dict[current_id[i]],pred_pos],axis=0)

        for k,v in dict.items():
            agent_list.append(v)
    return agent_list


# Calculate the average error of four aspects in the dataset
def getAllAvgScore(ds, G, args):
    avg_speed_real = getAvgSpeed(getAgentTrajList(ds))
    avg_speedChange_real = getAvgSpeedChange(getAgentTrajList(ds))
    avg_angleChange_real = getAvgAngleChange(getAgentTrajList(ds))
    avg_pddistance_real = getAvgPDDistance(getAgentTrajList(ds))
    agentlist = generateTrajByDataset(ds, G, args)
    avg_speed_fake = getAvgSpeed(agentlist)
    avg_speedChange_fake = getAvgSpeedChange(agentlist)
    avg_angleChange_fake = getAvgAngleChange(agentlist)
    avg_pddistance_fake = getAvgPDDistance(agentlist)
    print("real:",avg_pddistance_real,avg_speed_real,avg_speedChange_real,avg_angleChange_real)
    print("fake:",avg_pddistance_fake, avg_speed_fake, avg_speedChange_fake, avg_angleChange_fake)

    return abs(avg_pddistance_fake - avg_pddistance_real) / avg_pddistance_real,\
           abs(avg_speed_fake - avg_speed_real) / avg_speed_real,\
           abs(avg_speedChange_fake - avg_speedChange_real) / avg_speedChange_real, \
           abs(avg_angleChange_fake - avg_angleChange_real) / avg_angleChange_real

# Generate trajectories from file (for collision experiment trajectories)
def generateTrajbyFile(path, G,args):
    init_pos = np.loadtxt(path,
                     delimiter=",", dtype=np.float32)/100.
    start =init_pos[:,:2]
    dest =torch.from_numpy(init_pos[:,2:]).unsqueeze(dim=1).to(args.device) # Target position

    target_ext=init_pos[:,2:]-init_pos[:,:2]
    target_ext = target_ext / (np.expand_dims(np.linalg.norm(target_ext, axis=-1), -1))   # Direction
    second = start + 1.189*target_ext*0.4 # Initial speed of 1.189 (average speed in the dataset)

    pos=np.stack([start,second],axis=1)
    target_start=np.zeros_like(start)

    traj_rel=np.stack([target_start, 1.189*target_ext*0.4],axis=1)# Initial speed of 1.189 (average speed in the dataset)
    pos=torch.from_numpy(pos).to(args.device)
    traj_rel=torch.from_numpy(traj_rel).to(args.device)
    target = dest-pos
    mean = torch.norm(target, dim=-1).unsqueeze(-1)
    target = target / mean
    seq = torch.Tensor([[0, len(start)]]).to(dtype=torch.int32).to(args.device)
    col_label=torch.zeros([target.size(0),2,1]).to(args.device)
    G.eval()

    maxLen = 50
    for i in range(30):
        if pos.size(1) < maxLen:
            pred_speed = G(pos, traj_rel, target,col_label, seq)[:, -1:, :] * 0.4
        else:
            pred_speed = G(pos[:, -maxLen:, :], traj_rel[:, -maxLen:, :], target[:, -maxLen:, :],col_label[:, -maxLen:, :], seq)[:, -1:, :] * 0.4
        pos=torch.concat((pos,pos[:,-1:,:]+pred_speed),dim=1)
        traj_rel=torch.concat((traj_rel,pred_speed),dim=1)
        target = dest - pos
        mean = torch.norm(target, dim=-1).unsqueeze(-1)
        target = target / mean
        col_label = torch.concat((col_label, col_label[:, -1:, :]), dim=1)
    return pos


# Calculate the average error of four aspects in the dataset
def getAllAvgScoreByTraj(ds, traj):
    avg_speed_real = getAvgSpeed(getAgentTrajList(ds))
    avg_speedChange_real = getAvgSpeedChange(getAgentTrajList(ds))
    avg_angleChange_real = getAvgAngleChange(getAgentTrajList(ds))
    avg_pddistance_real = getAvgPDDistance(getAgentTrajList(ds))
    agentlist = traj
    avg_speed_fake = getAvgSpeed(agentlist)
    avg_speedChange_fake = getAvgSpeedChange(agentlist)
    avg_angleChange_fake = getAvgAngleChange(agentlist)
    avg_pddistance_fake = getAvgPDDistance(agentlist)
    print("real:",avg_pddistance_real,avg_speed_real,avg_speedChange_real,avg_angleChange_real)
    print("fake:",avg_pddistance_fake, avg_speed_fake, avg_speedChange_fake, avg_angleChange_fake)

    return abs(avg_pddistance_fake - avg_pddistance_real) / avg_pddistance_real,\
           abs(avg_speed_fake - avg_speed_real) / avg_speed_real,\
           abs(avg_speedChange_fake - avg_speedChange_real) / avg_speedChange_real, \
           abs(avg_angleChange_fake - avg_angleChange_real) / avg_angleChange_real



# Collision count calculation
def collision_count(traj,threshold=0.4):
    num = 0
    n_agents, n_positions, n_dims = traj.shape
    for i in range(n_agents):
        for j in range(n_positions):
            if traj[i, j, 0] <= 20 and traj[i, j, 1] <= 20 and traj[i, j, 0] >= 0 and traj[i, j, 1] >= 0:
                for k in range(i + 1, n_agents):
                    if traj[k, j, 0] <= 20 and traj[k, j, 1] <= 20 and traj[k, j, 0] >= 0 and traj[k, j, 1] >= 0:
                        distance = np.linalg.norm(traj[i, j] - traj[k, j])
                        if distance < threshold:
                            num += 1
    return num


def generateTrajExperience(path, G, args):
    init_pos = np.loadtxt(path,
                          delimiter=",", dtype=np.float32) / 100.
    if init_pos.ndim==1:
        init_pos=np.expand_dims(init_pos,axis=0)
    start = init_pos[:, :2]
    # The following lines are commented out but they are for loading different target positions
    # dest1 = torch.from_numpy(init_pos[:, 2:4]).unsqueeze(dim=1).to(args.device)  # Target position
    # dest2 = torch.from_numpy(init_pos[:, 4:6]).unsqueeze(dim=1).to(args.device)  # Target position
    # dest3 = torch.from_numpy(init_pos[:, 6:8]).unsqueeze(dim=1).to(args.device)  # Target position
    # dest4 = torch.from_numpy(init_pos[:, 10:12]).unsqueeze(dim=1).to(args.device)  # Target position
    # dest5 = torch.from_numpy(init_pos[:, 12:14]).unsqueeze(dim=1).to(args.device)  # Target position
    dest = init_pos[:, 2:14]  # Target positions

    target_ext = dest[:, :2] - start
    target_ext = target_ext / (
        np.expand_dims(np.linalg.norm(target_ext, axis=-1), -1))  # Direction of the first target position
    second = start + 1.189 * target_ext * 0.4  # Initial speed of 1.189 (average speed in the dataset) for the second point position

    pos = np.stack([start, second], axis=1)  # Positions of the first two points
    target_start = np.zeros_like(start)

    traj_rel = np.stack([target_start, 1.189 * target_ext * 0.4],
                        axis=1)  # Initial speed of 1.189 (average speed in the dataset), relative positions of the first two points
    pos = torch.from_numpy(pos).to(args.device)  # Initial input positions
    traj_rel = torch.from_numpy(traj_rel).to(args.device)  # Initial input relative positions
    target = torch.from_numpy(target_ext).unsqueeze(dim=1).to(args.device)  # Initial input direction
    target = torch.concat([target, target], dim=1)  # Initial input direction
    seq = torch.Tensor([[0, len(start)]]).to(dtype=torch.int32).to(
        args.device)  # Group start and end IDs, only one group
    col_label = torch.zeros([target.size(0), 2, 1]).to(args.device)  # Collision labels all set to non-collision
    G.eval()

    d=[0,0,0]
    maxLen=3
    for i in range(50):
        if pos.size(1) < maxLen:
            pred_speed = G(pos, traj_rel, target, col_label, seq)
        else:
            pred_speed = G(pos[:, -maxLen:, :], traj_rel[:, -maxLen:, :], target[:, -maxLen:, :], col_label[:, -maxLen:, :], seq)
        pred_speed=pred_speed[:, -1:,:] * 0.4


        for j in range(pred_speed.size(0)):
            if torch.norm(pred_speed[j])<0.2:
                pred_speed[j]=0.2/torch.norm(pred_speed[j])*pred_speed[j] # Speed compensation for low speed

        pos = torch.concat((pos, pos[:, -1:, :] + pred_speed), dim=1)
        cur_pos=pos[:,-1,:]
        cur_dest=[]
        for j in range(cur_pos.size(0)):
            if torch.norm(cur_pos[j]-torch.from_numpy(dest[j,d[j]*2:d[j]*2+2]).to(args.device))<1 and d[j]<4:
                d[j]=d[j]+1
                # next_target=torch.from_numpy(dest[j, d[j] * 2:d[j] * 2 + 2]).to(args.device)-cur_pos[j]
                # next_target=next_target/torch.norm(next_target, dim=-1).unsqueeze(-1)
                # pred_speed[j]=(torch.sum(pred_speed[j]**2,dim=-1)**0.5).unsqueeze(1)*next_target
            cur_dest.append(torch.from_numpy(dest[j,d[j]*2:d[j]*2+2]).to(args.device))
        cur_dest=torch.stack(cur_dest,dim=0)
        cur_target= cur_dest-cur_pos
        mean = torch.norm(cur_target, dim=-1).unsqueeze(-1)
        cur_target=(cur_target/mean).unsqueeze(1)
        target=torch.concat([target,cur_target],dim=1)

        traj_rel = torch.concat((traj_rel, pred_speed), dim=1)



        col_label = torch.concat((col_label, col_label[:, -1:, :]), dim=1)
    return pos


if __name__ == '__main__':

    args=parse_args()
    G=Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
        args.device)
    G.load_state_dict(torch.load(args.G_dict))
    print(getAllAvgScore('eth',G,args))
    # OutCsv(generateTrajByDataset('zara2',G,args),'zara2_our')
    #OutCsv(getAgentTrajList('zara2'),'zara2_real')
    #
    # traj=generateTrajbyFile("resources/collision exp pos and target.csv",G, args).cpu().detach().numpy()
    # OutCsv(traj,'col') # Output Generated Trajectories
    # #
    # print(collision_count(traj))
    # for i in range(5):
    #     OutCsv(generateTrajExperience('resources/maze exp1.csv', G, args).cpu().detach().numpy(),'maze'+str(i))








