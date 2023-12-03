import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import angle,collision_label
from config import parse_args
from sfmloader import gen_sfm_traj




def seq_collate(data):
    traj_list = [seq[0] for seq in data]
    _len = [len(seq) for seq in traj_list]  # Number of people in each group
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # start position, end position
    traj = torch.cat(traj_list, dim=0).permute(0, 2, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        traj, seq_start_end,
    ]  # input format:  batch,seq_len, input_size
    # Position, start index and end index of each group

    return tuple(out)

def train_seq_collate(data):

    traj_list = [seq[0] for seq in data]
    _len = [len(seq) for seq in traj_list]  # Number of people in each group
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # start position, end position
    seq_start_end = torch.LongTensor(seq_start_end)
    traj_sfm_list = [seq[1] for seq in data]

    traj_list = torch.cat(traj_list, dim=0)
    traj_sfm_list= torch.cat(traj_sfm_list, dim=0)
    out=[traj_list,traj_sfm_list,seq_start_end]
    return tuple(out)


def min_distance(traj):
    '''
    Calculate the lable of the collision (1 means no collision 0 means there is collision)
    '''
    for j in range(len(traj)-1):
        distance = ((traj[j:j + 1] - traj[j + 1:]) ** 2).sum(axis=1) ** 0.5
        min = distance.min()
        if min<0.4:
            return False
    return True

# Read file as numpy
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, traj_len=8,dataset='', delim='\t', type='train'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = './datasets/'+dataset+'/' + type + '/'
        self.traj_len = traj_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []

        # Extract all motion sequence groups (only consider all people who appear in all current consecutive frames, and the minimum number of people is greater than the position where the number of people to be considered is considered)
        for path in all_files:
            data = read_file(path, delim)  # Directly read text data segmentation (frame ID, person ID, Pos_x, Pos_y)
            frames = np.unique(data[:, 0]).tolist()  # Deduplicate by first column (get all frame IDs)
            frame_data = []  # All data are grouped by the first column (data list after grouping by frame, each item is n* (frame ID, person ID, Pos_x, Pos_y))
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(len(frames) - self.traj_len + 1 )#Number of sequences

            for idx in range(0, num_sequences  + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.traj_len], axis=0)  # Splice all the sequences of this group of motions
                pre_frame_data=np.zeros((1,4))#Last frame running information
                if idx!=0:
                    pre_frame_data = frame_data[idx - 1]

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # IDs of all persons involved in this group movement
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.traj_len+1))  # The absolute position of all people in this sequence, 2, full length (1+8) (the first position is used to store the last position)

                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):  # Iterate through all the people in this group of sports
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]  # All current movement status of the person
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # The current person’s movement start frame number
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # End frame number of the current person's group movement
                    if pad_end - pad_front != self.traj_len:  # Only continue execution if the person exists in all frames
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # Leave only the location information and transpose it
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front+1:pad_end+1] = curr_ped_seq  # Real coordinates, empty the first position (storage the previous moment position)
                    curr_ped_pre=pre_frame_data[pre_frame_data[:,1]==ped_id,:] #The current person's position in the previous frame (may not exist)
                    if len(curr_ped_pre)==1:
                        curr_seq[_idx, :, 0]=curr_ped_pre[0,2:] # Assign the current person’s position in the previous frame
                    else:
                        curr_seq[_idx, :, 0]=curr_seq[_idx, :, 1]#If the last position of the previous frame does not exist, use the last position of the current frame as the last position of the previous frame

                    num_peds_considered += 1

                if num_peds_considered > 0:  # If the number of people in the group for all frames is greater than the minimum number of people to be considered
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])



        self.num_seq = len(seq_list)  # Get all motion sequence groups (only consider all people appearing in all current frames, and the minimum number of people is greater than the position where the number of people to be considered is considered)
        seq_list = np.concatenate(seq_list, axis=0)  # All motion sequences

        # Convert numpy -> Torch Tensor
        self.traj = torch.from_numpy(seq_list).type(torch.float)  # All sequences
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  # The index position where each group of athletes starts
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]  # Starting index of each group of people

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.traj[start:end, :]
        ]
        return out

class TrainTrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, traj_len=8,dataset=''
    ):
        """
        """
        super(TrainTrajectoryDataset, self).__init__()

        self.data_dir = './datasets/' + dataset + '/' + str(traj_len) + '/'
        self.traj_len = traj_len

        self.traj=torch.load(self.data_dir+ 'real.pt' )
        self.seq_start_end = torch.load(self.data_dir + 'seq_start_end.pt')
        self.sfm = torch.load(self.data_dir + 'sfm.pt')

    def __len__(self):
        return self.seq_start_end.size(0)

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return self.traj[start:end,:,:],self.sfm[start:end,:,:]

def get_data_loader(args, type):
    if not type=='train':
        dset = TrajectoryDataset(
            traj_len=args.traj_len,
            dataset=args.dataset,
            delim=args.delim,
            type=type)

        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=seq_collate)
        return loader
    else:
        dset=TrainTrajectoryDataset(traj_len=args.traj_len,dataset=args.dataset)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_seq_collate)
        return loader


def rotation_2d_torch(x, theta, origin=None):
    '''旋转'''
    if origin is None:
        origin = x.reshape(-1, 2).mean(dim=0)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x  # Position after rotation


def preprocessing(traj, seq_start_end, type='train',withnoise=False):
    traj_rot = torch.zeros_like(traj).to(traj)
    if type == 'train':  # Random rotation of training data
        for se in seq_start_end:
            theta = torch.rand(1).to(traj) * np.pi * 2
            traj_rot[se[0]:se[1]] = rotation_2d_torch(traj[se[0]:se[1]], theta)
    else:
        traj_rot = traj
    if withnoise:
        noise_traj = torch.randn(traj_rot.size(0), traj_rot.size(1)-1, 2, requires_grad=False).to(traj_rot) * 0.05
        traj_rot[:,1:,:] = traj_rot[:,1:,:]  + noise_traj
    rel_traj = traj_rot[:, 1:, ] - traj_rot[:, :-1, ]  # relative position


    target = traj_rot[:,-1:,:]-traj_rot[:,1:,:]   # direction
    mean = torch.norm(target, dim=-1).unsqueeze(-1)
    target=target/mean
    target[torch.isnan(target)] = 0.
    #########################Used to adjust the target direction (set the target directly or set it based on the following three positions)#########################
    target=target.cpu()
    for i in range(target.size(0)):
        if torch.isnan(target[i][0][0]) and torch.isnan(target[i][0][1]):
            target[i][0]=0.
        for j in range(1,target.size(1)):
            if torch.isnan(target[i][j][0]) and torch.isnan(target[i][j][1]):
                target[i][j] = target[i][j - 1]
                continue

            angle1,angle2,angle3,angle4=180,180,180,180
            angle1 = angle(target[i][j-1],target[i][j]) #Current direction change angle
            if j+1<target.size(1):
                angle2 = angle(target[i][j-1],target[i][j+1])#Next direction change angle
            if j+2<target.size(1):
                angle3 = angle(target[i][j-1],target[i][j+2])#Change the angle in the next direction
            if j+3<target.size(1):
                angle4 = angle(target[i][j-1],target[i][j+3])#Change the angle in the next next direction
            if not (angle1>20 and angle2>20 and angle3>20 and angle4>20): #If the orientation of the subsequent positions changes, it is considered that the direction has changed.
                target[i][j]=target[i][j-1]
    target=target.to(mean)
    ################################################


    #target=torch.concat((target,target[:,-1:,:]),dim=1)
    collision=collision_label(traj_rot, seq_start_end)
    return traj_rot[:,1:,:], rel_traj, target,collision# absolute position, relative motion, direction





def init_train_data(ds_name,sfmspeed=2.2):
    # Construct training set folder
    args = parse_args()
    dset = TrajectoryDataset(
        traj_len=args.traj_len,
        dataset=ds_name,
        delim=args.delim,
        type='train')
    len = dset.__len__()
    list = []
    _len = []
    for i in range(len):
        item = dset.__getitem__(i)[0]
        list.append(dset.__getitem__(i)[0])
        _len.append(item.size(0))

    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]  # start position, end position
    traj = torch.cat(list, dim=0).permute(0, 2, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    data_dir = './datasets/' + ds_name + '/' + str(args.traj_len) + '/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    torch.save(traj, data_dir + 'real.pt')
    torch.save(seq_start_end, data_dir + 'seq_start_end.pt')
    collision = collision_label(traj, seq_start_end)
    sfm = []
    speed = (traj[:, 1:, ] - traj[:, :-1, ]) / 0.4
    speed = torch.concat([speed, speed[:, -1:, :]], dim=1)
    for i, se in enumerate(seq_start_end):
        if collision[se[0]:se[1]].max() < 1:
            sfm.append(traj[se[0]:se[1]])
        else:
            sfm.append(gen_sfm_traj(traj[se[0]:se[1]], speed[se[0]:se[1]],sfmspeed))
        print(i, "/", len,"penson num:",se[1]-se[0])
    sfm = torch.concat(sfm, dim=0)
    torch.save(sfm, data_dir + 'sfm.pt')

