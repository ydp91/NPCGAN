import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity

def load_agent_list_by_file(file):
    # Open CSV file
    with open('./resources/result/'+file, newline='') as csvfile:
        # Create a CSV reader
        reader = csv.reader(csvfile, delimiter=',')
        # Skip the first line
        next(reader)
        # Create an empty list to store results
        result = []
        # Iterate through each row and convert it to a Numpy array
        for row in reader:
            # Skip the first column
            data = np.array(row[1:], dtype=float)/100
            # Add the data to the results list
            data = np.reshape(data, (-1, 2))
            result.append(data)
        # Convert the results to a Numpy array
        result = np.array(result)

    # Return the result
    return result


def saveMoveInfo(g_out, x_max, x_min, y_max, y_min, args, epoch, type):
    """
    :param g_out:[person_count,seq_len,2]
    :param args:
    :return:
    """
    x_max = x_max.cpu()
    x_min = x_min.cpu()
    y_max = y_max.cpu()
    y_min = y_min.cpu()
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.out_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    g_out = g_out.view(-1, g_out.size(0), g_out.size(1), g_out.size(2))
    for i in range(g_out.size(0)):
        move = g_out[i].detach().clone().cpu()
        move[:, :, 0] = move[:, :, 0] * (x_max - x_min) + x_min
        move[:, :, 1] = move[:, :, 1] * (y_max - y_min) + y_min
        move = move.view(move.size(0), -1)
        csv = pd.DataFrame(move.numpy())
        csv.to_csv("%s/%s_%s_%s.csv" % (output_dir, type, epoch, i))


def getRealPos(base_pos, traj_rel):
    """
    :param traj_rel:[person_count,seq_len,2]
    :return:
    """
    displacement = torch.cumsum(traj_rel, dim=1)  # Accumulation of predicted trajectories
    return displacement + base_pos

def getRealPosBySpeed(base_pos, speed):
    """
    :param traj_rel:[person_count,seq_len,2]
    :return:
    """
    return speed + base_pos

# def collision_label(traj,seq_start_end):
#     '''
#     Calculate the lable of the collision 0 means no collision 1 means there is collision
#     '''
#     col_lab=[]
#     for se in seq_start_end:
#         current_group=traj[se[0]:se[1]]
#         n_agents, n_positions, n_dims = current_group.size()
#         for i in range(n_agents):
#             distance= torch.sum( (current_group[i:i+1]-current_group)**2,dim=-1)**0.5
#             distance[i]=1 #The distance from itself is set to 1, so that it does not collide with itself in subsequent settings.
#             col_lab.append((torch.min(distance,dim=0)[0]<0.4).float())
#     col_lab=torch.stack(col_lab,dim=0).unsqueeze(dim=-1)
#     return col_lab



def collision_label(traj,seq_start_end):
    '''
    Calculate whether the current speed will cause a collision if the speed of others remains unchanged.
    Calculate the lable of the collision  0 means no collision 1 means there is collision
    '''

    speed=traj[:,1:,:]-traj[:,:-1,:]

    col_lab=[]
    for se in seq_start_end:
        current_group=traj[se[0]:se[1]]
        current_group_speed=speed[se[0]:se[1]]
        n_agents = current_group.size(0)
        for i in range(n_agents):
            agent_i_pos=current_group[i:i+1,1:-1,:]+current_group_speed[i:i+1,1:,:] # Calculate the next position of the current agent based on its current speed
            agent_other_pos = current_group[:, 1:-1, :] + current_group_speed[i:i+1, :-1, :]  # Calculate the next position of other agents based on their previous speeds

            distance= torch.sum( (agent_i_pos-agent_other_pos)**2,dim=-1)**0.5
            distance[i]=1 # Set the distance to self as 1 to avoid self-collision
            col_lab.append((torch.min(distance,dim=0)[0]<0.4).float())
    col_lab=torch.stack(col_lab,dim=0).unsqueeze(dim=-1)
    return col_lab


def show(move):
    move = move.permute(1, 0, 2)
    for i in range(move.size(0)):
        plt.scatter(move[i, :, 0], move[i, :, 1])
        plt.show()


def saveModel(G, D,C, args,epoch=''):
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.model_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(G.state_dict(), "%s/G_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(D.state_dict(), "%s/D_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(C.state_dict(), "%s/C_%s_%s.pth" % (output_dir, args.dataset, epoch))


def loadModel(G, D, args):
    G.load_state_dict(args.G_dict)
    D.load_state_dict(args.D_dict)


def cal_ade(real, fake, mode="mean"):
    loss = real - fake
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

def cal_speed_num_error(real, fake, mode="mean"):
    loss = real - fake
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).mean(dim=1)
    if mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss

def cal_fde(real, fake, mode="mean"):
    loss = real[:, -1, :] - fake[:, -1, :]
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == "raw":
        return loss
    else:
        return torch.mean(loss)


def obs_to_srel(obs, seq_start_end):
    s_rel = torch.zeros_like(obs, requires_grad=False).to(obs)
    for se in seq_start_end:
        s_rel[se[0] : se[1], :, 0] = (
            obs[se[0] : se[1], :, 0]
            - (
                obs[se[0] : se[1], :, 0].min(dim=0)[0]
                + obs[se[0] : se[1], :, 0].max(dim=0)[0]
            )
            / 2
        )
        s_rel[se[0] : se[1], :, 1] = (
            obs[se[0] : se[1], :, 1]
            - (
                obs[se[0] : se[1], :, 1].min(dim=0)[0]
                + obs[se[0] : se[1], :, 1].max(dim=0)[0]
            )
            / 2
        )

    return s_rel

def angle(v1,v2):
    '''
    Calculate the angle between two unit vectors
    '''
    x=np.array(v1)
    y=np.array(v2)
    if np.isnan(x).max() or np.isnan(y).max():
        return 0


    # Compute the dot product of two vectors
    dot_value=x.dot(y)

    # Calculate the cos value of the included angle：
    cos_theta=dot_value
    if cos_theta>1:
        cos_theta=1
    if cos_theta < -1:
        cos_theta = -1

    # Find the included angle (in radians)：
    angle_radian=np.arccos(cos_theta)

    # Convert to angle value：
    angle_value=angle_radian*180/np.pi
    return angle_value


def writecsv(path,info):
    with open(path, 'a+') as f:
        f.write(','+ str(info))


#Trajectory plus random noise
def traj_addnoise(traj, seq_start_end):
    traj_rot = traj

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
                angle4 = angle(target[i][j-1],target[i][j+3])#Change the angle in the next,next direction
            if not (angle1>20 and angle2>20 and angle3>20 and angle4>20): #If the orientation of the subsequent positions changes, it is considered that the direction has changed.
                target[i][j]=target[i][j-1]
    target=target.to(mean)
    ################################################


    #target=torch.concat((target,target[:,-1:,:]),dim=1)
    collision=collision_label(traj_rot, seq_start_end)
    return traj_rot[:,1:,:], rel_traj, target,collision#absolute position, relative motion, direction


def mse(imageA, imageB):
    # Calculate the Mean Squared Error (MSE) similarity between two images
    # Note: Both images must have the same dimensions as the operation is based on corresponding pixels in the images
    # Subtract corresponding pixels and accumulate the result
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # Normalize the error
    err /= float(imageA.shape[0] * imageA.shape[1])

    # Return the result; a smaller value is better as it indicates higher similarity between the two images
    return err


def image_similarity(img1, img2):
    """
    """
    # load image
    image1 = cv2.imdecode(np.fromfile(img1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image2 = cv2.imdecode(np.fromfile(img2, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    if h1 != h2 or w1 != w2:
        image2 = cv2.resize(image2, (w1, h1))
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(gray1, gray2, full=True)
    # diff = (diff * 255).astype("uint8")
    return score

