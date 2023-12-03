import os
import numpy as np




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

#Get the trajectory information of each agent
def getAgentTrajList(dataset,delim='\t'):
    data_dir = './datasets/' + dataset + '/test/'
    #data_dir = './datasets/raw/train/biwi_eth_train.txt'
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    agent_data = []  # All data are grouped by the first column (data list after grouping by frame, each item is n* (frame ID, person ID, Pos_x, Pos_y))
    for path in all_files:
        data = read_file(path, delim)  # Directly read text data segmentation (frame ID, person ID, Pos_x, Pos_y)
        agents = np.unique(data[:, 1]).tolist()  # Everyone's ID
        for agent in agents:
            agent_data.append(data[agent == data[:, 1],2:])
    return agent_data



#Calculate average speed
def getAvgSpeed(agentList):
    speeds=[]
    for agent in agentList:
        agent=np.array(agent)
        agent_speed=(agent[1:]-agent[:-1])/0.4 #The partial velocity in the x and y directions at each moment
        speed=np.sqrt((agent_speed **2).sum(axis=-1))#speed at every moment
        speeds.append(speed.mean()) #The average speed of the agent
    return round(np.mean(speeds),3)



#Calculate average speed change
def getAvgSpeedChange(agentList):
    speedchanges = []
    for agent in agentList:
        if len(agent)>2:
            agent = np.array(agent)
            agent_speed = (agent[1:] - agent[:-1]) / 0.4  # The partial velocity in the x and y directions at each moment
            speed = np.sqrt((agent_speed ** 2).sum(axis=-1))  # speed at every moment
            speedchange=np.abs(speed[1:]-speed[:-1])/ 0.4 #acceleration at every moment
            speedchanges.append(speedchange.mean())
    return round(np.mean(speedchanges),3)

#Calculate average angle change
def getAvgAngleChange(agentList):
    angelChanges=[]
    for agent in agentList:
        if len(agent)>2:
            agent = np.array(agent)
            agent_angle = (agent[1:] - agent[:-1])
            agent_angle_norm=agent_angle/(np.expand_dims(np.linalg.norm(agent_angle,axis=-1),-1))
            angle_change=[]
            for i in range(1,len(agent_angle_norm)):
                angle_change.append(angle(agent_angle_norm[i-1],agent_angle_norm[i]))
            angelChanges.append(np.mean(angle_change))
    return round(np.mean(angelChanges)/0.4,3)

#Calculate the average vertical distance
def getAvgPDDistance(agentList):
    distances = []
    for agent in agentList:
        if len(agent) > 2:
            distance=[]
            for i in range(1,len(agent)-1):
                distance.append(get_distance_from_point_to_line(agent[i],agent[0],agent[-1]))
            distances.append(np.mean(distance))
        else:
            distances.append(0)
    return round(np.mean(distances),3)


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

    return angle_radian

#point to straight line distance
def get_distance_from_point_to_line(point, line_point1, line_point2):
    #When the coordinates of two points are the same point, the distance between the points is returned.
    if line_point1[0] == line_point2[0] and line_point1[1] == line_point2[1] :
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #Calculate the three parameters of a straight line
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #Calculate distance based on distance formula from point to straight line
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return  distance


def getDatasetPersonInfo(dataset, delim='\t'):
    data_dir = './datasets/' + dataset + '/test/'
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]

    info=[] #Result data (start frame set, end frame set, initial position set, target position set)
    for path in all_files:
        agent_data = [] # All data are grouped by the first column (data list after grouping by frame, each item is n* (frame ID, person ID, Pos_x, Pos_y))
        data = read_file(path, delim)  # Directly read text data segmentation (frame ID, person ID, Pos_x, Pos_y)
        agents = np.unique(data[:, 1]).tolist()  # Everyone's ID
        for agent in agents:
            agent_data.append(data[agent == data[:, 1], :])

        id,start,end,init,last,pos=[],[],[],[],[],[]#ID, start frame, end frame, initial position, target position, trajectory sequence
        for agent in agent_data:
            if len(agent)>2:
                if len(agent)==(agent[:, 0].max() / 10)-(agent[:,0].min()/10)+1:

                    id.append(agent[0][1])
                    start.append(agent[:,0].min()/10)
                    end.append(agent[:, 0].max() / 10)
                    init.append(agent[:2, 2:])
                    last.append(agent[-1, 2:])
                    pos.append(agent[:, 2:])

        info.append([id,start,end,init,last,pos])
    return info


def OutCsv(agentList,filename):
    with open("resources/"+filename+".csv",'w') as f:
        i=0
        f.write("ID")
        for j in range(0,len(agentList[0])):
            f.write(",x%s,y%s" % (str(j+1),str(j+2)))
        for agent in agentList:
            f.write('\n')
            i=i+1
            f.write("%s" %str(i))
            for pos in agent:
                f.write(",%s,%s" % (round(pos[0]*100,3),round(pos[1]*100,3)))
