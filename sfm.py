import torch
import numpy as np
from evalue import getDatasetPersonInfo,getAllAvgScoreByTraj
from evaluefunction import OutCsv
def social_force_update(locations, velocities, target_locations, time_step=0.1, alpha=200.0, beta=0.08, lambda_=0.35, desired_speed=2.2):
    """
    Update function for the Social Force Model using PyTorch.

    Args:
    locations (torch.Tensor): A tensor of shape [n, 2] representing the current locations (meters) of the agents.
    velocities (torch.Tensor): A tensor of shape [n, 2] representing the current velocities (m/s) of the agents.
    target_locations (torch.Tensor): A tensor of shape [n, 2] representing the target locations (meters) of the agents.
    time_step (float, optional): The time step for updating the locations and velocities (seconds). Default is 0.1.
    alpha (float, optional): The repulsive force strength (Newtons) between agents. Default is 200.0.
    beta (float, optional): The distance (meters) at which the repulsive force is felt. Default is 0.08.
    lambda_ (float, optional): A tuning parameter to control the agent's avoidance force. Default is 0.35.
    desired_speed (float, optional): The desired speed (m/s) of the agents. Default is 1.3.

    Returns:
    updated_locations (torch.Tensor): A tensor of shape [n, 2] representing the updated locations (meters) of the agents.
    updated_velocities (torch.Tensor): A tensor of shape [n, 2] representing the updated velocities (m/s) of the agents.
    """

    # Get the number of agents
    n = locations.size(0)

    # Calculate the desired directions
    desired_directions = (target_locations - locations) / (torch.norm(target_locations - locations, dim=-1, keepdim=True) + 1e-9)

    # Calculate the desired velocities
    desired_velocities = desired_speed * desired_directions

    # Calculate the difference in desired velocities and current velocities
    dv = desired_velocities - velocities

    # Calculate the social force
    social_force = torch.zeros_like(locations)
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate the difference in locations between agent i and agent j
                diff_loc = locations[i] - locations[j]
                # Calculate the distance between agent i and agent j
                distance = torch.norm(diff_loc)
                # Normalize the difference in locations
                normalized_diff_loc = diff_loc / (distance + 1e-9)
                # Calculate the social force between agent i and agent j
                social_force_ij = alpha * torch.exp(-distance / beta) * (normalized_diff_loc - lambda_ * velocities[i])
                # Add the social force to the total social force acting on agent i
                social_force[i] += social_force_ij

    # Update the velocities by adding the difference in velocities and social force, scaled by the time step
    updated_velocities = velocities + time_step * (dv + social_force)

    # Update the locations by adding the updated velocities, scaled by the time step
    updated_locations = locations + time_step * updated_velocities

    return updated_locations, updated_velocities



def get_collision_times(locations, collision_threshold=0.4):
    n = locations.size(0)
    collision_times = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (torch.norm(locations[i] - locations[j]) < collision_threshold):
                collision_times += 1
    return collision_times


def social_force_collision(dsname,dspeed=2.2):
    # The data set time step is 0.4
    time_step = 0.4
    dataset_collision_times = []
    dataset_sfm_trajs = []
    datainfo = getDatasetPersonInfo(dsname)
    # print(len(datainfo))
    for info in datainfo:
        # region DATA VISULIZATION
        # Create the figure
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # plt.ion()
        # endregion


        id, start, end, init, dest,pos = info
        # print(init)


        traj_dict = {}

        velocities_dict = {}
        for i in range(len(id)):
            traj_dict[id[i]] = init[i]
            velocities_dict[id[i]] = (init[i][1] - init[i][0]) / time_step
        # print(velocities_dict)


        collision_times = 0

        first_frame = min(start)
        cur_frame = first_frame

        last_frame = max(end)

        while cur_frame < last_frame - 1:
            cur_frame += 1
            curt_ids = []
            cur_trajs = []
            cur_velocities = []
            cur_targets = []


            for i in range(len(id)):
                if start[i] < cur_frame and end[i] > cur_frame:
                    curt_ids.append(id[i])  # id
                    cur_trajs.append(traj_dict[id[i]])
                    cur_velocities.append(velocities_dict[id[i]])
                    cur_targets.append(dest[i])
            if len(curt_ids) == 0:
                continue

            locations = [cur_trajs[i][-1] for i in range(len(cur_trajs))]
            locations = np.stack(locations, axis=0)
            locations = torch.from_numpy(locations).to(dtype=torch.float32)
            # print("location: " + str(locations.shape))

            # region DATA VISULIZATION
            # Draw the updated frame
            # ax1.cla()
            # ax1.scatter(locations[:,0], locations[:,1], c='r', marker='o', label='SFM')
            # ax1.set_title('SFM'+str(locations.shape[0]))
            # ax1.set_xlabel('X-axis')
            # ax1.set_ylabel('Y-axis')
            # ax1.legend()

            # ax2.cla()
            # ax2.scatter(locations[:,0], locations[:,1], c='b', marker='o', label='Real Data')
            # ax2.set_title('Sample Data'+str(locations.shape[0]))
            # ax2.set_xlabel('X-axis')
            # ax2.set_ylabel('Y-axis')
            # ax2.legend()
            # endregion


            # velocities = [(cur_trajs[i][-1]-cur_trajs[i][-2])/time_step for i in range(len(cur_trajs))]
            velocities = np.stack(cur_velocities, axis=0)
            velocities = torch.from_numpy(velocities).to(dtype=torch.float32)
            # print("Velocity: " + str(velocities.shape))


            target_locations = np.stack(cur_targets, axis=0)
            target_locations = torch.from_numpy(target_locations).to(dtype=torch.float32)
            # print("target_locations: " + str(target_locations.shape))


            collision_times += get_collision_times(locations, collision_threshold=0.4)


            sfm_multi_times = 5
            for _ in range(sfm_multi_times):
                locations, velocities = social_force_update(locations, velocities, target_locations,
                                                            time_step=time_step / sfm_multi_times,desired_speed=dspeed)
                # region DATA VISULIZATION
                # ax1.cla()
                # ax1.scatter(locations[:,0], locations[:,1], c='r', marker='o', label='SFM')
                # ax1.set_title('SFM'+str(locations.shape[0]))
                # ax1.set_xlabel('X-axis')
                # ax1.set_ylabel('Y-axis')
                # ax1.legend()
                # plt.pause(time_step/sfm_multi_times)
                # endregion


            if (cur_frame == last_frame):
                collision_times += get_collision_times(locations, collision_threshold=0.4)


            for i, cur_id in enumerate(curt_ids):
                traj_dict[cur_id] = np.concatenate([traj_dict[cur_id], locations[i].unsqueeze(0).numpy()], axis=0)
                velocities_dict[cur_id] = velocities[i].numpy()
                # print("dict shape: " + str(dict[cur_id].shape))
                # print("loc shape: " + str(locations[i].unsqueeze(0).numpy().shape))

        # region DATA VISULIZATION
        # plt.ioff()
        # plt.show()
        # endregion

        print(collision_times)
        dataset_collision_times.append(collision_times)
        dataset_sfm_trajs.append(list(traj_dict.values()))

    return dataset_collision_times, dataset_sfm_trajs


