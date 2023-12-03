from sfm import social_force_update
import torch



def gen_sfm_traj(traj, speed,desspeed=2.2):
    # Initialize the SFM parameters
    positions = traj[:, 0, :]
    velocities = speed[:, 0, :]
    target_locations = traj[:, -1, :]
    seq_len = traj.shape[1]

    sfm_tarj = positions.unsqueeze(1)

    iterate_scale = 5
    sfm_timestep = 0.4 / iterate_scale
    for i in range(iterate_scale*(seq_len-1)):
        positions, velocities = social_force_update(positions, velocities, target_locations, time_step=sfm_timestep,desired_speed=desspeed)
        if ((i+1) % iterate_scale == 0):
            # Save the positions every iterate_scale timesteps in oreder to keep the same shape between sfm_tarj and tarj
            sfm_tarj = torch.concat([sfm_tarj, positions.unsqueeze(1)], dim=1)
    return sfm_tarj


def gen_sfm_traj_batch(traj, speed,collision,seq_start_end):
    list=[]
    for se in seq_start_end:
        if collision[se[0]:se[1]].max()<1:
            list.append(traj[se[0]:se[1]])
        else:
            list.append(gen_sfm_traj( traj[se[0]:se[1]],speed[se[0]:se[1]]))
    return torch.concat(list,dim=0)


