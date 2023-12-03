import os

import torch
from torch import nn
from config import parse_args
from loader import get_data_loader, preprocessing
from loss import LossCompute
from model import Generator, Discrimiter,ColClassifier
from utils import cal_speed_num_error,saveModel,writecsv,traj_addnoise
from evalue import getAllAvgScore



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def train(args, G, D,C):
    if args.C_dict:
        C.load_state_dict(torch.load(args.C_dict))
    else:
        C.apply(init_weights)

    if args.D_dict:
        D.load_state_dict(torch.load(args.D_dict))
    else:
        D.apply(init_weights)

    if args.G_dict:
        G.load_state_dict(torch.load(args.G_dict))
    else:
        G.apply(init_weights)

    print(G)
    print(D)
    print(C)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    optimizer_C = torch.optim.Adam(C.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_step, gamma=args.lr_gamma)
    scheduler_C = torch.optim.lr_scheduler.StepLR(optimizer_C, step_size=args.lr_step, gamma=args.lr_gamma)
    train_dl = get_data_loader(args, 'train')
    lossfn = LossCompute(G, D, C, args)
    min_speed_error=100 #optimal index
    min_speed_error_epoch = 1  # Optimal index epoch
    min_all_error=[]

    iternum=0 #Number of iterations
    for i in range(args.epoch):
        G.train()
        for j, batch in enumerate(train_dl):
            iternum += 1
            batch = [tensor.cuda() for tensor in batch]
            (traj_init,tran_sfm, seq_start_end) = batch
            traj, rel_traj, target,collision = preprocessing(traj_init, seq_start_end,'train',False) #Trajectory, relative trajectory, direction
            #Train once with real data c
            optimizer_C.zero_grad()
            loss_c=lossfn.compute_ColClassifier_loss(traj, rel_traj, collision, seq_start_end)
            loss_c.backward()
            optimizer_C.step()

            # Noisy data training once c
            traj_n, rel_traj_n, target_n, collision_n=traj_addnoise(traj_init,seq_start_end)
            optimizer_C.zero_grad()
            loss_c = lossfn.compute_ColClassifier_loss(traj_n, rel_traj_n, collision_n, seq_start_end)
            loss_c.backward()
            optimizer_C.step()
            #Train once on real data D

            optimizer_D.zero_grad()
            loss_d, loss_real, loss_fake = lossfn.compute_discriminator_loss(traj, rel_traj, target,collision,  seq_start_end)
            loss_d.backward()
            optimizer_D.step()
            # Real data training once G
            optimizer_G.zero_grad()
            loss_g_all, loss_g_l2, loss_g,col_loss = lossfn.compute_generator_loss(traj, rel_traj, target,collision, seq_start_end)
            loss_g_all.backward()
            optimizer_G.step()

            print('Epoch:', i + 1, 'batch:', j)
            print('real_iter_num:',iternum)
            print("C_loss:", round(loss_c.item(), 3))
            print("D_loss_all:", round(loss_d.item(), 3),
                  "D_real:", round(loss_real, 3),
                  "D_fake:", round(loss_fake, 3))
            print("G_loss_all:", round(loss_g_all.item(), 3),
                  "G_loss:", round(loss_g, 3),
                  "G_loss_l2:",round(loss_g_l2, 3),
                  "G_col_loss:",round(col_loss, 3))
            iternum += 1
            #print("real col count:",collision_count(traj.detach().cpu().numpy()))
            traj, _, _, collision = preprocessing(tran_sfm, seq_start_end, 'train', False)
            #print("sfm col count:", collision_count(traj.detach().cpu().numpy()))
            # sfm data training once c
            optimizer_C.zero_grad()
            loss_c = lossfn.compute_ColClassifier_loss(traj, rel_traj, collision, seq_start_end)
            loss_c.backward()
            optimizer_C.step()

            # sfm data training once D
            optimizer_D.zero_grad()
            loss_d, loss_real, loss_fake = lossfn.compute_discriminator_loss(traj, rel_traj, target, collision,seq_start_end)
            loss_d.backward()
            optimizer_D.step()
            # sfm data training once G
            optimizer_G.zero_grad()
            loss_g_all, loss_g_l2, loss_g, col_loss = lossfn.compute_generator_loss(traj, rel_traj, target, collision,seq_start_end)
            loss_g_all.backward()
            optimizer_G.step()

            print('sfm_iter_num:', iternum)
            print("C_loss:", round(loss_c.item(), 3))
            print("D_loss_all:", round(loss_d.item(), 3),
                  "D_real:", round(loss_real, 3),
                  "D_fake:", round(loss_fake, 3))
            print("G_loss_all:", round(loss_g_all.item(), 3),
                  "G_loss:", round(loss_g, 3),
                  "G_loss_l2:", round(loss_g_l2, 3),
                  "G_col_loss:", round(col_loss, 3))
        scheduler_G.step()
        scheduler_D.step()
        scheduler_C.step()

        if (i + 1) % 10 == 0:
            saveModel(G, D,C, args,str(i+1))
        if args.dataset != 'raw':
            avg=getAllAvgScore(args.dataset, G, args)
            print('\033[0;32;40m\t errors:\033[0m',avg)

            if avg[1]<min_speed_error:
                min_speed_error=avg[1]
                min_speed_error_epoch=i+1
                min_all_error=avg
                saveModel(G, D,C, args, 'best')
            print('\033[0;32;40m\t min_speed_error_epoch:\033[0m', min_speed_error_epoch)
            print('\033[0;32;40m\t min_speed_error:\033[0m', min_speed_error)
            print('\033[0;32;40m\t min_all_error:\033[0m', min_all_error)


@torch.no_grad()
def evalue(args, G, type, times=1):
    G.eval()
    dl = get_data_loader(args, type)
    speed_num_error, fde = [], []
    for _ in range(times):
        real, fake = [], []
        for batch in dl:
            batch = [tensor.cuda() for tensor in batch]
            (traj, seq_start_end) = batch
            traj, rel_traj, target,col_label = preprocessing(traj, seq_start_end, 'eval')
            pred_speed = G(traj[:, :-1, :], rel_traj[:, :-1, :], target[:, :-1, :], col_label, seq_start_end)
            real.append(rel_traj[:, 1:, :]/0.4)
            fake.append(pred_speed)
        real = torch.concat(real, dim=0)  # *args.scale
        fake = torch.concat(fake, dim=0)  # *args.scale
        speed_num_error.append(round(cal_speed_num_error(real, fake).item(), 3))

    print('\033[0;32;40m\t' + type + ' ade:\033[0m', speed_num_error)
    print('\033[0;32;40m\t' + type + ' min_ade:\033[0m', min(speed_num_error))
    return min(speed_num_error)


if __name__ == '__main__':


    args = parse_args()
    G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
        args.device)
    D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    train(args, G, D,C)

    # args = parse_args()
    # args.dataset = 'univ'
    # G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # train(args, G, D, C)
    #
    # args = parse_args()
    # args.dataset = 'zara1'
    # G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # train(args, G, D, C)
    #
    # args = parse_args()
    # args.dataset = 'zara2'
    # G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # train(args, G, D, C)
    #
    # args = parse_args()
    # args.dataset = 'eth'
    # G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # train(args, G, D,C)
    #
    # args = parse_args()
    # args.dataset='hotel'
    # G = Generator(args.dim, args.mlp_dim, args.depth, args.heads, args.noise_dim, args.traj_len, args.dropout).to(
    #     args.device)
    # D = Discrimiter(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # C = ColClassifier(args.dim, args.mlp_dim, args.depth, args.heads, args.dropout).to(args.device)
    # train(args, G, D, C)

