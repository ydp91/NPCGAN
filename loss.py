import torch
import numpy as np
import random


class LossCompute:
    def __init__(self, netG, netD,netC, args):
        self.netG = netG
        self.netD = netD
        self.netC = netC
        self.device = args.device
        self.bce = torch.nn.BCELoss()
        self.l2_weight = args.l2_weight

    def compute_generator_loss(self, traj, rel_traj, target,col_label, seq_start_end):
        l2=[]
        for i in range(5):
            pred_speed = self.netG(traj[:,:-1,:], rel_traj[:,:-1,:], target[:,:-1,:],col_label, seq_start_end)#col Collision label, because it controls whether a collision occurs at the position of the next moment, so use [:,1:,0]
            loss_l2 =  self.l2_loss(rel_traj[:, 1:, :]/0.4,pred_speed,'')
            l2.append(loss_l2)
        l2=torch.stack(l2,dim=-1)
        l2=torch.min(l2,dim=-1)[0]
        l2_loss_sum = l2.mean()

        scores_fake = self.netD(traj[:, :-1, :], rel_traj[:, :-1, :], target[:, :-1, :], pred_speed, seq_start_end)
        discriminator_loss = self.gan_g_loss(scores_fake)

        col_fake=self.netC(traj[:,:-1,:],rel_traj[:, :-1, :] / 0.4,pred_speed, seq_start_end)
        col_loss=self.gan_c_loss(col_fake,col_label)

        return self.l2_weight * l2_loss_sum + discriminator_loss + col_loss, l2_loss_sum.item(), discriminator_loss.item(), col_loss.item()

    def compute_discriminator_loss(self, traj, rel_traj, target,col_label, seq_start_end):
        pred_speed = self.netG(traj[:, :-1, :], rel_traj[:, :-1, :], target[:, :-1, :],col_label, seq_start_end)
        scores_fake = self.netD(traj[:, :-1, :], rel_traj[:, :-1, :], target[:, :-1, :], pred_speed,seq_start_end)
        scores_real = self.netD(traj[:, :-1, :], rel_traj[:, :-1, :], target[:, :-1, :],rel_traj[:, 1:, :] / 0.4, seq_start_end)
        loss_real, loss_fake = self.gan_d_loss(scores_fake, scores_real)  # BCEloss
        return loss_real + loss_fake , loss_real.item(), loss_fake.item()

    def compute_ColClassifier_loss(self, traj, rel_traj, col_label, seq_start_end):
        col_fake=self.netC(traj[:, :-1, :],rel_traj[:, :-1, :] / 0.4,rel_traj[:, 1:, :] / 0.4,seq_start_end)
        col_loss=self.gan_c_loss(col_fake,col_label)
        return col_loss

    def l2_loss(self, pred_traj, pred_traj_gt, mode='mean'):
            loss = (pred_traj_gt - pred_traj) ** 2
            if mode == 'sum':
                return torch.sum(loss)
            elif mode == 'mean':
                return loss.sum(dim=2).mean(dim=1)
            elif mode == 'raw':
                return loss.sum(dim=2).sum(dim=1)
            else:
                return loss.sum(dim=2)

    def gan_g_loss(self, scores_fake):
        y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
        return self.bce(scores_fake, y_fake)

    def gan_d_loss(self, scores_fake, scores_real):
        y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
        y_fake = torch.ones_like(scores_fake) * random.uniform(0, 0.3)
        loss_real = self.bce(scores_real, y_real)
        loss_fake = self.bce(scores_fake, y_fake)
        return loss_real, loss_fake

    def gan_c_loss(self,col_score,col_label):
        return self.bce(col_score, col_label)



