import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim
from numpy import random

from datasets import Dataset

import torch.nn.functional as F
import torchvision.transforms as transforms
class CL(nn.Module):
    def __init__(self,temperature):
        super().__init__()
        # self.projection = encoder_MLP(rank, hidden_size)
        self.temperature = temperature

    def get_negative_mask(self, batch_size, labels1=None, labels2=None):
        if labels2 is None:
            labels1 = labels1.contiguous().view(-1, 1)
            if labels1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels1, labels1.T).float().cuda()
        else:
            labels1 = labels1.contiguous().view(-1, 1)
            mask1 = torch.eq(labels1, labels1.T).float().cuda()
            labels2 = labels2.contiguous().view(-1, 1)
            mask2 = torch.eq(labels2, labels2.T).float().cuda()
            mask = mask1*mask2
            mask = mask.float().cuda()
        mask = mask.repeat(2, 2)
        return mask

    def pos_loss(self, self_predictions, pos_predictions, labels1=None, labels2=None):
        pos_predictions = F.normalize(pos_predictions, dim=-1)
        self_predictions = F.normalize(self_predictions, dim=-1)
        mask = self.get_negative_mask(self_predictions.shape[0], labels1, labels2).cuda()
        out = torch.cat([self_predictions, pos_predictions], dim=0)
        similarity_m = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)
        pos = (similarity_m * mask) / self.temperature
        exp_logits = torch.exp(similarity_m / self.temperature)
        pos = pos.sum(1)
        pos = pos
        neg = exp_logits * ((~mask.bool()).float())
        neg = neg.sum(dim=-1)
        pos_loss = ( -pos + torch.log(neg)) / mask.sum(-1)
        pos_loss = pos_loss.mean()
        return pos_loss

    def forward(self, x1, x2, labels1=None, labels2=None):
        # x1 = self.projection(x1)
        # x2 = self.projection(x2)
        loss = self.pos_loss(x1, x2, labels1, labels2)
        return loss


class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_distance, neg_distance):
        return torch.mean(torch.relu(self.margin + pos_distance - neg_distance))

def distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2, p=2, dim=-1)
# 定义Negative Sampling Loss
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, pos_distance, neg_distance):
        return -torch.mean(torch.log(torch.sigmoid(pos_distance - neg_distance)))


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.elu=nn.ELU()
        self.linear=torch.nn.Linear(hidden_dim,hidden_dim)


        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()
    def get_negative_mask(self, batch_size, labels1=None, labels2=None):
        if labels2 is None:
            labels1 = labels1.contiguous().view(-1, 1)
            if labels1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels1, labels1.T).float().cuda()
        else:
            labels1 = labels1.contiguous().view(-1, 1)#contiguous()用于确保张量在内存中是连续存储的
            mask1 = torch.eq(labels1, labels1.T).float().cuda()#torch.eq() 是一个逐元素比较函数，用于比较两个张量的对应元素是否相等
            labels2 = labels2.contiguous().view(-1, 1)
            mask2 = torch.eq(labels2, labels2.T).float().cuda()
            mask = mask1*mask2
            mask = mask.float().cuda()
       # mask = mask.repeat(2, 2)
        return mask

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, labels1=None,labels2=None):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(B, N) 0-1张量，每个目标顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float 对比损失

        """
        #self.proj.device=z_sc.device
        pos = self.get_negative_mask(z_sc.shape[0], labels1, labels2).cuda()
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)

        sim_sc2mp = self.sim(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()

        batch = pos.shape[0]
        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-8)  # 不能改成/=
        loss_sc = -torch.log(torch.sum(sim_sc2mp[:batch] * pos, dim=1)).mean()

        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc[:batch] * pos, dim=1)).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_emb, negative_emb):
        # 计算欧氏距离
        distance = F.pairwise_distance(anchor_emb, negative_emb)

        # 计算对比损失
        loss = torch.mean((1 - distance) ** 2)

        return loss


class C_loss(torch.nn.Module):
    def __init__(self, temperature):
        super(C_loss, self).__init__()
        self.temperature = temperature

    def forward(self, self_predictions, pos_predictions, labels1):
        self_predictions = F.normalize(self_predictions, dim=-1)
        pos_predictions = F.normalize(pos_predictions, dim=-1)

        # Create positive pairs using labels1
        mask = labels1.unsqueeze(1) == labels1.unsqueeze(0)
        mask = mask.float()

        # Calculate cosine similarity
        similarity_m = F.cosine_similarity(self_predictions.unsqueeze(1), pos_predictions.unsqueeze(0), dim=-1)

        # Calculate negative loss
        neg_loss = -similarity_m / self.temperature
        neg_loss = torch.exp(neg_loss) * (1 - mask)
        neg_loss = neg_loss.sum() / (len(labels1) * (len(labels1) - 1))

        return neg_loss
