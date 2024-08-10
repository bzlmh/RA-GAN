import torch
import torch.nn as nn
import numpy as np
from models.Discriminator import Discriminator
import torch.optim as optim
import functools
import torch.nn.functional as F
def jaccard_loss(input, target):
    smooth = 1e-6
    intersection = torch.sum(input * target)
    union = torch.sum(input) + torch.sum(target) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return 1 - jaccard


def d_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    a = torch.sum(input * target, dim=1)
    b = torch.sum(input * input, dim=1) + 0.001
    c = torch.sum(target * target, dim=1) + 0.001
    epsilon = 0.0001
    d = (2 * a) / (b + c + epsilon)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


class Loss_Doc(nn.Module):
    def __init__(self, model, ganLoss=True):
        super(Loss_Doc, self).__init__()
        self.model = model  # 保存模型
        self.ganLoss = ganLoss
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.BCELoss()
    def forward(self, ori,mid_out, gt):
        fin_l1_loss = self.l1(mid_out, gt)
        fin_BCEloss = self.cross_entropy(mid_out, gt)
        fin_dice_loss = d_loss(mid_out, gt)
        fin_jaccard_loss = jaccard_loss(mid_out, gt)
        return fin_l1_loss, fin_BCEloss, fin_dice_loss, fin_jaccard_loss
