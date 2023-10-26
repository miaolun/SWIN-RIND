import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

# def attention_loss2(output,target):
#     num_pos = torch.sum(target == 1).float()
#     num_neg = torch.sum(target == 0).float()
#     alpha = num_neg / (num_pos + num_neg) * 1.0
#     eps = 1e-14
#     p_clip = torch.clamp(output, min=eps, max=1.0 - eps)
#
#     weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
#              (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
#     weight=weight.detach()
#
#     loss = F.binary_cross_entropy(output, target, weight, reduction='none')
#     loss = torch.sum(loss)
#     return loss

def attention_loss2(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    # gamma = 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))

    # weight = target  * alpha * ((1.0 - p_clip) ** gamma) + \
    #          (1.0 - target) * (1.0 - alpha) * (p_clip ** gamma)
    weight = weight.detach()

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss

def hed_part(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    # gamma = 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha + (1.0 - target) * (1.0 - alpha)

    # weight = target  * alpha * ((1.0 - p_clip) ** gamma) + \
    #          (1.0 - target) * (1.0 - alpha) * (p_clip ** gamma)
    weight = weight.detach()

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss

class hed_loss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, lamda=0.5):
        super(hed_loss, self).__init__()
        self.alpha = alpha

    def forward(self, output, label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            l = label[i]
            loss_focal = hed_part(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss

class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            l = label[i]
            loss_focal = attention_loss2(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sita = 1e-6

    def forward(self, inputs, targets):
        batch_size, c, height, width = targets.size()
        inputs = inputs.view(-1)
        targets = targets.contiguous().view(-1)
        assert inputs.size() == targets.size()

        fenmu = torch.sum(torch.square(inputs)) + torch.sum(torch.square(targets)) + self.sita
        fenzi = 2.0 * torch.sum(inputs * targets) + self.sita

        total_loss = fenzi / fenmu
        loss = total_loss
        return loss



class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss


if __name__ == '__main__':
    N = 4
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 1, H, W)).float()
    o_b = [torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W)]
    crientation = AttentionLoss2()
    total_loss = crientation(o_b, label)
    print('loss 2-1 :   '+ str(total_loss))





