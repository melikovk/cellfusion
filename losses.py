import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def yolo_loss(input, target, reduction='mean'):
    mask = target[:,0:1,:,:].byte()
    batch_size = target.size()[0]
    loss_conf = F.mse_loss(input[:,0,:,:], target[:,0,:,:], reduction='sum')
    loss_box = F.mse_loss(torch.masked_select(input[:,1:,:,:], mask), torch.masked_select(target[:,1:,:,:], mask), reduction='sum')
    if reduction == 'mean':
        loss = (loss_conf+loss_box)/batch_size
    else:
        loss = (loss_conf+loss_box)
    return loss
