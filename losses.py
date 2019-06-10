import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def yolo_loss(predict, target, reduction='mean'):
    loss_conf = F.mse_loss(predict[:,0,:,:], target[:,0,:,:], reduction='sum')
    loss_box = F.mse_loss(predict[:,1:,:,:]*target[:,0:1,:,:], target[:,1:,:,:]*target[:,0:1,:,:], reduction='sum')
    if reduction == 'mean':
        loss = (loss_conf+loss_box)/target.shape[0]
    else:
        loss = (loss_conf+loss_box)
    return loss

def object_detection_loss(predict, target, reduction='mean', confidence_loss = 'crossentropy',
    confidence_output = 'logits', size_transform = 'log', localization_weight = 1):
    """ Loss function for object detection for dense grid of predictions such
    as in YOLO and SSD detectors. Single anchor per grid cell.
    Parameters:
        predict: (batch_size, 5, h, w) 4D Tensor of predictions
        target:  (batch_size, 5, h, w) 4D Tensor of ground truth boxes
        reduction: {'mean'|'sum'} sum or average over the batch
        confidence_loss: {'crossentropy'|'mse'}
        confidence_output: {'logits'|'probability'}
        size_transform: {'log'|'sqrt'|'none'}
        localization_weight: Factor to multiply localization loss
                             before summing with confidence loss
                             default = 1
    Returns:
        (total_loss, {'confidence_loss':confidence_loss, 'localization_loss':localization_loss})
    """
    if confidence_output == 'logits':
        if confidence_loss == 'crossentropy':
            loss_conf = F.binary_cross_entropy_with_logits(predict[:,0,:,:], target[:,0,:,:], reduction='sum')
        elif confidence_loss == 'mse':
            loss_conf = F.mse_loss(torch.sigmoid(predict[:,0,:,:]), target[:,0,:,:], reduction='sum')
        else:
            raise ValueError("confidence_loss should be 'crossentropy' or 'mse'")
    elif confidence_output == 'probability':
        if confidence_loss == 'crossentropy':
            loss_conf = F.binary_cross_entropy(predict[:,0,:,:], target[:,0,:,:], reduction='sum')
        elif confidence_loss == 'mse':
            loss_conf = F.mse_loss(predict[:,0,:,:], target[:,0,:,:], reduction='sum')
        else:
            raise ValueError("confidence_loss should be 'crossentropy' or 'mse'")
    else:
        raise ValueError("confidence_output should be 'logits' or 'probability'")
    loss_box = F.mse_loss(predict[:,1:3,:,:]*target[:,0:1,:,:], target[:,1:3,:,:]*target[:,0:1,:,:], reduction='sum')
    if size_transform == 'log':
        loss_box += F.mse_loss(predict[:,3:,:,:].log()*target[:,0:1,:,:], target[:,3:,:,:].log()*target[:,0:1,:,:], reduction='sum')
    elif size_transform == 'sqrt':
        loss_box += F.mse_loss(predict[:,3:,:,:].sqrt()*target[:,0:1,:,:], target[:,3:,:,:].sqrt()*target[:,0:1,:,:], reduction='sum')
    elif size_transform == 'none':
        loss_box += F.mse_loss(predict[:,3:,:,:]*target[:,0:1,:,:], target[:,3:,:,:]*target[:,0:1,:,:], reduction='sum')
    else:
        raise ValueError("size_transform should be 'log', 'sqrt' or 'none'")
    if reduction == 'mean':
        loss_conf, loss_box = loss_conf/target.shape[0], loss_box/target.shape[0]
    elif reduction != 'sum':
        raise ValueError("reduction should be 'mean' or 'sum'")
    loss = loss_conf + localization_weight*loss_box
    return (loss, {'confidence_loss': loss_conf, 'localization_loss':localization_weight*loss_box})

def yolo1_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'mse',
        confidence_output = 'logits', size_transform = 'sqrt', localization_weight = localization_weight)

def yolo2_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'crossentropy',
        confidence_output = 'logits', size_transform = 'log', localization_weight = localization_weight)
