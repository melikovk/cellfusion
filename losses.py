import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def object_detection_loss(predict, target, reduction='mean', confidence_loss = 'crossentropy',
    size_transform = 'log', localization_weight = 1):
    """ Loss function for object detection for dense grid of predictions such
    as in YOLO and SSD detectors. Predictions are assumed to be groupped in the following way :
    objectness score's for all anchors, x's for all anchors, y's for all anchors
    followed by w's and h's for all anchors. Function ignores target anchors
    with negative scores, allowing setting anchor boxes in the target to -1 to be ignored
    in calculation of loss (both objectness and box coordinates)
    Parameters:
        predict: (batch_size, 5xNAnchors, h, w) 4D Tensor of predictions
        target:  (batch_size, 5xNAnchors, h, w) 4D Tensor of ground truth boxes
        reduction: {'mean'|'sum'} sum or average over the batch
        confidence_loss: {'crossentropy'|'mse'}
        confidence_output: {'logits'|'probability'}
        size_transform: {'log'|'sqrt'|'none'}
        localization_weight: Factor to multiply localization loss
                             before summing with confidence loss
                             default = 1
    Returns:
        {'loss': total_loss, 'confidence_loss':confidence_loss, 'localization_loss':localization_loss}
    """
    batch_size, _, w, h = predict.shape
    predict = predict.reshape(batch_size, 5, -1, w, h)
    target = target.reshape(batch_size, 5, -1, w, h)
    object_mask = target[:,0,...] > -0.5
    if confidence_loss == 'crossentropy':
        loss_conf = F.binary_cross_entropy_with_logits(torch.masked_select(predict[:,0,...], object_mask), torch.masked_select(target[:,0,...], object_mask), reduction='sum')
    elif confidence_loss == 'mse':
        loss_conf = F.mse_loss(torch.sigmoid(torch.masked_select(predict[:,0,...], object_mask)), torch.masked_select(target[:,0,...], object_mask), reduction='sum')
    else:
        raise ValueError("confidence_loss should be 'crossentropy' or 'mse'")
    box_mask = target[:,0:1,...] > 0.5
    loss_box = F.mse_loss(torch.masked_select(predict[:,1:3,...], box_mask), torch.masked_select(target[:,1:3,...], box_mask), reduction='sum')
    if size_transform == 'log':
        loss_box += F.mse_loss(torch.masked_select(predict[:,3:,...], box_mask), torch.masked_select(target[:,3:,...], box_mask).log(), reduction='sum')
    elif size_transform == 'sqrt':
        loss_box += F.mse_loss(torch.masked_select(predict[:,3:,...], box_mask), torch.masked_select(target[:,3:,...], box_mask).sqrt(), reduction='sum')
    elif size_transform == 'none':
        loss_box += F.mse_loss(torch.masked_select(predict[:,3:,...], box_mask), torch.masked_select(target[:,3:,...], box_mask), reduction='sum')
    else:
        raise ValueError("size_transform should be 'log', 'sqrt' or 'none'")
    if reduction == 'mean':
        loss_conf, loss_box = loss_conf/target.shape[0], loss_box/target.shape[0]
    elif reduction != 'sum':
        raise ValueError("reduction should be 'mean' or 'sum'")
    loss = loss_conf + localization_weight*loss_box
    return {'loss':loss, 'confidence_loss': loss_conf, 'localization_loss':localization_weight*loss_box}

def object_detection_loss_fast(predict, target, reduction='mean', confidence_loss = 'crossentropy',
    size_transform = 'log', localization_weight = 1, eps = 1e-5):
    """ This function is equivalent to object_detection_loss but uses multiplication by 0 for masking
    instead of torch.masked_select which is significantly slower.
    """
    eps = torch.tensor(eps).to(target.device)
    batch_size, _, w, h = predict.shape
    predict = predict.reshape(batch_size, 5, -1, w, h)
    target = target.reshape(batch_size, 5, -1, w, h)
    object_mask = ((target[:,0,...]) > -0.5).to(torch.float)
    if confidence_loss == 'crossentropy':
        loss_conf = F.binary_cross_entropy_with_logits(predict[:,0,...]*object_mask, target[:,0,...]*object_mask, reduction='sum')
    elif confidence_loss == 'mse':
        loss_conf = F.mse_loss(torch.sigmoid(predict[:,0,...])*object_mask, target[:,0,...]*object_mask, reduction='sum')
    else:
        raise ValueError("confidence_loss should be 'crossentropy' or 'mse'")
    box_mask = (target[:,0:1,...] > 0).to(torch.float)
    loss_box = F.mse_loss(predict[:,1:3,...]*box_mask, target[:,1:3,...]*box_mask, reduction='sum')
    if size_transform == 'log':
        loss_box += F.mse_loss(predict[:,3:,...]*box_mask, target[:,3:,...].max(eps).log()*box_mask, reduction='sum')
    elif size_transform == 'sqrt':
        loss_box += F.mse_loss(predict[:,3:,...]*box_mask, target[:,3:,...].sqrt()*box_mask, reduction='sum')
    elif size_transform == 'none':
        loss_box += F.mse_loss(predict[:,3:,...]*box_mask, target[:,3:,...]*box_mask, reduction='sum')
    else:
        raise ValueError("size_transform should be 'log', 'sqrt' or 'none'")
    if reduction == 'mean':
        loss_conf, loss_box = loss_conf/target.shape[0], loss_box/target.shape[0]
    elif reduction != 'sum':
        raise ValueError("reduction should be 'mean' or 'sum'")
    loss = loss_conf + localization_weight*loss_box
    return {'loss':loss, 'confidence_loss': loss_conf, 'localization_loss':localization_weight*loss_box}

def yolo1_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'mse',
        confidence_output = 'logits', size_transform = 'none', localization_weight = localization_weight)

def yolo2_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'crossentropy',
        confidence_output = 'logits', size_transform = 'log', localization_weight = localization_weight)
