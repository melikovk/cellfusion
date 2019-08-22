import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ObjectDetectionLoss:
    """ Loss function class for object detection for dense grid of predictions such
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
    def __init__(self, reduction='mean', confidence_loss = 'crossentropy', size_transform = 'none',
        localization_weight = 1, normalize_per_anchor = True, normalize_per_cell = True, **kwargs):
        assert reduction == 'mean' or reduction == 'sum', \
            "reduction should be 'mean' or 'sum'"
        assert confidence_loss in ['crossentropy', 'mse', 'focal_loss', 'focal_loss*'], \
            "confidence_loss should be 'crossentropy' or 'mse'"
        assert size_transform in {'none', 'log', 'sqrt'}, \
            "size_transform should be 'none', 'log' or 'sqrt'"
        self.reduction = reduction
        self.confidence_loss = confidence_loss
        self.size_transform = size_transform
        self.localization_weight = localization_weight
        self.normalize_per_anchor = normalize_per_anchor
        self.normalize_per_cell = normalize_per_cell
        self.kwargs = kwargs

    def __call__(self, predict, target):
        assert predict.shape == target.shape, \
            "prediction and target tensors should have the same shape"
        batch_size, _, w, h = predict.shape
        predict = predict.reshape(batch_size, 5, -1, w, h)
        target = target.reshape(batch_size, 5, -1, w, h)
        # Calculate object confidence loss
        object_mask = target[:,0,...] > -0.5
        if self.confidence_loss == 'crossentropy':
            loss_conf = torch.masked_select(F.binary_cross_entropy_with_logits(predict[:,0,...], target[:,0,...], reduction='none'), object_mask).sum()
        elif self.confidence_loss == 'mse':
            loss_conf = torch.masked_select(F.mse_loss(torch.sigmoid(predict[:,0,...]), target[:,0,...], reduction='none'), object_mask).sum()
        elif self.confidence_loss == 'focal_loss':
            loss_conf = torch.masked_select(_focal_loss(predict[:,0,...], target[:,0,...], **self.kwargs), object_mask).sum()
        else:
            loss_conf = torch.masked_select(_focal_loss_star(predict[:,0,...], target[:,0,...], **self.kwargs), object_mask).sum()
        # Calculate loaclization loss
        box_mask = target[:,0:1,...] > 0.5
        loss_box = torch.masked_select(F.mse_loss(predict[:,1:3,...], target[:,1:3,...], reduction='none'), box_mask).sum()
        # Transform box sizes if requested
        if self.size_transform == 'log':
            loss_box += F.mse_loss(torch.masked_select(predict[:,3:,...], box_mask).log(), torch.masked_select(target[:,3:,...], box_mask).log(), reduction='sum')
        elif self.size_transform == 'sqrt':
            loss_box += torch.masked_select(F.mse_loss(predict[:,3:,...].sqrt(), target[:,3:,...].sqrt(), reduction='none'), box_mask).sum()
        else:
            loss_box += torch.masked_select(F.mse_loss(predict[:,3:,...], target[:,3:,...], reduction='none'), box_mask).sum()
        if self.reduction == 'mean':
            loss_conf, loss_box = loss_conf/target.shape[0], loss_box/target.shape[0]
        if self.normalize_per_anchor:
            loss_conf = loss_conf/target.shape[2]
        if self.normalize_per_cell:
            loss_conf = loss_conf/(target.shape[-1]*target.shape[-2])
            loss_box = loss_box/(target.shape[-1]*target.shape[-2])
        loss = loss_conf + self.localization_weight*loss_box
        return {'loss':loss, 'confidence_loss': loss_conf, 'localization_loss':self.localization_weight*loss_box}

    def state_dict(self):
        state = {'reduction':self.reduction,
            'confidence_loss': self.confidence_loss,
            'size_transform':self.size_transform,
            'localization_weight':self.localization_weight,
            'normalize_per_anchor':self.normalize_per_anchor,
            'normalize_per_cell':self.normalize_per_cell}
        return state

    def load_state_dict(self, state):
        self.reduction = state['reduction']
        self.confidence_loss = state['confidence_loss']
        self.size_transform = state['size_transform']
        self.localization_weight = state['localization_weight']
        self.normalize_per_anchor = state['normalize_per_anchor']

def _focal_loss(predict, target, alpha=.25, gamma=2.0, reduction='none'):
    """ Loss function to calculate Focal Loss for object detection confidence
        FL(p_t) = -alpha_t(1-p_t)**gamma*log(p_t), where
        p_t = p if y == 1 else (1-p), p is predicted probabilty, y is target probability
        a_t = a if y == 1 else (1-a), a is a parameter
    """
    device = predict.device
    p = torch.sigmoid(predict)
    loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
    # This version is significantly slower for some reason when you run model training
    # loss = loss * torch.pow(torch.where(target > 0.5, p, 1-p), gamma)
    # loss = loss * torch.where(target > 0.5, torch.tensor(alpha).to(device), torch.tensor(1-alpha).to(device))
    loss = loss * torch.pow(1 - p * target + (1-p)*(1-target), gamma)
    loss = loss * (alpha * target + (1-alpha)*(1-target))
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        loss = loss.mean()
    return loss

def _focal_loss_star(predict, target, gamma=4.0, beta=0.0, reduction='none'):
    device = predict.device
    loss = F.binary_cross_entropy_with_logits(gamma*predict+beta, target, reduction='none')/gamma
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        loss = loss.mean()
    return loss

def yolo1_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'mse',
        size_transform = 'none', localization_weight = localization_weight)

def yolo2_loss(predict, target, reduction='mean', localization_weight = 1):
    return object_detection_loss(predict, target, reduction=reduction, confidence_loss = 'crossentropy',
        size_transform = 'log', localization_weight = localization_weight)
