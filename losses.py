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
    def __init__(self, clsnums = [], reduction='mean', confidence_loss = 'crossentropy', size_transform = 'none',
        localization_weight = 1., classification_weight = 1., normalize_per_anchor = True, normalize_per_cell = True, **kwargs):
        assert reduction == 'mean' or reduction == 'sum', \
            "reduction should be 'mean' or 'sum'"
        assert confidence_loss in ['crossentropy', 'mse', 'focal_loss', 'focal_loss*'], \
            "confidence_loss should be 'crossentropy' or 'mse'"
        assert size_transform in {'none', 'log', 'sqrt'}, \
            "size_transform should be 'none', 'log' or 'sqrt'"
        self.clsnums = clsnums
        self.reduction = reduction
        self.confidence_loss = confidence_loss
        self.size_transform = size_transform
        self.localization_weight = localization_weight
        self.classification_weight = classification_weight
        self.normalize_per_anchor = normalize_per_anchor
        self.normalize_per_cell = normalize_per_cell
        self.kwargs = kwargs

    def __call__(self, predict, target):
        batch_size, c, w, h = target.shape
        n_anchors = c // (5+len(self.clsnums))
        # Calculate object confidence loss
        predict_box = predict[:,-5*n_anchors:,...].reshape(batch_size,5,n_anchors,w,h)
        target_box = target[:,-5*n_anchors:,...].reshape(batch_size,5,n_anchors,w,h)
        obj_mask = target_box[:,0,...] > -0.5
        if self.confidence_loss == 'crossentropy':
            loss_conf = torch.masked_select(F.binary_cross_entropy_with_logits(predict_box[:,0,...],
                target_box[:,0,...], reduction='none'), obj_mask).sum()
        elif self.confidence_loss == 'mse':
            loss_conf = torch.masked_select(F.mse_loss(torch.sigmoid(predict_box[:,0,...]),
                target_box[:,0,...], reduction='none'), obj_mask).sum()
        elif self.confidence_loss == 'focal_loss':
            loss_conf = torch.masked_select(_focal_loss(predict_box[:,0,...],
                target_box[:,0,...], **self.kwargs), obj_mask).sum()
        else:
            loss_conf = torch.masked_select(_focal_loss_star(predict_box[:,0,...],
                target_box[:,0,...], **self.kwargs), box_mask).sum()
        # Calculate localization loss
        box_mask = target_box[:,0:1,...] > 0.5
        loss_box = torch.masked_select(F.mse_loss(predict_box[:,1:3,...],
            target_box[:,1:3,...], reduction='none'), box_mask).sum()
        # Transform box sizes if requested
        if self.size_transform == 'log':
            loss_box += F.mse_loss(torch.masked_select(predict_box[:,-2:,...], box_mask).log(),
                torch.masked_select(target_box[:,-2:,...], box_mask).log(), reduction='sum')
        elif self.size_transform == 'sqrt':
            loss_box += torch.masked_select(F.mse_loss(predict_box[:,-2:,...].sqrt(),
                target_box[:,-2:,...].sqrt(), reduction='none'), box_mask).sum()
        else:
            loss_box += torch.masked_select(F.mse_loss(predict_box[:,-2:,...],
                target_box[:,-2:,...], reduction='none'), box_mask).sum()
        # Calculate classification loss if needed
        loss_class = []
        ip = jp = 0
        it = jt = 0
        for cls_num in self.clsnums:
            ip, jp = jp, jp+cls_num*n_anchors
            it, jt = jt, jt+n_anchors
            loss_class.append(F.cross_entropy(predict[:,ip:jp,...].reshape((batch_size,cls_num,n_anchors,w,h)),
                target[:,it:jt,...].type(torch.long),reduction='sum', ignore_index=-1))
        # Calculate reductions
        if self.reduction == 'mean':
            loss_conf = loss_conf/batch_size
            loss_box = loss_box/batch_size
            loss_class = [l/batch_size for l in loss_class]
        if self.normalize_per_anchor:
            loss_conf = loss_conf/n_anchors
        if self.normalize_per_cell:
            s = w*h
            loss_conf = loss_conf/s
            loss_box = loss_box/s
            loss_class = [l/s for l in loss_class]
        loss = loss_conf + self.localization_weight*loss_box + self.classification_weight*sum(loss_class)
        loss_dict = {'loss':loss,
                     'confidence_loss': loss_conf,
                     'localization_loss':self.localization_weight*loss_box}
        for i, c_loss in enumerate(loss_class):
            loss_dict[f'classification_loss_{i}'] = c_loss
        return loss_dict

    def state_dict(self):
        state = {'reduction':self.reduction,
            'clsnums': self.clsnums,
            'confidence_loss': self.confidence_loss,
            'size_transform':self.size_transform,
            'localization_weight':self.localization_weight,
            'normalize_per_anchor':self.normalize_per_anchor,
            'normalize_per_cell':self.normalize_per_cell}
        return state

    def load_state_dict(self, state):
        self.clsnums = state['clsnums']
        self.reduction = state['reduction']
        self.confidence_loss = state['confidence_loss']
        self.size_transform = state['size_transform']
        self.localization_weight = state['localization_weight']
        self.normalize_per_anchor = state['normalize_per_anchor']

def _focal_loss(predict, target, alpha=.25, gamma=2.0, reduction='none'):
    """ Loss function to calculate Focal Loss for object detection confidence
        FL(p_t) = -a_t(1-p_t)**gamma*log(p_t), where
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
