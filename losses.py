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
        localization_weight = 1., classification_weight = 1., normalize_per_anchor = True, normalize_per_cell = True, **kwargs):
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
        self.classification_weight = classification_weight
        self.normalize_per_anchor = normalize_per_anchor
        self.normalize_per_cell = normalize_per_cell
        self.kwargs = kwargs

    def __call__(self, predict, target):
        if isinstance(predict, list):
            assert isinstance(target, list) and len(predict) == len(target), \
                "for class predictions lists of the same size are expected"
            add_class_loss = True
            predict_obj = predict[0]
            target_obj = target[0]
        else:
            predict_obj = predict
            target_obj = target
        assert predict_obj.shape == target_obj.shape, \
            "prediction and target tensors should have the same shape"
        batch_size, _, w, h = predict_obj.shape
        predict_obj = predict_obj.reshape(batch_size, 5, -1, w, h)
        target_obj = target_obj.reshape(batch_size, 5, -1, w, h)
        # Calculate object confidence loss
        object_mask = target_obj[:,0,...] > -0.5
        if self.confidence_loss == 'crossentropy':
            loss_conf = torch.masked_select(F.binary_cross_entropy_with_logits(predict_obj[:,0,...], target_obj[:,0,...], reduction='none'), object_mask).sum()
        elif self.confidence_loss == 'mse':
            loss_conf = torch.masked_select(F.mse_loss(torch.sigmoid(predict_obj[:,0,...]), target_obj[:,0,...], reduction='none'), object_mask).sum()
        elif self.confidence_loss == 'focal_loss':
            loss_conf = torch.masked_select(_focal_loss(predict_obj[:,0,...], target_obj[:,0,...], **self.kwargs), object_mask).sum()
        else:
            loss_conf = torch.masked_select(_focal_loss_star(predict_obj[:,0,...], target_obj[:,0,...], **self.kwargs), object_mask).sum()
        # Calculate localization loss
        box_mask = target_obj[:,0:1,...] > 0.5
        loss_box = torch.masked_select(F.mse_loss(predict_obj[:,1:3,...], target_obj[:,1:3,...], reduction='none'), box_mask).sum()
        # Transform box sizes if requested
        if self.size_transform == 'log':
            loss_box += F.mse_loss(torch.masked_select(predict_obj[:,3:,...], box_mask).log(), torch.masked_select(target_obj[:,3:,...], box_mask).log(), reduction='sum')
        elif self.size_transform == 'sqrt':
            loss_box += torch.masked_select(F.mse_loss(predict_obj[:,3:,...].sqrt(), target_obj[:,3:,...].sqrt(), reduction='none'), box_mask).sum()
        else:
            loss_box += torch.masked_select(F.mse_loss(predict_obj[:,3:,...], target_obj[:,3:,...], reduction='none'), box_mask).sum()
        # Calculate classification loss if needed
        if add_class_loss:
            loss_class = F.cross_entropy(predict[1].reshape((target[1].shape[0],-1)+target[1].shape[1:]),target[1],reduction='sum', ignore_index=-1)
        else:
            loss_class = 0
        # Calculate reductions
        if self.reduction == 'mean':
            loss_conf = loss_conf/target_obj.shape[0]
            loss_box = loss_box/target_obj.shape[0]
            loss_class = loss_class/target_obj.shape[0]
        if self.normalize_per_anchor:
            loss_conf = loss_conf/target_obj.shape[2]
        if self.normalize_per_cell:
            loss_conf = loss_conf/(target_obj.shape[-1]*target_obj.shape[-2])
            loss_box = loss_box/(target_obj.shape[-1]*target_obj.shape[-2])
            loss_class = loss_class/(target_obj.shape[-1]*target_obj.shape[-2])
        loss = loss_conf + self.localization_weight*loss_box + self.classification_weight*loss_class
        if add_class_loss:
            loss_dict = {'loss':loss,
                         'confidence_loss': loss_conf,
                         'localization_loss':self.localization_weight*loss_box,
                         'classification_loss':self.classification_weight*loss_class}
        else:
            loss_dict = {'loss':loss,
                         'confidence_loss': loss_conf,
                         'localization_loss':self.localization_weight*loss_box}
        return loss_dict

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
