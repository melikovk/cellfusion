import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
# from image.utils import iou as iou_np
import json

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

class CNNModel(nn.Module):

    def __init__(self, features, head):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        return self.head(self.features(x))

def iou(box, boxes):
    """ Given a rectangular box and an array of rectangular boxes
        returns IOU (intersection over union) metric for interestion
        of the first box with every box in the list
    Takes
        box: Tensor
        boxes: Tensor
    Returns
        iou: Tensor
    """
    left, top, width, height = box
    right, bottom = left + width, top + height
    zero = torch.tensor(0.0)
    iwidths = torch.max((torch.min(boxes[:,0]+boxes[:,2], right) - torch.max(boxes[:,0], left)),zero)
    iheights = torch.max((torch.min(boxes[:,1]+boxes[:,3], bottom) - torch.max(boxes[:,1], top)),zero)
    iareas = iwidths*iheights
    uareas = width*height + boxes[:,2]*boxes[:,3] - iareas
    return iareas/uareas

def nms(boxes, scores, iou_threshold):
    """ Given an array of rectangular boxes and confidence scores filters out boxes that
        overlap more than iou_threshold with boxes that higher score
    Takes
        boxes: Tensor
        scores: Tensor
        iou_threshold: float
    Returns
        keep_idx: Tensor
    """
    keep_idx = torch.ones_like(scores, dtype = torch.uint8)
    order = torch.argsort(scores, descending=True)
    for idx in order:
        if keep_idx[idx].item() == 0:
            continue
        remove = ((iou(boxes[idx], boxes) > iou_threshold) & (scores <= scores[idx])).nonzero().squeeze()
        keep_idx[remove] = 0
        keep_idx[idx] = 1
    return keep_idx.nonzero().squeeze()

def saveboxes(fpath, boxes, scores):
    """SSaves location bounding boxes to json file
    """
    records = []
    for idx in range(boxes.size()[0]):
        record = {}
        record['type'] = 0
        record['bounds'] = boxes[idx].numpy().astype(int).tolist()
        record['score'] = scores[idx].item()
        records.append(record)
    with open(fpath, 'w') as f:
        json.dump(records, f)

def localization_accuracy(reference, prediction, img_shape, iou_thresholds):
    def not_edge(box):
        w, h = img_shape
        xb, yb, wb, hb = box
        return xb>=0 and yb>=0 and xb+wb<w and yb+hb<h
    with open(reference, 'r') as f:
        true_box_records = json.load(f)
    true_boxes = torch.tensor([box['bounds'] for box in true_box_records if box['type'] == NUCLEUS], dtype = torch.float)
    with open(prediction, 'r') as f:
        predict_box_records = json.load(f)
    predict_boxes = torch.tensor([box['bounds'] for box in predict_box_records if box['type'] == NUCLEUS and not_edge(box['bounds'])], dtype = torch.float)
    predict_scores = torch.tensor([box['score'] for box in predict_box_records if box['type'] == NUCLEUS and not_edge(box['bounds'])], dtype = torch.float)
    ious = torch.stack([iou(box, true_boxes) for box in predict_boxes])
    best_ious, best_idxes = torch.max(ious, 1)
    matched = torch.zeros(true_boxes.size()[0])
    accs = []
    for threshold in iou_thresholds:
        matched[best_idxes[best_ious > threshold]] = 1
        tp = torch.sum(matched).item()
        accs.append(tp/(predict_boxes.size()[0]+true_boxes.size()[0]-tp))
        matched[:] = 0
    return accs
