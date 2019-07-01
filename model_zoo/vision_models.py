import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from image.metrics.localization import iou
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

def saveboxes(fpath, boxes, scores):
    """Saves location bounding boxes to json file
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
