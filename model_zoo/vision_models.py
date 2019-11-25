import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from image.metrics.localization import iou
import json
from inspect import signature
from image.datasets.utils import labels_to_boxes
from functools import partial
import math
import numpy as np

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

class ObjectDetectionModel(nn.Module):
    """ Dense grid prediction based (Yolo and SSD like) object detection model
    Takes class object of feature and head models and dicts with
    corresponding configuration parameters
    """
    def __init__(self, features_model, head_model, cell_anchors):
        super().__init__()
        self.features = features_model
        self.head = head_model
        self.cell_anchors = cell_anchors
        self.grid_size = self.features.grid_size

    def forward(self, x):
        return self.head(self.features(x))

    @torch.no_grad()
    def get_prediction(self, x, threshold = 0.5):
        """ Get prediction form network output
        """
        if isinstance(x, list):
            fmap_lbls = [labels_to_boxes(fmap, grid_size = self.grid_size//(2**i),
                cell_anchors = self.cell_anchors, threshold = threshold) for i, fmap in enumerate(x)]
            return [[torch.cat(lbls) for lbls in zip(*img_lbls)] for img_lbls in zip(*fmap_lbls)]
            # return fmap_lbls
        else:
            return labels_to_boxes(x, grid_size = self.grid_size, cell_anchors = self.cell_anchors, threshold = threshold)

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        """ Get prediction from image or batch of images
        """
        return self.get_prediction(self.forward(x), threshold)

    @torch.no_grad()
    def get_targets(self, x):
        """ Get targets from labels Tensor
        """
        if isinstance(x, list):
            fmap_lbls = [labels_to_boxes(fmap, grid_size = self.grid_size//(2**i),
                cell_anchors = self.cell_anchors) for i, fmap in enumerate(x)]
            out = [[torch.cat(lbls) for lbls in zip(*img_lbls)] for img_lbls in zip(*fmap_lbls)]
        else:
            out = labels_to_boxes(x, grid_size = self.grid_size, cell_anchors = self.cell_anchors)
        if len(self.head.clsnums) > 0:
            return [(box, cls) for box, score, cls in out]
        else:
            return [box for box, score in out]



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
