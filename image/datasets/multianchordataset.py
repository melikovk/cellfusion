import numpy as np
from ..metrics.localization import iou
from .cropdataset import CropDataset

class MultiAnchorDataset(CropDataset):
    """ Virtual class that extends CropDataset class. Base class for
    CropDatasets that use cell anchors for object box assignment. For each crop
    only objects that overlap crop window with IOU > threshold are used in label
    assignement.
    """
    def __init__(self, imgname, lblname, cell_anchors, window_overlap_threshold = .25, **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._cell_anchors = cell_anchors
        self._window_thresh = window_overlap_threshold

    def _get_boxes(self, idx):
        left_x, top_y = self._xys[idx]
        w, h = self._w, self._h
        box_idxs = iou(self._boxes, np.array([left_x, top_y, w, h]), denominator='first').squeeze()>self._window_thresh
        boxes = self._boxes[box_idxs]
        boxes[:,0:2] = boxes[:,0:2] - [left_x, top_y]
        boxcls = None if self._boxcls is None else self._boxcls[box_idxs]
        return boxes, boxcls
