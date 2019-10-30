import numpy as np
from ..metrics.localization import iou
from .utils import get_grid_anchors
from .multianchordataset import MultiAnchorDataset


class YoloDataset(MultiAnchorDataset):
    """ Concrete multianchor CropDataset.
    Each true box is assigned to only one anchor (anchor with highest IOU with
    the box) and we set ignore label (-1 for objectness) for anchors that have
    IOU higher than specified threshold with any true box, unless they have
    been assigned to true box
    """
    def __init__(self, imgname, lblname, grid_size = 32, anchor_ignore_threshold = 0.5, denominator = 'union', **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._ignore_thresh = anchor_ignore_threshold
        self._denominator = denominator
        self._grid_size = grid_size
        self._anchors = get_grid_anchors(self._cell_anchors, self._w//grid_size, self._h//grid_size).reshape(4,-1).T


    def _get_labels(self, idx):
        w, h = self._w//self._grid_size, self._h//self._grid_size
        n_anchors = self._cell_anchors.shape[0]
        anchors = self._anchors
        labels = np.zeros(anchors.shape[0])
        coordinates = np.zeros(4*anchors.shape[0])
        xs, ys, ws, hs = np.split(coordinates, 4)
        # Filter out boxes that overlap less than threshold with the window
        boxes, boxcls = self._get_boxes(idx)
        boxes = boxes/self._grid_size
        # If there are no true boxes in the window, return label with all background
        if boxes.shape[0] == 0:
            if boxcls is None:
                return np.zeros((5*n_anchors, w, h), dtype = np.float32)
            else:
                return np.concatenate([np.zeros((5*n_anchors, w, h), dtype=np.float32), np.full((n_anchors, w, h), -1, dtype=np.float32)])
        iou_matrix = iou(anchors, boxes, denominator=self._denominator)
        ignore_mask = (iou_matrix > self._ignore_thresh).any(axis=-1)
        labels[ignore_mask] = -1.0
        # Set labels for anchors that have maximum IOU with true boxes to 1.0
        match_ious, match_idxs = iou_matrix.max(axis=0), iou_matrix.argmax(axis=0)
        labels[match_idxs] = 1.0
        # Remaining anchors are background boxes
        xs[match_idxs] = (boxes[:,0] + boxes[:,2]/2 - anchors[match_idxs, 0])/anchors[match_idxs, 2]
        ys[match_idxs] = (boxes[:,1] + boxes[:,3]/2 - anchors[match_idxs, 1])/anchors[match_idxs, 3]
        ws[match_idxs] = boxes[:,2]/anchors[match_idxs, 2]
        hs[match_idxs] = boxes[:,3]/anchors[match_idxs, 3]
        # Set class labels if needed
        if boxcls is not None:
            clslbls = np.full_like(labels, -1)
            clslbls[match_idxs] = boxcls[np.arange(match_idxs.shape[0], dtype=np.long)]
            clslbls = clslbls.reshape((n_anchors, w, h))
        # Reshape
        labels = labels.reshape((n_anchors, w, h))
        coordinates = coordinates.reshape((4*n_anchors, w, h))
        if boxcls is None:
            return np.concatenate((labels, coordinates)).astype(np.float32)
        else:
            return np.concatenate((clslbls, labels, coordinates)).astype(np.float32)
