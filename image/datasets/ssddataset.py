import numpy as np
from ..metrics.localization import iou
from .utils import get_grid_anchors
from .multianchordataset import MultiAnchorDataset

class SSDDataset(MultiAnchorDataset):
    """ Concrete multianchor dataset that uses SSD like box assignement. We assign
    each true box to multiple anchors (anchor with highest IOU for this box and
    any other anchor with IOU higher than the threshold (0.5 to 0.7)).
    Anchors that have IOU with all true boxes below another threshold (around 0.3) are
    set to background. All other anchors are set to ignore and ignored during training.
    """
    def __init__(self, imgname, lblname, grid_size = 32, positive_anchor_threshold = 0.7,
        background_anchor_threshold = 0.5, denominator = 'union', **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._positive_thresh = positive_anchor_threshold
        self._bkg_thresh = background_anchor_threshold
        self._denominator = denominator
        self._grid_size = grid_size
        self._anchors = get_grid_anchors(self._cell_anchors, self._w/self._grid_size, self._h/self._grid_size).transpose(1,2,3,0)


    def _get_labels(self, idx):
        w, h = self._w//self._grid_size, self._h//self._grid_size
        n_anchors = self._anchors.shape[0]
        anchors = self._anchors.reshape(-1,4)
        # Filter out boxes that overlap less than threshold with the window
        boxes, boxcls = self._get_boxes(idx)
        boxes = boxes/self._grid_size
        # If there are no true boxes in the window, return label with all background
        if boxes.shape[0] == 0:
            if boxcls is None:
                return np.zeros((5*n_anchors, w, h))
            else:
                return np.zeros((5*n_anchors, w, h)), np.full((n_anchors, w, h), -1, dtype=np.long)
        labels = np.full(anchors.shape[0], -1)
        coordinates = np.zeros(4*anchors.shape[0])
        xs, ys, ws, hs = np.split(coordinates, 4)
        iou_matrix = iou(anchors, boxes, denominator=self._denominator)
        # Set background anchor labels to 0.0
        bkg_mask = (iou_matrix < self._bkg_thresh).all(axis=-1)
        labels[bkg_mask] = 0
        # Set labels for anchors that have maximum IOU with true boxes
        # to the index of the matching true box + 1
        match_ious, match_idxs = iou_matrix.max(axis=0), iou_matrix.argmax(axis=0)
        labels[match_idxs] = np.arange(1, match_idxs.shape[0]+1)
        # Select unmatched anchors
        unmatched_mask = labels == -1
        unmatched_labels = labels[unmatched_mask]
        # Find true boxes with maximal IOU with unmatched anchors
        match_ious, match_idx = iou_matrix[unmatched_mask].max(axis=-1), iou_matrix[unmatched_mask].argmax(axis=-1)
        # Assign anchors that have IOU with any true box higher than positive_threshold
        # to the index of the true box with highest IOU + 1
        unmatched_labels[match_ious > self._positive_thresh] = match_idx[match_ious > self._positive_thresh] + 1
        labels[unmatched_mask] = unmatched_labels
        # Remaining boxes are ignore boxes - they are not maximal and have intermediate
        # IOU with some true boxes (between bkg_thresh and positive_thresh)
        # Create mask with all matched anchors to set coordinates
        match_mask = labels > 0
        # Set box coordinates for positive anchors
        xs[match_mask] = (boxes[labels[match_mask]-1,0] + boxes[labels[match_mask]-1,2]/2 - anchors[match_mask, 0])/anchors[match_mask, 2]
        ys[match_mask] = (boxes[labels[match_mask]-1,1] + boxes[labels[match_mask]-1,3]/2 - anchors[match_mask, 1])/anchors[match_mask, 3]
        ws[match_mask] = boxes[labels[match_mask]-1,2]/anchors[match_mask, 2]
        hs[match_mask] = boxes[labels[match_mask]-1,3]/anchors[match_mask, 3]
        # Set class labels if needed
        if boxcls is not None:
            clslbls = np.full_like(labels, -1, dtype=np.long)
            clslbls[match_mask] = boxcls[labels[match_mask]-1]
            clslbls = clslbls.reshape((n_anchors, w, h))
        labels[labels>0] = 1
        labels = labels.reshape((n_anchors, w, h))
        coordinates = coordinates.reshape((4*n_anchors, w, h))
        if boxcls is None:
            return np.concatenate((labels, coordinates))
        else:
            return np.concatenate((labels, coordinates)), clslbls
