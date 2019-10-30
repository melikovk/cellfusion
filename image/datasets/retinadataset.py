import numpy as np
from ..metrics.localization import iou
from .utils import get_grid_anchors
from .multianchordataset import MultiAnchorDataset

class RetinaDataset(MultiAnchorDataset):
    """ Concrete multianchor CropDataset.

    """
    def __init__(self, imgname, lblname, grid_sizes, positive_anchor_threshold = 0.7,
        background_anchor_threshold = 0.5, denominator = 'union', **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._positive_thresh = positive_anchor_threshold
        self._bkg_thresh = background_anchor_threshold
        self._denominator = denominator
        self._grid_sizes = grid_sizes
        self._anchors = [get_grid_anchors(self._cell_anchors, self._w//gs, self._h//gs).reshape(4,-1).T
            for gs in grid_sizes]


    def _get_labels(self, idx):
        w, h = self._w, self._h
        anchors_per_cell = self._cell_anchors.shape[0]
        anchor_counts = np.cumsum([anc.shape[0] for anc in self._anchors])
        # Convert anchor sizes to pixels and merge into one array
        anchors = np.concatenate([anc*gs for anc, gs in zip(self._anchors, self._grid_sizes)])
        # Filter out boxes that overlap less than threshold with the window
        boxes, boxcls = self._get_boxes(idx)
        # boxes = boxes/self._grid_size
        # If there are no true boxes in the window, return label with all background
        if boxes.shape[0] == 0:
            if boxcls is None:
                return [np.zeros((5*anchors_per_cell, w//gs, h//gs)) for gs in self._grid_sizes]
            else:
                return np.zeros((5*n_anchors, w, h)), np.full((n_anchors, w, h), -1, dtype=np.long) # This is wrong
        # Initialize output arrays
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
            clslbls = np.full_like(labels, -1)
            clslbls[match_mask] = boxcls[labels[match_mask]-1]
            clslbls = np.split(clslbls, anchor_counts[:-1])
        labels[labels>0] = 1
        # labels = labels.reshape((n_anchors, w, h))
        # coordinates = coordinates.reshape((4*n_anchors, w, h))
        labels = np.split(labels, anchor_counts[:-1])
        xs = np.split(xs, anchor_counts[:-1])
        ys = np.split(ys, anchor_counts[:-1])
        ws = np.split(ws, anchor_counts[:-1])
        hs = np.split(hs, anchor_counts[:-1])
        if boxcls is None:
            out = [np.concatenate([l, bx, by, bw, bh]).reshape(5*anchors_per_cell, w//gs, h//gs)
                   for l, bx, by, bw, bh, gs in zip(labels, xs, ys, ws, hs, self._grid_sizes)]
            return out
        else:
            out = [np.concatenate(cl.reshape(anchors_per_cell,w//gs,h//gs),
                                  l.reshape(anchors_per_cell,w//gs, h//gs),
                                  c.reshape(4*anchors_per_cell,w//gs,h//gs))
                   for cl, l, c, gs in zip(clslbls, labels, coordinates, self._grid_sizes)]
            return out
