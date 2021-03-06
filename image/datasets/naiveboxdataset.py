import numpy as np
from . import CropDataset
from ..utils import centerinside

class NaiveBoxDataset(CropDataset):
    """ Simple CropDataset class.
    Does not use anchor boxes (i.e. grid_cell is the single anchor box).
    Assignes boxes to the grid cells based on the position of the box center.
    """
    def __init__(self, imgname, lblname, grid_size = 32, **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._grid_size = grid_size

    def _get_labels(self, idx):
        x, y = self._xys[idx]/self._grid_size
        boxes, boxcls = self._get_boxes(idx)
        boxes = boxes/self._grid_size
        w, h = self._w//self._grid_size, self._h//self._grid_size
        labels = np.zeros((5, w, h))
        if boxcls is not None:
            clslbls = np.full((w, h), -1, dtype = np.long)
        for bidx, box in enumerate(boxes):
            xbox, ybox, wbox, hbox = box
            xbox, ybox = xbox + wbox/2, ybox + hbox/2
            xidx, yidx = int(np.floor(xbox)), int(np.floor(ybox))
            xpos, ypos = xbox - xidx, ybox - yidx
            labels[:, xidx, yidx] = [1, xpos, ypos, wbox, hbox]
            if boxcls is not None:
                clslbls[xidx, yidx] = boxcls[bidx]
        return labels if boxcls is None else [labels, clslbls]

    def _get_boxes(self, idx):
        x, y = self._xys[idx]
        box_idxs = centerinside((x, y, self._w, self._h), self._boxes)
        boxes = self._boxes[box_idxs].reshape(-1,4)
        boxes[:,0:2] = boxes[:,0:2] - [x, y]
        boxcls = None if self._boxcls is None else self._boxcls[box_idxs]
        return boxes, boxcls
