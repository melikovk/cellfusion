import numpy as np
from . import CropDataset
from ..utils import centerinside

class NaiveBoxDataset(CropDataset):
    """ Simple dataset class to be used in pytorch
    Takes:
        data: Tuple(imgname, lblname, winsize, stride, bsize, transforms)
                imgname - path to image large image
                lblname - path to file with bounding boxes
                winsize - size of the window crops (w, h)
                stride - stride of the window crops
                bsize - size of the border to ignore
                transforms - transforms
        transform: callable object to transform each image
    Return:
        Dataset instance

    File with bounding boxes may contain boxes labelled as "ignore" if any ignore box overlaps
    window more than ignore_thresh the window is skipped, if any nucleus box overlaps window
    less than nuc_thresh window is also ignored, windows without any nuclei ignored as well
    """
    def _get_labels(self, idx):
        x, y = self._xys[idx]/self._grid_size
        boxes, boxcls = self._get_boxes(idx)
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
        x, y = self._xys[idx]/self._grid_size
        box_idxs = centerinside((x, y, self._w/self._grid_size, self._h/self._grid_size), self._boxes)
        boxes = self._boxes[box_idxs].reshape(-1,4)
        boxes[:,0:2] = boxes[:,0:2] - [x, y]
        boxcls = None if self._boxcls is None else self._boxcls[box_idxs]
        return boxes, boxcls
