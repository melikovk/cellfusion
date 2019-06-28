import json
import numpy as np
import skimage.io as io
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ..utils import centerinside
from ..metrics.localization import iou
import torch
import math

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

class RandomLoader(DataLoader):

    def __iter__(self):
        if isinstance(self.dataset, ConcatDataset):
            for dset in self.dataset.datasets:
                dset.reset()
        else:
            self.dataset.reset()
        return super().__iter__()

class CropDataset(Dataset):
    """ Base Dataset Class for all object detection datasets that crop subimages from larger image
    """
    def __init__(self, imgname, lblname, win_size=(224,224), border_size=32, grid_size=32, transforms=None, sample='random', length = None, seed=None, stride=None):
        self._img = io.imread(imgname).T
        self._w, self._h = win_size
        self._grid_size = grid_size
        self._border_size = border_size
        self._transforms = transforms
        # Create array ob object boxes
        with open(lblname, 'r') as f:
            boxes = json.load(f)
        self._boxes = np.array([box['bounds'] for box in boxes if box['type'] == NUCLEUS])/grid_size
        self._boxes[:,:2] = self._boxes[:,:2] + self._boxes[:,2:]/2
        self._xys = self._init_coordinates(sample=sample, length = length, seed=seed, stride=stride)

    def _init_coordinates(self,sample, length, seed, stride):
        if sample == 'random':
            self._seed = seed
            if length is None:
                length = self._img.shape[-1]*self._img.shape[-2] // (self._w*self._h)
            if self._seed is not None:
                np.random.seed(self._seed)
            xs = np.random.randint(self._border_size, self._img.shape[-2]-self._border_size-self._w, length)
            ys = np.random.randint(self._border_size, self._img.shape[-1]-self._border_size-self._h, length)
            return np.stack((xs,ys), axis=-1)
        elif sample == 'grid':
            self._seed = 0
            if stride is None:
                stride = (self._w, self._h)
            xrange = np.arange(self._border_size, self._img.shape[-2] - self._border_size - self._w, stride[0])
            yrange = np.arange(self._border_size, self._img.shape[-1] - self._border_size - self._h, stride[1])
            return np.stack(np.meshgrid(xrange, yrange, indexing = 'ij')).reshape(2,-1).T


    def _get_labels(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self._xys.shape[0]

    def __getitem__(self, idx):
        x, y = self._xys[idx]
        img = self._img[x:x+self._w, y:y+self._h]
        if self._transforms is not None:
            img = self._transforms(img)
        labels = self._get_labels(idx).astype(np.float32)
        return (img, labels)

    @staticmethod
    def labels_to_boxes(labels, grid_size, offset, threshold):
        raise NotImplementedError

    def reset(self):
        if self._seed is None:
            self._xys = self._init_coordinates(self._xys.shape[0])

class NaiveBoxDataset(CropDataset):
    def _get_labels(self, idx):
        x, y = self._xys[idx]/self._grid_size
        labels = np.zeros((5, self._w//self._grid_size, self._h//self._grid_size))
        boxes = self._get_boxes(idx)
        for box in boxes:
            xbox, ybox, wbox, hbox = box
            xidx, yidx = int(np.floor(xbox)), int(np.floor(ybox))
            xpos, ypos = xbox - xidx, ybox - yidx
            labels[:, xidx, yidx] = [1, xpos, ypos, wbox, hbox]
        return labels

    def _get_boxes(self, idx):
        x, y = self._xys[idx]/self._grid_size
        boxes = self._boxes[centerinside((x, y, self._w/self._grid_size, self._h/self._grid_size), self._boxes, lt_anchor='center')].reshape(-1,4)
        boxes[:,0:2] = boxes[:,0:2] - [x, y]
        return boxes

    @staticmethod
    def labels_to_boxes(labels, grid_size, threshold = 0.5, offset=(0,0)):
        """ Function to convert yolo type model output to bounding boxes
        Parameters:
            labels:     [5:width:height] Tensor of predictions
                        1st dimension stores [Pobj:Xcenter:Ycenter:W:H]
                        all dimensions are normalized to grid_size
            grid_size:  Size of the model grid
            offset:     offset of the crop in the image for multicrop predictions (in pixels)
            threshold:  Pobj threshold to use
        Returns:
            (ndarray(Xlt,Ylt,W,H), ndarray(Pobj)) all coordinates are float values in pixels
        """
        if isinstance(offset, int):
            offx = offy = offset
        else:
            offx, offy = offset
        _, wi, hi = labels.shape
        boxes = labels[1:].cpu().numpy()
        scores = labels[0].cpu().numpy()
        boxes[0] += np.arange(0, wi).reshape(-1,1) - boxes[2]/2
        boxes[1] += np.arange(0, hi).reshape(1,-1) - boxes[3]/2
        boxes = boxes*grid_size
        boxes[0] += offx
        boxes[1] += offy
        boxes = boxes.reshape((4,-1)).T
        scores = scores.reshape((1, -1)).squeeze()
        idx = (scores > threshold).nonzero()[0]
        return boxes[idx], scores[idx]

class MultiAnchorDataset(CropDataset):
    def __init__(self, imgname, lblname, scales=[1], anchors=[],**kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._anchors = self._init_anchors(scales, anchors)
        self._boxes[:,:2] = self._boxes[:,:2] - self._boxes[:,2:]/2

    def _init_anchors(self, scales, anchors):
        delta = self.anchors_delta(scales, anchors).reshape((-1,1,1,4))
        w, h = self._w//self._grid_size, self._h//self._grid_size
        anchors_grid = np.mgrid[0:w,0:h].transpose(1,2,0)
        anchors_grid = np.concatenate((anchors_grid, np.zeros_like(anchors_grid)),-1)
        anchors_grid = np.expand_dims(anchors_grid, axis=0)
        # anchor_centers = np.reshape(np.mgrid[0:delta.shape[0], 0:w, 0:h].transpose(1,2,3,0).reshape(-1,3)[:,1:]+0.5, (delta.shape[0], w, h, 2))
        return anchors_grid + delta

    @staticmethod
    def anchors_delta(scales, anchors):
        """ Create set of anchors for single cell
        Parameters:
            scales: [scale: float, ...]
            anchors: [(offset_x: float, offset_y: float, w_to_h_ration: float), ...]
        Return:
            corrections: ndarray([[delta_left, delta_top, width, height], ...])
        """
        corrections = []
        anchors = [(0.,0.,1.)] + anchors
        for s in scales:
            for dx, dy, whr in anchors:
                w = whr / math.sqrt(whr)
                h = 1 / math.sqrt(whr)
                corrections.append([0.5+s*(dx-w/2), 0.5+s*(dy-h/2), s*w, s*h])
        return np.array(corrections)

class YoloDataset(MultiAnchorDataset):
    """ Concrete multianchor dataset that uses Yolo like
    """
    def __init__(self, imgname, lblname, window_overlap_threshold = .25, anchor_ignore_threshold = 0.5, denominator = 'union', **kwargs):
        self._wot = window_overlap_threshold
        self._ait = anchor_ignore_threshold
        self._denominator = denominator
        super().__init__(imgname, lblname, **kwargs)

    def _get_boxes(self, idx):
        left_x, top_y = self._xys[idx]/self._grid_size
        w, h = self._w//self._grid_size, self._h//self._grid_size
        boxes = self._boxes[iou(self._boxes, np.array([left_x, top_y, w, h]), denominator='first').squeeze()>self._wot]
        boxes[:,0:2] = boxes[:,0:2] - [left_x, top_y]
        return boxes

    def _get_labels(self, idx):
        w, h = self._w//self._grid_size, self._h//self._grid_size
        n_anchors = self._anchors.shape[0]
        anchors = self._anchors.reshape(-1,4)
        labels = np.full(anchors.shape[0], -1.)
        coordinates = np.zeros(4*anchors.shape[0])
        xs, ys, ws, hs = np.split(coordinates, 4)
        # Filter out boxes that overlap less than threshold with the window
        boxes = self._get_boxes(idx)
        iou_matrix = iou(anchors, boxes, denominator=self._denominator)
        ignore_mask = (iou_matrix > self._ait).any(axis=-1)
        labels[ignore_mask] = 0
        match_ious, match_idxs = iou_matrix.max(axis=0), iou_matrix.argmax(axis=0)
        labels[match_idxs] = 1
        labels = labels.reshape((n_anchors, w, h))
        xs[match_idxs] = (boxes[:,0] + boxes[:,2]/2 - anchors[match_idxs, 0])/anchors[match_idxs, 2]
        ys[match_idxs] = (boxes[:,1] + boxes[:,3]/2 - anchors[match_idxs, 1])/anchors[match_idxs, 3]
        ws[match_idxs] = boxes[:,2]/anchors[match_idxs, 2]
        hs[match_idxs] = boxes[:,3]/anchors[match_idxs, 3]
        coordinates = coordinates.reshape((4*n_anchors, w, h))
        return np.concatenate((labels, coordinates))

    @staticmethod
    def labels_to_boxes(labels, grid_size, cell_anchors, threshold = 0.5, offset=(0,0)):
        """ Function to convert object loacalization model output to bounding boxes
        Parameters:
            labels:     [5*n_anchors:width:height] Tensor of predictions
                        1st dimension stores [Pobj:Xcenter:Ycenter:W:H]
                        all dimensions are normalized to grid_size
            grid_size:  Size of the model grid
            offset:     offset of the crop in the image for multicrop predictions (in pixels)
            threshold:  Pobj threshold to use
            cell_anchors: ndarray with anchors for single grid cell
                          should be the same as the one used to train the model
        Returns:
            (ndarray(Xlt,Ylt,W,H), ndarray(Pobj)) all coordinates are float values in pixels
        """
        if isinstance(offset, int):
            offx = offy = offset
        else:
            offx, offy = offset
        _, w, h = labels.shape
        # Create grid of all anchors
        anchors_grid = np.mgrid[0:w,0:h]
        anchors_grid = np.concatenate((anchors_grid, np.zeros_like(anchors_grid)))
        anchors_grid = np.expand_dims(anchors_grid, axis=1)
        anchors_grid = anchors_grid + cell_anchors.T.reshape((4,-1,1,1))
        anchors_grid = anchors_grid.reshape((4,-1))
        # Select booxes
        labels = labels.cpu().numpy().reshape((5, -1))
        idx = (labels[0] > threshold).nonzero()[0]
        scores = labels[0, idx]
        boxes = labels[1:,idx]
        anchors_grid = anchors_grid[:,idx]
        # Recalculate box sizes and positions
        boxes[:2] = (boxes[:2] - boxes[-2:]/2) * anchors_grid[-2:] + anchors_grid[:2]
        boxes[-2:] = boxes[-2:] * anchors_grid[-2:]
        boxes = boxes*grid_size
        boxes[0] += offx
        boxes[1] += offy
        return boxes.T, scores

def anchors_delta(scales, aspects):
    """ Create set of anchors for single cell
    Parameters:
        scales: [scale: float, ...]
        aspects: [(width: int, height: int), ...]
    Return:
        corrections: ndarray([[delta_left, delta_top, width, height], ...])
    """
    corrections = []
    for s in scales:
        for wa, ha in aspects:
            w = s * wa / math.sqrt(wa*ha)
            h = s * ha / math.sqrt(wa*ha)
            corrections.append([(1-w)/2, (1-h)/2, w, h])
    return np.array(corrections)

class NaiveGridDataset(NaiveBoxDataset):
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
    def __init__(self, imgname, lblname, **kwargs):
        super().__init__(imgname, lblname, sample='grid', **kwargs)

class NaiveRandomDataset(NaiveBoxDataset):
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
    """
    def __init__(self, imgname, lblname, **kwargs):
        super().__init__(imgname, lblname, sample='random', **kwargs)

YoloGridDataset = NaiveGridDataset

YoloRandomDataset = NaiveRandomDataset

def labelsToBoxes(labels, grid_size=32, offset=(0,0), threshold = 0.5):
    """ Function to convert yolo type model output to bounding boxes
    Parameters:
        labels:     [5:width:height] Tensor of predictions
                    1st dimension stores [Pobj:Xcenter:Ycenter:W:H]
                    all dimensions are normalized to grid_size
        grid_size:  Size of the model grid
        offset:     offset of the crop in the image for multicrop predictions (in pixels)
        threshold:  Pobj threshold to use
    Returns:
        (ndarray(Xlt,Ylt,W,H), ndarray(Pobj)) all coordinates are float values in pixels
    """
    if isinstance(offset, int):
        offx = offy = offset
    else:
        offx, offy = offset
    _, wi, hi = labels.shape
    # boxes = labels[1:].cpu().numpy()
    # scores = labels[0].cpu().numpy()
    boxes = labels[1:].copy()
    scores = labels[0].copy()
    boxes[0] += np.arange(0, wi).reshape(-1,1) - boxes[2]/2
    boxes[1] += np.arange(0, hi).reshape(1,-1) - boxes[3]/2
    boxes = boxes*grid_size
    boxes[0] += offx
    boxes[1] += offy
    boxes = boxes.reshape((4,-1)).T
    scores = scores.reshape((1, -1)).squeeze()
    idx = (scores > threshold).nonzero()
    return boxes[idx], scores[idx]

def iou_anchor(labels, denom = 'union'):
    bs = labels.reshape(5, -1).T[labels.reshape(5, -1)[0].nonzero()][:,1:].T
    iwidths = np.minimum(bs[0]+bs[2]/2, 1) - np.maximum(bs[0]-bs[2]/2, 0)
    iheights = np.minimum(bs[1]+bs[3]/2, 1) - np.maximum(bs[1]-bs[3]/2, 0)
    iareas = iwidths*iheights
    if denom == 'union':
        ious = iareas / (1 + bs[2]*bs[3] - iareas)
    elif denom == 'anchor':
        ious = iareas
    elif denom == 'box':
        ious = iareas / (bs[2]*bs[3])
    return ious
