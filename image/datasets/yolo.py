import json
import numpy as np
import skimage.io as io
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ..utils import centerinside
from ..metrics.localization import iou
import torch

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
    """ Base Dataset for all object detection datasets that crop small subimages from larger image
    """
    def __init__(self, imgname, lblname, win_size=(224,224), border_size=32, grid_size=32, transforms=None, **kwargs):
        super().__init__()
        self._img = io.imread(imgname).T
        self._w, self._h = win_size
        self._grid_size = grid_size
        self._border_size = border_size
        self._transforms = transforms
        # Create array ob object boxes
        with open(lblname, 'r') as f:
            boxes = json.load(f)
        self._boxes = np.array([box['bounds'] for box in boxes if box['type'] == NUCLEUS])/grid_size
        self._boxes[:,0:2] = self._boxes[:,0:2] + self._boxes[:,2:4]/2
        self._xys = self._init_coordinates(**kwargs)

    def _init_coordinates(self):
        raise NotImplementedError

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

    def labels_to_boxes(self, labels, grid_size = 32, offset=(0,0), threshold = 0.5):
        raise NotImplementedError

    def reset(self):
        pass

class GridCropDataset(CropDataset):
    def __init__(self, imgname, lblname, stride = None, **kwargs):
        super().__init__(imgname, lblname, **kwargs, stride=stride)

    def _init_coordinates(self, stride):
        if stride is None:
            stride = (self._w, self._h)
        xrange = np.arange(self._border_size, self._img.shape[0] - self._border_size - self._w, stride[0])
        yrange = np.arange(self._border_size, self._img.shape[1] - self._border_size - self._h, stride[1])
        return np.stack(np.meshgrid(xrange,yrange, indexing = 'ij')).reshape(2,-1).T

class RandomCropDataset(CropDataset):
    def __init__(self, imgname, lblname, length, seed=None, **kwargs):
        self._seed = seed
        super().__init__(imgname, lblname, **kwargs, length=length)

    def _init_coordinates(self, length):
        if self._seed is not None:
            np.random.seed(self._seed)
        xs = np.random.randint(self._border_size, self._img.shape[0]-self._border_size-self._w, length)
        ys = np.random.randint(self._border_size, self._img.shape[1]-self._border_size-self._h, length)
        return np.stack((xs,ys), axis=-1)

    def reset(self):
        if self._seed is None:
            self._xys = self._init_coordinates(self._xys.shape[0])

class NaiveBoxDataset(CropDataset):
    def _get_labels(self, idx):
        x, y = self._xys[idx]/self._grid_size
        labels = np.zeros((5, self._w//self._grid_size, self._h//self._grid_size))
        boxes = self._boxes[centerinside((x, y, self._w/self._grid_size, self._h/self._grid_size), self._boxes, lt_anchor='center')].reshape(-1,4)
        boxes[:,0] = boxes[:,0] - x
        boxes[:,1] = boxes[:,1] - y
        for box in boxes:
            xbox, ybox, wbox, hbox = box
            xidx, yidx = int(np.floor(xbox)), int(np.floor(ybox))
            xpos, ypos = xbox - xidx, ybox - yidx
            labels[:, xidx, yidx] = [1, xpos, ypos, wbox, hbox]
        return labels

    def labels_to_boxes(self, labels, threshold = 0.5, offset=(0,0)):
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
        boxes = boxes*self.grid_size
        boxes[0] += offx
        boxes[1] += offy
        boxes = boxes.reshape((4,-1)).T
        scores = scores.reshape((1, -1)).squeeze()
        idx = (scores > threshold).nonzero()[0]
        return boxes[idx], scores[idx]

class NaiveGridDataset(GridCropDataset, NaiveBoxDataset):
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
    pass

class NaiveRandomDataset(RandomCropDataset, NaiveBoxDataset):
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
    pass

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
