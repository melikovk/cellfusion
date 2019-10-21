import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import get_boxes_from_json

class CropDataset(Dataset):
    """ Base Dataset Class for all object detection datasets that crop subimages from larger image
    """
    def __init__(self, imgnames, lblname, clsname = None, win_size=(224,224), border_size=32,
        grid_size=32, point_transforms=[], geom_transforms=[], norm_transform=None,
        sample='random', length = None, seed=None, stride=None):
        self._fname = lblname
        self._img_orig = np.stack([np.asarray(Image.open(imgname)).T for imgname in imgnames])
        self._w, self._h = win_size
        self._grid_size = grid_size
        self._border_size = border_size
        self._point_transforms = point_transforms
        self._geom_transforms = geom_transforms
        self._norm_transform = norm_transform
        self._sample = sample
        self._length = length
        self._stride = stride
        self._seed = seed
        # Create array of object boxes
        if clsname is None:
            self._boxes_orig = get_boxes_from_json(lblname)
            self._boxcls = None
        else:
            self._boxes_orig, self._boxcls = get_boxes_from_json(lblname, clsname)
        self._img, self._boxes = self._img_orig.astype(np.float32), self._boxes_orig.astype(np.float32)/self._grid_size
        self._xys = self._init_coordinates()

    def _data_augmentation(self):
        img = self._img_orig.astype(np.float32)
        boxes = self._boxes_orig.astype(np.float32)
        for f in self._point_transforms:
            img = f(img)
        for f in self._geom_transforms:
            img, boxes = f(img, boxes)
        if self._norm_transform is not None:
            img = self._norm_transform(img)
        boxes = boxes/self._grid_size
        return img, boxes

    def _init_coordinates(self):
        if self._sample == 'random':
            if self._seed is not None:
                np.random.seed(self._seed)
            length = self._img.shape[-1]*self._img.shape[-2] // (self._w*self._h) if self._length is None else self._length
            xs = np.random.randint(self._border_size, self._img.shape[-2]-self._border_size-self._w, length)
            ys = np.random.randint(self._border_size, self._img.shape[-1]-self._border_size-self._h, length)
            return np.stack((xs,ys), axis=-1)
        elif self._sample == 'grid':
            stride = (self._w, self._h) if self._stride is None else self._stride
            xrange = np.arange(self._border_size, self._img.shape[-2] - self._border_size - self._w, stride[0])
            yrange = np.arange(self._border_size, self._img.shape[-1] - self._border_size - self._h, stride[1])
            return np.stack(np.meshgrid(xrange, yrange, indexing = 'ij')).reshape(2,-1).T

    def _get_crop(self, idx):
        x, y = self._xys[idx]
        return self._img[...,x:x+self._w, y:y+self._h]

    def _get_labels(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self._xys.shape[0]

    def __getitem__(self, idx):
        img = self._get_crop(idx)
        if len(img.shape) < 3:
            img = torch.unsqueeze(torch.from_numpy(img.astype(np.float32)), 0)
        else:
            img = torch.from_numpy(img.astype(np.float32))
        if self._boxcls is None:
            labels = torch.from_numpy(self._get_labels(idx).astype(np.float32))
        else:
            lbls  = self._get_labels(idx)
            try:
                labels, clslbls = lbls
            except ValueError:
                print(lbls.shape, idx, self._fname)
                raise
            labels = torch.from_numpy(labels.astype(np.float32))
            clslbls = torch.from_numpy(clslbls)
        return (img, labels) if self._boxcls is None else (img, [labels, clslbls])

    def reset(self):
        self._img, self._boxes = self._data_augmentation()
        self._xys = self._init_coordinates()
