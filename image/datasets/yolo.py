import json
import numpy as np
import skimage.io as io
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ..utils import centerinside, iosnd

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

class YoloGridDataset(Dataset):
    """ Simple dataset class to be used in pytorch
    Takes:
        data: Tuple(imgname, lblname, winsize, stride, bsize, transforms)
                imgname - path to image large image
                lblname - path to file with bounding boxes
                winsize - size of the window crops (w, h)
                stride - stride of the window crops
                bsize - size of the border to ignore
                ignore_thresh - maximum iou for intersection with "ignore" boxes
                nuc_thresh - minimum intersection with nuclei boxes
                transforms - transforms
        transform: callable object to transform each image
    Return:
        Dataset instance

    File with bounding boxes may contain boxes labelled as "ignore" if any ignore box overlaps
    window more than ignore_thresh the window is skipped, if any nucleus box overlaps window
    less than nuc_thresh window is also ignored, windows without any nuclei ignored as well
    """

    def __init__(self, imgname, lblname, winsize=(224,224), stride=(64,64), bsize=32, grid_size=32, ignore_thresh=0.5, nuc_thresh=.8, transforms=None):
        self.img = io.imread(imgname).T
        self.w = winsize[0]
        self.h = winsize[1]
        self.grid_size = grid_size
        self.transforms = transforms
        with open(lblname, 'r') as f:
            boxes = json.load(f)
        self.nucboxes = np.array([box['bounds'] for box in boxes if box['type'] == NUCLEUS])
        skipboxes = np.array([box['bounds'] for box in boxes if box['type'] == IGNORE]).reshape(-1,4) # reshape in case there are no ignore boxes
        self.xs = []
        self.ys = []
        for x0 in range(bsize, self.img.shape[0]-bsize-self.w, stride[0]):
            for y0 in range(bsize, self.img.shape[1]-bsize-self.h, stride[1]):
                window = (x0,y0,self.w, self.h)
                skipoverlaps = iosnd(window, skipboxes)
                nucs = self.nucboxes[centerinside(window, self.nucboxes)]
                if np.all(skipoverlaps <= ignore_thresh) and nucs.shape[0]>0 and np.all(iosnd(window, nucs)>=nuc_thresh):
                    self.xs.append(x0)
                    self.ys.append(y0)

    def __len__(self):
        return len(self.xs)

    def _getlabels(self, idx):
        x, y = self.xs[idx], self.ys[idx]
        labels = np.zeros((5, self.w//self.grid_size, self.h//self.grid_size))
        nucboxes = self.nucboxes[centerinside((x,y,self.w, self.h), self.nucboxes)]
        nucboxes[:,0] = nucboxes[:,0] - x
        nucboxes[:,1] = nucboxes[:,1] - y
        for box in nucboxes:
            xbox, ybox, wbox, hbox = box
            xbox, ybox = xbox+(wbox-1)//2, ybox+(hbox-1)//2
            xidx, yidx = xbox//self.grid_size, ybox//self.grid_size
            xpos, ypos = xbox%self.grid_size/self.grid_size, ybox%self.grid_size/self.grid_size
            labels[:, xidx, yidx] = [1, xpos, ypos, wbox/self.grid_size, hbox/self.grid_size]
        return labels

    def __getitem__(self, idx):
        img = self.transforms(self.img[self.xs[idx]:self.xs[idx]+self.w, self.ys[idx]:self.ys[idx]+self.h])
        labels = self._getlabels(idx).astype(np.float32)
        return (img, labels)

class YoloRandomDataset(Dataset):
    """ Simple dataset class to be used in pytorch
    Takes:
        data: Tuple(imgname, lblname, winsize, stride, bsize, transforms)
                imgname - path to image large image
                lblname - path to file with bounding boxes
                winsize - size of the window crops (w, h)
                stride - stride of the window crops
                bsize - size of the border to ignore
                ignore_thresh - maximum iou for intersection with "ignore" boxes
                nuc_thresh - minimum intersection with nuclei boxes
                transforms - transforms
        transform: callable object to transform each image
    Return:
        Dataset instance
    """

    def __init__(self, imgname, lblname, seed=None, winsize=(224,224), bsize=0, grid_size=32, length=1000, transforms=None):
        self.img = io.imread(imgname).T
        self.w = winsize[0]
        self.h = winsize[1]
        self.bsize = bsize
        self.grid_size = grid_size
        self.length = length
        self.transforms = transforms
        self.seed = seed
        imgw, imgh = self.img.shape
        # read nucleus boxes and normalize sizes to grid_size
        with open(lblname, 'r') as f:
            boxes = json.load(f)
        self.nucboxes = np.array([box['bounds'] for box in boxes if box['type'] == NUCLEUS])/grid_size
        # Convert left corners into center positions
        self.nucboxes[:,0] = self.nucboxes[:,0] + self.nucboxes[:,2]/2
        self.nucboxes[:,1] = self.nucboxes[:,1] + self.nucboxes[:,3]/2
        np.random.seed(seed)
        self.xs = np.random.randint(bsize, imgw-bsize-self.w, length)
        self.ys = np.random.randint(bsize, imgh-bsize-self.h, length)

    def __len__(self):
        return self.length

    def _getlabels(self, idx):
        x, y = self.xs[idx]/self.grid_size, self.ys[idx]/self.grid_size
        labels = np.zeros((5, self.w//self.grid_size, self.h//self.grid_size))
        nucboxes = self.nucboxes[centerinside((x,y,self.w/self.grid_size, self.h/self.grid_size), self.nucboxes, lt_anchor='center')].reshape(-1,4)
        nucboxes[:,0] = nucboxes[:,0] - x
        nucboxes[:,1] = nucboxes[:,1] - y
        for box in nucboxes:
            xbox, ybox, wbox, hbox = box
            xidx, yidx = int(np.floor(xbox)), int(np.floor(ybox))
            xpos, ypos = xbox - xidx, ybox - yidx
            labels[:, xidx, yidx] = [1, xpos, ypos, wbox, hbox]
        return labels

    def __getitem__(self, idx):
        if self.transforms is not None:
            img = self.transforms(self.img[self.xs[idx]:self.xs[idx]+self.w, self.ys[idx]:self.ys[idx]+self.h])
        else:
            img = self.img[self.xs[idx]:self.xs[idx]+self.w, self.ys[idx]:self.ys[idx]+self.h]
        labels = self._getlabels(idx).astype(np.float32)
        return (img, labels)

    def reset(self):
        if self.seed is None:
            imgw, imgh = self.img.shape
            self.xs = np.random.randint(self.bsize, imgw-self.bsize-self.w, self.length)
            self.ys = np.random.randint(self.bsize, imgh-self.bsize-self.h, self.length)

def labelsToBoxes(labels, grid_size=32, offset=(0,0), threshold = 0.5):
    """ Function to convert yolo type model output to bounding boxes
    Parameters:
        labels:     [batch_size:5:width:height] tensor of values
                    2nd dimension stores [Pobj:Xcenter:Ycenter:W:H]
                    all dimensions are normalized to grid_size
        grid_size:  Size of the model grid
        offset:     offset of the crop in the image for multicrop predictions
        threshold:  Pobj threshold to use
    Returns:
        ([(Xlt,Ylt,W,H)], [Pobj]) all coordinates are int values in pixels
    """
    if isinstance(offset, int):
        offx = offy = offset
    else:
        offx, offy = offset
    _, wi, hi = labels.shape
    boxes = []
    scores = []
    for xi in range(wi):
        for yi in range(hi):
            if labels[0, xi, yi] > threshold:
                w, h = labels[3,xi,yi]*grid_size, labels[4,xi,yi]*grid_size
                x = xi*grid_size + labels[1,xi,yi]*grid_size - w/2
                y = yi*grid_size + labels[2,xi,yi]*grid_size - h/2
                boxes.append((offx+np.round(x), offy+np.round(y), np.round(w), np.round(h)))
                scores.append(labels[0,xi,yi])
    return boxes, scores

class randomIdx:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __getitem__(self, idx):
        return np.random.randint(self.low, self.high)
