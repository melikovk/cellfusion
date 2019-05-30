import json
import glob
import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
import skimage.io as io
import skimage.transform as transform
import os.path
from typing import List, Tuple, NewType
import torch
from torch.utils.data import Dataset
from tensorflow.keras.utils import Sequence
from skimage.exposure import adjust_gamma

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

Rect = Tuple[int,int,int,int]

class CropDataset(Dataset):
    """ Simple dataset class to be used in pytorch
    Takes:
        data: Tuple(imgs, labels)
                imgs is a numpy array withh all image crops
                labels - numpy array with corresponding class labels
        transform: callable object to transform each image
    Return:
        Dataset instance
    """
    def __init__(self, imgs, labels=None, transforms=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.transforms is not None:
            if type(idx) is slice:
                start = 0 if idx.start is None else idx.start
                stop =  len(self.imgs) if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step
                imgs = np.stack([self.transforms(self.imgs[i]) for i in range(start, stop, step)], axis=0)
            else:
                imgs = self.transforms(self.imgs[idx])
        else:
            imgs = self.imgs[idx]
        if self.labels is None:
            return imgs
        else:
            return (imgs, self.labels[idx])

# class CropSequence(Sequence):
#     def __init__(self, data, batch_size):
#         self.dataset = CropDataset(data)
#         self.batch_size = batch_size
#
#     def __len__(self):
#         return len(self.dataset)//self.batch_size
#
#     def __getitem__(self, idx):
#         return self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]

def getCropBox(box: Tuple, size: int, bsize: int, imgsize: Tuple)->Tuple:
    """ Returns a crop box around the region of interest (ROI)
    Takes
        box: (left_x: int, top_y: int, width: int, height: int) - initial ROI box
        size: int  - desired size of the square region
        bsize: int - border size around the target square region
        imgsize: (width: int, height: int) - size of the image from which crops are made
    Returns
        (left_x: int, top_y: int, right_x: int, bottom_y: int)
        or None if desired crop box cannot be created

    Functions finds the biggest possible crop that includes initial region of interest
    and is as close in size to the square with the side size+bsize """
    x0, y0, w, h = box
    # Find center of ROI
    xc, yc = x0+w//2, y0+h//2
    # Attempt the crop box with bsize border around all sides
    size = max(w, h, size)+2*bsize
    # Crop if hit the edge of the image
    x0, y0 = max(0, xc-size//2), max(0, yc-size//2)
    x1, y1 = min(imgsize[0]-1, xc+size//2), min(imgsize[1]-1, yc+size//2)
    size = min(xc-x0, x1-xc, yc-y0, y1-yc)
    # If after crop size is smaller than the biggest dimension of initial ROI return None
    if 2*size+1 < max(w,h):
        # print(f'size={size}, w={w}, h={h}')
        crop = None
    else:
        crop = (xc-size, yc-size, xc+size, yc+size)
    return crop

def getStrictCropBox(box: Rect, size: int, imgsize: Tuple[int, int])->Tuple[int,int,int,int]:
    """ Returns a crop box around the region of interest (ROI)
    Takes
        box: (left_x: int, top_y: int, width: int, height: int) - initial ROI box
        size: int  - desired size of the square region
        imgsize: (width: int, height: int) - size of the image from which crops are made
    Returns
        (left_x: int, top_y: int, right_x: int, bottom_y: int)
        or None if desired crop box cannot be created
    """
    x0, y0, w, h = box
    if w > size or h > size:
        return None
    xc, yc = x0+w//2, y0+h//2
    if xc-size//2 < 0 or yc-size//2 < 0 or xc+size//2 > imgsize[0]-1 or yc+size//2 > imgsize[1]-1:
        return None
    crop = (xc-size//2, yc-size//2, xc+size//2, yc+size//2)
    return crop

def getPartialCropBoxes(box:Rect, size: int, imgsize: Tuple[int,int])->List[Tuple[int,int,int,int]]:
    x0, y0, w, h = box
    xc, yc = x0+w//2, y0+h//2
    # offsets = [(size,size), (size, size//2), (size, 0), (size//2, size), (size//2,0), (0,size), (0, size//2), (0, 0)]
    offsets = [(size,size), (size, 0), (0,size), (0, 0)]
    crops = [(xc-xoff, yc-yoff, xc-xoff+size, yc-yoff+size) for xoff, yoff in offsets if
             xc-xoff>=0 and yc-yoff>=0 and xc-xoff+size<imgsize[0] and yc-yoff+size<imgsize[1]]
    return crops

def showChannels(img, cmaps=None):
    if cmaps is None:
        cols = ['Blues_r', 'Reds_r', 'Greens_r']
    elif isinstance(cmaps, list):
        cols = cmaps
    else:
        cols = [cmaps]*3
    f, axes = plt.subplots(1, 3)
    for i in range(3):
        axes[i].imshow(img[:,:,i], cmap=cols[i])
        axes[i].set_axis_off()
    return f

def getPartialNucCrops(fpaths: List, size: int)->Tuple:
    """
    """
    cropImgs = []
    for path in fpaths:
        fname = os.path.split(path)[1][:11]
        fdir = os.path.split(path)[0]
        img = io.imread(fdir+'/'+fname+'.tif').T
        with open(path, 'r') as f:
            initboxes = json.load(f)
        initboxes = [box['bounds'] for box in initboxes if box['type'] == NUCLEUS]
        newboxes = []
        for box in initboxes:
            newbox = getPartialCropBoxes(box, size, img.shape)
            newboxes = newboxes+newbox
        crops = np.stack([img[x0:x1,y0:y1] for x0, y0, x1, y1 in newboxes], axis = 0)
        cropImgs.append(crops)
    return np.concatenate(cropImgs)

def getnucleusCrops(fpaths: List, size: int)->Tuple:
    """
    """
    cropImgs = []
    for path in fpaths:
        fname = os.path.split(path)[1][:11]
        fdir = os.path.split(path)[0]
        img = io.imread(fdir+'/'+fname+'.tif').T
        with open(path, 'r') as f:
            initboxes = json.load(f)
        initboxes = [box['bounds'] for box in initboxes if box['type'] == NUCLEUS]
        newboxes = []
        for box in initboxes:
            newbox = getStrictCropBox(box, size, img.shape)
            if newbox is not None:
                newboxes.append(newbox)
        crops = np.stack([img[x0:x1,y0:y1] for x0, y0, x1, y1 in newboxes], axis = 0)
        cropImgs.append(crops)
    return np.concatenate(cropImgs)

def getbkgCrops(fpaths: List, size: int, count=1000, threshold=.05)->Tuple:
    """
    """
    def isgoodbox(arr):
        left, top = arr
        return np.all(iofst((left,top,size,size), nucboxes)<threshold)

    cropImgs = []
    for path in fpaths:
        fname = os.path.split(path)[1][:11]
        fdir = os.path.split(path)[0]
        img = io.imread(fdir+'/'+fname+'.tif').T
        with open(path, 'r') as f:
            nucboxes = json.load(f)
        nucboxes = np.array([box['bounds'] for box in nucboxes if box['type'] == NUCLEUS or box['type'] == IGNORE])
        bkgboxes = np.stack((np.random.randint(img.shape[0]-size, size=count),
                          np.random.randint(img.shape[1]-size, size=count)), axis=-1)
        bkgboxes = bkgboxes[np.apply_along_axis(isgoodbox, 1, bkgboxes)]
        crops = np.stack([img[x0:x0+size,y0:y0+size] for x0, y0 in bkgboxes], axis = 0)
        cropImgs.append(crops)
    return np.concatenate(cropImgs)

def getfusionCrops(fpaths: List, size: int, bsize: int)->Tuple:
    """
    """
    def getCrops(path):
        fname = os.path.split(path)[1][:11]
        fdir = os.path.split(os.path.split(path)[0])[0]
        img = np.stack([io.imread(fdir+c+fname+'.tif').T for c in colors], axis=-1)
        with open(path, 'r') as f:
            boxes = json.load(f)
        boxes = [box for box in boxes if box['type'] == NUCLEUS]
        crops, labels = [], []
        for box in boxes:
            crop = getCropBox(box['bounds'], size, bsize, img.shape[:-1])
            if crop is not None:
                crops.append(crop)
                labels.append(box['fusion'])
        cropImgs = np.stack([cv2.resize(img[x0:x1,y0:y1,:], (size+2*bsize, size+2*bsize)) for x0, y0, x1, y1 in crops], axis = 0)
        labels = np.array(labels)
        return (cropImgs, labels)
    colors = ['/nuclei/', '/red/', '/green/']
    boxes, labels = zip(*[getCrops(path) for path in fpaths])
    return (np.concatenate(boxes), np.concatenate(labels))

def centerinside(box:Rect, boxes, lt_anchor = 'topleft'):
    """ Given a rectangular box and a list of rectangular boxes
        determines if centers of boxes in the list are inside
        the first box.
    Takes
        box: Rect(left: int, top: int, width: int, height: int)
        boxes: numpy array with dimensions num_of_boxes x 4 (left, top, width, height) or (center_x, center_y, width, height)
        lt_anchor: determines if box anchor (first two values) is top left corner 'topleft' or center of the box 'center'
    Returns
        List[Bool]
    """
    x, y, w, h = box
    if lt_anchor == 'topleft':
        xs = boxes[:,0] + boxes[:,2]/2
        ys = boxes[:,1] + boxes[:,3]/2
    elif lt_anchor == 'center':
        xs = boxes[:,0]
        ys = boxes[:,1]
    else:
        raise ValueError("lt_anchor should be {'topleft'|'center'}")
    return np.all(np.stack((xs>=x, xs<x+w, ys>=y, ys<y+h)), axis=0)

def iou(box:Rect, boxes):
    """ Given a rectangular box and a list of rectangular boxes
        returns IOU (intersection over union) metric for interestion
        of the first box with every box in the list
    Takes
        box: Rect(left: int, top: int, width: int, height: int)
        boxes: numpy array with dimensions num_of_boxes x 4 (left, top, width, height)
    Returns
        iou: numpy array
    """
    left, top, width, height = box
    right, bottom = left + width, top + height
    iwidths = np.maximum((np.minimum(boxes[:,0]+boxes[:,2], right) - np.maximum(boxes[:,0], left)),0)
    iheights = np.maximum((np.minimum(boxes[:,1]+boxes[:,3], bottom) - np.maximum(boxes[:,1], top)),0)
    iareas = iwidths*iheights
    uareas = width*height + boxes[:,2]*boxes[:,3] - iareas
    return iareas/uareas

def iou_new(boxes1, boxes2):
    """ Given 2 list of rectangular boxes
        returns IOU (intersection over union) metric for interestion
        of every box in the first list with every box in the second list
        Lists should be in the form of n x 4 ndarray or torch tensor
        where n is the number of boxes in the list and each box is represented
        as left, top, width, height
    Takes
        boxes1: numpy array or torch tensor with dimensions
                num_of_boxes x 4 (left, top, width, height)
        boxes2: numpy array or torch tensor with dimensions
                num_of_boxes x 4 (left, top, width, height)
    Returns
        iou: numpy array
    """
    # validate arguments
    assert isinstance(boxes1, np.ndarray) or torch.is_tensor(boxes1), \
        "first argument should be ndarray or pytorch Tensor"
    assert isinstance(boxes2, type(boxes1)), \
        "second argument should have the same type as the first"
    assert len(boxes1.shape)<3 and len(boxes2.shape)<3, \
        "Array or tensor should be 1D (length 4) or 2D (n x 4)"
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4, \
        "Each box should have 4 coordinates: left, top, width, height"
    # convert 1D to 2D and ndarray to Tensor
    if len(boxes1.shape)==1:
        boxes1 = boxes1.reshape((1,4))
    if len(boxes2.shape)==1:
        boxes2 = boxes2.reshape((1,4))
    isndarray = isinstance(boxes1, np.ndarray)
    if isndarray:
        boxes1 = torch.from_numpy(boxes1)
        boxes2 = torch.from_numpy(boxes2)
    x1, y1, w1, h1 = boxes1.split(1, dim=1)
    x2, y2, w2, h2 = boxes2.split(1, dim=1)
    zero = torch.zeros(1, dtype = boxes1.dtype, device=boxes1.device)
    iwidths = torch.max(torch.min(x1+w1, (x2+w2).t()) - torch.max(x1, x2.t()),zero)
    iheights = torch.max(torch.min(y1+h1, (y2+h2).t()) - torch.max(y1, y2.t()) ,zero)
    iareas = iwidths*iheights
    uareas = w1*h1 + (w2*h2).t() - iareas
    if isndarray:
        return (iareas/uareas).numpy()
    else:
        return iareas/uareas

def iofst(box:Rect, boxes:List[Rect])->List[float]:
    """ Given to rectangular boxe and a list of rectangular boxes
        returns area of intersection of first box with every box int the list
        over the area of the first box
    Takes
        box: Rect(left: int, top: int, width: int, height: int)
        boxes: List[Rect(left: int, top: int, width: int, height: int)]
    Returns
        iof: List[float]
    """
    left, top, width, height = box
    right, bottom = left + width, top + height
    iwidths = np.maximum((np.minimum(boxes[:,0]+boxes[:,2], right) - np.maximum(boxes[:,0], left)),0)
    iheights = np.maximum((np.minimum(boxes[:,1]+boxes[:,3], bottom) - np.maximum(boxes[:,1], top)),0)
    iareas = iwidths*iheights
    return iareas/(width*height)

def iosnd(box:Rect, boxes):
    """ Given to rectangular boxe and a list of rectangular boxes
        returns area of intersection of first box with every box int the list
        over the area of the second box
    Takes
        box: Rect(left: int, top: int, width: int, height: int)
        boxes: numpy array with dimensions num_of_boxes x 4 (left, top, width, height)
    Returns
        iof: numpy array
    """
    left, top, width, height = box
    right, bottom = left + width, top + height
    iwidths = np.maximum((np.minimum(boxes[:,0]+boxes[:,2], right) - np.maximum(boxes[:,0], left)),0)
    iheights = np.maximum((np.minimum(boxes[:,1]+boxes[:,3], bottom) - np.maximum(boxes[:,1], top)),0)
    iareas = iwidths*iheights
    sndareas = boxes[:,-1]*boxes[:,-2]
    return iareas/sndareas
