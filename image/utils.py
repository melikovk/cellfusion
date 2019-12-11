import json
import os.path
import glob
import numpy as np
import cv2
import skimage
import skimage.io as io
import skimage.transform as transform
from skimage.exposure import adjust_gamma
from matplotlib import pyplot as plt
import os.path
from typing import List, Tuple, NewType
import torch
from torch.utils.data import Dataset
from .metrics.localization import iou
import xml.etree.ElementTree as ET
import tifffile as tif
import czifile as czi

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

Rect = Tuple[int,int,int,int]

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
        return np.all(iou(np.array([left,top,size,size]), nucboxes, denominator='first', keepdim=False)<threshold)

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

def read_cell_counter_xml(fname):
    counter_tree = ET.parse(fname)
    markers = []
    for marker_type in counter_tree.iter('Marker_Type'):
        mtype = int(marker_type.find('Type').text)
        for marker in marker_type.iter('Marker'):
            x = int(marker.find('MarkerX').text)
            y = int(marker.find('MarkerY').text)
            markers.append([x,y,mtype])
    return markers

def load_markers(fname):
    """ Load fusion markers created using Cell Counter plugin in ImageJ
    Marker_Type 1 corresponds to all nuclei
    Marker_Type 2 corresponds to nuclei in fused cells
    """
    tree = ET.parse(fname)
    markers = []
    for marker in tree.findall(".//Marker_Type[Type='1']/Marker"):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        markers.append((x,y,0))
    for marker in tree.findall(".//Marker_Type[Type='2']/Marker"):
        x = int(marker.find('MarkerX').text)
        y = int(marker.find('MarkerY').text)
        markers.append((x,y,1))
    return np.array(markers)

def mark_fused_boxes(markers, boxes):
    """Given an array of fusion markers and an array of nucleus boxes
    return fusion labels for the boxes. Fusion markers are expected to be
    inside nucleus boxes.
    """
    fusion = markers[markers[:,2] == 1]
    fusion_boxes = np.any(np.all(np.stack([boxes[:,0:1] < fusion[:,0:1].T,
                             boxes[:,0:1] + boxes[:,2:3] > fusion[:,0:1].T,
                             boxes[:,1:2] < fusion[:,1:2].T,
                             boxes[:,1:2] + boxes[:,3:4] > fusion[:,1:2].T], axis=-1), axis=-1), axis=-1)
    return fusion_boxes.astype(int)

CHANNELS_TO_DIRNAMES = {'DAPI1%': 'nuclei', 'DAPI5%': 'nuclei', 'DAPI10%': 'nuclei',
                        'DAPI20%': 'nuclei','DAPI50%': 'nuclei','DAPI90%': 'nuclei',
                        'Trans1%': 'phase', 'Trans5%': 'phase', 'Trans10%': 'phase',
                        'Trans20%': 'phase','Trans50%': 'phase','Trans90%': 'phase',
                        'DAPI': 'nuclei', 'AF568': 'red', 'AF488': 'green',
                        'T-PMT': 'phase'}

def split_imagej_channels(imgpath, filedir, index, channel_to_dirname = CHANNELS_TO_DIRNAMES):
     with tif.TiffFile(imgpath) as implus:
        fname = f'image{index:06d}.tif'
        img = implus.asarray()
        channels = implus.micromanager_metadata['Summary']['ChNames']
        pixel_size = implus.micromanager_metadata['Summary']['PixelSize_um']
        for i, chname in enumerate(channels):
            try:
                fpath = os.path.join(filedir, channel_to_dirname[chname], fname)
            except KeyError:
                print(f"Unexpected channel {chname} in image {imgpath}")
                return
            metadata = {'filename':imgpath, 'pixel_size':pixel_size}
            tif.imwrite(fpath, img[i], imagej=True, metadata=metadata)

def split_czi_channels(imgpath, filedir, channel_to_dirname = CHANNELS_TO_DIRNAMES):
     with czi.CziFile(imgpath) as implus:
        fname = os.path.basename(imgpath)[:-4]+'.tif'
        img = implus.asarray().squeeze()
        metadata = ET.fromstring(implus.metadata())
        channels = ['']*img.shape[0]
        for chnl in metadata.iterfind('.//Channel[@Id]'):
            channels[int(chnl.get('Id')[-1:])]=chnl.get('Name').split('-T')[0]
        pixel_size = 0
        for child in metadata.find(".//Distance[Value][@Id]"):
            if child.tag == 'Value':
                pixel_size = float(child.text)*1e6
        for i, chname in enumerate(channels):
            try:
                fpath = os.path.join(filedir, channel_to_dirname[chname], fname)
            except KeyError:
                print(f"Unexpected channel {chname} in image {imgpath}")
                return
            metadata = {'filename':imgpath, 'pixel_size':pixel_size}
            tif.imwrite(fpath, img[i], imagej=True, metadata=metadata)
