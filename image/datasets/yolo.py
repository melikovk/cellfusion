import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ..utils import centerinside
from ..metrics.localization import iou
import torch
import math
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from matplotlib import patches
import torch.multiprocessing as mp
import os


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
            # cpu_num = os.cpu_count()
            # with mp.Pool(cpu_num) as pool:
            #     self.dataset.datasets = pool.map(reset_dataset, self.dataset.datasets)
        else:
            self.dataset.reset()
        return super().__iter__()

def get_boxes_from_json(fname, clsname = None):
    with open(fname, 'r') as f:
        boxes = json.load(f)
    if clsname is None:
        return np.array([box['bounds'] for box in boxes])
    else:
        return tuple(map(np.array, zip(*[(box['bounds'], box[clsname]) for box in boxes])))

def save_boxes_to_json(boxes, fname, scores=None, clsname=None, clslbl=None, clsscores=None):
    if scores is not None:
        probs = expit(scores).astype(float)
    if clsscores is not None:
        clsprobs = softmax(clsscores, axis=1)
    records = [{'bounds': box.round().astype(int).tolist()} for box in boxes]
    if scores is not None:
        for idx in range(len(records)):
            records[idx]['p'] = probs[idx]
    if clsname is not None:
        if clslbl is not None:
            for idx in range(len(records)):
                records[idx][clsname] = int(clslbl[idx])
        if clsscores is not None:
            for idx in range(len(records)):
                records[idx]['p_'+clsname] = clsscores[idx].tolist()
        if clslbl is None and clsscores is None:
            raise ValueError("If class name is provided at least one of the class "
                              "labels or class scores should be provided")
    with open(fname, 'w') as f:
        json.dump(records, f)

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
        # self._needs_reset = False

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
        # if self._needs_reset:
        #     self._img, self._boxes = self._data_augmentation()
        #     self._xys = self._init_coordinates()
        #     self._needs_reset = False
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
        # self._needs_reset = True
        self._img, self._boxes = self._data_augmentation()
        self._xys = self._init_coordinates()


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

class MultiAnchorDataset(CropDataset):
    def __init__(self, imgname, lblname, cell_anchors, window_overlap_threshold = .25, **kwargs):
        super().__init__(imgname, lblname, **kwargs)
        self._cell_anchors = cell_anchors
        self._window_thresh = window_overlap_threshold
        self._anchors = get_grid_anchors(self._cell_anchors, self._w/self._grid_size, self._h/self._grid_size).transpose(1,2,3,0)

    def _get_boxes(self, idx):
        left_x, top_y = self._xys[idx]/self._grid_size
        w, h = self._w//self._grid_size, self._h//self._grid_size
        box_idxs = iou(self._boxes, np.array([left_x, top_y, w, h]), denominator='first').squeeze()>self._window_thresh
        boxes = self._boxes[box_idxs]
        boxes[:,0:2] = boxes[:,0:2] - [left_x, top_y]
        boxcls = None if self._boxcls is None else self._boxcls[box_idxs]
        return boxes, boxcls

class YoloDataset(MultiAnchorDataset):
    """ Concrete multianchor dataset that uses Yolo like box assignement. We assign
    each box to only one anchor (anchor with highest iou with the box) and
    set ignore label (-1 for objectness) for anchors that have iou higher than specified threshold
    with any true box.
    """
    def __init__(self, imgname, lblname, anchor_ignore_threshold = 0.5, denominator = 'union', **kwargs):
        self._ignore_thresh = anchor_ignore_threshold
        self._denominator = denominator
        super().__init__(imgname, lblname, **kwargs)

    def _get_labels(self, idx):
        w, h = self._w//self._grid_size, self._h//self._grid_size
        n_anchors = self._anchors.shape[0]
        anchors = self._anchors.reshape(-1,4)
        labels = np.zeros(anchors.shape[0])
        coordinates = np.zeros(4*anchors.shape[0])
        xs, ys, ws, hs = np.split(coordinates, 4)
        # Filter out boxes that overlap less than threshold with the window
        boxes, boxcls = self._get_boxes(idx)
        iou_matrix = iou(anchors, boxes, denominator=self._denominator)
        ignore_mask = (iou_matrix > self._ignore_thresh).any(axis=-1)
        labels[ignore_mask] = -1.0
        # Set labels for anchors that have maximum IOU with true boxes to 1.0
        match_ious, match_idxs = iou_matrix.max(axis=0), iou_matrix.argmax(axis=0)
        labels[match_idxs] = 1.0
        # Remaining anchors are background boxes
        xs[match_idxs] = (boxes[:,0] + boxes[:,2]/2 - anchors[match_idxs, 0])/anchors[match_idxs, 2]
        ys[match_idxs] = (boxes[:,1] + boxes[:,3]/2 - anchors[match_idxs, 1])/anchors[match_idxs, 3]
        ws[match_idxs] = boxes[:,2]/anchors[match_idxs, 2]
        hs[match_idxs] = boxes[:,3]/anchors[match_idxs, 3]
        # Set class labels if needed
        if boxcls is not None:
            clslbls = np.full_like(labels, -1, dtype=np.long)
            clslbls[match_idxs] = boxcls[np.arange(match_idxs.shape[0], dtype=np.long)]
            clslbls = clslbls.reshape((n_anchors, w, h))
        # Reshape
        labels = labels.reshape((n_anchors, w, h))
        coordinates = coordinates.reshape((4*n_anchors, w, h))
        if boxcls is None:
            return np.concatenate((labels, coordinates))
        else:
            return np.concatenate((labels, coordinates)), clslbls

class SSDDataset(MultiAnchorDataset):
    """ Concrete multianchor dataset that uses SSD like box assignement. We assign
    each true box to multiple anchors (anchor with highest IOU for this box and
    any other anchor with IOU higher than the threshold (0.5 to 0.7)).
    Anchors that have IOU with all true boxes below another threshold (around 0.3) are
    set to background. All other anchors are set to ignore and ignored during training.
    """
    def __init__(self, imgname, lblname, positive_anchor_threshold = 0.7,
                 background_anchor_threshold = 0.5, denominator = 'union', **kwargs):
        self._positive_thresh = positive_anchor_threshold
        self._bkg_thresh = background_anchor_threshold
        self._denominator = denominator
        super().__init__(imgname, lblname, **kwargs)

    def _get_labels(self, idx):
        w, h = self._w//self._grid_size, self._h//self._grid_size
        n_anchors = self._anchors.shape[0]
        anchors = self._anchors.reshape(-1,4)
        # Filter out boxes that overlap less than threshold with the window
        boxes, boxcls = self._get_boxes(idx)
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


def get_cell_anchors(scales, anchors):
    """ Create set of anchors for single cell on the grid
    All values are normalized to cell sizes (width and height)
    Parameters:
        scales: [scale: float, ...]
        anchors: [(offset_x: float, offset_y: float, w_to_h_ration: float), ...]
                 Offsets are between anchor center and grid cell center
    Return:
        cell_anchors: ndarray([[left, top, width, height], ...])
                      Left-top corner of the anchors are relative to the left-top
                      corner of the grid cell
    """
    cell_anchors = []
    anchors = [(0.,0.,1.)] + anchors
    for s in scales:
        for dx, dy, whr in anchors:
            w = whr / math.sqrt(whr)
            h = 1 / math.sqrt(whr)
            cell_anchors.append([0.5+s*(dx-w/2), 0.5+s*(dy-h/2), s*w, s*h])
    return np.array(cell_anchors)

def get_grid_anchors(cell_anchors, w, h):
    """Create a grid of anchors give array of cell anchors and width and height
    of the grid. Returns ndarray of shape (4, NAnchors, w, h)
    """
    anchors_grid = np.mgrid[0:w,0:h]
    anchors_grid = np.concatenate((anchors_grid, np.zeros_like(anchors_grid)))
    anchors_grid = np.expand_dims(anchors_grid, axis=1)
    anchors_grid = anchors_grid + cell_anchors.T.reshape((4,-1,1,1))
    return anchors_grid

def labels_to_boxes(labels, grid_size, cell_anchors, clslbls=None, threshold = 0.5, offset=(0,0)):
    """ Function to convert object loacalization model output to bounding boxes
    Parameters:
        labels:     [5*n_anchors:width:height] or
                    [Batch_size:5*n_anchors:width:height] Tensor of predictions
                    1st dimension stores [Pobj:Xcenter:Ycenter:W:H]
                    all dimensions are normalized to grid_size
        grid_size:  Size of the model grid
        cell_anchors: ndarray with cell anchors
        offset:     offset of the crop in the image for multicrop predictions (in pixels)
        threshold:  Pobj threshold to use
        cell_anchors: ndarray with anchors for single grid cell
                      should be the same as the one used to train the model
    Returns:
        (ndarray(Xlt,Ylt,W,H), ndarray(Pobj)) all coordinates are float values in pixels
        if given batch return list of the predictions for each image
    """
    if len(labels.shape) == 4:
        if clslbls is None:
            return [labels_to_boxes(img, grid_size, cell_anchors, clslbls, threshold, offset) for img in labels]
        else:
            return [labels_to_boxes(img, grid_size, cell_anchors, imgcls, threshold, offset) for img, imgcls in zip(labels, clslbls)]
    if isinstance(offset, int):
        offx = offy = offset
    else:
        offx, offy = offset
    _, w, h = labels.shape
    # Create grid of all anchors
    anchors_grid = get_grid_anchors(cell_anchors, w, h).reshape((4,-1))
    # Select booxes
    labels = labels.cpu().numpy().reshape((5, -1))
    logit_threshold = math.log(threshold/(1-threshold))
    idx = (labels[0] > logit_threshold).nonzero()[0]
    scores = labels[0, idx]
    boxes = labels[1:, idx]
    anchors_grid = anchors_grid[:, idx]
    # Recalculate box sizes and positions
    boxes[:2] = (boxes[:2] - boxes[-2:]/2) * anchors_grid[-2:] + anchors_grid[:2]
    boxes[-2:] = boxes[-2:] * anchors_grid[-2:]
    boxes = boxes*grid_size
    boxes[0] += offx
    boxes[1] += offy
    # get class labels or class scores if needed
    if clslbls is not None:
        boxcls = clslbls.cpu().numpy().reshape((-1, labels.shape[-1]))[:, idx].T
    return (boxes.T, scores) if clslbls is None else (boxes.T, scores, boxcls)

def show_boxes(image, labels, grid_size, cell_anchors, threshold=0.5):
    """
        Function to display image and bounding boxes given the output of CropDataset
    """
    f, ax = plt.subplots()
    img = image.numpy()[0].T
    boxes, scores = labels_to_boxes(labels, grid_size, cell_anchors, threshold)
    if len(boxes) > 0:
        xmin, ymin = min(0, boxes[:,0].min()), min(0, boxes[:,1].min())
        xmax, ymax = max(img.shape[0],(boxes[:,0]+boxes[:,2]).max()), max(img.shape[1],(boxes[:,1]+boxes[:,3]).max())
    else:
        xmin, ymin, xmax, ymax = 0, 0, img.shape[0], img.shape[1]
    ax.imshow(img)
    ax.tick_params(left= True,bottom= True)
    ax.set_axis_off()
    for box in boxes:
        rect = patches.Rectangle(box[0:2], *box[-2:], fill=False, edgecolor='red')
        ax.add_patch(rect)
#     rect = patches.Rectangle((-xmin,-ymin), *img.shape, fill=False, edgecolor='green')
#     ax.add_patch(rect)
    ax.set(xlim=(xmin-5,xmax+5), ylim=(ymax+5, ymin-5))

# def reset_dataset(dset):
#     dset.reset()
#     return dset
