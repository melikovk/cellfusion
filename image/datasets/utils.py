import numpy as np
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
from matplotlib import patches
import json
import math
import torch

NUCLEUS = 0
BKG = 1
IGNORE = 2
FUSION = 1
NOFUSION = 0

def get_boxes_from_json(fname, clsname = None):
    """ Return bounding box coordinates and class labels from JSON file.
    JSON file should contain a list of dictionaries with 1 dictionary for each
    object. Dictionary should have a key 'bounds' containing left, top, width
    and height of the bounding box and a key with the same name as clsname if
    clsname is not None.
    Args:
        filename: string
        clsname: string
    Returns:
        boxes: ndarray(N,4)
        clslbl: ndarray(N,)
    """
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
    """Create a grid of anchors given array of cell anchors and width and height
    of the grid. Returns ndarray of shape (4, NAnchors, w, h)
    """
    anchors_grid = np.mgrid[0:w,0:h]
    anchors_grid = np.concatenate((anchors_grid, np.zeros_like(anchors_grid)))
    anchors_grid = np.expand_dims(anchors_grid, axis=1)
    anchors_grid = anchors_grid + cell_anchors.T.reshape((4,-1,1,1))
    return anchors_grid

def labels_to_boxes(labels, grid_size, cell_anchors, threshold = 0.5, offset=(0,0)):
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
        return [labels_to_boxes(img, grid_size, cell_anchors, threshold, offset) for img in labels]
    if isinstance(offset, int):
        offx = offy = offset
    else:
        offx, offy = offset
    _, w, h = labels.shape
    # Create grid of all anchors
    anchors_grid = torch.from_numpy(get_grid_anchors(cell_anchors, w, h).reshape((4,-1))).to(device=labels.device, dtype=labels.dtype)
    # Select booxes
    n_anchors = cell_anchors.shape[0]
    box_labels = labels[-5*n_anchors:,...].reshape((5, -1))
    logit_threshold = math.log(threshold/(1-threshold))
    idx = (box_labels[0] > logit_threshold)
    scores = box_labels[0, idx]
    boxes = box_labels[1:, idx]
    anchors_grid = anchors_grid[:, idx]
    # Recalculate box sizes and positions
    boxes[:2] = (boxes[:2] - boxes[-2:]/2) * anchors_grid[-2:] + anchors_grid[:2]
    boxes[-2:] = boxes[-2:] * anchors_grid[-2:]
    boxes = boxes*grid_size
    boxes[0] += offx
    boxes[1] += offy
    # get class labels or class scores if needed
    if labels.shape[0] > 5*n_anchors:
        boxcls = labels[:-5*n_anchors].reshape((-1, box_labels.shape[-1]))[:, idx].T
        return boxes.T, scores, boxcls
    else:
        return boxes.T, scores

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
