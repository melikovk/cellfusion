import numpy as np
import torch

def iou(boxes1, boxes2, usegpu = False, gpu = 0, keepdim = False, denominator = 'union'):
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
        usegpu: True | False
        gpu:    int - GPU device number to use
        keepdim: True | False
                if True result is always 2D
        denominator: 'union' | 'first'
                Calculate IOU or normalize to the area of the 1st box
    Returns
        iou: numpy array
    """
    # parse arguments
    assert isinstance(boxes1, np.ndarray) or torch.is_tensor(boxes1), \
        "first argument should be ndarray or pytorch Tensor"
    assert isinstance(boxes2, type(boxes1)), \
        "second argument should have the same type as the first"
    assert len(boxes1.shape)<3 and len(boxes2.shape)<3, \
        "Array or tensor should be 1D (length 4) or 2D (n x 4)"
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4, \
        "Each box should have 4 coordinates: left, top, width, height"
    assert denominator == 'union' or denominator == 'first', \
        "denominator should have value {'union'|'first'}"
    if usegpu:
        assert gpu >= 0 and gpu < torch.cuda.device_count(), \
            "Invalid GPU device number"
    # convert 1D to 2D and ndarray to Tensor
    b1, b2 = boxes1, boxes2
    if len(b1.shape)==1:
        b1 = b1.reshape((1,4))
    if len(b2.shape)==1:
        b2 = b2.reshape((1,4))
    if isinstance(boxes1, np.ndarray):
        b1 = torch.from_numpy(b1)
        b2 = torch.from_numpy(b2)
    if usegpu:
        b1 = b1.cuda(gpu)
        b2 = b2.cuda(gpu)
    zero = torch.zeros(1, dtype = b1.dtype, device=b1.device)
    # Calculate metric
    x1, y1, w1, h1 = b1.split(1, dim=1)
    x2, y2, w2, h2 = b2.split(1, dim=1)
    iwidths = torch.max(torch.min(x1+w1, (x2+w2).t()) - torch.max(x1, x2.t()),zero)
    iheights = torch.max(torch.min(y1+h1, (y2+h2).t()) - torch.max(y1, y2.t()) ,zero)
    iareas = iwidths*iheights
    if denominator == 'union':
        denom = w1*h1 + (w2*h2).t() - iareas
    else:
        denom = w1*h1
    result = iareas/denom
    if not keepdim:
        result = result.squeeze()
    if isinstance(boxes1, np.ndarray):
        return result.cpu().numpy()
    else:
        return result.to(boxes1.device)

def mean_iou_image(predict, target, scores=None):
    """ Calculates average IOU per image for intersection between
    predicted and target bounding boxes. All boxes are assumed to predict
    same class of objects. If scores are given than they will be used
    to sort prediction boxes in ascending order of their prediction
    confidence score. See documentation of match_boxes() function
    on the deatils of the matching procedure. If no prediction matches
    the target box it will contribute 0 IOU to the mean
    Parameters:
        predict: (n,4) Tensor of box predictions
        target: (n, 4) Tensor of box targets
        scores: (n,) Tensor of confidence scores for predictions
    Returns:
        mean_iou: (1,) Tensor with mean IOU
    """
    if torch.is_tensor(scores):
        predict = predict[scores.argsort()]
    match_ious, _, _ = match_boxes(predict, target)
    return match_ious.sum()/(predict.shape[0]+target.shape[0]-match_ious.shape[0])

def precision_recall_jaccard(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """ Calculates precision, recall and Jaccard score for object detection given
    specified IOU threshold for intersection between predicted and target
    bounding boxes. All boxes are assumed to predict same class of objects.
    If scores are given than they will be used to sort prediction boxes in
    ascending order of their prediction confidence score. See documentation
    of match_boxes() function on the deatils of the matching procedure.
    Parameters:
        predict: (n,4) Tensor of box predictions
        target: (n, 4) Tensor of box targets
        iou_threshold: float IOU above which prediction is counted as positive
        scores: (n,) Tensor of confidence scores for predictions
        score_threshold: float Remove predicted boxes with score below the threshold
    Returns:
        (precision, recall, f1): ((1,), (1,), (1,)) Tuple of Tensors
    """
    if torch.is_tensor(scores):
        n_score = torch.sum(scores > score_threshold, dtype=torch.long)
        predict = predict[scores.argsort()[:n_score]]
    match_ious, _, _ = match_boxes(predict, target)
    tp = torch.sum(match_ious > iou_threshold, dtype=torch.float)
    precision = tp/predict.shape[0]
    recall = tp/target.shape[0]
    jaccard = tp/(predict.shape[0]+target.shape[0]-tp)
    return precision, recall, jaccard

def precision(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard() function"""
    return precision_recall(predict, target, iou_threshold, scores, score_threshold)[0]

def recall(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard() function"""
    return precision_recall(predict, target, iou_threshold, scores, score_threshold)[1]

def f1(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Calculates F1 = 2*precision*recall/(precision+recall)
    For additional information refer to documentation on precision_recall() function"""
    precision, recall = precision_recall(predict, target, iou_threshold, scores, score_threshold)
    return 2*precision*recall/(precision+recall)

def jaccard_score(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard() function"""
    return precision_recall(predict, target, iou_threshold, scores, score_threshold)[0]

def match_boxes(predict, target):
    """Assigns predicted bounding boxes to ground truth boxes. Returns IOUs
    and indexes of matched boxes. Gready box assignement used.
    Each predicted box can be assigned to only 1 target box (box with maximal IOU).
    If multiple predicted boxes are assigned to the same target box
    only the box with maximal prediction score is used (other prediction
    boxes will have IOU of 0). The prediction boxes should be sorted
    in ascending order of their prediction confidence score.
    Parameters:
        predict: (n,4) Tensor of box predictions
        target: (n, 4) Tensor of box targets
    Returns:
        (ious, p_idxs, t_idxs): Tuple of tensors
    """
    dev = predict.device
    ious, idxs = iou(predict, target, keepdim=True).max(dim=-1)
    match_ious = torch.zeros(target.shape[0], device=dev).index_put_((idxs,),ious)
    p_idxs = -torch.ones(target.shape[0], dtype=torch.long, device=dev)
    p_idxs.index_put_((idxs,),torch.arange(predict.shape[0], device=dev))
    t_idxs = torch.arange(target.shape[0], device=dev)
    matched = match_ious.nonzero().squeeze()
    return match_ious[matched], p_idxs[matched], t_idxs[matched]
