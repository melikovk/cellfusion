import numpy as np
import torch

def _check_box_args(boxes1, boxes2):
    assert isinstance(boxes1, np.ndarray) or torch.is_tensor(boxes1), \
        "first argument should be ndarray or Tensor"
    assert isinstance(boxes2, type(boxes1)), \
        "second argument should have the same type as the first"
    assert len(boxes1.shape)<3 and len(boxes2.shape)<3, \
        "ndarray or Tensor should be 1D (length 4) or 2D (n x 4)"
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4, \
        "Each box should have 4 coordinates: left, top, width, height"
    b1, b2 = boxes1, boxes2
    if len(b1.shape)==1:
        b1 = b1.reshape((1,4))
    if len(b2.shape)==1:
        b2 = b2.reshape((1,4))
    return b1, b2

def iou(boxes1, boxes2, denominator = 'union'):
    """ Given 2 list of rectangular boxes returns IOU (intersection over union) or
        IOF (area of intersection over area of the first box) metric for
        every box in the first list with every box in the second list.
        Lists should be in the form of (n, 4) Tensor or ndarray where n is
        the number of boxes in the list and each box is represented as
        [left, top, width, height]
    Takes
        boxes1: ndarray or Tensor with dimensions (n1, 4)
                (left, top, width, height)
        boxes2: ndarray or Tensor with dimensions (n1, 4)
                (left, top, width, height)
        denominator: 'union' | 'first'
                Calculate IOU or normalize to the area of the 1st box
    Returns
        iou: (n1, n2) ndarray or Tensor
    """
    # Parse arguments
    b1, b2 = _check_box_args(boxes1, boxes2)
    assert denominator == 'union' or denominator == 'first', \
        "denominator should have value {'union'|'first'}"
    # Calculate Metric
    if isinstance(b1, np.ndarray):
        return iou_numpy(b1, b2, denominator)
    else:
        return iou_torch(b1, b2, denominator)

def iou_torch(boxes1, boxes2, denominator = 'union'):
    """ Given 2 list of rectangular boxes returns IOU (intersection over union) or
        IOF (area of intersection over area of the first box) metric for
        every box in the first list with every box in the second list.
        Lists should be in the form of (n, 4) Tensor where n is the number
        of boxes in the list and each box is represented as
        [left, top, width, height]
    Takes
        boxes1: Tensor with dimensions (n1, 4)
                (left, top, width, height)
        boxes2: Tensor with dimensions (n2, 4)
                (left, top, width, height)
        denominator: 'union' | 'first'
                Calculate IOU or normalize to the area of the 1st box
    Returns
        iou: (n1, n2) Tensor
    """
    zero = torch.zeros(1, dtype = boxes1.dtype, device=boxes1.device)
    x1, y1, w1, h1 = boxes1.split(1, dim=1)
    x2, y2, w2, h2 = boxes2.split(1, dim=1)
    # print(x1.shape)
    iwidths = torch.max(torch.min(x1+w1, (x2+w2).t()) - torch.max(x1, x2.t()), zero)
    iheights = torch.max(torch.min(y1+h1, (y2+h2).t()) - torch.max(y1, y2.t()), zero)
    iareas = iwidths*iheights
    if denominator == 'union':
        denom = w1*h1 + (w2*h2).t() - iareas
    else:
        denom = w1*h1
    return iareas/denom

def iou_numpy(boxes1, boxes2, denominator = 'union'):
    """ Given 2 list of rectangular boxes returns IOU (intersection over union) or
        IOF (area of intersection over area of the first box) metric for
        every box in the first list with every box in the second list.
        Lists should be in the form of n x 4 ndarray where n is the number
        of boxes in the list and each box is represented as
        [left, top, width, height]
    Takes
        boxes1: ndarray with dimensions (n1, 4)
                (left, top, width, height)
        boxes2: ndarray with dimensions (n2, 4)
                (left, top, width, height)
        denominator: 'union' | 'first'
                Calculate IOU or normalize to the area of the 1st box
    Returns
        iou: (n1, n2) ndarray
    """
    x1, y1, w1, h1 = np.split(boxes1, 4, axis=1)
    x2, y2, w2, h2 = np.split(boxes2, 4, axis=1)
    # print(x1.T.shape)
    iwidths = np.maximum(np.minimum(x1+w1, (x2+w2).T) - np.maximum(x1, x2.T), 0)
    iheights = np.maximum(np.minimum(y1+h1, (y2+h2).T) - np.maximum(y1, y2.T), 0)
    iareas = iwidths*iheights
    if denominator == 'union':
        denom = w1*h1 + (w2*h2).T - iareas
    else:
        denom = w1*h1
    return iareas/denom

def mean_iou_img(predict, target, scores=None):
    """ Calculates average IOU per image for intersection between
    predicted and target bounding boxes. All boxes are assumed to predict
    same class of objects. If scores are given than they will be used
    to sort prediction boxes in ascending order of their prediction
    confidence score. See documentation of match_boxes() function
    on the deatils of the matching procedure. If no prediction matches
    the target box it will contribute 0 IOU to the mean
    Parameters:
        predict: (n,4) ndarray or Tensor of box predictions
        target: (n, 4) ndarray or Tensor of box targets
        scores: (n,) ndarray or Tensor of confidence scores for predictions
    Returns:
        mean_iou: (1,) Tensor with mean IOU or float IOU
    """
    # Parse arguments
    p, t = _check_box_args(predict, target)
    if scores is not None:
        assert isinstance(scores, type(p)), \
            "If provided scores should have the same type as predict and target (ndarray or Tensor)"
        p = p[scores.argsort()]
    if torch.is_tensor(p):
        match_ious, _, _ = match_boxes_torch(p, t)
    else:
        match_ious, _, _ = match_boxes_numpy(p, t)
    return match_ious.sum()/(p.shape[0]+t.shape[0]-match_ious.shape[0])

def tp_fp_fn_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """ Calculates True Positive, False Positive and False Negative boxes
    for object detection given specified IOU threshold for intersection between
    predicted and target bounding boxes. All boxes are assumed to predict same
    class of objects. If scores are given than they will be used to sort
    prediction boxes in ascending order of their prediction confidence score.
    See documentation of match_boxes() function on the deatils of the matching procedure.
    Parameters:
        predict: (n,4) ndarray or Tensor of box predictions
        target: (n, 4) ndarray or Tensor of box targets
        iou_threshold: float IOU above which prediction is counted as positive
        scores: (n,) ndarray or Tensor of confidence scores for predictions
        score_threshold: float Remove predicted boxes with score below the threshold
                         Ignored if scores are not provided
    Returns:
        (precision, recall, f1): ((1,), (1,), (1,)) Tuple of Tensors or floats
    """
    p, t = _check_box_args(predict, target)
    if scores is not None:
        assert isinstance(scores, type(p)), \
            "If provided scores should have the same type as predict and target (ndarray or Tensor)"
        n_score = (scores > score_threshold).sum()
        p = p[scores.argsort()[:n_score]]
    if t.shape[0] == 0:
        return 0, p.shape[0], 0
    if torch.is_tensor(p):
        match_ious, _, _ = match_boxes_torch(p, t)
    else:
        match_ious, _, _ = match_boxes_numpy(p, t)
    tp = (match_ious > iou_threshold).sum()
    fp = p.shape[0] - tp
    fn = t.shape[0] - tp
    return tp, fp, fn

def precision_recall_jaccard_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """ Calculates precision, recall and Jaccard score for object detection given
    specified IOU threshold for intersection between predicted and target
    bounding boxes. All boxes are assumed to predict same class of objects.
    If scores are given than they will be used to sort prediction boxes in
    ascending order of their prediction confidence score. See documentation
    of match_boxes() function on the deatils of the matching procedure.
    Parameters:
        predict: (n,4) ndarray or Tensor of box predictions
        target: (n, 4) ndarray or Tensor of box targets
        iou_threshold: float IOU above which prediction is counted as positive
        scores: (n,) ndarray or Tensor of confidence scores for predictions
        score_threshold: float Remove predicted boxes with score below the threshold
    Returns:
        (precision, recall, f1): ((1,), (1,), (1,)) Tuple of Tensors or floats
    """
    tp, fp, fn = tp_fp_fn_img(predict, target, iou_threshold, scores, score_threshold)
    if torch.is_tensor(tp):
        tp = tp.float()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    jaccard = tp/(tp+fp+fn)
    return precision, recall, jaccard

def precision_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard_img() function"""
    tp, fp, fn = tp_fp_fn_img(predict, target, iou_threshold, scores, score_threshold)
    if torch.is_tensor(tp):
        tp = tp.float()
    return tp/(tp+fp)

def recall_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard() function"""
    tp, fp, fn = tp_fp_fn_img(predict, target, iou_threshold, scores, score_threshold)
    if torch.is_tensor(tp):
        tp = tp.float()
    return tp/(tp+fn)

def f1_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Calculates F1 = 2*precision*recall/(precision+recall)
    For additional information refer to documentation on precision_recall() function"""
    tp, fp, fn = tp_fp_fn_img(predict, target, iou_threshold, scores, score_threshold)
    if torch.is_tensor(tp):
        tp = tp.float()
    return 2*tp/(2*tp+fp+fn)

def jaccard_score_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """Refer to documentation on precision_recall_jaccard() function"""
    tp, fp, fn = tp_fp_fn_img(predict, target, iou_threshold, scores, score_threshold)
    if torch.is_tensor(tp):
        tp = tp.float()
    return tp/(tp+fp+fn)

def average_precision_img(predict, target, iou_threshold = 0.5, scores = None, score_threshold = 0):
    """ Calculates average precision (area under precision vs recall curve)
    for object detection given specified IOU threshold for intersection between
    predicted and target bounding boxes. All boxes are assumed to predict
    same class of objects. If scores are given than they will be used to sort
    prediction boxes in ascending order of their prediction confidence score.
    See documentation of match_boxes() function on the deatils of the matching procedure.
    Parameters:
        predict: (n,4) Tensor of box predictions
        target: (n, 4) Tensor of box targets
        iou_threshold: float IOU above which prediction is counted as positive
        scores: (n,) Tensor of confidence scores for predictions
        score_threshold: float Remove predicted boxes with score below the threshold
    Returns:
        ap: ((1,) Tensor
    """
    p, t = _check_box_args(predict, target)
    if scores is not None:
        assert isinstance(scores, type(p)), \
            "If provided scores should have the same type as predict and target (ndarray or Tensor)"
        n_score = (scores > score_threshold).sum()
        p = p[scores.argsort()[:n_score]]
    if torch.is_tensor(p):
        match_ious, p_idxs, _ = match_boxes_numpy(p.cpu().numpy(), t.cpu().numpy())
    else:
        match_ious, p_idxs, _ = match_boxes_numpy(p, t)
    npredict, ntarget = p.shape[0], t.shape[0]
    tp_boxes = np.zeros(npredict)
    tp_boxes.put((p_idxs,), match_ious)
    # We can safely use put above since matching indexes are quaranteed to be unique
    tp_boxes = np.flip(tp_boxes > iou_threshold)
    precisions = tp_boxes.cumsum(axis=0)/np.arange(1, npredict+1)
    ap = precisions[tp_boxes].sum() / ntarget
    if torch.is_tensor(p):
        ap = torch.from_numpy(np.array(ap)).to(p)
    return ap

def match_boxes(predict,target):
    """Assigns predicted bounding boxes to ground truth boxes. Returns IOUs
    and indexes of matched boxes. Gready box assignement used.
    Each predicted box can be assigned to only 1 target box (box with maximal IOU).
    If multiple predicted boxes are assigned to the same target box
    only the box with maximal prediction score is used (other prediction
    boxes will have IOU of 0). The prediction boxes should be sorted
    in ascending order of their prediction confidence score.
    Parameters:
        predict: (n, 4) Tensor or ndarray of box predictions
        target: (n, 4) Tensor or ndarray of box targets
    Returns:
        (ious, p_idxs, t_idxs): tuple of Tensors or ndarrays
    """
    predict, target = _check_box_args(predict, target)
    iou_matrix = iou(predict, target)
    if torch.is_tensor(predict):
        ious, idxs = iou_matrix.max(dim = -1)
        ious = ious.cpu().numpy()
        idxs = idxs.cpu().numpy()
    else:
        ious, idxs = iou_matrix.max(axis=-1), iou_matrix.argmax(axis=-1)
    match_ious = np.zeros(target.shape[0], dtype=ious.dtype)
    for v, i in zip(ious, idxs):
        match_ious[i] = v
    p_idxs = np.zeros(target.shape[0], dtype=np.long)
    for v, i in enumerate(idxs):
        p_idxs[i] = v
    t_idxs = np.arange(target.shape[0])
    matched = match_ious.nonzero()[0]
    if torch.is_tensor(predict):
        match_ious = torch.from_numpy(match_ious[matched]).to(predict.device)
        p_idxs = torch.from_numpy(p_idxs[matched]).to(predict.device)
        t_idxs = torch.from_numpy(t_idxs[matched]).to(predict.device)
    else:
        match_ious = match_ious[matched]
        p_idxs = p_idxs[matched]
        t_idxs = t_idxs[matched]
    return match_ious, p_idxs, t_idxs

def nms(boxes, scores, iou_threshold):
    """ Given an array of rectangular boxes and confidence scores filters out boxes
        that overlap more than iou_threshold with boxes that have higher score.
        Returns an ndarray or Tensor of indexes of retained boxes sorted in the
        descending order of box confidence scores
    Takes
        boxes: (n,4) ndarray or Tensor
        scores: (n,) ndarray or Tensor
        iou_threshold: float
    Returns
        keep_idx: ndarray or Tensor with indexes of retained boxes
    """
    assert isinstance(boxes, np.ndarray) or torch.is_tensor(boxes), \
        "boxes should be ndarray or Tensor"
    assert isinstance(scores, type(boxes)), \
        "scores should have the same type as the first"
    assert scores.shape[0] == boxes.shape[0], \
        "Number of scores and boxes should be the same"
    ious = iou(boxes, boxes) > iou_threshold
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
        ious = ious.cpu().numpy()
    keep_mask = np.ones_like(scores, dtype = bool)
    order = np.flip(scores.argsort())
    keep_idx = []
    for idx in order:
        if keep_mask[idx]:
            keep_mask[ious[idx]] = False
            keep_idx.append(idx)
    if torch.is_tensor(boxes):
        return torch.tensor(keep_idx).to(boxes.device, dtype = torch.long)
    else:
        return np.array(keep_idx)

def precision_recall_f1(predict, target, iou_thresholds, nms_threshold = 0.8):
    """ Calculates precision, recall and F1 score on a batch of predictions.
    """
    if isinstance(predict, list):
        assert isinstance(target, list) and len(predict) == len(target), \
            "If predict and target are lists theshould have same length"
    else:
        predict, target = [predict], [target]
    counts = np.zeros((len(iou_thresholds), 3), dtype=np.int)
    for i in range(len(predict)):
        pboxes, pscores = predict[i]
        tboxes, tscores = target[i]
        tboxes = tboxes[nms(tboxes, tscores,.95)]
        pboxes = pboxes[nms(pboxes, pscores, nms_threshold)]
        for i, thresh in enumerate(iou_thresholds):
            counts[i] += tp_fp_fn_img(pboxes, tboxes, iou_threshold=thresh)
    results = {}
    for i, thresh in enumerate(iou_thresholds):
        tp, fp, fn = counts[i]
        results[f"Precision@IOU {thresh:{0}.{2}}"] = tp/(tp+fp) if tp+fp !=0 else 0
        results[f"Recall@IOU {thresh:{0}.{2}}"] = tp/(tp+fn) if tp+fn !=0 else 0
        results[f"F1@IOU {thresh:{0}.{2}}"] = 2*tp/(2*tp+fp+fn) if tp+fn !=0 else 0
    return results

def precision_recall_meanIOU(predict, target, iou_thresholds, nms_threshold = 0.8):
    if isinstance(predict, list):
        assert isinstance(target, list) and len(predict) == len(target), \
            "If predict and target are lists theshould have same length"
    else:
        predict, target = [predict], [target]
    counts = np.zeros((len(iou_thresholds), 3), dtype=np.int)
    ious = [[]]
    for i in range(len(predict)):
        pboxes, pscores = predict[i]
        tboxes = target[i]
        tboxes = tboxes[nms(tboxes, np.ones(tboxes.shape[0]),.95)]
        pboxes = pboxes[nms(pboxes, pscores, nms_threshold)]
        if pboxes.shape[0] > 0 and tboxes.shape[0] > 0:
            ious.append(match_boxes_numpy(pboxes, tboxes)[0])
        for i, thresh in enumerate(iou_thresholds):
            counts[i] += tp_fp_fn_img(pboxes, tboxes, iou_threshold=thresh)
    results = {}
    for i, thresh in enumerate(iou_thresholds):
        tp, fp, fn = counts[i]
        results[f"Precision@IOU {thresh:{0}.{2}}"] = tp/(tp+fp) if tp+fp !=0 else 0
        results[f"Recall@IOU {thresh:{0}.{2}}"] = tp/(tp+fn) if tp+fn !=0 else 0
    ious = np.concatenate(ious)
    results["meanIOU"] = ious.mean() if len(ious) > 0 else 0
    return results


class PrecisionRecallF1MeanIOU:
    def __init__(self, iou_thresholds=[0.5,0.8,.95], nms_threshold = 0.8):
        self.iou_thresholds = iou_thresholds
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def __call__(self, predict, target):
        if isinstance(predict, list):
            assert isinstance(target, list) and len(predict) == len(target), \
                "If predict and target are lists they should have same length"
        else:
            predict, target = [predict], [target]
        counts = np.zeros((len(self.iou_thresholds), 3), dtype=np.int)
        ious = []
        for img_idx in range(len(predict)):
            pboxes, pscores = predict[img_idx]
            tboxes = target[img_idx]
            tboxes = tboxes[nms(tboxes, torch.ones(tboxes.shape[0]),.95).flip((0,))]
            pboxes = pboxes[nms(pboxes, pscores, self.nms_threshold).flip((0,))]
            if pboxes.shape[0] > 0 and tboxes.shape[0] > 0:
                match_ious, p_idxs, t_idxs = match_boxes(pboxes, tboxes)
                ious.append(match_ious.cpu().numpy())
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    tp = (match_ious > thresh).sum().item()
                    fp = pboxes.shape[0] - tp
                    fn = tboxes.shape[0] - tp
                    counts[t_idx] += [tp, fp, fn]
            else:
                ious.append(np.array([]))
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    counts[t_idx] += [0, pboxes.shape[0], tboxes.shape[0]]
        results = {}
        for t_idx, thresh in enumerate(self.iou_thresholds):
            tp, fp, fn = counts[t_idx]
            results[f"Precision@IOU {thresh:{0}.{2}}"] = tp/(tp+fp) if tp+fp !=0 else 0
            results[f"Recall@IOU {thresh:{0}.{2}}"] = tp/(tp+fn) if tp+fn !=0 else 0
            results[f"F1@IOU {thresh:{0}.{2}}"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn !=0 else 0
        ious = np.concatenate(ious)
        results["meanIOU"] = ious.mean() if len(ious) > 0 else 0
        return results

    def state_dict(self):
        state = {'iou_thresholds': self.iou_thresholds,
        'nms_threshold': self.nms_threshold}
        return state

    def load_state_dict(self, state):
        self.iou_thresholds = state['iou_thresholds']
        self.nms_threshold = state['nms_threshold']


class PrecisionRecallF1ClassF1MeanIOU:
    def __init__(self, iou_thresholds=[0.5,0.8,.95], nms_threshold = 0.8):
        self.iou_thresholds = iou_thresholds
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def __call__(self, predict, target):
        if isinstance(predict, list):
            assert isinstance(target, list) and len(predict) == len(target), \
                "If predict and target are lists they should have same length"
        else:
            predict, target = [predict], [target]
        counts = np.zeros((len(self.iou_thresholds), 3), dtype=np.int)
        cls_counts = np.zeros((len(self.iou_thresholds), 3), dtype=np.int)
        ious = []
        for img_idx in range(len(predict)):
            pboxes, pscores, pclsscores = predict[img_idx]
            tboxes, tclslbl = target[img_idx]
            tidxs = nms(tboxes, torch.ones(tboxes.shape[0]),.95).flip((0,))
            tboxes = tboxes[tidxs]
            pidxs = nms(pboxes, pscores, self.nms_threshold).flip((0,))
            pboxes = pboxes[pidxs]
            if pboxes.shape[0] > 0 and tboxes.shape[0] > 0:
                tclslbl = tclslbl[tidxs]
                pclslbl = torch.argmax(pclsscores, axis=1)
                pclslbl = pclslbl[pidxs].reshape((-1,1)) # WARNING
                match_ious, p_idxs, t_idxs = match_boxes(pboxes, tboxes)
                ious.append(match_ious.cpu().numpy())
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    match_idxs = match_ious > thresh
                    tp = match_idxs.sum().item()
                    fp = pboxes.shape[0] - tp
                    fn = tboxes.shape[0] - tp
                    counts[t_idx] += [tp, fp, fn]
                    cls_tp = (pclslbl[p_idxs[match_idxs]] == tclslbl[t_idxs[match_idxs]]).sum().item()
                    cls_fp = pboxes.shape[0] - cls_tp
                    cls_fn = tboxes.shape[0] - cls_tp
                    cls_counts[t_idx] = [cls_tp, cls_fp, cls_fn]
            else:
                ious.append(np.array([]))
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    counts[t_idx] += [0, pboxes.shape[0], tboxes.shape[0]]
                    cls_counts[t_idx] += [0, pboxes.shape[0], tboxes.shape[0]]
        results = {}
        for t_idx, thresh in enumerate(self.iou_thresholds):
            tp, fp, fn = counts[t_idx]
            results[f"Precision@IOU {thresh:{0}.{2}}"] = tp/(tp+fp) if tp+fp !=0 else 0
            results[f"Recall@IOU {thresh:{0}.{2}}"] = tp/(tp+fn) if tp+fn !=0 else 0
            results[f"F1@IOU {thresh:{0}.{2}}"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn !=0 else 0
            cls_tp, cls_fp, cls_fn = cls_counts[t_idx]
            results[f"ClassF1@IOU {thresh:{0}.{2}}"] = 2*cls_tp/(2*cls_tp+cls_fp+cls_fn) if 2*cls_tp+cls_fp+cls_fn !=0 else 0
        ious = np.concatenate(ious)
        results["meanIOU"] = ious.mean() if len(ious) > 0 else 0
        return results

    def state_dict(self):
        state = {'iou_thresholds': self.iou_thresholds,
        'nms_threshold': self.nms_threshold}
        return state

    def load_state_dict(self, state):
        self.iou_thresholds = state['iou_thresholds']
        self.nms_threshold = state['nms_threshold']

class PrecisionRecallF1BalancedClassAccuracyMeanIOU:
    """ Balanced class accuracy is calculated as macro-average of per class recall
    """
    def __init__(self, iou_thresholds=[0.5,0.8,.95], nms_threshold = 0.8):
        self.iou_thresholds = iou_thresholds
        self.nms_threshold = nms_threshold

    @torch.no_grad()
    def __call__(self, predict, target):
        if isinstance(predict, list):
            assert isinstance(target, list) and len(predict) == len(target), \
                "If predict and target are lists they should have same length"
        else:
            predict, target = [predict], [target]
        counts = np.zeros((len(self.iou_thresholds), 3), dtype=np.int)
        cls_accuracy = np.zeros(len(self.iou_thresholds), dtype=np.int)
        ious = []
        img_count = 0
        for img_idx in range(len(predict)):
            pboxes, pscores, pclsscores = predict[img_idx]
            tboxes, tclslbl = target[img_idx]
            tidxs = nms(tboxes, torch.ones(tboxes.shape[0]),.95).flip((0,))
            tboxes = tboxes[tidxs]
            pidxs = nms(pboxes, pscores, self.nms_threshold).flip((0,))
            pboxes = pboxes[pidxs]
            if pboxes.shape[0] > 0 and tboxes.shape[0] > 0:
                tclslbl = tclslbl[tidxs]
                cls_lbls, cls_counts = torch.unique(tclslbl, return_counts=True)
                cls_recalls = np.zeros(cls_lbls.shape[0])
                pclslbl = torch.argmax(pclsscores, axis=1)
                pclslbl = pclslbl[pidxs].reshape((-1,1)) # WARNING
                match_ious, p_idxs, t_idxs = match_boxes(pboxes, tboxes)
                ious.append(match_ious.cpu().numpy())
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    match_idxs = match_ious > thresh
                    tp = match_idxs.sum().item()
                    fp = pboxes.shape[0] - tp
                    fn = tboxes.shape[0] - tp
                    counts[t_idx] += [tp, fp, fn]
                    for cls_idx in range(cls_lbls.shape[0]):
                        cls_recalls[cls_idx] = (pclslbl[p_idxs[match_idxs]] == cls_lbls[cls_idx]).sum().item() / cls_counts[cls_idx].item()
                    cls_accuracy[t_idx] += cls_recalls.mean()
            else:
                ious.append(np.array([]))
                img_count +=1
                for t_idx, thresh in enumerate(self.iou_thresholds):
                    counts[t_idx] += [0, pboxes.shape[0], tboxes.shape[0]]
        results = {}
        for t_idx, thresh in enumerate(self.iou_thresholds):
            tp, fp, fn = counts[t_idx]
            results[f"Precision@IOU {thresh:{0}.{2}}"] = tp/(tp+fp) if tp+fp !=0 else 0
            results[f"Recall@IOU {thresh:{0}.{2}}"] = tp/(tp+fn) if tp+fn !=0 else 0
            results[f"F1@IOU {thresh:{0}.{2}}"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn !=0 else 0
            cls_tp, cls_fp, cls_fn = cls_counts[t_idx]
            results[f"ClassAccuracy@IOU {thresh:{0}.{2}}"] = cls_accuracy[t_idx]/img_count if img_count != 0 else 0
        ious = np.concatenate(ious)
        results["meanIOU"] = ious.mean() if len(ious) > 0 else 0
        return results

    def state_dict(self):
        state = {'iou_thresholds': self.iou_thresholds,
        'nms_threshold': self.nms_threshold}
        return state

    def load_state_dict(self, state):
        self.iou_thresholds = state['iou_thresholds']
        self.nms_threshold = state['nms_threshold']
