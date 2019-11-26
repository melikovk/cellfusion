import numpy as np
import torch
import os
from PIL import Image
import cv2
from image.datasets.utils import get_cell_anchors, get_boxes_from_json

def get_filenames(datadir, channels, boxes = 'boxes/', suffix = ''):
    """ This function takes name of a directory, list of channel names and name
    of folder with boxes. It expects to find files train.txt and test.txt in the
    provided directory. Both files contain file paths. Each file path is a basename
    and we expect to find images with names:
    filedir/channels[i]/filename
    and file with box annotations:
    filedir//boxes/filename+boxes.json
    return list of tuples, where each tuple containes
    list of image files and box file
    suffix can be specified to use files train+suffix.txt and test+suffix.txt.
    """

    def check_paths(fpaths):
        return os.path.exists(fpaths[1]) and all(map(os.path.exists, fpaths[0]))

    with open(datadir+'train'+suffix+'.txt') as f:
        train_names = list(filter(check_paths, (([os.path.join(fdir, channel, fname+'.tif')
            for channel in channels], os.path.join(fdir, boxes, fname+'boxes.json'))
            for fdir, fname in (os.path.split(fpath.strip()) for fpath in f.readlines()))))
    with open(datadir+'test'+suffix+'.txt') as f:
        test_names = list(filter(check_paths,(([os.path.join(fdir, channel, fname+'.tif')
            for channel in channels], os.path.join(fdir, boxes, fname+'boxes.json'))
            for fdir, fname in (os.path.split(fpath.strip()) for fpath in f.readlines()))))
    return train_names, test_names

def predict_boxes(model, imgnames, transforms=None, nms_threshold=1.0, p_threshold = 0.5, upscale = 1):
    """ Given model and name of image file predict boxes
    """
    model.eval()
    img  = np.stack([np.asarray(Image.open(imgname)).T.astype(np.float32) for imgname in imgnames])
    w, h = img.shape[-2:]
    w1 = (int(w*upscale)//model.grid_size)*model.grid_size
    h1 = (int(h*upscale)//model.grid_size)*model.grid_size
    wfactor = w / w1
    hfactor = h / h1
    img = np.stack([cv2.resize(channel, dsize=(h1,w1), interpolation=cv2.INTER_CUBIC) for channel in img])
    device = next(model.parameters()).device
    if transforms is None:
        input = torch.from_numpy(img).reshape(1, img.shape[0], w1, h1).to(device)
    else:
        input = torch.from_numpy(transforms(img)).reshape(1, img.shape[0], w1, h1).to(device)
    if len(model.head.clsnums) > 0:
        boxes, scores, clsscores = model.predict(input)[0]
    else:
        boxes, scores = model.predict(input, p_threshold)[0]
    if nms_threshold < 1.0:
        keep_idx = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        if len(model.head.clsnums) > 0:
            clsscores = clsscores[keep_idx]
    boxes = boxes * torch.tensor([wfactor, hfactor, wfactor, hfactor]).reshape(1,4).to(device)
    # if upscale is not None:
    #     boxes = boxes / upscale
    if len(model.head.clsnums) > 0:
        return boxes, scores, clsscores
    else:
        return boxes, scores

def evaluate_model(model, fnames, eval_func, clsname=None, transfer_to_cpu=False, **kwargs):
    """ Evaluate function on a set of files. Expects list of tuples (imgname, boxname)"""
    predictions = []
    targets = []
    for imgnames, boxname in fnames:
        print(boxname)
        # p = predict_boxes(model, imgnames, **kwargs)
        p = predict_boxes(model, imgnames, **kwargs)
        if transfer_to_cpu:
            p = [arr.cpu() for arr in p]
        predictions.append(p)
        t = get_boxes_from_json(boxname, clsname)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(device=p[0].device, dtype = p[0].dtype)
        else:
            [torch.from_numpy(arr).to(device=p[0].device, dtype = p[0].dtype) for arr in t]
        targets.append(t)
    return eval_func(predictions, targets)
