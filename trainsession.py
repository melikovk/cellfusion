import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict
from tensorboardX import SummaryWriter
import skimage.io as io
import numpy as np
from image.datasets.yolo import labels_to_boxes, get_cell_anchors, get_boxes_from_json
from image.metrics.localization import nms
from image.cv2transforms import AutoContrast, Gamma
from skimage.transform import rescale
from importlib import import_module
from PIL import Image
import math
# # Apex
# from apex import amp


autocontrast = lambda x: AutoContrast()(x).astype(np.float32)

def get_filenames(datadir, channel, boxes = 'boxes/', suffix = ''):
    channel += '/'
    with open(datadir+'train'+suffix+'.txt') as f:
        train_names = [(datadir+channel+name[:-1]+'.tif', datadir+boxes+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    with open(datadir+'test'+suffix+'.txt') as f:
        test_names = [(datadir+channel+name[:-1]+'.tif', datadir+boxes+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    return train_names, test_names

def return_zero():
    return 0.0

def predict_nuclei(image, model, grid_size = 32, conf_threshold = 0.5, iou_threshold = 0.8, offset= (0,0), transform = autocontrast):
    if isinstance(image, str):
        img = io.imread(image).T
    else:
        img = image
    w, h = img.shape
    w = w // grid_size * grid_size
    h = h // grid_size * grid_size
    img = img[:w,:h]
    model.eval()
    device = next(model.parameters()).device
    input = torch.from_numpy(transform(img).reshape((1, -1, w, h))).to(device)
    labels = model(input).detach().cpu().squeeze()
    boxes, scores = labels_to_boxes(labels, grid_size = grid_size, cell_anchors = get_cell_anchors([1],[]), offset = offset, threshold = conf_threshold)
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    idxs = nms(boxes, scores, iou_threshold)
    return boxes[idxs], scores[idxs]

def predict_nuclei_zoom2x(fpath, model, grid_size = 32, conf_threshold = 0.5, iou_threshold = 0.8, transform = autocontrast):
    img = io.imread(fpath).T
    img = rescale(img, 2, order=1)
    w, h = img.shape
    cw, ch = w // 2, h // 2
    offsets = [(0,0), (0, ch), (cw,0), (cw, ch)]
    boxes = []
    scores = []
    for offw, offh in offsets:
        cropboxes, cropscores = predict_nuclei(img[offw:offw+cw, offh:offh+ch], model, grid_size, conf_threshold, iou_threshold, (offw, offh), transform)
        boxes.append(cropboxes)
        scores.append(cropscores)
    boxes = torch.floor(torch.cat(boxes)/2)
    scores = torch.cat(scores)
    idxs = nms(boxes, scores, iou_threshold)
    return boxes[idxs], scores[idxs]

def predict_yolo(fpath, model, grid_size = 32):
    img = io.imread(fpath).T
    w, h = img.shape
    w = w // grid_size * grid_size
    h = h // grid_size * grid_size
    img = img[:w,:h]
    device = next(model.parameters()).device
    transform = lambda x: torch.from_numpy(AutoContrast()(Gamma(0.1)(x)).astype(np.float32)).reshape((1,1,w,h)).to(device)
    input = transform(img)
    labels = model(input).detach().cpu()
    return labels

class SessionSaver:
    """ A class for saving pytorch training sesssion
    It requires a full path (path + filename) where to save the training sesssion file.
    """
    def __init__(self, path, frequency = 1, overwrite = True, bestonly = True, metric = 'loss', ascending = False):
        if path.endswith('.tar'):
            self.path = path[:-4]
        else:
            self.path = path
        self.frequency = frequency
        self.overwrite = overwrite
        self.bestonly = bestonly
        self.metric = metric
        self.direction = -1 if ascending else 1
        self.bestmetric = None


    def save(self, session, epoch, metrics):
        if epoch % self.frequency != 0:
            return
        if not self.bestonly:
            if self.overwrite:
                fname = self.path+'.tar'
            else:
                fname = f'{self.path}_{epoch}.tar'
            torch.save(session.state_dict(), fname)
        elif self.bestmetric is None or metrics[self.metric]*self.direction < self.bestmetric*self.direction:
            self.bestmetric = metrics[self.metric]
            if self.overwrite:
                fname = self.path+'.tar'
            else:
                fname = f'{self.path}_{epoch}.tar'
            torch.save(session.state_dict(), fname)

    def state_dict(self):
        state = {'bestmetric':self.bestmetric,
            'path': self.path,
            'frequency':self.frequency,
            'overwrite':self.overwrite,
            'bestonly':self.bestonly,
            'metric': self.metric,
            'direction': self.direction}
        return state

    def load_state_dict(self, state):
        self.bestmetric = state['bestmetric']
        self.path = state['path']
        self.frequency = state['frequency']
        self.overwrite = state['overwrite']
        self.bestonly = state['bestonly']
        self.metric = state['metric']
        self.direction = state['direction']


class TrainSession:
    """ Class to train and save pytorch models
    Parameters:
        model: pyTorch model
        lossfunc: function to calculate loss. Should return dict of the form
                  {'loss': total_loss, 'loss_name1': loss1, ...}.
                  Total loss will be optimized.
        optimizer: pyTorch Optimizer to use for optimization
        parameters: iterable with model parameters or dict objects to optimize
                    refer to specific Optimizer documentation for details
        acc_func: function to report model accuracy metric should return a dict
                  of the form {'metrics_name':metric, ...}
        opt_defaults: dict with the default parameters of the optimizer
                      refer to specific Optimizer documentation
        scheduler: learning rate scheduler
        scheduler_params: parameters of the scheduler
        log_dir: Directory to log training progress
        saver: SessionSaver object to save the training session during training
        device: device on which to perform training
    """
    def __init__(self, model, lossfunc, optimizer, parameters, acc_func, opt_defaults = None,
                 scheduler=None, scheduler_params=None, log_dir=None, saver=None,
                 device = None):
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer(parameters) if opt_defaults is None else optimizer(parameters, **opt_defaults)
        # # Apex
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2")
        # # Apex end
        self.scheduler =  None if scheduler is None else scheduler(self.optimizer, **scheduler_params)
        self.scheduler_params = scheduler_params
        self.acc_func = acc_func
        self.log_dir = log_dir
        self.saver = saver
        self.epoch = 0
        self.epochs_left = 0

    def train_step(self, inputs, labels):
        # with torch.autograd.detect_anomaly():
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.lossfunc(outputs, labels)
        loss['loss'].backward()
        # Apex
        # with amp.scale_loss(loss['loss'], self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        self.optimizer.step()
        return (loss, outputs)

    @torch.no_grad()
    def evaluate(self, data, update_saver = True):
        self.model.eval()
        loss = defaultdict(return_zero)
        size = 0
        accuracy = defaultdict(return_zero)
        for inputs, labels in data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            batch_loss = self.lossfunc(outputs, labels)
            for k, v in batch_loss.items():
                loss[k] += v.item() * inputs.size(0)
            batch_acc = self.acc_func(self.model.get_prediction(outputs), self.model.get_targets(labels))
            for k, v in batch_acc.items():
                accuracy[k] += v * inputs.size(0)
            size += inputs.size(0)
        for k, v in loss.items():
            loss[k] = loss[k]/size
        for k, v in accuracy.items():
            accuracy[k] = accuracy[k]/size
        if self.saver and update_saver:
            self.saver.save(self, self.epoch, {**loss, **accuracy})
        return (loss, accuracy)

    def train(self, train_data, valid_data, epochs = None):
        if epochs is None:
             epochs = self.epochs_left
        if self.log_dir:
            writer = SummaryWriter(self.log_dir, purge_step = self.epoch)
        for epoch in range(epochs):
            self.epoch += 1
            self.epochs_left = epochs-epoch-1
            self.model.train()
            train_loss = defaultdict(return_zero)
            size = 0
            # train_acc = 0.0
            train_acc = defaultdict(return_zero)
            if self.scheduler:
                self.scheduler.step(epoch)
            pbar = tqdm(total = len(train_data), leave = False)
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_loss, outputs = self.train_step(inputs, labels)
                # statistics
                for k, v in batch_loss.items():
                    train_loss[k] += v.item() * inputs.size(0)
                # train_acc += self.acc_func(outputs, labels).item()
                with torch.no_grad():
                    batch_acc = self.acc_func(self.model.get_prediction(outputs), self.model.get_targets(labels))
                for k, v in batch_acc.items():
                    train_acc[k] += v * inputs.size(0)
                size += inputs.size(0)
                pbar.update(1)
            for k, v in train_loss.items():
                train_loss[k] = train_loss[k]/size
            # train_acc = train_acc / size
            for k, v in train_acc.items():
                train_acc[k] = train_acc[k]/size
            valid_loss, valid_acc = self.evaluate(valid_data)
            message = (f'Epoch {epoch+1} of {epochs} took {tqdm.format_interval(pbar.last_print_t-pbar.start_t)}\n'
                       f'Train Loss: {train_loss["loss"]:.4f}, Validation Loss: {valid_loss["loss"]:.4f}\n'
                       f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            tqdm.write(message)
            if self.log_dir:
                metrics = {}
                for k, v in train_loss.items():
                    metrics[k+'/train'] = v
                for k, v in valid_loss.items():
                    metrics[k+'/validation'] = v
                for k, v in train_acc.items():
                    metrics[k+'/train'] = v
                for k, v in valid_acc.items():
                    metrics[k+'/validation'] = v
                for metric, value in metrics.items():
                    writer.add_scalar(metric, value, self.epoch)
            pbar.close()

    def update_lr(self, factor):
        """ Update initial_lr if scheduler is used or optimizer lr if not using
        scheduler. In both cases lr of all parameter groups is multiplied by the
        same factor.
        Arguments:
            factor (float): factor by which to multiply lr
        Examples::
            >>> self.update_lr(0.5)
        """
        if self.scheduler:
            for group in self.optimizer.param_groups:
                group['initial_lr'] = group['initial_lr']*factor
            for i in range(len(self.scheduler.base_lrs)):
                self.scheduler.base_lrs[i] *= factor
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr']*factor

    def set_lr(self, lrs):
        """ Set initial_lrs if scheduler is used or optimizer lr if not using scheduler.
        Takes single float or list of floats. If list is give its length should match
        the number of parameter groups.
        Arguments:
            lrs (float or [float]): learning rates
        Examples::
            >>> self.set_lr(.01)
            >>> self.set_lr([0.01, 0.02, 0.001])
        """
        if not isinstance(lrs, list):
            lrs = [lrs]*len(self.optimizer.param_groups)
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError(f"Number of the learning rates ({len(lrs)}) does not match "
                             f"number of the parameter groups ({len(self.optimizer.param_groups)})")
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group['initial_lr'] = lr
        self.scheduler.base_lrs = lrs

    def state_dict(self):
        state = {'epoch': self.epoch,
                 'epochs_left': self.epochs_left,
                 'model': self.model.state_dict(),
                 'model_type': {'name':type(self.model).__name__, 'module':type(self.model).__module__},
                 'model_config': self.model.config,
                 'optimizer': _map_param_ids_to_names(self.optimizer.state_dict(), self.model),
                 'optimizer_type': {'name':type(self.optimizer).__name__, 'module':type(self.optimizer).__module__},
                 'lossfunc':self.lossfunc.state_dict(),
                 'lossfunc_type':{'name':type(self.lossfunc).__name__, 'module':type(self.lossfunc).__module__},
                 'acc_func':self.acc_func.state_dict(),
                 'acc_func_type':{'name':type(self.acc_func).__name__, 'module':type(self.acc_func).__module__}}
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
            state['scheduler_type'] = {'name':type(self.scheduler).__name__, 'module':type(self.scheduler).__module__}
            state['scheduler_params'] = self.scheduler_params
        if self.log_dir:
            state['log_dir'] = self.log_dir
        if self.saver:
            state['saver'] = self.saver.state_dict()
        return state

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.epochs_left = state['epochs_left']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lossfunc.load_state_dict(state['lossfunc'])
        self.acc_func.load_state_dict(state['acc_func'])
        if self.scheduler and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        if not self.log_dir and 'log_dir' in state:
            self.log_dir = state['log_dir']
        if self.saver and 'saver' in state:
            self.saver.load_state_dict(state['saver'])

    @classmethod
    def restore_from_state_dict(cls, state, device = None):
        model = getattr(import_module(state['model_type']['module']), state['model_type']['name'])(**state['model_config'])
        lossfunc = getattr(import_module(state['lossfunc_type']['module']), state['lossfunc_type']['name'])()
        acc_func = getattr(import_module(state['acc_func_type']['module']), state['acc_func_type']['name'])()
        named_params = dict(model.named_parameters())
        param_groups = [{'params':[named_params[p] for p in g['params']]} for g in state['optimizer']['param_groups']]
        optimizer = getattr(import_module(state['optimizer_type']['module']), state['optimizer_type']['name'])
        scheduler = getattr(import_module(state['scheduler_type']['module']), state['scheduler_type']['name']) if 'scheduler' in state else None
        scheduler_params = state['scheduler_params'] if 'scheduler' in state else None
        log_dir = state['log_dir'] if 'log_dir' in state else None
        saver = SessionSaver(" ") if 'saver' in state else None
        session = cls(model = model, lossfunc = lossfunc, optimizer = optimizer, parameters = param_groups,
            acc_func = acc_func, opt_defaults = None, scheduler = scheduler, scheduler_params = scheduler_params,
            log_dir = log_dir, saver = saver, device = device)
        session.load_state_dict(state)
        return session

def _map_param_ids_to_names(opt_state, model):
    reverse_dict = {id(parameter):name for name, parameter in model.named_parameters()}
    new_state = opt_state.copy()
    for g in new_state['param_groups']:
        g['params'] = [reverse_dict[i] for i in g['params']]
    return new_state

def predict_boxes(model, imgname, transforms=None, nms_threshold=None, upscale = None):
    """ Given model and name of image file predict boxes
    """
    model.eval()
    img  = Image.open(imgname)
    w, h = img.size
    if upscale is not None:
        w = int(w*upscale)
        h = int(h*upscale)
        img = img.resize((w,h), resample = Image.NEAREST)
    w1 = (w//model.grid_size)*model.grid_size
    h1 = (h//model.grid_size)*model.grid_size
    wfactor = w / w1
    hfactor = h / h1
    img = img.resize((w1, h1))
    device = next(model.parameters()).device
    if transforms is None:
        input = torch.from_numpy(np.asarray(img).T.astype(np.float32)).reshape(1, 1, w1, h1).to(device)
    else:
        input = torch.from_numpy(transforms(np.asarray(img).T).astype(np.float32)).reshape(1, 1, w1, h1).to(device)
    boxes, scores = model.predict(input)[0]
    if nms_threshold is not None:
        keep_idx = nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
    boxes = boxes * np.array([wfactor, hfactor, wfactor, hfactor]).reshape(1,4)
    if upscale is not None:
        boxes = boxes / upscale
    return boxes, scores

def evaluate_model(model, fnames, eval_func, **kwargs):
    """ Evaluate function on a set of files. Expects list of tuples (imgname, boxname)"""
    predictions = []
    targets = []
    for imgname, boxname in fnames:
        p = predict_boxes(model, imgname, **kwargs)
        predictions.append(p)
        t = get_boxes_from_json(boxname, 1)
        targets.append(t)
    return eval_func(predictions, targets)
