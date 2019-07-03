import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict
from tensorboardX import SummaryWriter
import skimage.io as io
import numpy as np
from image.datasets.yolo import labels_to_boxes, get_cell_anchors
from image.metrics.localization import nms
from image.cv2transforms import AutoContrast, Gamma
from skimage.transform import rescale

autocontrast = lambda x: AutoContrast()(x).astype(np.float32)

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
    def __init__(self, path, frequency = 1, overwrite = True, bestonly = True, metric = 'loss'):
        if path.endswith('.tar'):
            self.path = path[:-4]
        else:
            self.path = path
        self.frequency = frequency
        self.overwrite = overwrite
        self.bestonly = bestonly
        self.metric = metric
        self.mult = -1 if metric == 'accuracy' else 1
        self.bestmetric = None

    def save(self, session, epoch, metrics):
        if epoch % self.frequency != 0:
            return
        if not self.bestonly:
            if self.overwrite:
                fname = self.path+'.tar'
            else:
                fname = f'{self.path}_{epoch}.tar'
            session.save(fname)
        elif self.bestmetric is None or metrics[self.metric]*self.mult < self.bestmetric*self.mult:
            self.bestmetric = metrics[self.metric]
            if self.overwrite:
                fname = self.path+'.tar'
            else:
                fname = f'{self.path}_{epoch}.tar'
            session.save(fname)

    def state_dict(self):
        state = {'bestmetric':self.bestmetric}
        return state

    def load_state_dict(self, state):
        self.bestmetric = state['bestmetric']


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
                 scheduler=None, scheduler_params=None, log_dir=None, saver=None, device = None):
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer(parameters) if opt_defaults is None else optimizer(parameters, **opt_defaults)
        self.scheduler =  None if scheduler is None else scheduler(self.optimizer, **scheduler_params)
        self.acc_func = acc_func
        self.log_dir = log_dir
        self.saver = saver
        self.epoch = 0
        self.epochs_left = 0

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.lossfunc(outputs, labels)
        loss['loss'].backward()
        self.optimizer.step()
        return (loss, outputs)

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        loss = defaultdict(lambda:0.0)
        size = 0
        accuracy = defaultdict(lambda:0.0)
        for inputs, labels in data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            batch_loss = self.lossfunc(outputs, labels)
            for k, v in batch_loss.items():
                loss[k] += v.item() * inputs.size(0)
            # accuracy += self.acc_func(outputs, labels).item()
            batch_acc = self.acc_func(outputs, labels)
            for k, v in batch_acc.items():
                accuracy[k] += v * inputs.size(0)
            size += inputs.size(0)
        for k, v in loss.items():
            loss[k] = loss[k]/size
        for k, v in accuracy.items():
            accuracy[k] = accuracy[k]/size
        return (loss, accuracy)

    def train(self, train_data, valid_data, epochs = None):
        if epochs is None:
             epochs = self.epochs_left
        if self.log_dir:
            writer = SummaryWriter(self.log_dir)
        for epoch in range(epochs):
            self.epoch += 1
            self.epochs_left = epochs-epoch-1
            self.model.train()
            train_loss = defaultdict(lambda:0.0)
            size = 0
            # train_acc = 0.0
            train_acc = defaultdict(lambda:0.0)
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
                    batch_acc = self.acc_func(outputs, labels)
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
                       f'Train Loss: {train_loss["loss"]:.4f}, Validation Loss: {valid_loss["loss"]:.4f}')
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
                # metrics['accuracy/train'] = train_acc
                # metrics['accuracy/validation'] = valid_acc
                for metric, value in metrics.items():
                    writer.add_scalar(metric, value, self.epoch)
            if self.saver:
                self.saver.save(self, self.epoch, {'loss':valid_loss['loss'], 'accuracy':valid_acc})
            pbar.close()

    def update_lr(self, factor):
        if self.scheduler:
            for group in self.optimizer.param_groups:
                group['initial_lr'] = group['initial_lr']*factor
            for i in range(len(self.scheduler.base_lrs)):
                self.scheduler.base_lrs[i] *= factor
        else:
            for group in self.optimizer.param_groups:
                group['lr'] = group['lr']*factor

    def save(self, path):
        state = {'epoch': self.epoch,
                 'epochs_left': self.epochs_left,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        if self.scheduler:
            state['scheduler'] = self.scheduler.state_dict()
        if self.log_dir:
            state['log_dir'] = self.log_dir
        if self.saver:
            state['saver'] = self.saver.state_dict()
        torch.save(state, path)

    def load(self, path):
        state = torch.load(path)
        self.epoch = state['epoch']
        self.epochs_left = state['epochs_left']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        if not self.log_dir and 'log_dir' in state:
            self.log_dir = state['log_dir']
        if self.saver and 'saver' in state:
            self.saver.load_state_dict(state['saver'])

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
