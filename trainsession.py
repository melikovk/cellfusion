import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import skimage.io as io
import numpy as np
from image.datasets.yolo import labelsToBoxes
from model_zoo.vision_models import nms
from image.cv2transforms import AutoContrast, Gamma
from skimage.transform import rescale

autocontrast = lambda x: AutoContrast()(x).astype(np.float32)

def class_accuracy(outputs, labels):
    _, preds = torch.max(outputs.detach(), 1)
    return torch.sum(preds == labels.detach())

def iou_accuracy(outputs, labels):
    """ Estimates intersection over union accuracy metrics
    This is simplified function as id does not perform non-maximal supression
    and will penalize if the wrong cell predicts correct bounding box
    """
    X = outputs.detach()
    Y = labels.detach()
    batch_size = X.size(0)
    count = torch.sum(torch.ge(X[:,0,:,:], 0.5))
    ileft = torch.max(X[:,1,:,:], Y[:,1,:,:])
    iright = torch.min(X[:,1,:,:]+X[:,3,:,:], Y[:,1,:,:]+Y[:,3,:,:])
    itop = torch.max(X[:,2,:,:], Y[:,2,:,:])
    ibottom = torch.min(X[:,2,:,:]+X[:,4,:,:], Y[:,2,:,:]+Y[:,4,:,:])
    iwidths = torch.max(iright-ileft, torch.zeros_like(ileft))
    iheights = torch.max(ibottom-itop, torch.zeros_like(itop))
    iareas = iwidths*iheights*torch.ge(X[:,0,:,:], 0.5).float()
    uareas = X[:,3,:,:]*X[:,4,:,:] + Y[:,3,:,:]*Y[:,4,:,:] - iareas
    return torch.sum(iareas/uareas)/count*batch_size if count.item() > 0 else torch.tensor(0)

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
    boxes, scores = labelsToBoxes(labels, grid_size = grid_size, offset = offset, threshold = conf_threshold)
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
        loss.backward()
        self.optimizer.step()
        return (loss, outputs)

    def evaluate(self, data):
        self.model.eval()
        loss = 0.0
        size = 0
        accuracy = 0.0
        for inputs, labels in data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs).detach()
            loss += self.lossfunc(outputs, labels).item() * inputs.size(0)
            accuracy += self.acc_func(outputs, labels).item()
            size += inputs.size(0)
        return (loss/size, accuracy/size)

    def train(self, train_data, valid_data, epochs = None):
        if epochs is None:
             epochs = self.epochs_left
        if self.log_dir:
            writer = SummaryWriter(self.log_dir)
        for epoch in range(epochs):
            self.epoch += 1
            self.epochs_left = epochs-epoch-1
            self.model.train()
            loss = 0.0
            size = 0
            accuracy = 0.0
            if self.scheduler:
                self.scheduler.step(epoch)
            pbar = tqdm(total = len(train_data), leave = False)
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_loss, outputs = self.train_step(inputs, labels)
                # statistics
                loss += batch_loss.item() * inputs.size(0)
                accuracy += self.acc_func(outputs, labels).item()
                size += inputs.size(0)
                pbar.update(1)
            train_loss = loss / size
            train_acc = accuracy / size
            valid_loss, valid_acc = self.evaluate(valid_data)
            message = (f'Epoch {epoch+1} of {epochs} took {tqdm.format_interval(pbar.last_print_t-pbar.start_t)}\n'
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n'
                       f'Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_acc:.4f}')
            tqdm.write(message)
            if self.log_dir:
                metrics = {'loss/train': train_loss,
                           'loss/validation': valid_loss,
                           'accuracy/train': train_acc,
                           'accuracy/validation': valid_acc}
                for metric, value in metrics.items():
                    writer.add_scalar(metric, value, self.epoch)
            if self.saver:
                self.saver.save(self, self.epoch, {'loss':valid_loss, 'accuracy':valid_acc})
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
