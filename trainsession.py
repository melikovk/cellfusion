import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm
from collections import OrderedDict, defaultdict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from image.datasets.utils import get_cell_anchors, get_boxes_from_json
from image.metrics.localization import nms
from importlib import import_module
from PIL import Image
import math
import os.path
import cv2
import matplotlib.pyplot as plt
from session_saver import SessionSaver

# # Apex
# from apex import amp

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

    def train_step(self, inputs, labels, batch_size_multiplier, accumulate=True):
        # with torch.autograd.detect_anomaly():
        outputs = self.model(inputs)
        loss = self.lossfunc(outputs, labels)
        (loss['loss']/batch_size_multiplier).backward()
        # Apex
        # with amp.scale_loss(loss['loss'], self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        if not accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return (loss, outputs)

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        loss = defaultdict(float)
        size = 0
        accuracy = defaultdict(float)
        for inputs, labels in data:
            inputs = inputs.to(self.device)
            if isinstance(labels, list):
                labels = [label.to(self.device) for label in labels]
            else:
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
        return (loss, accuracy)

    def train(self, train_data, valid_data, total_epochs = None, start_epoch=0, batch_size_multiplier=1):
        if total_epochs is None:
             total_epochs = start_epoch + self.epochs_left
        if self.log_dir:
            writer = SummaryWriter(self.log_dir, purge_step = self.epoch+1)
        for epoch in range(start_epoch, total_epochs):
            self.epoch += 1
            self.epochs_left = total_epochs-epoch-1
            self.model.train()
            self.optimizer.zero_grad()
            train_loss = defaultdict(float)
            batch_accumulator = 0
            size = 0
            train_acc = defaultdict(float)
            if self.scheduler:
                self.scheduler.step(epoch)
            pbar = tqdm(total = len(train_data), leave = False)
            for inputs, labels in train_data:
                inputs = inputs.to(self.device)
                if isinstance(labels, list):
                    labels = [label.to(self.device) for label in labels]
                else:
                    labels = labels.to(self.device)
                if batch_accumulator + 1 < batch_size_multiplier:
                    batch_loss, outputs = self.train_step(inputs, labels, batch_size_multiplier)
                    batch_accumulator += 1
                else:
                    batch_loss, outputs = self.train_step(inputs, labels, batch_size_multiplier, accumulate=False)
                    batch_accumulator = 0
                # statistics
                for k, v in batch_loss.items():
                    train_loss[k] += v.item() * inputs.size(0)
                with torch.no_grad():
                    batch_acc = self.acc_func(self.model.get_prediction(outputs), self.model.get_targets(labels))
                for k, v in batch_acc.items():
                    train_acc[k] += v * inputs.size(0)
                size += inputs.size(0)
                pbar.update(1)
            for k, v in train_loss.items():
                train_loss[k] = train_loss[k]/size
            for k, v in train_acc.items():
                train_acc[k] = train_acc[k]/size
            valid_loss, valid_acc = self.evaluate(valid_data)
            message = (f'Epoch {epoch+1} of {total_epochs} took {tqdm.format_interval(pbar.last_print_t-pbar.start_t)}\n'
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
            if self.saver is not None:
                if not self.saver.save(self, self.epoch, {**valid_loss, **valid_acc}):
                    pbar.close()
                    return
            pbar.close()

    def reset_metrics(self, valid_data):
        if self.saver is not None:
            valid_loss, valid_acc = self.evaluate(valid_data)
            self.saver.reset()
            self.saver.save(self, self.epoch, {**valid_loss, **valid_acc})

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
        if self.scheduler:
            for group, lr in zip(self.optimizer.param_groups, lrs):
                group['initial_lr'] = lr
            for i, group in enumerate(self.optimizer.param_groups):
                self.scheduler.base_lrs[i] = group['initial_lr']
        else:
            for group, lr in zip(self.optimizer.param_groups, lrs):
                group['lr'] = lr

    def state_dict(self):
        state = {'epoch': self.epoch,
                 'epochs_left': self.epochs_left,
                 'model': self.model.state_dict(),
                 'model_type': {'name':type(self.model).__name__, 'module':type(self.model).__module__},
                 'model_config': self.model.config,
                 'param_groups': map_ids_to_names(self.optimizer.state_dict()['param_groups'], self.model),
                 'optimizer': self.optimizer.state_dict(),
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
        param_groups = [group.copy() for group in state['param_groups']]
        for g in param_groups:
            g['params'] = [named_params[name] for name in g['params']]
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

def map_ids_to_names(param_groups, model):
    reverse_dict = {id(parameter):name for name, parameter in model.named_parameters()}
    new_groups = [group.copy() for group in param_groups]
    for g in new_groups:
        g['params'] = [reverse_dict[id] for id in g['params']]
    return new_groups
