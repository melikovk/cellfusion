import glob
import os.path
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from trainsession import TrainSession, SessionSaver
import numpy as np
import torchvision.transforms as transforms
import image.cv2transforms as cv2transforms
from model_zoo import mobilenet_v2, cnn_heads
from losses import yolo1_loss, yolo2_loss, object_detection_loss
from model_zoo.vision_models import ObjectDetectionModel, saveboxes
from image.datasets.yolo import YoloDataset, RandomLoader, labels_to_boxes, get_cell_anchors, SSDDataset
from image.metrics.localization import precision_recall_f1, precision_recall_meanIOU
from model_zoo import catalog
import argparse
from functools import partial

""" Training and model parameters are defined below """

# Model to train
MODEL_CLASS = catalog.mobilenet_v2_base_1ch_split_yolo
# Location of data files
DATADIR = '~/workspace/fusion_data/Automate/20x/'
# Imaging chanel to use (i.e. name of the subfolder with images )
CHANNEL = 'nuclei'
# Data transformation to apply to images
DATA_TRANSFORMS = transforms.Compose([
    cv2transforms.AutoContrast(),
    cv2transforms.Typecast(np.float32),
    imgToTensor ])
# Location to save training session. Set to None to disable session saving
MODELDIR = '~/workspace/cellfusion/models/flnuc_mobilenet2_split_base_lw5'
# Location to save log data (loss and accuracy log). Set to None to disable logging
LOGDIR = '~/workspace/cellfusion/models/flnuc_mobilenet2_split_base_lw5'
# Set Parameters of trainning Dataset below
TRAIN_DATASET_CLASS = SSDDataset
TRAIN_DATASET_PARAMETERS = {
'win_size': (224, 224),
'border_size': 32,
'positive_anchor_threshold': 0.7,
'background_anchor_threshold': 0.5,
'denominator': 'union',
'window_overlap_threshold': .25,
'grid_size': 32,
'sample': 'random', # random or grid
'length': 500, # works only when sample == 'random'
'seed': None,  # works only when sample == 'random'
'stride': None, # Works only when sample == 'grid'
'transforms': data_transforms }
# Set Parameters of trainning Dataset below
TEST_DATASET_CLASS = SSDDataset
TEST_DATASET_PARAMETERS = {
'win_size': (224, 224),
'border_size': 32,
'positive_anchor_threshold': 0.7,
'background_anchor_threshold': 0.5,
'denominator': 'union',
'window_overlap_threshold': .25,
'grid_size': 32,
'sample': 'random', # random or grid
'length': 100, # works only when sample == 'random'
'seed': None,  # works only when sample == 'random'
'stride': None, # Works only when sample == 'grid'
'transforms': data_transforms }
# Set loss function and it parameters below
LOSS_FUNCTION = object_detection_loss
LOSS_PARAMETERS = {
'reduction': 'mean',
'confidence_loss': 'crossentropy',
'size_transform': 'none',
'localization_weight': 5.0,
'normalize_per_anchor': True }
# Set optimizer at default parameter below
OPTIMIZER = optim.Adam
OPTIMIZER_DEFAULTS = {
'lr': 0.01,
'weight_decay':1e-5}
# Set accuracy function and its parameters below
ACCURACY_FUNCTION = precision_recall_meanIOU
ACCURRACY_PARAMETERS = {
'iou_thresholds': [0.5, 0.7, 0.9],
'nms_threshold': 0.8}
# Set scheduler and scheduler options below
SCHEDULER = optim.lr_scheduler.CosineAnnealingLR,
SCHEDULER_PARAMETERS = {
'T_max': 100 }
# Set training device
DEVICE = 'cuda:0'
# Set train dataloader class and dataloader parameters
TRAIN_DATALOADER_CLASS = RandomLoader
TRAIN_DATALOADER_PARAMETERS = {
'batch_size': 32,
'shuffle': True,
'num_workers': 2 }
# Set test dataloader class and dataloader parameters
TEST_DATALOADER_CLASS = RandomLoader
TEST_DATALOADER_PARAMETERS = {
'batch_size': 32,
'shuffle': True,
'num_workers': 2 }
LEARNING_RATE_MULTIPLIER = 0.1
NUMBER_OF_TRAINING_CYCLES = 5

def imgToTensor(img):
    return torch.unsqueeze(torch.from_numpy(img), 0)

def get_filenames(datadir, channel):
    chanel += '/'
    with open(datadir+'train.txt') as f:
        train_names = [(datadir+chanel+name[:-1]+'.tif', datadir+'boxes/'+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    with open(datadir+'test.txt') as f:
        test_names = [(datadir+chanel+name[:-1]+'.tif', datadir+'boxes/'+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    return train_names, test_names

def train_model(modelclass, datadir, channel, data_transforms, train_dataset_class,
    train_dataset_parameters, test_dataset_class, test_dataset_parameters, train_dataloader_class,
    train_dataloader_parameters, test_dataloader_class, test_dataloader_parameters,
    loss_function, loss_parameters, optimizer, optimizer_defaults, accuracy_function,
    accuracy_parameters, scheduler, scheduler_parameters, modeldir, logdir, device,
    lr_multiplier, n_cycles):
    """ Procedure that trains mobilenet_v2 based nucleus recognition models
    Parameters:
        datadir: folder with train and test data
        device:  device to use for training
    """
    train_names, test_names = get_filenames(datadir, channel)
    model = modelclass()
    trainDataset = ConcatDataset([train_dataset_class(*names,
        cell_anchors = model.cell_anchors,
        transforms = data_transforms,
        **train_dataset_parameters) for names in train_names])
    testDataset = ConcatDataset([test_dataset_class(*names,
        cell_anchors = model.cell_anchors,
        transforms = data_transforms,
        **test_dataset_parameters) for names in test_names])
    trainDataLoader = train_dataloader_class(trainDataset, **train_dataloader_parameters}
    testDataLoader = test_dataloader_class(ttestDataset, **test_dataloader_parameters)
    lossfunc = partial(loss_function, **loss_parameters)
    accuracy = partial(accuracy_function, **accuracy_parameters)
    saver = None if modeldir is None else SessionSaver(modeldir)
    session = TrainSession(model,
                           lossfunc,
                           optimizer,
                           model.parameters(),
                           accuracy,
                           log_dir = logdir,
                           opt_defaults = optimizer_defaults,
                           scheduler = scheduler,
                           scheduler_params = scheduler_parameters,
                           saver = saver,
                           device = torch.device(device))
    for i in range(n_cycles):
        session.train(trainDataLoader, testDataLoader, t_max)
        session.update_lr(lr_multiplier)

if __name__ == "__main__":
    train_model(modelclass = MODEL_CLASS,
        datadir = DATADIR,
        channel = CHANNEL,
        data_transforms = DATA_TRANSFORMS,
        train_dataset_class = TRAIN_DATASET_CLASS,
        train_dataset_parameters = TRAIN_DATASET_PARAMETERS,
        test_dataset_class = TEST_DATASET_CLASS,
        test_dataset_parameters = TEST_DATASET_PARAMETERS,
        train_dataloader_class = TRAIN_DATASET_CLASS,
        train_dataset_parameters = TRAIN_DATALOADER_CLASS,
        test_dataloader_class = TEST_DATALOADER_CLASS,
        test_dataloader_parameters = TEST_DATALOADER_PARAMETERS,
        loss_function = LOSS_FUNCTION,
        loss_parameters = LOSS_PARAMETERS,
        accuracy_function = ACCURACY_FUNCTION,
        accuracy_parameters = ACCURRACY_PARAMETERS,
        scheduler = SCHEDULER,
        scheduler_parameters = SCHEDULER_PARAMETERS,
        modeldir = MODELDIR,
        logdir = LOGDIR,
        device = DEVICE,
        lr_multiplier = LEARNING_RATE_MULTIPLIER,
        n_cycle = NUMBER_OF_TRAINING_CYCLES)
