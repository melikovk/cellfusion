import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import ConcatDataset
from trainsession import TrainSession, SessionSaver, get_filenames
import torchvision.transforms as transforms
import image.cv2transforms as cv2transforms
from losses import ObjectDetectionLoss
from model_zoo.vision_models import ObjectDetectionModel
from image.datasets.yolo import YoloDataset, RandomLoader, SSDDataset
from image.metrics.localization import PrecisionRecallF1MeanIOU
from model_zoo import catalog
from functools import partial
from image.random_transforms import RandomGamma, RandomContrast, RandomFlip, RandomZoom, AutoContrast
import argparse
import os.path

def train_model(datadir, modelfile, dataset_type, device, num_cycles, cycle_length, lr_multiplier, init_lr):
    """ Procedure that trains object recognition models
    Parameters:
        datadir: folder with train and test data
        device:  device to use for training
    """

    state = torch.load(modelfile. map_location='cpu')

    model = getattr(import_module(state['model_type']['module']), state['model_type']['name'])(state['model_config'])

    train_names, test_names = get_filenames(datadir, 'nuclei', 'new_curated_boxes_without_dead/')

    point_transforms = [RandomGamma(), RandomContrast()]

    geom_transforms = [RandomFlip(), RandomZoom()]

    norm_transform = AutoContrast()

    train_dataset_parameters = {
    'win_size': (224, 224),
    'border_size': 32,
    'window_overlap_threshold': .9,
    'grid_size': model.features.grid_size,
    'length': 500}
    if dataset_type == 'SSD':
        train_dataset_parameters['positive_anchor_threshold'] = 0.5
        train_dataset_parameters['background_anchor_threshold'] = 0.3
    elif dataset_type == 'YOLO':
        train_dataset_parameters['anchor_ignore_threshold'] = 0.5
    else:
        raise ValueError('dataset_type should be either SSD or YOLO')
    test_dataset_parameters = train_dataset_parameters.copy()
    test_dataset_parameters['win_size'] = (800, 800)
    test_dataset_parameters['length'] = 10

    if dataset_type == 'SSD':
        dataset_class = SSDDataset
    else:
        dataset_class = YoloDataset

    trainDataset = ConcatDataset([dataset_class(*names,
        cell_anchors = model.cell_anchors,
        point_transforms = point_transforms,
        geom_transforms = geom_transforms,
        norm_transform = norm_transform,
        **train_dataset_parameters) for names in train_names])

    testDataset = ConcatDataset([dataset_class(*names,
        cell_anchors = model.cell_anchors,
        point_transforms = point_transforms,
        geom_transforms = geom_transforms,
        norm_transform = norm_transform,
        **test_dataset_parameters) for names in test_names])

    train_dataloader_parameters = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 2 }

    test_dataloader_parameters = {
    'batch_size': 2,
    'shuffle': True,
    'num_workers': 2 }

    trainDataLoader = RandomLoader(trainDataset, **train_dataloader_parameters)
    testDataLoader = RandomLoader(testDataset, **test_dataloader_parameters)

    loss_parameters = {
    'confidence_loss': 'crossentropy',
    'size_transform': 'none',
    'localization_weight': 10.0,
    'normalize_per_anchor': True }

    lossfunc = ObjectDetectionLoss(**loss_parameters)

    accuracy_parameters = {
    'iou_thresholds': [0.5, 0.7, 0.9],
    'nms_threshold': 0.8}

    accuracy = PrecisionRecallF1MeanIOU(**accuracy_parameters)

    saver = SessionSaver(modeldir)

    optimizer = optim.Adam

    optimizer_defaults = {
    'lr': init_lr,
    'weight_decay':1e-5}

    scheduler = optim.lr_scheduler.CosineAnnealingLR
    scheduler_parameters = {
    'T_max': cycle_length }

    logdir = state['log_dir']

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
    session.epoch = state['epoch']
    for i in range(num_cycles):
        session.train(trainDataLoader, testDataLoader, cycle_length)
        session.update_lr(lr_multiplier)

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser('Training script for object detection models')
    main_parser.add_argument('datadir')
    main_parser.add_argument('modelfile')
    main_parser.add_argument('--dataset_type', default='SSD')
    main_parser.add_argument('--init_lr', type=float, default=.01)
    main_parser.add_argument('--cycle_length', type=int, default = 100)
    main_parser.add_argument('--lr_multiplier', type=float, default=.5)
    main_parser.add_argument('--num_cycles', type=int, default=5)
    main_parser.add_argument('--device', type=int, required=True)
    main_args = main_parser.parse_args()
    main_args.device = torch.device(f'cuda:{main_args.device}')
    train_model(**vars(main_args))
