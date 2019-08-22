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
from image.metrics.localization import PrecisionRecallMeanIOU
from model_zoo import catalog
from functools import partial
from image.random_transforms import RandomGamma, RandomContrast, RandomFlip, RandomZoom, AutoContrast
import argparse
import os.path

def resume_model_training(datadir, modelfile, dataset_type, device, num_cycles, cycle_length, lr_multiplier, init_lr):
    """ Procedure that trains object recognition models
    Parameters:
        datadir: folder with train and test data
        device:  device to use for training
    """

    session_state = torch.load(modelfile, map_location='cpu')

    session = TrainSession.restore_from_state_dict(session_state, device)

    train_names, test_names = get_filenames(datadir, 'nuclei', 'new_curated_boxes_without_dead/')

    point_transforms = [RandomGamma(), RandomContrast()]

    geom_transforms = [RandomFlip(), RandomZoom()]

    norm_transform = AutoContrast()

    train_dataset_parameters = {
    'win_size': (224, 224),
    'border_size': 32,
    'window_overlap_threshold': .7,
    'grid_size': session.model.features.grid_size,
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
        cell_anchors = session.model.cell_anchors,
        point_transforms = point_transforms,
        geom_transforms = geom_transforms,
        norm_transform = norm_transform,
        **train_dataset_parameters) for names in train_names])

    testDataset = ConcatDataset([dataset_class(*names,
        cell_anchors = session.model.cell_anchors,
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR
    scheduler_parameters = {
    'T_max': cycle_length }

    logdir = os.path.join('runs', os.path.split(modelfile)[1])

    # Finish last unfinished training cycle
    session.train(trainDataLoader, testDataLoader)
    # Initialize and assign new scheduler
    session.scheduler = scheduler(session.optimizer, **scheduler_parameters)
    # Start new training cycles
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
    resume_model_training(**vars(main_args))
