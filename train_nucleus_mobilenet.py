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
from image.datasets.yolo import YoloGridDataset, YoloRandomDataset, RandomLoader, labels_to_boxes, get_cell_anchors
from image.metrics.localization import precision_recall_f1_batch, precision_recall_meanIOU_batch
from model_zoo import catalog
import argparse
from functools import partial

_MODEL_SELECTION = {'base': catalog.mobilenet_v2_1ch_object_detect_base,
                    'full': catalog.mobilenet_v2_1ch_object_detect_full,
                    'shrink': catalog.mobilenet_v2_1ch_object_detect_shrink,
                    'deep': catalog.mobilenet_v2_1ch_object_detect_deep,
                    'split_base': catalog.mobilenet_v2_1ch_object_detect_split_base}


def imgToTensor(img):
    return torch.unsqueeze(torch.from_numpy(img), 0)

def train_nucleus_mobilenet(modelchoice, datadir, modeldir, logdir, device = 'cuda:0',
                            chanel='nuclei', init_lr = 0.01, batch = 32,
                            t_max = 20, lr_mult = 0.5, n_cycles = 10, size_transform = 'log',
                            confidence_loss='crossentropy', localization_weight=1.):
    """ Procedure that trains mobilenet_v2 based nucleus recognition models
    Parameters:
        datadir: folder with train and test data
        device:  device to use for training
    """
    chanel += '/'
    with open(datadir+'train.txt') as f:
        train_names = [(datadir+chanel+name[:-1]+'.tif', datadir+'boxes/'+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    with open(datadir+'test.txt') as f:
        test_names = [(datadir+chanel+name[:-1]+'.tif', datadir+'boxes/'+name[:-1]+'boxes.json')
                        for name in f.readlines()]
    yolo_transforms = transforms.Compose([cv2transforms.AutoContrast(),
                                        cv2transforms.Typecast(np.float32),
                                        imgToTensor])
    trainDataset = ConcatDataset([YoloRandomDataset(*names,
                                        win_size = (224, 224),
                                        border_size = 32,
                                        length = 500,
                                        transforms = yolo_transforms) for names in train_names])
    validDataset = ConcatDataset([YoloGridDataset(*names,
                                        win_size=(224, 224),
                                        border_size = 32,
                                        transforms = yolo_transforms) for names in test_names])
    model = _MODEL_SELECTION[modelchoice]()
    labels_to_boxes_func = partial(labels_to_boxes, cell_anchors = get_cell_anchors([1],[]))
    session = TrainSession(model,
                           partial(object_detection_loss, confidence_loss = confidence_loss, size_transform=size_transform, localization_weight=localization_weight),
                           optim.Adam,
                           model.parameters(),
                           partial(precision_recall_meanIOU_batch, labeltoboxesfunc = labels_to_boxes_func, iou_thresholds=[0.5, 0.7, 0.9]),
                           log_dir = logdir,
                           opt_defaults = {'lr':init_lr,'weight_decay':1e-5},
                           scheduler = optim.lr_scheduler.CosineAnnealingLR,
                           scheduler_params = {'T_max':t_max},
                           saver = SessionSaver(modeldir),
                           device = torch.device(device))
    image_datasets = {'train': trainDataset, 'val': validDataset}
    dataloaders = {x: RandomLoader(image_datasets[x], batch_size=batch, shuffle=True, num_workers=2) for x in ['train', 'val']}
    # session.train(dataloaders['train'], dataloaders['val'], 3)
    for i in range(n_cycles):
        session.train(dataloaders['train'], dataloaders['val'], t_max)
        session.update_lr(lr_mult)

if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(description="Train mobilenet_v2 model for nucleus detection")
    main_parser.add_argument('model', choices = _MODEL_SELECTION.keys(), default= 'base',help = 'Mobilenet model variant to train')
    main_parser.add_argument('datadir', help = 'Directory containing training and test data')
    main_parser.add_argument('modeldir', help = 'File name prefix to save model and train session')
    main_parser.add_argument('logdir', help = 'Directory to log training progress')
    main_parser.add_argument('--gpu', type = int, default = 0, choices = range(torch.cuda.device_count()), help = 'GPU device to use')
    main_parser.add_argument('--chanel', default = 'nuclei', choices = ['nuclei','phase'], help = 'Image chanel to use')
    main_parser.add_argument('--init_lr', type = float, default = 0.01, help = 'Initial learning rate')
    main_parser.add_argument('--batch', type = int, default = 32, help = 'Batch size')
    main_parser.add_argument('--t_max', type = int, default = 20, help = 'Cycle length for scheduler')
    main_parser.add_argument('--lr_mult', type = float, default = 0.5, help = 'Factor to adjust learning rate for consecutive cycles')
    main_parser.add_argument('--n_cycles', type = int, default = 10, help = 'Number of cycles of training')
    main_parser.add_argument('--size_transform', choices = ['log','sqrt','none'], default= 'log', help = 'Transformation of the box size for loss calculation')
    main_parser.add_argument('--confidence_loss', choices = ['mse','corssentropy'], default= 'crossentropy', help = 'Transformation of the box size for loss calculation')
    main_parser.add_argument('--localization_weight', type = float, default = 1.0, help = 'Multiplier for the localization loss')
    main_args = main_parser.parse_args()
    train_nucleus_mobilenet(modelchoice = main_args.model,
                            datadir = main_args.datadir,
                            modeldir = main_args.modeldir,
                            logdir = main_args.logdir,
                            device = f'cuda:{main_args.gpu}',
                            chanel = main_args.chanel,
                            init_lr = main_args.init_lr,
                            batch = main_args.batch,
                            t_max = main_args.t_max,
                            lr_mult = main_args.lr_mult,
                            n_cycles = main_args.n_cycles,
                            size_transform = main_args.size_transform,
                            confidence_loss = main_args.confidence_loss,
                            localization_weight = main_args.localization_weight)
