import glob
import os.path
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from trainsession import TrainSession, class_accuracy, SessionSaver, iou_accuracy
import numpy as np
import torchvision.transforms as transforms
import image.cv2transforms as cv2transforms
from model_zoo import mobilenet_v2, cnn_heads
from losses import yolo_loss
from model_zoo.vision_models import CNNModel, iou, nms, saveboxes, localization_accuracy
from image.datasets.yolo import YoloGridDataset, YoloRandomDataset, RandomLoader, labelsToBoxes
from model_zoo import catalog
import argparse

_MODEL_SELECTION = {'base': catalog.mobilenet_v2_1ch_object_detect_base,
                    'full': catalog.mobilenet_v2_1ch_object_detect_full,
                    'shrink': catalog.mobilenet_v2_1ch_object_detect_shrink,
                    'deep': mobilenet_v2_1ch_object_detect_deep}


def imgToTensor(img):
    return torch.unsqueeze(torch.from_numpy(img), 0)

def train_nucleus_mobilenet(modelchoice, datadir, modeldir, logdir, device = 'cuda:0',
                            chanel='nuclei', init_lr = 0.01, batch = 32,
                            t_max = 20, lr_mult = 0.5, n_cycles = 10):
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
                                        winsize = (224, 224),
                                        bsize = batch,
                                        length = 500,
                                        transforms = yolo_transforms) for names in train_names])
    validDataset = ConcatDataset([YoloRandomDataset(*names,
                                        winsize=(224, 224),
                                        bsize = batch,
                                        length = 500,
                                        transforms = yolo_transforms) for names in test_names])
    model = _MODEL_SELECTION[modelchoice]()
    session = TrainSession(model,
                           yolo_loss,
                           optim.Adam,
                           model.parameters(),
                           iou_accuracy,
                           log_dir = logdir,
                           opt_defaults = {'lr':init_lr,'weight_decay':1e-5},
                           scheduler = optim.lr_scheduler.CosineAnnealingLR,
                           scheduler_params = {'T_max':t_max},
                           saver = SessionSaver(modeldir),
                           device = torch.device(device))
    image_datasets = {'train': trainDataset, 'val': validDataset}
    dataloaders = {x: RandomLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=2) for x in ['train', 'val']}
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
    main_parser.add_argument('--chanel', type = int, default = 'nuclei', choices = ['nuclei','phase'], help = 'Image chanel to use')
    main_parser.add_argument('--init_lr', type = float, default = 0.01, help = 'Initial learning rate')
    main_parser.add_argument('--batch', type = int, default = 32, help = 'Batch size')
    main_parser.add_argument('--t_max', type = int, default = 20, help = 'Cycle length for scheduler')
    main_parser.add_argument('--lr_mult', type = float, default = 0.5, help = 'Factor to adjust learning rate for consecutive cycles')
    main_parser.add_argument('--n_cycles', type = int, default = 10, help = 'Number of cycles of training')
    main_args = main_parser.parse_args()
    print(main_args)
    # train_nucleus_mobilenet(modelchoice = main_args.model,
    #                         datadir = main_args.datadir,
    #                         modeldir = main_args.modeldir,
    #                         logdir = main_args.logdir,
    #                         device = f'cuda:{main_args.gpu}',
    #                         chanel = main_args.chanel,
    #                         init_lr = main_args.init_lr,
    #                         batch = main_args.batch,
    #                         t_max = main_args.t_max,
    #                         lr_mult = main_args.lr_mult,
    #                         n_cycles = main_args.n_cycles)
