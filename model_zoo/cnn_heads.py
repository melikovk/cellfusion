import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict

class AdaptiveMaxAvgPool2D(nn.Module):
    """ 2D pooling layer that concatenates results of AdaptiveMaxPool2d and AdaptiveAvgPool2d
        Parameters:
            output_size: int or (int, int)- the size of the output 2D image HxH or HxW
    """
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self,x):
        return torch.cat([self.maxpool(x), self.avgpool(x)], dim=1)

_pool = {'max': nn.AdaptiveMaxPool2d,
         'avg': nn.AdaptiveAvgPool2d,
         'maxavg': AdaptiveMaxAvgPool2D}

_activation = {'relu': nn.ReLU,
               'relu6': nn.ReLU6}

class Classifier(nn.Module):

    def __init__(self, classes, in_features, pool = 'max', activation = 'relu', hidden = [512], bn_args={'momentum':0.01}, act_args={}):
        super().__init__()
        if pool == 'maxavg':
            in_features *= 2
        self.fc0 = _pool[pool](1)
        self.bn0 = nn.BatchNorm1d(in_features, **bn_args)
        self.activ0 = _activation[activation](**act_args)
        hidden = [in_features] + hidden
        for i in range(1, len(hidden)):
            self.add_module(f'fc{i}', nn.Linear(hidden[i-1], hidden[i]))
            self.add_module(f'bn{i}', nn.BatchNorm1d(hidden[i], **bn_args))
            self.add_module(f'activ{i}', _activation[activation](**act_args))
        self.out = nn.Linear(hidden[-1], classes)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.reshape(x, (x.shape[0], -1))
        for m in list(self.children())[1:]:
            x = m(x)
        return x

class ObjectDetectionHead(nn.Module):
    """ Fully convolutional object detection module.
    It is intended to be used on top of convolutional feature extractor.
    Outputs prediction of object presence, coordinates of the bounding box and it's sizes on a dense grid.
    Coordinates of the bounding box are relative to the grid cell and thus are within [0,1] interval.
    Parameters:
        in_features: int - size of the feature dimension of the feature extractor
        activation: {relu, relu6}
        hidden_features: [int] - size of features dimension of hidden layers
        hidden_kernel: [int] - size of kernel for Conv2d layers
        bn_arg: dict - BatchNorm2d parameters
        act_args: dict - Activation layers parameters
    Ouputs:
        5xHxWxB tensor of predictions
    """
    def __init__(self, in_features, activation='relu', hidden_features=[1024, 1024], hidden_kernel=[3, 3], bn_args={'momentum':0.01}, act_args={}):
        assert len(hidden_features) == len(hidden_kernel), "You should provide kernel_size for each hidden convolutional layer"
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_features, **bn_args)
        self.activ0 = _activation[activation](**act_args)
        hidden_features = [in_features] + hidden_features
        for i in range(1, len(hidden_features)):
            self.add_module(f'conv2d_{i}', nn.Conv2d(hidden_features[i-1], hidden_features[i], hidden_kernel[i-1], padding = hidden_kernel[i-1]//2))
            self.add_module(f'bn_{i}', nn.BatchNorm2d(hidden_features[i], **bn_args))
            self.add_module(f'activ_{i}', _activation[activation](**act_args))
        self.out = nn.Conv2d(hidden_features[-1], 5, 1)

    def forward(self, x):
        for m in list(self.children()):
            x = m(x)
        x = torch.cat([torch.sigmoid(x[:,:3,:,:]),x[:,-2:,:,:]], dim = 1)
        return x



#
# if __name__=="__main__":
#    main()
