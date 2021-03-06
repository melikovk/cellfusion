import torch
import torch.nn as nn
import torch.nn.init as init
from model_zoo.mobilenet_v2 import Block as ResidualBottleneckBlock

class NormTanh(nn.Module):
    """ Tanh normalized to be within min_val and max_mal """

    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        assert max_val > min_val
        self.register_buffer('min_val', torch.tensor(min_val))
        self.register_buffer('max_val', torch.tensor(max_val))


    def forward(self, x):
        return (torch.tanh(x)+self.min_val+1)*self.max_val/(self.min_val+2)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'

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

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_kernels, activation, bn_args, act_args):
        super().__init__()
        hidden_features = [in_features] + hidden_features
        for i in range(1, len(hidden_features)):
            self.add_module(f'conv_{i}', nn.Conv2d(hidden_features[i-1], hidden_features[i], hidden_kernels[i-1], padding = hidden_kernels[i-1]//2))
            self.add_module(f'bn_{i}', nn.BatchNorm2d(hidden_features[i], **bn_args))
            self.add_module(f'activ_{i}', _activation[activation](**act_args))
        # Parameter initialization
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                init.ones_(layer.weight)
                init.zeros_(layer.bias)
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                init.zeros_(layer.bias)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

class YoloHead(nn.Module):
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
        probability: {False | True} Return probability instead of logits (default is False)
        coordinate_transform: {tanh|sigmoid} transformation of box coordinate
    Ouputs:
        5*NAnchorsxHxWxB tensor of predictions
    """
    def __init__(self, in_features, anchors = 1, activation='relu', clsnums=[],
        hidden_features=[1024, 1024], hidden_kernels=[3, 3], bn_args={'momentum':0.01},
        act_args={}, coordinate_transform = 'tanh', eps = 1e-5):

        assert len(hidden_features) == len(hidden_kernels), \
            "You should provide kernel_size for each hidden convolutional layer"
        assert coordinate_transform == 'tanh' or coordinate_transform == 'sigmoid', \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.anchors = anchors
        self.register_buffer('eps', torch.tensor(eps))
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh(min_val=-1.0, max_val=2.0)
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=-1.0, max_val=2.0)
        self.conv_block = ConvolutionalBlock(in_features, hidden_features, hidden_kernels, activation, bn_args, act_args)
        self.out = Conv2d(hidden_features[-1], anchors*(5+sum(clsnums)), 1)
        # Initialize weights
        init.kaiming_uniform_(self.out.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.out.bias)

    def forward(self, x):
        n = self.anchors
        x = self.out(self.conv_block(x))
        return torch.cat([x[:,:-4*n,:,:],self.coord_func(x[:,-4*n:-2*n,:,:]),torch.max(x[:,-2*n:,:,:], self.eps)], dim = 1)

class YoloHeadSplit(nn.Module):
    """ Fully convolutional object detection module.
    It is intended to be used on top of convolutional feature extractor.
    Outputs prediction of object presence, coordinates of the bounding box and it's sizes on a dense grid.
    Coordinates of the bounding box are relative to the grid cell and thus are within [0,1] interval.
    Uses similar but independent subnets for objectness classification and bounding box regression
    Parameters:
        in_features: int - size of the feature dimension of the feature extractor
        activation: {relu, relu6}
        hidden_features: [int] - size of features dimension of hidden layers
        hidden_kernel: [int] - size of kernel for Conv2d layers
        bn_arg: dict - BatchNorm2d parameters
        act_args: dict - Activation layers parameters
        coordinate_transform: {tanh|sigmoid} transformation of box coordinate
    Ouputs:
        5*NAnchorsxHxWxB tensor of predictions
    """
    def __init__(self, in_features, anchors=1, activation='relu', clsnums=[],
        hidden_features=[256, 256, 256, 256], hidden_kernels=[3, 3, 3, 3],
        bn_args={'momentum':0.01}, act_args={}, coordinate_transform = 'tanh', eps = 1e-5):

        assert len(hidden_features) == len(hidden_kernels), \
            "You should provide kernel_size for each hidden convolutional layer"
        assert coordinate_transform in ['hardtanh', 'sigmoid', 'tanh'], \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))
        self.anchors = anchors
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh(min_val=-1.0, max_val=2.0)
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=-1.0, max_val=2.0)
        self.obj_cls_subnet = ConvolutionalBlock(in_features, hidden_features, hidden_kernels, activation, bn_args, act_args)
        self.obj_cls_out = nn.Conv2d(hidden_features, self.anchors*(1+sum(clsnums)), 1)
        self.box_subnet = ConvolutionalBlock(in_features, hidden_features, hidden_kernels, activation, bn_args, act_args)
        self.box_out = nn.Conv2d(hidden_features, 4*self.anchors, 1)
        # Initialize Weights
        init.kaiming_uniform_(self.obj_cls_out.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.box_out.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.obj_cls_out.bias)
        init.zeros_(self.box_out.bias)

    def forward(self, x):
        n = self.anchors
        x_obj_cls = self.obj_cls_out(self.obj_cls_subnet(x))
        x_box = self.box_out(self.box_subnet(x))
        return torch.cat([x_obj_cls, self.coord_func(x_box[:,:2*n,:,:]), torch.max(x_box[:,2*n:,:,:], self.eps)], dim = 1)

class YoloHeadSplitResBtlneck(nn.Module):
    def __init__(self, in_features, anchors=1, clsnums=[], repeats = 4,
        hidden_features = 256, expansion = 3, bn_args={'momentum':0.01},
        act_args={}, coordinate_transform='tanh', eps=1e-5):

        assert coordinate_transform in ['hardtanh', 'sigmoid', 'tanh'], \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))
        self.anchors = anchors
        self.clsnums = clsnums
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh(min_val=-1.0, max_val=2.0)
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=-1.0, max_val=2.0)
        self.obj_cls_subnet = ResidualBottleneckBlock(in_features, out_channels=hidden_features,
            repeats=repeats, stride=1, expansion=expansion, bn_args = bn_args)
        self.obj_cls_out = nn.Conv2d(hidden_features, self.anchors*(1+sum(clsnums)), 1)
        self.box_subnet = ResidualBottleneckBlock(in_features, out_channels=hidden_features,
            repeats=repeats, stride=1, expansion=expansion, bn_args = bn_args)
        self.box_out = nn.Conv2d(hidden_features, 4*self.anchors, 1)
        # Initialize Weights
        init.kaiming_uniform_(self.obj_cls_out.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.box_out.weight, mode='fan_in', nonlinearity='relu')
        init.constant_(self.obj_cls_out.bias, -7.)
        init.zeros_(self.box_out.bias)

    def forward(self,x):
        n = self.anchors
        x_obj_cls = self.obj_cls_out(self.obj_cls_subnet(x))
        x_box = self.box_out(self.box_subnet(x))
        return torch.cat([x_obj_cls, self.coord_func(x_box[:,:2*n,:,:]), torch.max(x_box[:,2*n:,:,:], self.eps)], dim = 1)
