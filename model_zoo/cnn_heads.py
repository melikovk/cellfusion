import torch
import torch.nn as nn
import torch.nn.init as init
from model_zoo.mobilenet_v2 import Block as ResidualBottleneckBlock

class NormTanh(nn.Module):
    """ Tanh normalized to be within min_val and max_mal """
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        assert self.max_val > self.min_val

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
    def __init__(self, in_features, out_features, hidden_features, hidden_kernels, activation, bn_args, act_args):
        super().__init__()
        hidden_features = [in_features] + hidden_features
        for i in range(1, len(hidden_features)):
            self.add_module(f'conv_{i}', nn.Conv2d(hidden_features[i-1], hidden_features[i], hidden_kernels[i-1], padding = hidden_kernels[i-1]//2))
            self.add_module(f'bn_{i}', nn.BatchNorm2d(hidden_features[i], **bn_args))
            self.add_module(f'activ_{i}', _activation[activation](**act_args))
        self.out = nn.Conv2d(hidden_features[-1], out_features, 1)
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
        probability: {False | True} Return probability instead of logits (default is False)
        coordinate_transform: {tanh|sigmoid} transformation of box coordinate
    Ouputs:
        5*NAnchorsxHxWxB tensor of predictions
    """
    def __init__(self, in_features, anchors = 1, activation='relu', hidden_features=[1024, 1024],
                 hidden_kernels=[3, 3], bn_args={'momentum':0.01}, act_args={},
                 coordinate_transform = 'tanh', eps = 1e-5):
        assert len(hidden_features) == len(hidden_kernels), \
            "You should provide kernel_size for each hidden convolutional layer"
        assert coordinate_transform == 'tanh' or coordinate_transform == 'sigmoid', \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.anchors = anchors
        self.register_buffer('eps', torch.tensor(eps))
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh()
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.bn0 = nn.BatchNorm2d(in_features, **bn_args)
        self.activ0 = _activation[activation](**act_args)
        self.conv_block = ConvolutionalBlock(in_features, 5*self.anchors,hidden_features, hidden_kernels, activation, bn_args, act_args)

    def forward(self, x):
        n = self.anchors
        x = self.conv_block(self.activ0(self.bn0(x)))
        x = torch.cat([x[:,:n,:,:],self.coord_func(x[:,n:3*n,:,:]),torch.max(x[:,3*n:,:,:], self.eps)], dim = 1)
        return x

class ObjectDetectionHeadSplit(nn.Module):
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
    def __init__(self, in_features, anchors=1, activation='relu', hidden_features=[256, 256, 256, 256],
                 hidden_kernels=[3, 3, 3, 3], bn_args={'momentum':0.01}, act_args={},
                 coordinate_transform = 'tanh', eps = 1e-5):
        assert len(hidden_features) == len(hidden_kernels), \
            "You should provide kernel_size for each hidden convolutional layer"
        assert coordinate_transform in ['hardtanh', 'sigmoid', 'tanh'], \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))
        self.anchors = anchors
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh()
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.bn0 = nn.BatchNorm2d(in_features, **bn_args)
        self.activ0 = _activation[activation](**act_args)
        self.object_subnet = ConvolutionalBlock(in_features, self.anchors, hidden_features, hidden_kernels, activation, bn_args, act_args)
        self.box_subnet = ConvolutionalBlock(in_features, 4*self.anchors, hidden_features, hidden_kernels, activation, bn_args, act_args)
        # Parameter initialization
        init.ones_(self.bn0.weight)
        init.zeros_(self.bn0.bias)

    def forward(self, x):
        n = self.anchors
        x = self.activ0(self.bn0(x))
        x_obj = self.object_subnet(x)
        x_box = self.box_subnet(x)
        x = torch.cat([x_obj, self.coord_func(x_box[:,:2*n,:,:]), torch.max(x_box[:,2*n:,:,:], self.eps)], dim = 1)
        return x

class ObjectDetectionHeadSplitResBtlneck(nn.Module):
    def __init__(self, in_features, anchors=1, clsnums=None, activation='relu',
        repeats = 4, hidden_features = 256, expansion = 3, bn_args={'momentum':0.01},
        act_args={}, coordinate_transform='tanh', eps=1e-5):

        assert coordinate_transform in ['hardtanh', 'sigmoid', 'tanh'], \
            "coordinate_transform should be 'hardtanh', 'tanh' or 'sigmoid'"
        super().__init__()
        self.register_buffer('eps', torch.tensor(eps))
        self.anchors = anchors
        if coordinate_transform == 'tanh':
            self.coord_func = NormTanh()
        elif coordinate_transform == 'sigmoid':
            self.coord_func = nn.Sigmoid()
        else:
            self.coord_func = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.bn0 = nn.BatchNorm2d(in_features, **bn_args)
        self.activ0 = _activation[activation](**act_args)
        self.object_subnet = ResidualBottleneckBlock(in_features, out_channels=hidden_features,
            repeats=repeats, stride=1, expansion=expansion, bn_args = bn_args)
        self.object_out = nn.Conv2d(hidden_features, self.anchors, 1)
        self.box_subnet = ResidualBottleneckBlock(in_features, out_channels=hidden_features,
            repeats=repeats, stride=1, expansion=expansion, bn_args = bn_args)
        self.box_out = nn.Conv2d(hidden_features, 4*self.anchors, 1)
        if clsnums is not None:
            self.cls_out = nn.Conv2d(hidden_features, self.anchors*clsnums, 1)
        # Parameter initialization
        init.ones_(self.bn0.weight)
        init.zeros_(self.bn0.bias)

    def forward(self,x):
        n = self.anchors
        x = self.activ0(self.bn0(x))
        x_obj_subnet = self.object_subnet(x)
        x_obj = self.object_out(x_obj_subnet)
        x_box = self.box_out(self.box_subnet(x))
        x = torch.cat([x_obj, self.coord_func(x_box[:,:2*n,:,:]),
            torch.max(x_box[:,2*n:,:,:], self.eps)], dim = 1)
        if hasattr(self, 'cls_out'):
            x_cls = self.cls_out(x_obj_subnet)
            return [x, x_cls]
        else:
            return x


#
# if __name__=="__main__":
#    main()
