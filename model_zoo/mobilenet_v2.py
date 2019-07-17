import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 2, expansion = 6, bn_momentum = .01):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, momentum = bn_momentum)
        self.activation_expand = nn.ReLU6()
        self.conv_dwise = nn.Conv2d(channels, channels, 3, stride = stride, padding = 1, groups = channels)
        self.bn_dwise = nn.BatchNorm2d(channels, momentum = bn_momentum)
        self.activation_dwise = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, out_channels, 1)
        self.bn_shrink = nn.BatchNorm2d(out_channels, momentum = bn_momentum)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dwise, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.constant_(layer.bias, 0.0)
        for layer in [self.bn_expand, self.bn_dwise, self.bn_shrink]:
            init.constant_(layer.weight, 1.0)
            init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.activation_expand(self.bn_expand(self.conv_expand(x)))
        x = self.activation_dwise(self.bn_dwise(self.conv_dwise(x)))
        x = self.bn_shrink(self.conv_shrink(x))
        return x

class ResidualBottleneck(Bottleneck):

    def __init__(self, in_channels, expansion = 6, bn_momentum = .01):
        super().__init__(in_channels, in_channels, 1, expansion, bn_momentum)

    def forward(self, x):
        return x + super().forward(x)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, repeat = 1, stride = 2, expansion = 6, bn_momentum = .01):
        super().__init__()
        self.linear_btlneck = Bottleneck(in_channels, out_channels, stride, expansion, bn_momentum)
        for i in range(1, repeat):
            self.add_module(f'res_btlneck_{i}', ResidualBottleneck(out_channels, expansion, bn_momentum))

    def forward(self,x):
        for m in self.children():
            x = m(x)
        return x

class MobileNetV2(nn.Module):

    def __init__(self, in_channels, out_channels=(32,16,24,32,64,96,160,320), repeats=(1,2,3,4,3,3,1), strides=(1,2,2,2,1,2,1), expansions=(1,6,6,6,6,6,6), bn_momentum=.01):
        super().__init__()
        self.grid_size = np.prod(strides)*2
        self.conv_init = nn.Conv2d(in_channels, out_channels[0], 3, stride=2, padding=1)
        for i in range(len(expansions)):
            self.add_module(f'block_{i+1}',Block(out_channels[i], out_channels[i+1], repeats[i], strides[i], expansions[i], bn_momentum))

    def forward(self,x):
        for m in self.children():
            x = m(x)
        return x
