import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict, deque
import numpy as np
import re

class SepConv2d(nn.Module):
    """ Separable convolution block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bn_args={'momentum':0.01}):
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
            padding = kernel_size//2, groups=in_channels)
        self.bn_dw = nn.BatchNorm2d(in_channels, **bn_args)
        self.relu_dw = nn.ReLU6()
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_1x1 = nn.BatchNorm2d(out_channels, **bn_args)
        # Parameter Initialization
        for layer in [self.conv_dw, self.conv_1x1]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_dw, self.bn_1x1]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.bn_1x1(self.conv_1x1(self.relu_dw(self.bn_dw(self.conv_dw(x)))))
        return out

class MBConv2d(nn.Module):
    """  Linear Mobile Bottleneck Block """

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, expansion = 3, bn_args={'momentum':0.01}):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, **bn_args)
        self.relu_expand = nn.ReLU6()
        self.conv_dw = nn.Conv2d(channels, channels, kernel_size, stride = stride, padding = kernel_size//2, groups = channels)
        self.bn_dw = nn.BatchNorm2d(channels, **bn_args)
        self.relu_dw = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, out_channels, 1)
        self.bn_shrink = nn.BatchNorm2d(out_channels, **bn_args)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dw, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_dw, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.relu_expand(self.bn_expand(self.conv_expand(x)))
        out = self.relu_dw(self.bn_dw(self.conv_dw(out)))
        out = self.bn_shrink(self.conv_shrink(out))
        return out

class RMBConv2d(nn.Module):
    """  Residual Mobile Bottleneck Block """

    def __init__(self, in_channels, kernel_size, expansion = 3, bn_args={'momentum':0.01}):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, **bn_args)
        self.relu_expand = nn.ReLU6()
        self.conv_dw = nn.Conv2d(channels, channels, kernel_size, padding = kernel_size//2, groups = channels)
        self.bn_dw = nn.BatchNorm2d(channels, **bn_args)
        self.relu_dw = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, in_channels, 1)
        self.bn_shrink = nn.BatchNorm2d(in_channels, **bn_args)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dw, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_dw, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.relu_expand(self.bn_expand(self.conv_expand(x)))
        out = self.relu_dw(self.bn_dw(self.conv_dw(out)))
        out = self.bn_shrink(self.conv_shrink(out))
        out = out + x
        return out

class SqueezeExcitation(nn.Module):
    """ Squeeze-Excitation module"""
    def __init__(self, in_channels, se_ratio):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//se_ratio, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//se_ratio, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return out

class RMBSEConv2d(nn.Module):
    """  Residual Mobile Bottleneck Block with squeeze-excitation"""

    def __init__(self, in_channels, kernel_size, expansion = 3, se_ratio = 8, bn_args={'momentum':0.01}):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, **bn_args)
        self.relu_expand = nn.ReLU6()
        self.conv_dw = nn.Conv2d(channels, channels, kernel_size, padding = kernel_size//2, groups = channels)
        self.bn_dw = nn.BatchNorm2d(channels, **bn_args)
        self.relu_dw = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, in_channels, 1)
        self.bn_shrink = nn.BatchNorm2d(in_channels, **bn_args)
        self.se = SqueezeExcitation(channels, se_ratio)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dw, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_dw, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.relu_expand(self.bn_expand(self.conv_expand(x)))
        out = self.relu_dw(self.bn_dw(self.conv_dw(out)))
        out = torch.mul(out, self.se(out))
        out = self.bn_shrink(self.conv_shrink(out))
        out = out + x
        return out

class MNasNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, repeats,
        stride = 1, expansion = 6, se_ratio = 0, bn_args={'momentum':0.01}):

        super().__init__()
        self.stride = stride
        self.conv0 = SepConv2d(in_channels, out_channels, kernel_size, stride, bn_args)
        self.relu0 = nn.ReLU6()
        if se_ratio == 0:
            for i in range(repeats):
                self.add_module(f'btlneck_{i}', RMBConv2d(out_channels, kernel_size, expansion, bn_args))
        else:
            for i in range(repeats):
                self.add_module(f'SEbtlneck_{i}', RMBSEConv2d(out_channels, kernel_size, expansion, se_ratio, bn_args))

    def forward(self,x):
        for m in self._modules.values():
            x = m(x)
        return x

class MNasNet(nn.Module):

    def __init__(self, in_channels, features_num = 1, out_channels=(32,16,24,32,64,96,160,320),
        repeats=(1,2,3,4,2,3,1), strides=(1,2,2,2,1,2,1), kernel_sizes=(3,3,5,3,3,5,3),
        se_ratios=(0,0,8,0,8,8,0), expansions=(1,6,3,6,6,6,6), bn_args={'momentum':0.01}):

        super().__init__()
        self.grid_size = np.prod(strides)*2
        self.features_num = features_num
        self.conv0 = nn.Conv2d(in_channels, out_channels[0], 3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(out_channels[0], **bn_args)
        self.relu0 = nn.ReLU6()
        self.blocks = nn.ModuleList()
        for i in range(len(expansions)):
            self.blocks.append(MNasNetBlock(out_channels[i], out_channels[i+1],
                kernel_sizes[i], repeats[i], strides[i], expansions[i], se_ratios[i], bn_args))
        # Parameter initialization
        init.kaiming_uniform_(self.conv0.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv0.bias)
        init.ones_(self.bn0.weight)
        init.zeros_(self.bn0.bias)

    def forward(self,x):
        feature_maps = deque(maxlen=self.features_num)
        x = self.relu0(self.bn0(self.conv0(x)))
        feature_maps.append(x)
        for block in self.blocks:
            x = block(x)
            if block.stride == 1:
                feature_maps.pop()
            feature_maps.append(x)
        if self.features_num > 1:
            return list(reversed(feature_maps))
        else:
            return feature_maps.pop()

class MNasNet1(MNasNet):

    def __init__(self, in_channels, features_num, bn_args={'momentum':0.01}):
        super().__init__(in_channels, features_num, bn_args=bn_args)

class MNasNet2(MNasNet):

    def __init__(self, in_channels, features_num, bn_args={'momentum':0.01}):
        super().__init__(in_channels, features_num, out_channels=(48,24,36,48,96,144,240,480),
            expansions=(1,3,6,6,3,6,3), bn_args=bn_args)

class EfficientNet1(MNasNet):

    def __init__(self, in_channels, features_num, bn_args={'momentum':0.01}):
        super().__init__(in_channels, features_num, out_channels=(32,16,24,40,80,112,192,320),
            repeats=(1,2,2,3,3,4,1), strides=(1,2,2,1,2,2,1), kernel_sizes=(3,3,5,3,5,5,3),
            se_ratios=(8,8,8,8,8,8,8), expansions=(1,6,6,6,6,6,6), bn_args=bn_args)

class EfficientNet2(MNasNet):

    def __init__(self, in_channels, features_num, bn_args={'momentum':0.01}):
        super().__init__(in_channels, features_num, out_channels=(48,24,36,48,96,144,240,480),
            repeats=(2,4,4,6,6,8,2), strides=(1,2,2,1,2,2,1), kernel_sizes=(3,3,5,3,5,5,3),
            se_ratios=(8,8,8,8,8,8,8), expansions=(1,6,6,6,6,6,6), bn_args=bn_args)
