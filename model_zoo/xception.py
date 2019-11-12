import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict, deque
import numpy as np
import re

class SepConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, bn_args={'momentum':0.01}):
        super().__init__()
        self.conv_dwise = nn.Conv2d(in_channels, in_channels, 3, padding = 1, groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels, **bn_args)
        # Parameter initialization
        for layer in [self.conv_dwise, self.conv_1x1]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        init.ones_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def forward(self, x):
        out = self.bn(self.conv_1x1(self.conv_dwise(x)))
        return out

class StridedBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_args={'momentum':0.01}):
        super().__init__()
        self.activation_1 = nn.ReLU6()
        self.sepconv_1 = SepConv2d(in_channels, out_channels, bn_args)
        self.activation_2 = nn.ReLU6()
        self.sepconv_2 = SepConv2d(out_channels, out_channels, bn_args)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.reslink = nn.Conv2d(in_channels, out_channels,1,stride=2)

    def forward(self, x):
        out = self.reslink(x)+self.maxpool(self.sepconv_2(self.activation_2(self.sepconv_1(self.activation_1(x)))))
        return out

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, bn_args={'momentum':0.01}):
        super().__init__()
        self.activation_1 = nn.ReLU6()
        self.sepconv_1 = SepConv2d(in_channels, out_channels, bn_args)
        self.activation_2 = nn.ReLU6()
        self.sepconv_2 = SepConv2d(out_channels, out_channels, bn_args)
        self.activation_3 = nn.ReLU6()
        self.sepconv_3 = SepConv2d(out_channels, out_channels, bn_args)

    def forward(self, x):
        out = x
        for m in self._modules.values():
            out = m(out)
        return out + x

class Xception(nn.Module):

    def __init__(self, in_channels, features_num = 1, bn_args={'momentum':0.01}):

        super().__init__()
        self.features_num = features_num
        self.conv_0 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn_0 = nn.BatchNorm2d(32, **bn_args)
        self.relu_0 = nn.ReLU6()
        self.conv_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64, **bn_args)
        self.entryflow_1 = StridedBlock(64, 128, bn_args)
        self.entryflow_2 = StridedBlock(128, 256, bn_args)
        self.entryflow_3 = StridedBlock(256, 728, bn_args)
        self.middleflow = nn.Sequential(*[Block(728,728, bn_args) for i in range(8)])
        self.exitflow = nn.Sequential(
            StridedBlock(728,1024,bn_args),
            SepConv2d(1024, 1536, bn_args),
            nn.ReLU6(),
            SeparableConv2d(1536, 2048, bn_args)
            )
        # # Parameter initialization
        init.kaiming_uniform_(self.conv_0.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv_0.bias)
        init.kaiming_uniform_(self.conv_1.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv_1.bias)
        init.ones_(self.bn_0.weight)
        init.zeros_(self.bn_0.bias)
        init.ones_(self.bn_1.weight)
        init.zeros_(self.bn_1.bias)


    def forward(self,x):
        feature_maps = deque(maxlen=self.features_num)
        out = self.bn_1(self.conv_1(self.relu_0(self.bn_0(self.conv_0(x)))))
        feature_maps.append(out)
        out = self.entryflow_1(out)
        feature_maps.append(out)
        out = self.entryflow_2(out)
        feature_maps.append(out)
        out = self.entryflow_3(out)
        feature_maps.append(out)
        out = self.exitflow(self.middleflow(out))
        feature_maps.append(out)
        if self.features_num > 1:
            return list(reversed(feature_maps))
        else:
            return feature_maps.pop()
