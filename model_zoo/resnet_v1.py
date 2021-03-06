import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict, deque

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, bn_args={'momentum':0.01}):
        super().__init__()
        channels = in_channels // 4
        self.conv_shrink = nn.Conv2d(in_channels, channels, 1)
        self.bn_shrink = nn.BatchNorm2d(channels, **bn_args)
        self.activation_shrink = nn.ReLU6()
        self.conv_3x3 = nn.Conv2d(channels, channels, 3, stride = 1, padding = 1)
        self.bn_3x3 = nn.BatchNorm2d(channels, **bn_args)
        self.activation_3x3 = nn.ReLU6()
        self.conv_expand = nn.Conv2d(channels, in_channels, 1)
        self.bn_expand = nn.BatchNorm2d(in_channels, **bn_args)
        self.activation_sum = nn.ReLU6()
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_3x3, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_3x3, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.activation_shrink(self.bn_shrink(self.conv_shrink(x)))
        out = self.activation_3x3(self.bn_3x3(self.conv_3x3(out)))
        out = self.bn_expand(self.conv_expand(out))
        return self.activation_sum(out+x)

class DownsampleBottleneck(nn.Module):
    def __init__(self, in_channels, expansion=2, bn_args={'momentum':0.01}):
        super().__init__()
        channels = in_channels // 4
        out_channels = in_channels * expansion
        self.conv_shrink = nn.Conv2d(in_channels, channels, 1)
        self.bn_shrink = nn.BatchNorm2d(channels, **bn_args)
        self.activation_shrink = nn.ReLU6()
        self.conv_3x3 = nn.Conv2d(channels, channels, 3, stride = 2, padding = 1)
        self.bn_3x3 = nn.BatchNorm2d(channels, **bn_args)
        self.activation_3x3 = nn.ReLU6()
        self.conv_expand = nn.Conv2d(channels, out_channels, 1)
        self.bn_expand = nn.BatchNorm2d(out_channels, **bn_args)
        self.activation_sum = nn.ReLU6()
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding = 1)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_3x3, self.conv_shrink, self.conv_shortcut]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_3x3, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        out = self.activation_shrink(self.bn_shrink(self.conv_shrink(x)))
        out = self.activation_3x3(self.bn_3x3(self.conv_3x3(out)))
        out = self.bn_expand(self.conv_expand(out))
        return self.activation_sum(out+self.conv_shortcut(x))

class Block(nn.Module):
    def __init__(self, in_channels, repeats, expansion = 2, bn_args={'momentum':0.01}):
        super().__init__()
        self.btlnecks = nn.ModuleList()
        out_channels = in_channels * expansion
        self.btlnecks.append(DownsampleBottleneck(in_channels, expansion, bn_args))
        for _ in range(repeats-1):
            self.btlnecks.append(ResidualBottleneck(out_channels, bn_args))

    def forward(self, x):
        for module in self.btlnecks:
            x = module(x)
        return x

class ResNetV1(nn.Module):
    def __init__(self, in_channels, repeats, expansions, features_num = 1, bn_args={'momentum':0.01}):
        super().__init__()
        channels = 64
        self.features_num = features_num
        self.grid_size = 2**(len(repeats)+1)
        self.conv_init = nn.Conv2d(in_channels, channels, kernel_size = 7, stride = 2, padding = 3)
        self.bn_init = nn.BatchNorm2d(channels, **bn_args)
        self.activation_init = nn.ReLU6()
        self.blocks = nn.ModuleList()
        for reps, expansion in zip(repeats, expansions):
            self.blocks.append(Block(channels, reps, expansion, bn_args))
            channels = channels * expansion

    def forward(self, x):
        feature_maps = deque(maxlen=self.features_num)
        x = self.activation_init(self.bn_init(self.conv_init(x)))
        feature_maps.append(x)
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)
        if self.features_num > 1:
            return feature_maps
        else:
            return feature_maps.pop()

class ResNet50V1(ResNetV1):
    def __init__(self, in_channels, features_num=1, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [3,4,6,3], [4,2,2,2], features_num, bn_args)

class ResNet101V1(ResNetV1):
    def __init__(self, in_channels, features_num=1, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [3,4,23,3], [4,2,2,2], features_num, bn_args)

class ResNet152V1(ResNetV1):
    def __init__(self, in_channels, features_num=1, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [3,8,36,3], [4,2,2,2], features_num, bn_args)
