import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict, deque
import numpy as np

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 2, expansion = 6, gn_args={'num_groups':8}):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.gn_expand = nn.GroupNorm(num_channels=channels, **gn_args)
        self.activation_expand = nn.ReLU6()
        self.conv_dwise = nn.Conv2d(channels, channels, 3, stride = stride, padding = 1, groups = channels)
        self.gn_dwise = nn.GroupNorm(num_channels=channels, **gn_args)
        self.activation_dwise = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, out_channels, 1)
        self.gn_shrink = nn.GroupNorm(num_channels=out_channels, **gn_args)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dwise, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.gn_expand, self.gn_dwise, self.gn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        x = self.activation_expand(self.gn_expand(self.conv_expand(x)))
        x = self.activation_dwise(self.gn_dwise(self.conv_dwise(x)))
        x = self.gn_shrink(self.conv_shrink(x))
        return x

class ResidualBottleneck(Bottleneck):

    def __init__(self, in_channels, expansion = 6, gn_args={'num_groups':8}):
        super().__init__(in_channels, in_channels, 1, expansion, gn_args)

    def forward(self, x):
        return x + super().forward(x)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, repeats = 1, stride = 2, expansion = 6, gn_args={'num_groups':8}):
        super().__init__()
        self.stride = stride
        self.linear_btlneck = Bottleneck(in_channels, out_channels, stride, expansion, gn_args)
        self.linear_activation = nn.ReLU6()
        for i in range(1, repeats):
            self.add_module(f'res_btlneck_{i}', ResidualBottleneck(out_channels, expansion, gn_args))

    def forward(self,x):
        for m in self._modules.values():
            x = m(x)
        return x

class MobileNetV2gn(nn.Module):

    def __init__(self, in_channels, features_num = 1, out_channels=(32,16,24,32,64,96,160,320),
        repeats=(1,2,3,4,3,3,1), strides=(1,2,2,2,1,2,1), expansions=(1,6,6,6,6,6,6), gn_args={'num_groups':8}):

        super().__init__()
        self.grid_size = np.prod(strides)*2
        self.features_num = features_num
        self.conv_init = nn.Conv2d(in_channels, out_channels[0], 3, stride=2, padding=1)
        self.gn_init = nn.GroupNorm(num_channels=out_channels[0], **gn_args)
        self.activation_init = nn.ReLU6()
        self.blocks = nn.ModuleList()
        for i in range(len(expansions)):
            self.blocks.append(Block(out_channels[i], out_channels[i+1], repeats[i], strides[i], expansions[i], gn_args))
        # Parameter initialization
        init.kaiming_uniform_(self.conv_init.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv_init.bias)
        init.ones_(self.gn_init.weight)
        init.zeros_(self.gn_init.bias)

    def forward(self,x):
        feature_maps = deque(maxlen=self.features_num)
        x = self.activation_init(self.gn_init(self.conv_init(x)))
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
