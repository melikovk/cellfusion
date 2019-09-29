import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 2, expansion = 6, bn_args={'momentum':0.01}):
        super().__init__()
        channels = expansion*in_channels
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, **bn_args)
        self.activation_expand = nn.ReLU6()
        self.conv_dwise = nn.Conv2d(channels, channels, 3, stride = stride, padding = 1, groups = channels)
        self.bn_dwise = nn.BatchNorm2d(channels, **bn_args)
        self.activation_dwise = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, out_channels, 1)
        self.bn_shrink = nn.BatchNorm2d(out_channels, **bn_args)
        # Parameter initialization
        for layer in [self.conv_expand, self.conv_dwise, self.conv_shrink]:
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)
        for layer in [self.bn_expand, self.bn_dwise, self.bn_shrink]:
            init.ones_(layer.weight)
            init.zeros_(layer.bias)

    def forward(self, x):
        x = self.activation_expand(self.bn_expand(self.conv_expand(x)))
        x = self.activation_dwise(self.bn_dwise(self.conv_dwise(x)))
        x = self.bn_shrink(self.conv_shrink(x))
        return x

class ResidualBottleneck(Bottleneck):

    def __init__(self, in_channels, expansion = 6, bn_args={'momentum':0.01}):
        super().__init__(in_channels, in_channels, 1, expansion, bn_args)

    def forward(self, x):
        return x + super().forward(x)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, repeats = 1, stride = 2, expansion = 6, bn_args={'momentum':0.01}):
        super().__init__()
        self.stride = stride
        self.linear_btlneck = Bottleneck(in_channels, out_channels, stride, expansion, bn_args)
        self.linear_activation = nn.ReLU6()
        for i in range(1, repeats):
            self.add_module(f'res_btlneck_{i}', ResidualBottleneck(out_channels, expansion, bn_args))

    def forward(self,x):
        for m in self._modules.values():
            x = m(x)
        return x

# class MobileNetV2(nn.Module):
#
#     def __init__(self, in_channels, out_channels=(32,16,24,32,64,96,160,320), repeats=(1,2,3,4,3,3,1), strides=(1,2,2,2,1,2,1), expansions=(1,6,6,6,6,6,6), bn_args={'momentum':0.01}):
#         super().__init__()
#         self.grid_size = np.prod(strides)*2
#         self.conv_init = nn.Conv2d(in_channels, out_channels[0], 3, stride=2, padding=1)
#         self.bn_init = nn.BatchNorm2d(out_channels[0], **bn_args)
#         self.activation_init = nn.ReLU6()
#         for i in range(len(expansions)):
#             self.add_module(f'block_{i+1}',Block(out_channels[i], out_channels[i+1], repeats[i], strides[i], expansions[i], bn_args))
#         # Parameter initialization
#         init.kaiming_uniform_(self.conv_init.weight, mode='fan_in', nonlinearity='relu')
#         init.zeros_(self.conv_init.bias)
#         init.ones_(self.bn_init.weight)
#         init.zeros_(self.bn_init.bias)
#
#     def forward(self,x):
#         for m in self._modules.values():
#             x = m(x)
#         return x

class MobileNetV2(nn.Module):

    def __init__(self, in_channels, features_num = [-1], out_channels=(32,16,24,32,64,96,160,320),
        repeats=(1,2,3,4,3,3,1), strides=(1,2,2,2,1,2,1), expansions=(1,6,6,6,6,6,6), bn_args={'momentum':0.01}):

        super().__init__()
        self.grid_size = np.prod(strides)*2
        self.features_num = features_num
        self.conv_init = nn.Conv2d(in_channels, out_channels[0], 3, stride=2, padding=1)
        self.bn_init = nn.BatchNorm2d(out_channels[0], **bn_args)
        self.activation_init = nn.ReLU6()
        self.blocks = nn.ModuleList()
        for i in range(len(expansions)):
            self.blocks.append(Block(out_channels[i], out_channels[i+1], repeats[i], strides[i], expansions[i], bn_args))
        # Parameter initialization
        init.kaiming_uniform_(self.conv_init.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv_init.bias)
        init.ones_(self.bn_init.weight)
        init.zeros_(self.bn_init.bias)

    def forward(self,x):
        feature_maps = []
        x = self.activation_init(self.bn_init(self.conv_init(x)))
        feature_maps.append(x)
        for block in self.blocks:
            x = block(x)
            if block.stride > 1:
                feature_maps.append(x)
            else:
                feature_maps[-1] = x
        return feature_maps[-self.features_num:][-1::-1]

def convert_parameters(parameters):
    new_params = OrderedDict()
    def repl_func(match):
        block_num = match.groups()[1]
        return f'blocks.{int(block_num)-1}'
    for pname, p in parameters.items():
        new_params[re.sub("(block_)(\d)", repl_func, pname)] = p
    return new_params
