import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict, deque
import numpy as np

""" Densely connected convolutional model mostly similar to the paper
"Densely Connected Convolutional Networks" by Huang et al. We use
depthwise separable convolution module similar to MobileNetV2 Bottleneck
to create new feature maps
"""

class NewFeatures(nn.Module):
    def __init__(self, in_channels, growth_rate = 32, expansion = 4, bn_args={'momentum':0.01}):
        super().__init__()
        channels = expansion*growth_rate
        self.conv_expand = nn.Conv2d(in_channels, channels, 1)
        self.bn_expand = nn.BatchNorm2d(channels, **bn_args)
        self.activation_expand = nn.ReLU6()
        self.conv_dwise = nn.Conv2d(channels, channels, 3, padding = 1, groups = channels)
        self.bn_dwise = nn.BatchNorm2d(channels, **bn_args)
        self.activation_dwise = nn.ReLU6()
        self.conv_shrink = nn.Conv2d(channels, growth_rate, 1)
        self.bn_shrink = nn.BatchNorm2d(growth_rate, **bn_args)
        self.activation_shrink = nn.ReLU6()
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
        x = self.activation_shrink(self.bn_shrink(self.conv_shrink(x)))
        return x

class DenseLayer(NewFeatures):
    def __init__(self, in_channels, growth_rate = 32, expansion = 4, bn_args={'momentum':0.01}):
        super().__init__(in_channels, growth_rate = growth_rate, expansion = expansion, bn_args = bn_args)

    def forward(self, x):
        return torch.cat([x, super().forward(x)], dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression = 2, bn_args={'momentum':0.01}):
        super().__init__()
        out_channels = in_channels // compression
        self.conv  = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels, **bn_args)
        self.activation = nn.ReLU6()
        self.pool = nn.AvgPool2d(2)
        # Parameter initialization
        init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv.bias)
        init.ones_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def forward(self, x):
        return self.pool(self.activation(self.bn(self.conv(x))))

class DenseBlock(nn.Module):
    def __init__(self, in_channels, repeats, growth_rate = 32, expansion = 4, bn_args={'momentum':0.01}):
        super().__init__()
        for i in range(repeats):
            self.add_module(f'dense_layer_{i}', DenseLayer(in_channels + i*growth_rate,
                 growth_rate = growth_rate, expansion=expansion, bn_args = bn_args))

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, in_channels, repeats, features_num = 1, growth_rate = 32, expansion = 4, compression = 2, bn_args={'momentum':0.01}):
        super().__init__()
        out_channels = 2*growth_rate
        self.features_num = features_num
        self.grid_size = 4 * 2**(len(repeats)-1)
        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 3)
        self.bn_init = nn.BatchNorm2d(out_channels, **bn_args)
        self.activation_init = nn.ReLU6()
        self.pooling_init = nn.MaxPool2d(3, stride=2, padding = 1)
        for i in range(len(repeats)-1):
            self.add_module(f'dense_block_{i}', DenseBlock(out_channels, repeats = repeats[i],
                 growth_rate = growth_rate, expansion=expansion, bn_args = bn_args))
            out_channels += growth_rate*repeats[i]
            self.add_module(f'transition_layer_{i}', TransitionLayer(out_channels, compression = compression, bn_args=bn_args))
            out_channels = out_channels // 2
        i = len(repeats)-1
        self.add_module(f'dense_block_{i}', DenseBlock(out_channels, repeats = repeats[i],
             growth_rate = growth_rate, expansion=expansion, bn_args = bn_args))
        # Parameter initialization
        init.kaiming_uniform_(self.conv_init.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv_init.bias)
        init.ones_(self.bn_init.weight)
        init.zeros_(self.bn_init.bias)

    def forward(self, x):
        feature_maps = deque(maxlen=self.features_num)
        for m in self._modules.values():
            x = m(x)
            if isinstance(m, DenseBlock):
                feature_maps.append(x)
        if self.features_num > 1:
            return list(reversed(feature_maps))
        else:
            return feature_maps.pop()

class DenseNet121(DenseNet):
    def __init__(self, in_channels, features_num=1, growth_rate = 32, expansion = 4, compression = 2, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [6,12,24,16], features_num, growth_rate, expansion, compression, bn_args)

class DenseNet169(DenseNet):
    def __init__(self, in_channels, features_num=1, growth_rate = 32, expansion = 4, compression = 2, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [6,12,32,32], features_num, growth_rate, expansion, compression, bn_args)

class DenseNet201(DenseNet):
    def __init__(self, in_channels, features_num=1, growth_rate = 32, expansion = 4, compression = 2, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [6,12,48,32], features_num ,growth_rate, expansion, compression, bn_args)

class DenseNet264(DenseNet):
    def __init__(self, in_channels, features_num=1, growth_rate = 32, expansion = 4, compression = 2, bn_args={'momentum':0.01}):
        super().__init__(in_channels, [6,12,64,48], features_num, growth_rate, expansion, compression, bn_args)
