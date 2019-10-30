import torch
import torch.nn as nn
import torch.nn.init as init
from .yolo_heads import YoloHeadSplitResBtlneck

class Upscale(nn.Module):

    def __init__(self, in_features, upscale_factor = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_features, in_features*upscale_factor*upscale_factor,
            kernel_size=3, padding=1, groups=in_features)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        # Initialize parameters
        init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.conv.bias)

    def forward(self,x):
        return self.shuffle(self.conv(x))

class RetinaHead(nn.Module):

    def __init__(self, in_features, **kwargs):

        super().__init__()
        self.class_and_box_subnet = YoloHeadSplitResBtlneck(in_features[0], **kwargs)
        self.upscale = nn.ModuleList([Upscale(in_features[0]) for _ in range(1, len(in_features))])
        self.lateral = nn.ModuleList([nn.Conv2d(in_features[i], in_features[0], kernel_size=1)
            for i in range(1, len(in_features))])

    def forward(self, x):
        feature_maps = []
        feature_maps.append(x[0])
        for xi, up, lateral in zip(x[1:], self.upscale, self.lateral):
            fmap = up(feature_maps[-1])+lateral(xi)
            feature_maps.append(fmap)
        return [self.class_and_box_subnet(fmap) for fmap in feature_maps]
