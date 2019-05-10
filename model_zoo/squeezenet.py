import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(type(self), self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000, remove_top=False):
        super(type(self), self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        # Final convolution is initialized differently form the rest
        if not remove_top:
            self.num_classes = num_classes
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            init.normal_(final_conv.weight, mean=0.0, std=0.01)
            if final_conv.bias is not None:
                init.constant_(final_conv.bias, 0)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AvgPool2d(13, stride=1)
            )
        else:
            self.forward = self._forward_notop

    def _forward_notop(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        return x

def squeezenet1_0(pretrained=False, remove_top=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        remove_top (bool): If True, removes the final calssifier from the model
    """
    model = SqueezeNet(version=1.0, remove_top=remove_top, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['squeezenet1_0'])
        if remove_top or model.num_classes!=1000:
            state_dict = OrderedDict(filter(lambda x: 'classifier' not in x[0],
                                     state_dict.items()))
        model.load_state_dict(state_dict, strict=False)
    return model


def squeezenet1_1(pretrained=False, remove_top=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        remove_top (bool): If True, removes the final calssifier from the model
    """
    model = SqueezeNet(version=1.1, remove_top=remove_top, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['squeezenet1_1'])
        if remove_top or model.num_classes!=1000:
            state_dict = OrderedDict(filter(lambda x: 'classifier' not in x[0],
                                     state_dict.items()))
        model.load_state_dict(state_dict, strict=False)
    return model
