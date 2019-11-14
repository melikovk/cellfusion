from .mobilenet_v2 import MobileNetV2
from .mobilenet_v2gn import MobileNetV2gn
from .resnet_v2 import ResNet50V2
from .yolo_heads import YoloHead, YoloHeadSplit, YoloHeadSplitResBtlneck
from .retina_head import RetinaHead
from .vision_models import ObjectDetectionModel
from image.datasets.utils import get_cell_anchors
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet264
import numpy as np

ANCHORS = {'anchors0': get_cell_anchors(scales=[1], anchors=[]),
           'anchors1': get_cell_anchors(scales = 2**(np.arange(0,4)/3), anchors = [(.5,.5,1.0),(0.0 ,0.5, 2.), (0.5, 0.0 ,.5)]),
           'anchors2': get_cell_anchors(scales = [1.5], anchors = [(.5,.5,1.0),(0.0 ,0.5, 2.), (0.5, 0.0 ,.5)])}

class MobileNetJointHead(ObjectDetectionModel):
    """ Object detection model with feature MobileNetv2 feature extractor
        And fully convolutional object detection head
    """
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), YoloHead(**head_params, anchors = len(cell_anchors)), cell_anchors)

class MobileNetSplitHead(ObjectDetectionModel):
    """ Object detection model with feature MobileNetv2 feature extractor
        And fully convolutional object detection head with independent subnets for
        box predictions and obejctness predictions
    """
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), YoloHeadSplit(anchors = len(cell_anchors), **head_params), cell_anchors)

class MobileNetSplitResBtlneckHead(ObjectDetectionModel):
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), YoloHeadSplitResBtlneck(anchors = len(cell_anchors), **head_params), cell_anchors)

class MobileNetGNSplitResBtlneckHead(ObjectDetectionModel):
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2gn(**features_params), YoloHeadSplitResBtlneck(anchors = len(cell_anchors), **head_params), cell_anchors)

class ResNet50V2SplitResBtlneckHead(ObjectDetectionModel):
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(ResNet50V2(**features_params), YoloHeadSplitResBtlneck(anchors = len(cell_anchors), **head_params), cell_anchors)

class DenseNetSplitHead(ObjectDetectionModel):
    """ Object detection model with DenseNet feature extractor
        and fully convolutional object detection head with independent subnets for
        box predictions and obejctness predictions
    """
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        if features_params['name'] == 'densenet121':
            features_model = DenseNet121
            head_params['in_features'] = 1024
        elif features_params['name'] == 'densenet169':
            features_model = DenseNet169
            head_params['in_features'] = 1664
        elif features_params['name'] == 'densenet201':
            features_model = DenseNet201
            head_params['in_features'] = 1920
        elif features_params['name'] == 'densenet264':
            features_model = DenseNet264
            head_params['in_features'] = 2688
        else:
            raise ValueError('Incorrect value for densenet parameter')
        super().__init__(features_model(features_params['in_channels']), YoloHeadSplit(anchors = len(cell_anchors), **head_params), cell_anchors)

class MobileNetRetinaHead(ObjectDetectionModel):
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), RetinaHead(anchors = len(cell_anchors), **head_params), cell_anchors)
