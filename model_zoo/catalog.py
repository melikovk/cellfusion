from .mobilenet_v2 import MobileNetV2
from .cnn_heads import ObjectDetectionHead, ObjectDetectionHeadSplit
from .vision_models import ObjectDetectionModel
from image.datasets.yolo import get_cell_anchors
import numpy as np

ANCHORS = {'anchors0': get_cell_anchors(scales=[1], anchors=[]),
           'anchors1': get_cell_anchors(scales = 2**(np.arange(0,4)/3), anchors = [(.5,.5,1.0),(0.0 ,0.5, 2.), (0.5, 0.0 ,.5)])}

class MobileNetJointHead(ObjectDetectionModel):
    """ Object detection model with feature MobileNetv2 feature extractor
        And fully convolutional object detection head
    """
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_params': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), ObjectDetectionHead(**head_params, anchors = len(cell_anchors)), cell_anchors)

class MobileNetSplitHead(ObjectDetectionModel):
    """ Object detection model with feature MobileNetv2 feature extractor
        And fully convolutional object detection head with independent subnets for
        box predictions and obejctness predictions
    """
    def __init__(self, anchors, features_params, head_params):
        self.config = {'anchors': anchors,
                       'features_param': features_params,
                       'head_params': head_params}
        cell_anchors = ANCHORS[anchors].copy()
        super().__init__(MobileNetV2(**features_params), ObjectDetectionHeadSplit( anchors = len(cell_anchors), **head_params), cell_anchors)

# class MobilenetBase1ch(ObjectDetectionModel):
#     """ Object detection model for 1 chanel images with feature extractor
#         that is similar to 1x MobileNet_v2 in original paper.
#         Uses simple 2 layer fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1}
#         head_params = {'in_features': 320, 'anchors': len(cell_anchors)}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHead(**head_params), cell_anchors)
#
# class MobilenetBase1chSplitHead(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that is similar to 1x MobileNet_v2 in original paper.
#         Uses split fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors)}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHeadSplit(**head_params), cell_anchors)
#
# class MobilenetBase1chSplitHeadSigmoid(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that is similar to 1x MobileNet_v2 in original paper.
#         Uses split fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors), 'coordinate_transform':'sigmoid'}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHeadSplit(**head_params), cell_anchors)
#
# class MobilenetFull1ch(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that has 1.5 times more features in each bottleneck layer relative to
#         1x MobileNet_v2 in original paper.
#         Uses simple 2 layer fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1, 'out_channels': (48,24,36,48,96,144,240,480)}
#         head_params = {'in_features':480, 'anchors': len(cell_anchors)}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHead(**head_params), cell_anchors)
#
# class MobilenetShrink1ch(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that has expansion factor set to 3 vs 6 in 1x MobileNet_v2 from original paper.
#         Uses simple 2 layer fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1, 'expansions': (1,3,3,3,3,3,3)}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors)}
#         supre().__init__(MobileNetV2(**features_params), ObjectDetectionHead(**head_params), cell_anchors)
#
# class MobilenetDeep1ch(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that has 2 times more repeats in each block (i.e. ~ 2x deeper)
#         than 1x MobileNet_v2 from original paper.
#         Uses simple 2 layer fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS1.copy()
#         features_params = {'in_channels': 1, 'repeats': (2,4,6,8,6,6,2)}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors)}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHead(**head_params), cell_anchors)
#
# class MobilenetBase1chSplitHeadSingleAnchor(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that is similar to 1x MobileNet_v2 in original paper.
#         Uses split fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS0.copy()
#         features_params = {'in_channels': 1}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors)}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHeadSplit(**head_params), cell_anchors)
#
# class MobilenetBase1chSplitHeadSingleAnchorSigmoid(ObjectDetectionModel):
#     """ Creates an object detection model for 1 chanel images with feature extractor
#         that is similar to 1x MobileNet_v2 in original paper.
#         Uses split fully convolutional object detection head
#     """
#     def __init__(self):
#         cell_anchors = ANCHORS0.copy()
#         features_params = {'in_channels': 1}
#         head_params = {'in_features':320, 'anchors': len(cell_anchors), 'coordinate_transform': 'sigmoid'}
#         super().__init__(MobileNetV2(**features_params), ObjectDetectionHeadSplit(**head_params), cell_anchors)
