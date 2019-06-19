from . import mobilenet_v2, cnn_heads, vision_models

_features_params = {'in_channels': 1,
                   'out_channels': (32,16,24,32,64,96,160,320),
                   'repeats': (1,2,3,4,3,3,1),
                   'strides': (1,2,2,2,1,2,1),
                   'expansions': (1,6,6,6,6,6,6),
                   'bn_momentum': 0.01}
_head_params = {'in_features': 320,
               'activation': 'relu',
               'hidden_features': [1024, 1024],
               'hidden_kernel': [3, 3],
               'bn_args': {'momentum':0.01},
               'act_args': {},
               'probability': False,
               'coordinate_transform': 'hardtanh',
               'eps': 1e-5}

_split_head_params = {'in_features': 320,
               'activation': 'relu',
               'hidden_features': [256]*4,
               'hidden_kernel': [3]*4,
               'bn_args': {'momentum':0.01},
               'act_args': {},
               'probability': False,
               'coordinate_transform': 'hardtanh',
               'eps': 1e-5}

def mobilenet_v2_1ch_object_detect_base():
    """ Creates an object detection model for 1 chanel images with feature extractor
        that is similar to 1x MobileNet_v2 in original paper.
        Uses simple 2 layer fully convolutional object detection head
    """
    features_params = _features_params.copy()
    head_params = _head_params.copy()
    model = vision_models.CNNModel(mobilenet_v2.MobileNetV2(**features_params), cnn_heads.ObjectDetectionHead(**head_params))
    return model

def mobilenet_v2_1ch_object_detect_split_base():
    """ Creates an object detection model for 1 chanel images with feature extractor
        that is similar to 1x MobileNet_v2 in original paper.
        Uses simple 2 layer fully convolutional object detection head
    """
    features_params = _features_params.copy()
    head_params = _split_head_params.copy()
    model = vision_models.CNNModel(mobilenet_v2.MobileNetV2(**features_params), cnn_heads.ObjectDetectionHeadSplit(**head_params))
    return model

def mobilenet_v2_1ch_object_detect_full():
    """ Creates an object detection model for 1 chanel images with feature extractor
        that has 1.5 times more features in each bottleneck layer relative to
        1x MobileNet_v2 in original paper.
        Uses simple 2 layer fully convolutional object detection head
    """
    features_params = _features_params.copy()
    head_params = _head_params.copy()
    features_params['out_channels'] = (48,24,36,48,96,144,240,480)
    head_params['in_features'] = 480
    model = vision_models.CNNModel(mobilenet_v2.MobileNetV2(**features_params), cnn_heads.ObjectDetectionHead(**head_params))
    return model

def mobilenet_v2_1ch_object_detect_shrink():
    """ Creates an object detection model for 1 chanel images with feature extractor
        that has expansion factor set to 3 vs 6 in 1x MobileNet_v2 from original paper.
        Uses simple 2 layer fully convolutional object detection head
    """
    features_params = _features_params.copy()
    head_params = _head_params.copy()
    features_params['expansions'] = (1,3,3,3,3,3,3)
    model = vision_models.CNNModel(mobilenet_v2.MobileNetV2(**features_params), cnn_heads.ObjectDetectionHead(**head_params))
    return model

def mobilenet_v2_1ch_object_detect_deep():
    """ Creates an object detection model for 1 chanel images with feature extractor
        that has 2 times more repeats in each block (i.e. ~ 2x deeper)
        than 1x MobileNet_v2 from original paper.
        Uses simple 2 layer fully convolutional object detection head
    """
    features_params = _features_params.copy()
    head_params = _head_params.copy()
    features_params['repeats'] = (2,4,6,8,6,6,2)
    model = vision_models.CNNModel(mobilenet_v2.MobileNetV2(**features_params), cnn_heads.ObjectDetectionHead(**head_params))
    return model
