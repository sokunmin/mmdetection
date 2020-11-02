# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from mmcv.cnn import ContextBlock, ConvWS2d, GeneralizedAttention, NonLocal2d, conv_ws_2d, build_plugin_layer
from mmcv.ops import (Conv2d, ConvTranspose2d,
                      CornerPool,
                      Linear,
                      MaskedConv2d, MaxPool2d,
                      RoIAlign, RoIPool, SAConv2d,
                      SigmoidFocalLoss, SimpleRoIAlign, batched_nms,
                      deform_conv,
                      get_compiler_version,
                      get_compiling_cuda_version, modulated_deform_conv, nms,
                      nms_match, point_sample, rel_roi_point_to_rel_img_point,
                      roi_align, roi_pool, sigmoid_focal_loss, soft_nms,
                      DeformConv2d, DeformConv2dPack, DeformRoIPool,
                      DeformRoIPoolPack, ModulatedDeformRoIPoolPack,
                      ModulatedDeformConv2d, ModulatedDeformConv2dPack, deform_roi_pool)
from mmdet.ops.nms import simple_nms

# yapf: enable

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'DeformConv2d', 'DeformConv2dPack', 'DeformRoIPool', 'DeformRoIPoolPack',
    'ModulatedDeformRoIPoolPack', 'ModulatedDeformConv2d',
    'ModulatedDeformConv2dPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pool', 'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'MaskedConv2d', 'ContextBlock', 'GeneralizedAttention', 'NonLocal2d',
    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
    'conv_ws_2d', 'build_plugin_layer', 'batched_nms', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d', 'simple_nms'
]

