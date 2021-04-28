import math
from typing import List

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from timm.models.efficientnet import create_conv2d, EfficientNetBuilder, FeatureInfo, FeatureHooks
from timm.models.efficientnet_builder import \
    decode_arch_def, round_channels, get_act_layer

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


@BACKBONES.register_module()
class EfficientNetV2(nn.Module):
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, depth_multiplier, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck',
                 in_chans=3, stem_size=32, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', fix_stem=False, act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        arch_def = [
            ['er_r2_k3_s1_e1_c24'],
            ['er_r4_k3_s2_e4_c48'],
            ['er_r4_k3_s2_e4_c64'],
            ['ir_r6_k3_s2_e4_c128_se0.25'],
            ['ir_r9_k3_s1_e6_c160_se0.25'],
            ['ir_r15_k3_s2_e6_c272_se0.25'],
        ]
        block_args = decode_arch_def(arch_def, depth_multiplier)
        act_layer = get_act_layer('silu')

        super(EfficientNetV2, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, feature_location=feature_location, verbose=False)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())