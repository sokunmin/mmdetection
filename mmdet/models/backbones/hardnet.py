import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES


class HarDBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 growth_rate,
                 growth_mult,
                 num_layers,
                 keep_base=False,
                 norm_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.growth_mult = growth_mult
        self.n_layers = num_layers
        self.keep_base = keep_base
        self.links = []
        self.out_channels = 0

        layers_ = []
        for i in range(num_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, growth_mult)
            self.links.append(link)
            layers_.append(ConvModule(inch, outch, 3, 1, 1, norm_cfg=norm_cfg))

            if (i % 2 == 0) or (i == num_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def get_link(self, layer, base_channel, growth_rate, growth_mult):
        if layer == 0:
            return base_channel, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= growth_mult
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            out_channel = self.get_link(i, base_channel, growth_rate, growth_mult)[0]
            in_channels += out_channel
        return out_channels, in_channels, link

    def get_out_channel(self):
        return self.out_channels

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keep_base) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


@BACKBONES.register_module
class HarDNet(nn.Module):
    def __init__(self,
                 stem_channels=(48, 96),
                 stage_layers=(8, 16, 16, 16, 16),
                 stage_channels=(192, 256, 320, 480, 720),
                 growth_rate=(24, 24, 28, 36, 48),
                 growth_mult=1.7,
                 down_ratio=4,
                 skip_nodes=(1, 3, 8, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HarDNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio)) - 1
        self.skip_nodes = skip_nodes
        self.base_layers = nn.ModuleList([])
        self.base_layers.append(
            ConvModule(3, stem_channels[0], 3, 2, 1, norm_cfg=norm_cfg))
        self.base_layers.append(
            ConvModule(stem_channels[0], stem_channels[1], 3, 1, 1, norm_cfg=norm_cfg))
        self.base_layers.append(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        # Build all HarDNet blocks
        num_stages = len(stage_channels)
        in_channel = stem_channels[1]
        for i in range(num_stages):
            block = HarDBlock(in_channel, growth_rate[i], growth_mult, stage_layers[i],
                              norm_cfg=norm_cfg)
            in_channel = block.get_out_channel()
            self.base_layers.append(block)
            if i != num_stages - 1:
                self.base_layers.append(ConvModule(in_channel, stage_channels[i], 1, 1, 0,
                                                   norm_cfg=norm_cfg))
            in_channel = stage_channels[i]
            if i == 0:
                self.base_layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))
            elif i != num_stages - 1 and i != 1 and i != 3:
                self.base_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self.base_layers, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        xs = []

        for i in range(len(self.base_layers)):
            x = self.base_layers[i](x)
            if i in self.skip_nodes:
                xs.append(x)

        return x, xs
