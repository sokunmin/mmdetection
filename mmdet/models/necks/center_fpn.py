import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init, build_upsample_layer, normal_init, kaiming_init
from mmcv.runner import auto_fp16
from mmcv.ops import ModulatedDeformConv2dPack, CARAFEPack

from ..builder import NECKS


class Upsample2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample_cfg=dict(
                     type='deconv',
                     kernel_size=4,
                     stride=2,
                     padding=1,
                     output_padding=0,
                     bias=False
                 ),
                 with_last_norm=False,
                 with_last_relu=False):
        super(Upsample2d, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        assert self.upsample_cfg.type in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        self.dcn = ModulatedDeformConv2dPack(in_channels, out_channels, 3, stride=1,
                                             padding=1, dilation=1, deformable_groups=1)
        self.dcn_bn = nn.BatchNorm2d(out_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_cfg.type == 'deconv':
            upsample_cfg_.update(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self.upsample_cfg.kernel_size,
                stride=self.upsample_cfg.stride,
                padding=self.upsample_cfg.padding,
                output_padding=self.upsample_cfg.output_padding,
                bias=self.upsample_cfg.bias)
        elif self.upsample_cfg.type == 'pixel_shuffle':
            upsample_cfg_.update(
                in_channels=out_channels,
                out_channels=out_channels,
                scale_factor=2,
                upsample_kernel=self.upsample_kernel)
        elif self.upsample_cfg.type == 'carafe':
            upsample_cfg_.update(channels=out_channels, scale_factor=2)
        else:
            # suppress warnings
            align_corners = (None if self.upsample_cfg.type == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=2,
                mode=self.upsample_cfg.type,
                align_corners=align_corners)
        self.upsample = build_upsample_layer(upsample_cfg_)
        if with_last_norm:
            self.upsample_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_last_norm = with_last_norm
        self.with_last_relu = with_last_relu
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        if self.with_last_norm:
            x = self.upsample_bn(x)
        if self.with_last_relu:
            x = self.relu(x)
        return x


@NECKS.register_module()
class CenterFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 level_index=-1,
                 reverse_levels=False,
                 upsample_cfg=None,
                 with_last_norm=False,
                 with_last_relu=False,
                 shortcut_convs=None):
        super(CenterFPN, self).__init__()
        assert isinstance(in_channels, (tuple, list))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level_index = level_index
        self.reverse_levels = reverse_levels
        self.with_upsamples = upsample_cfg is not None
        self.upsample_cfg = upsample_cfg.copy() if self.with_upsamples else None
        self.with_last_norm = with_last_norm
        self.with_last_relu = with_last_relu
        self.shortcut_convs = shortcut_convs
        self.with_shortcuts = shortcut_convs is not None and len(shortcut_convs) > 0
        self.fp16_enabled = False
        self._init_layers()

    def build_upsamples(self, cfg):
        upsamples = nn.ModuleList()
        num_layers = len(self.in_channels) - 1
        for i in range(num_layers):
            upsamples.append(Upsample2d(
                in_channels=self.in_channels[i],
                out_channels=self.in_channels[i+1] if i < num_layers else self.out_channels,
                upsample_cfg=cfg,
                with_last_norm=self.with_last_norm,
                with_last_relu=self.with_last_relu))
        return upsamples

    def build_shortcuts(self):
        shortcuts = nn.ModuleList()
        num_layers = len(self.in_channels) - 1
        for i in range(len(self.in_channels) - 1):
            if self.with_shortcuts:
                assert num_layers == len(self.shortcut_convs)
                layers = []
                num_convs = self.shortcut_convs[i]
                for j in range(num_convs):
                    layers.append(nn.Conv2d(self.in_channels[i+1],
                                            self.in_channels[i+1] if i < num_layers else self.out_channels,
                                            3, 1, 1))
                    if j < num_convs - 1 or self.with_last_relu:
                        layers.append(nn.ReLU(inplace=True))
                shortcuts.append(nn.Sequential(*layers))
            else:
                shortcuts.append(nn.Sequential())
        return shortcuts

    def _init_layers(self):
        if self.with_upsamples:
            self.upsamples = self.build_upsamples(self.upsample_cfg)
            self.shortcuts = self.build_shortcuts()

    def init_weights(self):
        if self.with_shortcuts:
            for _, m in self.shortcuts.named_modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def forward(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        if self.reverse_levels:
            feats = feats[::-1]  # > reverse order: [512, 256, 128, 64]
        x = feats[self.level_index]
        if self.with_upsamples:
            assert len(self.upsamples) == len(self.shortcuts)
            for i, upsample, shortcut in \
                    zip(range(len(self.upsamples)), self.upsamples, self.shortcuts):
                x = upsample(x)
                if self.with_shortcuts:
                    residual = shortcut(feats[i+1])
                    x = x + residual
        return x