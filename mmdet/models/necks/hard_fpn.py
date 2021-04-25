import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init

from ..backbones.hardnet import HarDBlock
from ..builder import NECKS


class TransitionUp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
            x,
            size=(skip.size(2), skip.size(3)),
            mode="bilinear",
            align_corners=True)
        if concat:
            out = torch.cat([out, skip], 1)
        return out


@NECKS.register_module()
class HarDFPN(nn.Module):
    def __init__(self,
                 in_channels=(784, 768),
                 feat_channels=(256, 80),
                 out_channels=(256, 80),
                 stage_layers=(8, 8, 4),
                 stage_channels=(224, 160, 96),
                 growth_rate=(64, 48, 28),
                 growth_mult=1.7,
                 skip_channels=(96, 214, 458, 784),
                 skip_level=3,
                 sc=(32, 32, 0),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HarDFPN, self).__init__()
        assert isinstance(in_channels, (tuple, list))
        self.sc = np.array(sc)
        self.stage_layers = stage_layers
        self.stage_channels = stage_channels
        self.growth_rate = growth_rate
        self.growth_mult = growth_mult
        self.skip_channels = skip_channels
        self.skip_level = skip_level
        self._init_layers(in_channels, feat_channels, out_channels, norm_cfg=norm_cfg)

    def _init_layers(self, in_channels, feat_channels, out_channels, norm_cfg):
        self.last_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.last_proj = ConvModule(
            in_channels[0], feat_channels[0], 1, 1, 0, norm_cfg=norm_cfg)
        self.last_block = HarDBlock(
            in_channels[1], feat_channels[1], self.growth_mult, 8, norm_cfg=norm_cfg)

        self.trans_up_blocks = nn.ModuleList([])
        self.dense_up_blocks = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])
        self.avg9x9 = nn.AvgPool2d(kernel_size=(9, 9), stride=1, padding=(4, 4))
        prev_channel = self.last_block.get_out_channel()

        last_stage_channel = np.array(self.stage_channels) + self.sc
        for i in range(3):
            skip_channel = self.skip_channels[3 - i]
            self.trans_up_blocks.append(TransitionUp())
            if i < self.skip_level:
                cur_channel = prev_channel + skip_channel
            else:
                cur_channel = prev_channel
            self.conv1x1_up.append(
                ConvModule(cur_channel, last_stage_channel[i], 1, 1, 0, norm_cfg=norm_cfg))
            cur_channel = last_stage_channel[i]
            cur_channel -= self.sc[i]
            cur_channel *= 3

            block = HarDBlock(cur_channel, self.growth_rate[i], self.growth_mult, self.stage_layers[i],
                              norm_cfg=norm_cfg)

            self.dense_up_blocks.append(block)
            prev_channel = block.get_out_channel()

        prev_channel += self.sc.sum()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, feats):
        x, xs = feats
        x_sc = []
        x = self.last_proj(x)
        x = self.last_pool(x)
        x2 = self.avg9x9(x)
        x3 = x / (x.sum((2, 3), keepdim=True) + 0.1)
        x = torch.cat([x, x2, x3], 1)
        x = self.last_block(x)

        for i in range(3):
            skip_x = xs[3 - i]
            x = self.trans_up_blocks[i](x, skip_x, (i < self.skip_level))
            x = self.conv1x1_up[i](x)
            if self.sc[i] > 0:
                end = x.shape[1]
                x_sc.append(x[:, end - self.sc[i]:, :, :].contiguous())
                x = x[:, :end - self.sc[i], :, :].contiguous()
            x2 = self.avg9x9(x)
            x3 = x / (x.sum((2, 3), keepdim=True) + 0.1)
            x = torch.cat([x, x2, x3], 1)
            x = self.dense_up_blocks[i](x)

        scs = [x]
        for i in range(3):
            if self.sc[i] > 0:
                scs.insert(0, F.interpolate(
                    x_sc[i], size=(x.size(2), x.size(3)),
                    mode="bilinear", align_corners=True))
        x = torch.cat(scs, 1)
        return x