_base_ = './ttfnet_r34_aug_10x.py'

# model settings
model = dict(
    type='TTFNet',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(64, 128, 256, 512),
        head_conv_channel=128,
        wh_conv_channel=64,
        num_hm_convs=2,
        num_wh_convs=2,
        num_classes=80,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        loss_cls=dict(
            type='CenterFocalLoss',
            gamma=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0)))