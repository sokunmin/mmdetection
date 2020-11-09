_base_ = './ttfnet_d53_aug_10x.py'

# model settings
model = dict(
    type='TTFNet',
    pretrained=None,
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_eval=False),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(128, 256, 512, 1024),
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