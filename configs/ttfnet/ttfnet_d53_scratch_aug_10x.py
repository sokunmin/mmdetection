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
        in_channels=128,
        feat_channels=(128, 64),
        stacked_convs=(2, 2),
        num_classes=80,
        wh_offset_base=16,
        area_cfg=dict(
            type='log',
            agnostic=True,
            gaussian=True,
            alpha=0.54,
            beta=0.54
        ),
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0)))