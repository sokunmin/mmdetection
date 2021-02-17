_base_ = './ttfnet_r34_10x_aug.py'

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
    neck=dict(
        type='CenterFPN',
        in_channels=(512, 256, 128, 64),
        out_channels=64,
        level_index=0,
        reverse_levels=True,
        with_last_norm=False,
        with_last_relu=False,
        upsample_cfg=dict(type='bilinear'),
        shortcut_convs=(1, 2, 3)),
    bbox_head=dict(
        type='TTFHead',
        num_classes=80,
        in_channels=64,
        feat_channels=(128, 64),
        stacked_convs=(2, 2),
        offset_base=16,
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