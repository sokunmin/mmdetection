_base_ = './ttfnet_dla34_2x.py'

# model settings
model = dict(
    type='TTFNet',
    pretrained='./pretrain/dla34-ba72cf86.pth',
    backbone=dict(
        type='DLASeg',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        down_ratio=4,
        last_level=5),
    neck=dict(
        type='CenterFPN',
        in_channels=(64, 64, 64),
        out_channels=64
    ),
    bbox_head=dict(
        type='TTFHead',
        num_classes=1,
        in_channels=64,
        stacked_convs=(2, 1),
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

classes = ('person', )
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))