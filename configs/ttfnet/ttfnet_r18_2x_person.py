_base_ = './ttfnet_r18_2x.py'

# model settings
model = dict(
    type='TTFNet',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        in_channels=64,
        feat_channels=(128, 64),
        stacked_convs=(2, 1),
        num_classes=1,
        wh_offset_base=16,
        area_cfg=dict(
            type='log',
            agnostic=True,
            gaussian=True,
            alpha=0.54,
            beta=0.54
        ),
        upsample_cfg=dict(type='bilinear',
                          in_channels=(512, 256, 128, 64),
                          with_last_relu=False),
        shortcut_cfg=dict(in_channels=(256, 128, 64),
                          out_channels=(256, 128, 64),
                          kernel_size=3,
                          padding=1,
                          levels=(1, 2, 3)),
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0)))
classes = ('person', )
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))