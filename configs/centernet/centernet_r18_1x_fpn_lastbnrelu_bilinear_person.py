_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/coco_detection.py'
]

# model settings
model = dict(
    type='CenterNet',
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
    neck=dict(
        type='CenterFPN',
        in_channels=(512, 256, 128, 64),
        out_channels=64,
        level_index=0,
        reverse_levels=True,
        with_last_norm=True,
        with_last_relu=True,
        upsample_cfg=dict(type='bilinear')),
    bbox_head=dict(
        type='CenterHead',
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        num_feat_levels=1,
        corner_emb_channels=0,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.1)))
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    min_overlap=0.7,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    kp_score_thr=0.1,
    max_per_img=100)
# dataset settings, SEE: Normalize RGB https://aishack.in/tutorials/normalized-rgb/
img_norm_cfg = dict(
    # NOTE: add `norm_rgb=True` if eval offical pretrained weights
    # mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=False, norm_rgb=True)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False, norm_rgb=True)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomLighting', scale=0.1),
    dict(type='RandomCenterCropPad',
         crop_size=(512, 512),
         ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
         test_mode=False,
         test_pad_mode=None,
         **img_norm_cfg),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg')),
        ])
]
classes = ('person', )
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(classes=classes, pipeline=train_pipeline),
    val=dict(classes=classes, pipeline=test_pipeline),
    test=dict(classes=classes, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0004,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 5,
    step=[90, 120])
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 140
cudnn_benchmark = True
find_unused_parameters = True