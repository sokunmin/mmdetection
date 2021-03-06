_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/coco_detection.py'
]

dataset_type = 'CocoPersonDataset'
data_root = 'data/coco/'

# model settings
model = dict(
    type='CenterNet',
    pretrained='./weights/dla/imagenet-dla34.pth',
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
        type='CenterHead',
        num_classes=1,
        in_channels=64,
        feat_channels=256,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.1)),
    keypoint_head=dict(
        type='CenterPoseHead',
        num_classes=17,
        in_channels=64,
        feat_channels=256,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0),
        loss_joint=dict(type='L1Loss', loss_weight=1.0)))
cudnn_benchmark = True
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
    mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=False)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=False, norm_rgb=True)
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False, norm_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         with_keypoint=True,
         with_mask_ignore=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomLighting', scale=0.1),
    dict(type='AffineTransform',
         img_scale=(512, 512),
         ratio_mode='range',
         flip_ratio=0.5,
         flip_direction='horizontal',
         shift_pixels=50,
         scale_ratio=0.8,
         scale_range=(0.7, 1.3),
         min_occupation_ratio=0.6,
         rotate_angle=1,
         img_pad_value=(124, 127, 127),
         mask_pad_value=dict(mask=(0,), mask_ignore=(0, 255)),
         with_mask2bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels',
               'gt_masks', 'gt_masks_ignore', 'gt_keypoints'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize'),
            # dict(type='RandomFlip'),
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
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/person_keypoints_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0004,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[270, 300])
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['bbox', 'keypoints'], multitask=True)
# runtime settings
total_epochs = 320