_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/coco_detection.py'
]

dataset_type = 'CocoPersonDataset'
data_root = 'data/coco/'

# model settings
model = dict(
    type='CenterPoseNet',
    pretrained='./pretrain/dla34-ba72cf86.pth',
    backbone=dict(
        type='DLASeg',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        down_ratio=4,
        last_level=5),
    neck=None,
    bbox_head=dict(
        type='CenterPoseHead',
        in_channels=64,
        feat_channels=256,
        use_dla=True,
        down_ratio=4,
        num_classes=1,
        stacked_convs=1,
        ct_head_cfg=dict(
            wh_out_channels=2,
            reg_out_channels=2,
            offset_base=16.,
            area_process='log',
            with_agnostic=True,
            with_gaussian=True
        ),
        kp_head_cfg=dict(
            ct_out_channels=17,
            reg_out_channels=34,
            offset_out_channels=2
        ),
        alpha=0.54,
        loss_cls=dict(
            type='CenterFocalLoss',
            gamma=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0)))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    kp_score_thr=0.1,
    max_per_img=100)
# dataset settings
img_norm_cfg = dict(
    # CenterNet: BGR format + normalize RGB
    mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=False, norm_rgb=True)
    # TTFNet mean/std
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            # dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ('person', )
data = dict(
    samples_per_gpu=24,
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
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0004,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[18, 22])
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 24