_base_ = './ttfnet_dla34_2x.py'
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[90, 110])
checkpoint_config = dict(interval=10)
total_epochs = 120
