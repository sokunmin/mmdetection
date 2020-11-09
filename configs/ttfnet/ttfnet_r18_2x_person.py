_base_ = './ttfnet_r18_2x.py'

classes = ('person', )
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))