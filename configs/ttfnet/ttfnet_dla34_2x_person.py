_base_ = './ttfnet_dla34_2x.py'

classes = ('person', )
data = dict(
    samples_per_gpu=24,
    workers_per_gpu=2,
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))