import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss, weight_reduce_loss


@weighted_loss
def smooth_loss(pred, target):
    assert len(pred) == len(target)
    pred1, pred2 = pred
    target1, target2 = target
    losses = []
    for p1, p2, t1, t2 in zip(pred1, pred2, target1, target2):  # > #imgs
        num_objs = len(t1)
        loss = smooth_loss_single(p1, t1, p2, t2,
                                  avg_factor=num_objs)
        losses.append(loss)
    losses = torch.stack(losses)
    return losses


def smooth_loss_single(pred1,
                       target1,
                       pred2,
                       target2,
                       weight=None,
                       reduction='mean',
                       avg_factor=None):
    assert len(pred1) == len(target1) == len(pred2) == len(target2)

    loss = pred1[0].new_tensor(0.0)
    for p1, t1, p2, t2 in zip(pred1, target1, pred2, target2):  # > #objs
        if len(p1) > 0:
            loss += torch.sum((p1 - t1) ** 2) + torch.sum((p2 - t2) ** 2)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class SmoothLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(SmoothLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * smooth_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss