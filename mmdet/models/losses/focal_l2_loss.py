import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def focal_l2_loss(pred, target, gamma=2.0, alpha=0.1, beta=0.02, target_thr=0.01):
    assert gamma >= 0 and alpha >= 0 and beta >= 0
    assert pred.size()[1:] == target.size()[1:] and target.numel() > 0
    pt = torch.where(target >= target_thr, pred - alpha, 1 - pred - beta)
    scale_factor = torch.abs((1 - pt) ** gamma)
    loss = (pred - target) ** 2 * scale_factor
    return loss   # (#stack, B, C, H, W)


@LOSSES.register_module
class FocalL2Loss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.1,
                 beta=0.02,
                 target_thr=0.01,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(FocalL2Loss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.target_thr = target_thr
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)  # default: 'mean'
        loss = self.loss_weight * focal_l2_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            beta=self.beta,
            target_thr=self.target_thr,
            reduction=reduction,
            avg_factor=avg_factor)  # `avg_factor` is assigned in `head.loss()`
        return loss
