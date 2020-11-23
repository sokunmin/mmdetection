import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def ct_focal_loss(pred, target, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        target: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)  # reduce punishment
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss)


@LOSSES.register_module()
class CenterFocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(CenterFocalLoss, self).__init__()
        self.gamma = gamma
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
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * ct_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss