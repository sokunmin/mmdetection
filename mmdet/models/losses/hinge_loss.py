import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# @weighted_loss
def lovasz_hinge_loss_single(pred, target, activation='relu', map2inf=False):
    if len(target) == 0:
        # only void pixels, the gradients should be 0
        return pred.sum() * 0.
    if map2inf:
        pred = torch.log(pred) - torch.log(1 - pred)

    signs = 2. * target.float() - 1.
    errors = (1. - pred * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = target[perm]
    grad = lovasz_grad(gt_sorted)
    if activation == 'elu':
        loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    elif activation == 'relu':
        loss = torch.dot(F.relu(errors_sorted), grad)
    else:
        raise KeyError('No activation named ', activation, ' found.')
    return loss


@weighted_loss
def lovasz_hinge_loss(pred, target, crop_masks, activation='relu', map2inf=False):
    """
    Binary Lovasz hinge loss
      pred: [P] Variable, logits at each prediction (between -\infty and +\infty)
      target: [P] Tensor, binary ground truth labels (0 or 1)
    """
    losses = []
    for m, p, t in zip(crop_masks, pred, target): # > imgs
        num_objs = t.size()[0]
        loss = t.new_tensor(0.0)
        for i in range(num_objs):
            if len(p[i]) > 0:
                loss += lovasz_hinge_loss_single(p[i][m[i]].view(-1),
                                                 t[i][m[i]].view(-1),
                                                 activation=activation,
                                                 map2inf=map2inf)
        if num_objs > 0:
            loss /= num_objs
        losses.append(loss)
    losses = torch.stack(losses)
    return losses


@LOSSES.register_module
class LovaszHingeLoss(nn.Module):
    """
    https://github.com/bermanmaxim/LovaszSoftmax
    """

    def __init__(self, activation='relu', map2inf=False, reduction='mean', loss_weight=1.0):
        super(LovaszHingeLoss, self).__init__()
        self.activation = activation
        self.map2inf = map2inf
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                crop_masks,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = self.loss_weight * lovasz_hinge_loss(
            pred,
            target,
            weight,
            crop_masks=crop_masks,
            activation=self.activation,
            map2inf=self.map2inf,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss