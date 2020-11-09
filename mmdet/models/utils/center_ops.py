import torch


def topk_channel(scores, topk):
    batch, cls, height, width = scores.size()
    # `scores`: (B, #cls, H, W) -> (B, #cls, HxW) -> (B, #cls, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), topk)

    topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
    topk_ys = torch.true_divide(topk_inds, width).int().float()  # (B, #cls, topk)
    topk_xs = (topk_inds % width).int().float()  # (B, #cls, topk)

    return topk_scores, topk_inds, topk_ys, topk_xs


def topk(scores, topk):
    batch, cls, height, width = scores.size()
    # both are (batch, #cls, topk) -> (B, #cls, HxW) -> (B, #cls, topk)
    # topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), topk)
    # topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
    # topk_ys = torch.true_divide(topk_inds, width).int().float()  # (B, #cls, topk)
    # topk_xs = (topk_inds % width).int().float()  # (B, #cls, topk)

    topk_scores, topk_inds, topk_ys, topk_xs = topk_channel(scores, topk)

    # both are (batch, topk). select topk from (B, #cls x topk)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
    topk_clses = torch.true_divide(topk_ind, topk).int()
    topk_ind = topk_ind.unsqueeze(2)  # > (B, topk) -> (B, topk, 1)
    # `topk_inds`: (B, #cls, topk) -> (B, #cls x topk, 1) ∩ (B, topk, 1) -> (B, topk)
    topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
    topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # (B, K) -> (B, K, 1) -> (B, K, dim)
    feat = feat.gather(1, ind)  # (B, HxW, dim) ∩ (B, K, dim)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, HxW, C)
    feat = gather_feat(feat, ind)  # > (B, K, C)
    return feat
