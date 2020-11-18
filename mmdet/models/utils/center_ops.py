import torch


def topk_channel(scores, topk):
    batch, cls, height, width = scores.size()
    # `scores`: (B, #cls, H, W) -> (B, #cls, HxW) -> (B, #cls, K)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), topk)

    topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
    topk_ys = torch.floor_divide(topk_inds, width).float()  # (B, #cls, topk)
    topk_xs = (topk_inds % width).float()  # (B, #cls, topk)

    return topk_scores, topk_inds, topk_ys, topk_xs


def topk(scores, topk):
    batch, cls, height, width = scores.size()
    topk_scores, topk_inds, topk_ys, topk_xs = topk_channel(scores, topk)

    # both are (batch, topk). select topk from (B, #cls x topk)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
    topk_clses = torch.floor_divide(topk_ind, topk)
    # `topk_inds`: (B, #cls, topk) -> (B, #cls x topk, 1) ∩ (B, topk, 1) -> (B, topk)
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, topk)
    topk_ys = gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, topk)
    topk_xs = gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, topk)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# def gather_feat(feat, ind, mask=None):
#     dim = feat.size(2)
#     ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # (B, K) -> (B, K, 1) -> (B, K, dim)
#     feat = feat.gather(1, ind)  # (B, HxW, dim) ∩ (B, K, dim)
#     if mask is not None:
#         mask = mask.unsqueeze(2).expand_as(feat)
#         feat = feat[mask]
#         feat = feat.view(-1, dim)
#     return feat

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)  # (B, K) -> (B, K, #cls)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, HxW, C)
    feat = gather_feature(feat, ind)  # > (B, K, C)
    return feat
