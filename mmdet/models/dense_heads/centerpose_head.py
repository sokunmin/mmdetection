import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, build_upsample_layer, build_norm_layer
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.ops import ModulatedDeformConv2dPack
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import force_fp32
import numpy as np

from mmdet.core import multi_apply, calc_region, simple_nms
from .base_dense_head import BaseDenseHead
from ..builder import HEADS, build_loss
from ..utils.center_ops import transpose_and_gather_feat, topk, topk_channel
from ...core.bbox.iou_calculators.iou2d_calculator import bbox_areas


@HEADS.register_module
class CenterPoseHead(BaseDenseHead):

    def __init__(self,
                 in_channels=64,
                 feat_channels=256,
                 use_dla=False,
                 down_ratio=4,
                 num_classes=1,
                 stacked_convs=1,
                 loss_cls=dict(
                     type='CenterFocalLoss',
                     gamma=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0)):
        super(CenterPoseHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_dla = use_dla
        self.num_classes = num_classes
        self.num_keypoints = 17
        self.stacked_convs = stacked_convs

        self.limbs = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False
        self.down_ratio = down_ratio

        # > ct heads
        self.ct_hm_head = self.build_head(self.num_classes)
        self.ct_kp_reg_head = self.build_head(34)
        self.bbox_wh_head = self.build_head(2)
        self.bbox_reg_head = self.build_head(2)
        self.kp_hm_head = self.build_head(17)
        self.kp_reg_head = self.build_head(2)

    def build_head(self, out_channel):
        head_convs = [ConvModule(
            self.in_channels, self.feat_channels, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True))]
        for i in range(1, self.stacked_convs):
            head_convs.append(ConvModule(
                self.feat_channels, self.feat_channels, 3,
                padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))

        head_convs.append(nn.Conv2d(self.feat_channels, out_channel, 1))
        return nn.Sequential(*head_convs)

    def init_weights(self):
        # TOCHECK: cmp w/ centerpose/ttfnet
        for _, m in self.ct_hm_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        # > centernet
        bias_cls = bias_init_with_prob(0.1)
        # > ttfnet
        # bias_cls = bias_init_with_prob(0.01)
        normal_init(self.ct_hm_head[-1], std=0.01, bias=bias_cls)
        normal_init(self.kp_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.bbox_wh_head)
        self._init_head_weights(self.bbox_reg_head)
        self._init_head_weights(self.ct_kp_reg_head)
        self._init_head_weights(self.kp_reg_head)

    def forward(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]  # (64, H, W)

        ct_hm = self.ct_hm_head(x)  # > (64, H, W) -> (128, H, W) -> (#cls, H, W)
        ct_wh = self.bbox_wh_head(x)
        ct_offset = self.bbox_reg_head(x)
        ct_kp = self.ct_kp_reg_head(x)

        kp_ct = self.kp_hm_head(x)
        kp_reg = self.kp_reg_head(x)
        # `ct`: 1, `ct_offset`:2, `ct_wh`: 2, `ct_kp`:34, `kp_ct`: 17, `kp_offset`: 2
        return ct_hm, ct_offset, ct_wh, ct_kp, kp_ct, kp_reg

    def get_bboxes(self,
                   pred_ct,
                   pred_ct_offset,
                   pred_ct_wh,
                   pred_ct_kp,
                   pred_kp_ct,
                   pred_kp_ct_offset,
                   img_metas,
                   rescale=False):
        # `ct`: 1, `ct_offset`:2, `ct_wh`: 2, `ct_kp`:34, `kp_ct`: 17, `kp_offset`: 2
        batch, cls, height, width = pred_ct.size()
        ct_hm = pred_ct.detach().sigmoid_()  # > (#cls, H, W)
        ct_offset = pred_ct_offset.detach()  # > (B, 2, H, W)
        ct_wh = pred_ct_wh.detach()  # > (B, 2, H, W)
        ct2kp_offset = pred_ct_kp.detach()  # > (B, #kp x 2, H, W)
        kp_hm = pred_kp_ct.detach().sigmoid_()  # > (B, #kp, H, W)
        kp_ct_offset = pred_kp_ct_offset.detach()  # > (B, 2, H, W)
        num_topk = self.test_cfg.max_per_img
        num_joints = self.num_keypoints

        # > perform nms & topk over center points
        ct_hm = simple_nms(ct_hm)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = topk(ct_hm, topk=num_topk)

        # [BBox + Scores]
        # > refine center points by offsets
        # > [2] CenterNet's way
        ct_offset = transpose_and_gather_feat(ct_offset, ct_inds)
        ct_offset = ct_offset.view(batch, num_topk, 2)  # (B, K, 2)
        ct_xs = ct_xs.view(batch, num_topk, 1) + ct_offset[:, :, 0:1]  # (B, K, 1)
        ct_ys = ct_ys.view(batch, num_topk, 1) + ct_offset[:, :, 1:2]  # (B, K, 1)

        # > `width/height`: (B, 2, H, W) ∩ (B, K) -> (B, HxW, 2) ∩ (B, K, 2) -> (B, K, 2)
        ct_wh = transpose_and_gather_feat(ct_wh, ct_inds)
        ct_wh = ct_wh.view(batch, num_topk, 2)  # (B, K, 2)

        # > `classes & scores`
        bbox_clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        bbox_scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)

        # > `bboxe`s: (B, topk, 4)
        bboxes = torch.cat([ct_xs - ct_wh[..., 0:1] / 2,
                            ct_ys - ct_wh[..., 1:2] / 2,
                            ct_xs + ct_wh[..., 0:1] / 2,
                            ct_ys + ct_wh[..., 1:2] / 2], dim=2)

        # > center to joint offsets (out: 34)
        # `ct2kp_loc` ∩ `ct_inds`: (B, 34, H, W) ∩ (B, K) -> (B, HxW, 34) ∩ (B, K, 34) -> (B, K, 34)
        ct2kp_offset = transpose_and_gather_feat(ct2kp_offset, ct_inds)
        ct2kp_offset = ct2kp_offset.view(batch, num_topk, num_joints * 2)
        # (B, K) -> (B, K, 1) -> (B, K, #kp)
        ct2kp_offset[..., ::2] += ct_xs.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_offset[..., 1::2] += ct_ys.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_offset = ct2kp_offset.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()

        # > NOTE: `ct_det_kps`: regressed joint locations, (B, #kp, K, K, 2)
        ct_det_kps = ct2kp_offset.unsqueeze(3).expand(batch, num_joints, num_topk, num_topk, 2)

        # [Pose]
        # > centers as joints (out:17)
        kp_hm = simple_nms(kp_hm)  # (B, #kp, H, W)
        # `kp_ct`: (B, #kp, H, W) -> (B, #kp, HxW) -> `kp_ct_score/kp_ct_inds`: (B, 17, K)
        kp_ct_scores, kp_ct_inds, kp_ct_ys, kp_ct_xs = topk_channel(kp_hm, topk=num_topk)

        # > `kp_ct_offset`: refine joint by offsets
        # `kp_ct_offset`: (B, 2, H, W) ∩ (B, 17xK) -> (B, HxW, 2) ∩ (B, 17xK, 2) -> (B, 17xK, 2)
        kp_ct_offset = transpose_and_gather_feat(kp_ct_offset, kp_ct_inds.view(batch, -1))
        kp_ct_offset = kp_ct_offset.view(batch, num_joints, num_topk, 2)
        kp_ct_xs = kp_ct_xs + kp_ct_offset[..., 0]
        kp_ct_ys = kp_ct_ys + kp_ct_offset[..., 1]

        # > assign negative values to lower-score entries
        mask = (kp_ct_scores > self.test_cfg.kp_score_thr).float()  # (B, 17, K)
        kp_ct_scores = (1 - mask) * -1 + mask * kp_ct_scores
        kp_ct_xs = (1 - mask) * (-10000) + mask * kp_ct_xs
        kp_ct_ys = (1 - mask) * (-10000) + mask * kp_ct_ys

        # > NOTE:`kp_det_kps`: detected keypoints, (B, #kp, K, K, 2)
        kp_det_kps = torch.stack([kp_ct_xs, kp_ct_ys], dim=-1).unsqueeze(2)
        kp_det_kps = kp_det_kps.expand(batch, num_joints, num_topk, num_topk, 2)

        # NOTE: assign each regressed location to its closest detectio keypoint `argmin(ct_kps_loc - kp_kps_loc)^2`
        # (B, #kp, K, K, 2) - (B, #kp, K, K, 2)
        dist = (((ct_det_kps - kp_det_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_dist_ind = dist.min(dim=3)

        # `kp_ct_score`: (B, #kp, K) -> (B, #kp, K, 1)
        kp_ct_scores = kp_ct_scores.gather(2, min_dist_ind).unsqueeze(-1)
        min_dist = min_dist.unsqueeze(-1)  # (B, #kp, K, 1)
        # `min_ind`: (B, #kp, K, 1) -> (B, #kp, K, 1, 1) -> (B, #kp, K, 1, 2)
        min_dist_ind = min_dist_ind.view(batch, num_joints, num_topk, 1, 1)
        min_dist_ind = min_dist_ind.expand(batch, num_joints, num_topk, 1, 2)

        kp_det_kps = kp_det_kps.gather(3, min_dist_ind)  # (B, #kp, K, K, 2)
        kp_det_kps = kp_det_kps.view(batch, num_joints, num_topk, 2)  # (B, #kp, K, 2)

        # `bboxes`: (B, K, 4) -> `l/t/r/b`: (B, K) -> (B, 1, K, 1) -> (B, #kp, K, 1)
        l = bboxes[..., 0].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        t = bboxes[..., 1].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        r = bboxes[..., 2].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        b = bboxes[..., 3].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)

        # `kp_det_kps`: (B, #kp, K, 2), `kp_ct_score`: (B, #kp, K, 1), `min_dist`: (B, #kp, K, 1)
        mask = (kp_det_kps[..., 0:1] < l) + (kp_det_kps[..., 0:1] > r) + \
               (kp_det_kps[..., 1:2] < t) + (kp_det_kps[..., 1:2] > b) + \
               (kp_ct_scores < self.test_cfg.kp_score_thr) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        # negative `mask`: (B, #kp, K, 1) -> (B, #kp, K, 2)
        mask = (mask > 0).float().expand(batch, num_joints, num_topk, 2)

        # `keypoints`: (B, #kp, K, 2) -> (B, K, #kp x 2)
        kps = (1 - mask) * kp_det_kps + mask * ct2kp_offset
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, num_topk, num_joints * 2)
        kp_scores = torch.transpose(kp_ct_scores.squeeze(dim=3), 1, 2)  # (B, K, #kp)

        # `bboxes`: (B, K, 4), `bbox_scores`: (B, K, 1)
        # `kps`: (B, K, 34), `kp_scores`: (B, K, 17)
        bbox_result_list = []
        kp_result_list = []
        for batch_i in range(bboxes.shape[0]):  # > #imgs
            img_shape = img_metas[batch_i]['pad_shape']  # (512, 512, 3)
            bboxes_per_img = bboxes[batch_i]
            bbox_scores_per_img = bbox_scores[batch_i]  # (B, K, 1) -> (K, 1)
            labels_per_img = bbox_clses[batch_i]  # (B, K, 1) -> (K, 1)
            kps_per_img = kps[batch_i].view(kps.size(1), num_joints, 2)  # (B, K, #kp x 2) -> (K, #kp, 2)
            kp_scores_per_img = kp_scores[batch_i]  # (B, K, #kp) -> (K, #kp)

            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)
            # TOCHECK: Clamp keypoints out of scope?
            # kps_per_img[:, 0::2] = kps_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            # kps_per_img[:, 1::2] = kps_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)
            bboxes_per_img *= self.down_ratio
            kps_per_img *= self.down_ratio
            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)
                kps_per_img /= kps_per_img.new_tensor(scale_factor[:2])
            bboxes_per_img = torch.cat([bboxes_per_img, bbox_scores_per_img], dim=1)  # (K, 4 + 1)
            labels_per_img = labels_per_img.squeeze(-1)  # (K, 1) -> (K,)
            kp_scores_per_img = kp_scores_per_img > self.test_cfg.kp_score_thr
            kp_scores_per_img = kp_scores_per_img.int().unsqueeze(-1)
            kps_per_img = torch.cat([kps_per_img, kp_scores_per_img], dim=2).view(num_topk, -1)  # (K, #kp x 3)
            kps_per_img = torch.cat([kps_per_img, bbox_scores_per_img], dim=1)  # (K, #kp x 3 + 1)
            # TODO: add concat() to (K, 56) for softnms39().
            bbox_result_list.append((bboxes_per_img, labels_per_img))
            kp_result_list.append(kps_per_img)

        return bbox_result_list, kp_result_list

    @force_fp32(apply_to=('pred_heatmap', 'pred_wh'))
    def loss(self,
             pred_heatmap,  # (B, #cls, H, W)
             pred_wh,  # (B, 4, H, W)
             gt_bboxes,  # (B, #obj, 4)
             gt_labels,  # (B, #obj)
             img_metas,
             gt_bboxes_ignore=None):
        pass