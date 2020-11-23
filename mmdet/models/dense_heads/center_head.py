import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, build_upsample_layer, build_norm_layer
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.ops import ModulatedDeformConv2dPack
from mmcv.runner import force_fp32
import numpy as np
import math
from mmdet.core import multi_apply, calc_region
from ..builder import HEADS, build_loss
# from ..utils.center_ops import gather_feature, transpose_and_gather_feat, topk, topk_channel
from .corner_head import CornerHead
from ..utils import gaussian_radius, gen_gaussian_target

# TOCHECK:
#  1. SmoothL1Loss vs. L1Loss
#  2. GaussianFocalLoss vs. CenterFocalLoss
#  3. FocalL2Loss vs. GaussianFocalLoss vs. L1Loss
#  4. FPN style


@HEADS.register_module
class CenterHead(CornerHead):

    def __init__(self,
                 *args,
                 max_objs=128,
                 dcn_cfg=dict(
                     in_channels=(512, 256, 128, 64),
                     kernels=(4, 4, 4),
                     strides=(2, 2, 2),
                     paddings=(1, 1, 1),
                     out_paddings=(0, 0, 0)
                 ),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 **kwargs):
        self.max_objs = max_objs
        self.dcn_cfg = dcn_cfg
        super(CenterHead, self).__init__(*args, **kwargs)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False
        self.limbs = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

    def build_upsample(self, in_channels, kernels, strides, paddings, out_paddings):
        upsamples = []
        for i in range(len(in_channels) - 1):
            upsamples.append(UpsampleDeconv(
                in_channels[i],
                in_channels[i+1],
                kernels[i],
                strides[i],
                paddings[i],
                out_paddings[i]
            ))
        return nn.Sequential(*upsamples)

    def build_head(self, out_channel):
        head_convs = [ConvModule(
            self.in_channels, self.in_channels, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True))]
        for i in range(1, self.num_feat_levels):
            head_convs.append(ConvModule(
                self.in_channels, self.in_channels, 3,
                padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))

        head_convs.append(nn.Conv2d(self.in_channels, out_channel, 1))
        return nn.Sequential(*head_convs)

    def _init_layers(self):
        # > upsample
        self.upsamples = self.build_upsample(self.dcn_cfg.in_channels,
                                             self.dcn_cfg.kernels,
                                             self.dcn_cfg.strides,
                                             self.dcn_cfg.paddings,
                                             self.dcn_cfg.out_paddings)

        # > ct heads
        self.ct_hm_head = self.build_head(self.num_classes)
        self.bbox_wh_head = self.build_head(2)
        self.bbox_reg_head = self.build_head(2)

    def init_weights(self):
        # TOCHECK: cmp w/ centerpose/ttfnet
        for _, m in self.ct_hm_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        bias_cls = bias_init_with_prob(0.1)
        normal_init(self.ct_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.bbox_wh_head)
        self._init_head_weights(self.bbox_reg_head)

    def _init_head_weights(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        x = feats[-1]  # (B, 512, H, W)
        x = self.upsamples(x)  # (B, 64, H, W)
        ct_cls = self.ct_hm_head(x)  # > (B, #cls, 128, 128)
        ct_wh = self.bbox_wh_head(x)  # > (B, 2, 128, 128)
        ct_offset = self.bbox_reg_head(x)  # > (B, 2, 128, 128)
        return ct_cls, ct_offset, ct_wh

    def get_bboxes(self,
                   pred_ct_cls,
                   pred_ct_offset,
                   pred_ct_wh,
                   img_metas,
                   rescale=False):

        batch, cls, feat_h, feat_w = pred_ct_cls.size()
        # `ct`: #cls, `ct_offset`: 2, `ct_wh`: 2
        ct_cls = pred_ct_cls.detach().sigmoid_()  # > (#cls, H, W)
        ct_offset = pred_ct_offset.detach()  # > (B, 2, H, W)
        ct_wh = pred_ct_wh.detach()  # > (B, 2, H, W)
        num_topk = self.test_cfg.max_per_img

        # > perform nms & topk over center points
        ct_cls = self._local_maximum(ct_cls, kernel=3)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_cls, k=num_topk)

        if ct_offset is not None:
            ct_offset = self._transpose_and_gather_feat(ct_offset, ct_inds)  # (B, 2, H, W) -> (B, K, 2)
            ct_offset = ct_offset.view(batch, num_topk, 2)
            ct_xs = ct_xs.view(batch, num_topk, 1) + ct_offset[:, :, 0:1]
            ct_ys = ct_ys.view(batch, num_topk, 1) + ct_offset[:, :, 1:2]
        else:
            ct_xs = ct_xs.view(batch, num_topk, 1) + 0.5
            ct_ys = ct_ys.view(batch, num_topk, 1) + 0.5

        ct_wh = self._transpose_and_gather_feat(ct_wh, ct_inds)  # (B, 2, H, W) -> (B, K, 2)
        ct_wh = ct_wh.view(batch, num_topk, 2)  # (B, K, 2)

        # > `classes & scores`
        clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)

        # > `bboxes`: (B, topk, 4)
        half_w, half_h = ct_wh[..., 0:1] / 2, ct_wh[..., 1:2] / 2
        bboxes = torch.cat([ct_xs - half_w, ct_ys - half_h,
                            ct_xs + half_w, ct_ys + half_h],
                           dim=2)  # (B, K, 4)
        # feat_bboxes = bboxes.clone()
        # bboxes[..., 0::2] = bboxes[..., 0::2] / width_ratio
        # bboxes[..., 1::2] = bboxes[..., 1::2] / height_ratio
        result_list = []
        for img_id in range(len(img_metas)):  # > #img
            result_list.append(self.get_bboxes_single(
                bboxes[img_id],
                scores[img_id],
                clses[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale
            ))
        return result_list

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_size,
                          rescale=False):
        # `bboxes`: (B, K, 4), `bbox_scores`: (B, K, 1)
        pad_h, pad_w = img_meta['pad_shape'][:2]
        width_ratio = float(feat_size[1] / pad_w)
        height_ratio = float(feat_size[0] / pad_h)
        feat_bboxes = bboxes.clone()
        bboxes[:, 0::2] = (bboxes[:, 0::2] / width_ratio).clamp(min=0, max=pad_w - 1)
        bboxes[:, 1::2] = (bboxes[:, 1::2] / height_ratio).clamp(min=0, max=pad_h - 1)
        if rescale:
            scale_factor = img_meta['scale_factor']
            bboxes /= bboxes.new_tensor(scale_factor)
        bboxes = torch.cat([bboxes, scores], dim=1)  # (K, 4 + 1)
        labels = labels.squeeze(-1)  # (K, 1) -> (K,)
        # TOCHECK: remove `feat_bboxes`
        return bboxes, labels#, feat_bboxes

    @force_fp32(apply_to=('pred_heatmap', 'pred_reg', 'pred_wh'))
    def loss(self,
             pred_heatmap,  # (B, #cls, H, W)
             pred_wh,  # (B, 2, H, W)
             pred_reg,  # (B, 2, H, W)
             gt_bboxes,  # (B, #obj, 4)
             gt_labels,  # (B, #obj)
             img_metas,
             gt_bboxes_ignore=None):
        img_shape = img_metas[0]['pad_shape'][:2]
        feat_shape = tuple(pred_heatmap.size()[2:])
        all_targets = self.get_all_targets(gt_bboxes, gt_labels, img_shape, feat_shape)  # heatmap, box_target, reg_weight
        loss_cls, loss_wh, loss_reg = self.loss_calc(pred_heatmap, pred_wh, pred_reg, *all_targets)
        return {'loss_heatmap': loss_cls, 'loss_wh': loss_wh, 'loss_reg': loss_reg}

    def target_single_image(self, gt_boxes, gt_labels, img_shape, feat_shape):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        img_h, img_w = img_shape  # > input size: (512, 512)
        feat_h, feat_w = feat_shape  # > feat size: (128, 128)
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        num_objs = gt_boxes.size(0)
        # `H/W`: 128
        gt_heatmap = gt_boxes.new_zeros(self.num_classes, feat_h, feat_w)  # > (#cls, H, W)
        gt_wh = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        gt_reg = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        gt_inds = gt_boxes.new_zeros(self.max_objs)  # > (#max_obj,)
        gt_mask = gt_boxes.new_zeros(self.max_objs)  # > (#max_obj,)

        feat_gt_boxes = gt_boxes.clone()  # > (#obj, 4)
        feat_gt_boxes[:, [0, 2]] *= width_ratio
        feat_gt_boxes[:, [1, 3]] *= height_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(
            feat_gt_boxes[:, [0, 2]], min=0, max=feat_w - 1)  # x_min, x_max
        feat_gt_boxes[:, [1, 3]] = torch.clamp(
            feat_gt_boxes[:, [1, 3]], min=0, max=feat_h - 1)  # y_min, y_max
        feat_gt_centers = (feat_gt_boxes[:, :2] + feat_gt_boxes[:, 2:]) / 2  # > (#obj, 2)
        feat_gt_centers_int = feat_gt_centers.to(torch.int)  # > (#obj, 2)
        gt_inds[:num_objs] = feat_gt_centers_int[..., 1] * feat_h + feat_gt_centers_int[..., 0]
        gt_reg[:num_objs] = feat_gt_centers - feat_gt_centers_int
        gt_mask[:num_objs] = 1

        feat_gt_wh = torch.zeros_like(feat_gt_centers)  # > (#obj, 2)
        feat_gt_wh[..., 0] = feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0]
        feat_gt_wh[..., 1] = feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1]
        gt_wh[:num_objs] = feat_gt_wh

        for box_id in range(num_objs):
            radius = gaussian_radius(feat_gt_wh[box_id], min_overlap=self.train_cfg.min_overlap)
            radius = max(0, int(radius))
            cls_ind = gt_labels[box_id]
            gt_heatmap[cls_ind] = gen_gaussian_target(heatmap=gt_heatmap[cls_ind],
                                                      center=feat_gt_centers_int[box_id],
                                                      radius=radius)
        return gt_heatmap, gt_wh, gt_reg, gt_inds, gt_mask

    def get_all_targets(self,
                        gt_boxes,
                        gt_labels,
                        img_shape,
                        feat_shape):
        """
        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        with torch.no_grad():
            # `heatmap`: (B, #cls, H, W), `wh_target`: (B, 2, H, W), `reg_target`: (B, 2, H, W)
            heatmap, wh_target, reg_target, ind_target, target_mask = multi_apply(
                self.target_single_image,
                gt_boxes,  # (B, #obj, 4)
                gt_labels,  # (B, #obj)
                img_shape=img_shape,  # (img_H, img_W)
                feat_shape=feat_shape  # (H, W)
            )
            heatmap = torch.stack(heatmap, dim=0)
            wh_target = torch.stack(wh_target, dim=0)
            reg_target = torch.stack(reg_target, dim=0)
            ind_target = torch.stack(ind_target, dim=0).to(torch.long)
            target_mask = torch.stack(target_mask, dim=0)
            # heatmap, wh_target = [torch.stack(t, dim=0).detach() for t in [heatmap, wh_target]]
            # reg_target = torch.stack(reg_target, dim=0).detach()
            return heatmap, wh_target, reg_target, ind_target, target_mask

    def loss_calc(self,
                  pred_hm,
                  pred_reg,
                  pred_wh,
                  heatmap_target,
                  wh_target,
                  reg_target,
                  ind_target,
                  target_mask):
        """

        Args:
            pred_hm: tensor, (batch, #cls, h, w).
            pred_reg: tensor, (batch, 2, h, w)
            pred_wh: tensor, (batch, 2, h, w)
            heatmap_target: tensor, same as pred_hm.
            reg_target: tensor, same as pred_reg.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        H, W = pred_hm.shape[2:]
        eps = 1e-12
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = heatmap_target.eq(1).float().sum()
        # > class loss
        loss_cls = self.loss_cls(pred_hm, heatmap_target, avg_factor=1 if num_pos == 0 else num_pos)
        target_mask = target_mask.unsqueeze(2)

        # > width/height loss
        pred_wh = self._transpose_and_gather_feat(pred_wh, ind_target)
        wh_mask = target_mask.expand_as(pred_wh).float()
        loss_wh = self.loss_bbox(pred_wh * wh_mask,
                                 wh_target * wh_mask,
                                 avg_factor=wh_mask.sum() + 1e-4)

        # > offset loss
        pred_reg = self._transpose_and_gather_feat(pred_reg, ind_target)
        reg_mask = target_mask.expand_as(pred_reg).float()
        loss_reg = self.loss_offset(pred_reg * reg_mask,
                                    reg_target * reg_mask,
                                    avg_factor=reg_mask.sum() + 1e-4)
        return loss_cls, loss_wh, loss_reg


# @HEADS.register_module()
class CenterPoseHead(CornerHead):

    def __init__(self,
                 *args,
                 in_channels=256,
                 loss_pose=None,
                 **kwargs):
        super(CenterPoseHead, self).__init__(*args, **kwargs)
        self.loss_pose = build_loss(loss_pose)
        self.in_channels = in_channels
        self._init_layers()
        self.fp16_enabled = False

    def build_head(self, out_channel):
        head_convs = [ConvModule(
            self.in_channels, self.in_channels, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True))]
        for i in range(1, self.num_feat_levels):
            head_convs.append(ConvModule(
                self.in_channels, self.in_channels, 3,
                padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))

        head_convs.append(nn.Conv2d(self.in_channels, out_channel, 1))
        return nn.Sequential(*head_convs)

    def _init_layers(self):
        self.ct_kp_reg_head = self.build_head(self.num_classes * 2)
        self.kp_hm_head = self.build_head(self.num_classes)
        self.kp_reg_head = self.build_head(2)

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.1)
        normal_init(self.kp_hm_head[-1], std=0.01, bias=bias_cls)
        # TOCHECK: cmp w/ centerpose/ttfnet
        self._init_head_weights(self.ct_kp_reg_head)
        self._init_head_weights(self.kp_reg_head)

    def _init_head_weights(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        x = feats[-1]  # (64, H, W)
        ct_kp_reg = self.ct_kp_reg_head(x)
        kp_hm = self.kp_hm_head(x)
        kp_reg = self.kp_reg_head(x)
        return ct_kp_reg, kp_hm, kp_reg

    def get_keypoints(self,
                      pred_ct_kp,
                      pred_kp_hm,
                      pred_kp_ct_reg,
                      pred_bbox_scores,
                      pred_ct_xs,
                      pred_ct_ys,
                      pred_ct_inds,
                      feat_bboxes,
                      img_metas,
                      rescale=False):
        batch, _, feat_h, feat_w = pred_kp_hm.size()
        num_topk = self.test_cfg.max_per_img
        num_joints = self.num_classes
        # pad_h, pad_w = img_metas[0]['pad_shape'][:2]
        batch, cls, feat_h, feat_w = pred_ct_kp.size()
        # width_ratio = float(feat_w / pad_w)
        # height_ratio = float(feat_h / pad_h)
        # batch, _, height, width = pred_ct_kp.size()
        ct2kp_offset = pred_ct_kp.detach()  # > (B, #kp x 2, H, W)
        kp_hm = pred_kp_hm.detach().sigmoid_()  # > (B, #kp, H, W)
        kp_offset = pred_kp_ct_reg.detach()  # > (B, 2, H, W)

        # > offsets of center to keypoints
        ct2kp_offset = self._transpose_and_gather_feat(ct2kp_offset, pred_ct_inds)
        ct2kp_offset = ct2kp_offset.view(batch, num_topk, num_joints * 2)
        ct2kp_offset[..., ::2] += pred_ct_xs.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_offset[..., 1::2] += pred_ct_ys.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_offset = ct2kp_offset.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()
        ct_det_kps = ct2kp_offset.unsqueeze(3).expand(batch, num_joints, num_topk, num_topk, 2)

        # > keypoint heatmaps, TOCHECK: remove `self._topk`
        kp_hm = self._local_maximum(kp_hm, kernel=3)
        kp_scores, kp_inds, kp_ys, kp_xs = self._topk(kp_hm, k=num_topk)

        # > keypoint regressed locations
        kp_offset = self._transpose_and_gather_feat(kp_offset, kp_inds.view(batch, -1))
        kp_offset = kp_offset.view(batch, num_joints, num_topk, 2)
        kp_xs = kp_xs + kp_offset[..., 0]
        kp_ys = kp_ys + kp_offset[..., 1]

        # > keep positives
        mask = (kp_scores > self.test_cfg.kp_score_thr).float()
        kp_scores = (1 - mask) * -1 + mask * kp_scores
        kp_xs = (1 - mask) * (-10000) + mask * kp_xs
        kp_ys = (1 - mask) * (-10000) + mask * kp_ys

        kp_det_kps = torch.stack([kp_xs, kp_ys], dim=-1).unsqueeze(2)
        kp_det_kps = kp_det_kps.expand(batch, num_joints, num_topk, num_topk, 2)

        # NOTE: assign each regressed location to its closest detectio keypoint `argmin(ct_kps_loc - kp_kps_loc)^2`
        dist = (((ct_det_kps - kp_det_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_dist_ind = dist.min(dim=3)
        min_dist = min_dist.unsqueeze(-1)  # (B, #kp, K, 1)

        kp_scores = kp_scores.gather(2, min_dist_ind).unsqueeze(-1)
        # `min_ind`: (B, #kp, K, 1) -> (B, #kp, K, 1, 1) -> (B, #kp, K, 1, 2)
        min_dist_ind = min_dist_ind.view(batch, num_joints, num_topk, 1, 1)
        min_dist_ind = min_dist_ind.expand(batch, num_joints, num_topk, 1, 2)

        kp_det_kps = kp_det_kps.gather(3, min_dist_ind)  # (B, #kp, K, K, 2)
        kp_det_kps = kp_det_kps.view(batch, num_joints, num_topk, 2)  # (B, #kp, K, 2)

        l = feat_bboxes[..., 0].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        t = feat_bboxes[..., 1].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        r = feat_bboxes[..., 2].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        b = feat_bboxes[..., 3].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        mask = (kp_det_kps[..., 0:1] < l) + (kp_det_kps[..., 0:1] > r) + \
               (kp_det_kps[..., 1:2] < t) + (kp_det_kps[..., 1:2] > b) + \
               (kp_scores < self.test_cfg.kp_score_thr) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, num_topk, 2)

        kps = (1 - mask) * kp_det_kps + mask * ct2kp_offset
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, num_topk, num_joints * 2)
        kp_scores = torch.transpose(kp_scores.squeeze(dim=3), 1, 2)  # (B, K, #kp)

        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self.get_keypoints_single(
                kps[img_id],
                kp_scores[img_id],
                pred_bbox_scores[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale
            ))
        return result_list

    def get_keypoints_single(self,
                             keypoints,
                             scores,
                             bbox_scores,
                             img_meta,
                             feat_size,
                             rescale=False):
        pad_h, pad_w = img_meta['pad_shape'][:2]
        num_topk = self.test_cfg.max_per_img
        width_ratio = float(feat_size[1] / pad_w)
        height_ratio = float(feat_size[0] / pad_h)
        keypoints = keypoints.view(keypoints.size(0), self.num_classes, 2)
        keypoints[..., 0] /= width_ratio
        keypoints[..., 1] /= height_ratio
        if rescale:
            scale_factor = img_meta['scale_factor']
            keypoints /= keypoints.new_tensor(scale_factor)
        scores = (scores > self.test_cfg.kp_score_thr).int().unsqueeze(-1)

        keypoints = torch.cat([keypoints, scores], dim=2).view(num_topk, -1)  # (K, #kp x 3)
        keypoints = torch.cat([keypoints, bbox_scores], dim=1)  # (K, #kp x 3 + 1)
        # TOCHECK: add concat() to (K, 56) for softnms39().
        return keypoints

    def _topk(self, scores, k=20):
        # TOCHECK: use `CornerHead._topk()`
        batch, cls, height, width = scores.size()
        # `scores`: (B, #cls, H, W) -> (B, #cls, HxW) -> (B, #cls, K)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), k)

        topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
        topk_ys = (topk_inds // width).float()  # (B, #cls, topk)
        topk_xs = (topk_inds % width).float()  # (B, #cls, topk)

        return topk_scores, topk_inds, topk_ys, topk_xs


class UpsampleDeconv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 deconv_kernel,
                 deconv_stride,
                 deconv_pad,
                 deconv_out_pad):
        super(UpsampleDeconv, self).__init__()
        self.dcn = ModulatedDeformConv2dPack(in_channels, out_channels, 3, stride=1,
                                             padding=1, dilation=1, deformable_groups=1)
        self.dcn_bn = nn.BatchNorm2d(out_channels)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=deconv_kernel,
            stride=deconv_stride,
            padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self.up_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self._deconv_init()

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x


class ShortcutConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 activation_last=False):
        super(ShortcutConv2d, self).__init__()
        assert len(kernel_sizes) == len(paddings)

        layers = []
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            inc = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(inc, out_channels, kernel_size, padding=padding))
            if i < len(kernel_sizes) - 1 or activation_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y
