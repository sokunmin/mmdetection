import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, xavier_init, build_upsample_layer
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32
from mmcv.ops.carafe import CARAFEPack

from mmdet.core import multi_apply
from .corner_head import CornerHead
from ..builder import HEADS, build_loss
from ..utils import gaussian_radius, gen_gaussian_target


def build_convs(in_channel, feat_channel, stacked_convs):
    head_convs = []
    for i in range(stacked_convs):
        chn = in_channel if i == 0 else feat_channel
        head_convs.append(ConvModule(
            chn, feat_channel, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))
    return head_convs


def build_share_convs(in_channel, feat_channel, stacked_convs):
    if stacked_convs == 0:
        return nn.Identity()
    return nn.Sequential(*build_convs(in_channel, feat_channel, stacked_convs))


def build_head(in_channel, feat_channel, stacked_convs, out_channel):
    head_convs = build_convs(in_channel, feat_channel, stacked_convs)
    head_convs.append(nn.Conv2d(feat_channel, out_channel, 1))
    return nn.Sequential(*head_convs)


@HEADS.register_module
class CenterHead(CornerHead):
    def __init__(self,
                 *args,
                 stacked_convs=1,
                 share_stacked_convs=0,
                 feat_channels=256,
                 max_objs=128,
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.max_objs = max_objs
        self.share_stacked_convs = share_stacked_convs
        self.with_share_convs = share_stacked_convs > 0
        super(CenterHead, self).__init__(*args, **kwargs)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_embedding = None
        self.fp16_enabled = False

    def _init_layers(self):
        feat_channels = self.feat_channels
        stacked_convs = self.stacked_convs
        head_in_channel = feat_channels if self.with_share_convs else self.in_channels
        self.share_convs = build_share_convs(
            self.in_channels, feat_channels, self.share_stacked_convs)
        self.ct_hm_head = build_head(
            head_in_channel, feat_channels, stacked_convs, self.num_classes)
        self.ct_wh_head = build_head(
            head_in_channel, feat_channels, stacked_convs, 2)
        self.ct_reg_head = build_head(
            head_in_channel, feat_channels, stacked_convs, 2)

    def init_weights(self):
        for _, m in self.ct_hm_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        bias_cls = bias_init_with_prob(0.1)
        normal_init(self.ct_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.ct_wh_head)
        self._init_head_weights(self.ct_reg_head)

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
        if self.with_share_convs:
            feats = self.share_convs(feats)
        ct_hm = self.ct_hm_head(feats)  # > (B, #cls, 128, 128)
        ct_wh = self.ct_wh_head(feats)  # > (B, 2, 128, 128)
        ct_offset = self.ct_reg_head(feats)  # > (B, 2, 128, 128)
        if self.with_share_convs:
            return ct_hm, ct_offset, ct_wh, feats
        return ct_hm, ct_offset, ct_wh

    def get_bboxes(self,
                   pred_ct_cls,
                   pred_ct_offset,
                   pred_ct_wh,
                   img_metas,
                   rescale=False,
                   with_nms=True):
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
            refined_ct_xs = ct_xs.view(batch, num_topk, 1) + ct_offset[:, :, 0:1]
            refined_ct_ys = ct_ys.view(batch, num_topk, 1) + ct_offset[:, :, 1:2]
        else:
            refined_ct_xs = ct_xs.view(batch, num_topk, 1) + 0.5
            refined_ct_ys = ct_ys.view(batch, num_topk, 1) + 0.5

        ct_wh = self._transpose_and_gather_feat(ct_wh, ct_inds)  # (B, 2, H, W) -> (B, K, 2)
        ct_wh = ct_wh.view(batch, num_topk, 2)  # (B, K, 2)

        # > `classes & scores`
        clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)

        half_w, half_h = ct_wh[..., 0:1] / 2, ct_wh[..., 1:2] / 2
        bboxes = torch.cat([refined_ct_xs - half_w, refined_ct_ys - half_h,
                            refined_ct_xs + half_w, refined_ct_ys + half_h],
                           dim=2)  # (B, K, 4)
        feat_bboxes = bboxes.clone()
        result_list = []
        for img_id in range(len(img_metas)):  # > #img
            result_list.append(self.get_bboxes_single(
                bboxes[img_id],
                scores[img_id],
                clses[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale,
                with_nms=with_nms
            ))
        return result_list, (clses, scores, feat_bboxes, ct_inds, ct_xs, ct_ys, ct_wh)

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_shape,
                          rescale=False,
                          with_nms=True):
        # `bboxes`: (B, K, 4), `bbox_scores`: (B, K, 1)
        pad_h, pad_w = img_meta['pad_shape'][:2]
        width_stride = float(pad_w / feat_shape[1])  # > 4
        height_stride = float(pad_h / feat_shape[0])
        bboxes[:, 0::2] = (bboxes[:, 0::2] * width_stride).clamp(min=0, max=pad_w - 1)
        bboxes[:, 1::2] = (bboxes[:, 1::2] * height_stride).clamp(min=0, max=pad_h - 1)
        if rescale:
            scale_factor = img_meta['scale_factor']
            bboxes /= bboxes.new_tensor(scale_factor)
        bboxes = torch.cat([bboxes, scores], dim=1)  # (K, 4 + 1)
        labels = labels.squeeze(-1)  # (K, 1) -> (K,)
        # TOCHECK: `self._bboxes_nms` in `Cornerhead`
        return bboxes, labels

    def interprocess(self, *inputs, training=False):
        if training:
            """
            [IN] bbox_head() -> (
                preds: 
                    bbox=(ct_hm, ct_offset, ct_wh) / (ct_hm, ct_offset, ct_wh, feats) 
                gts: (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
                metas: 
                    bbox=None
            ) -> (gt_boxes, gt_labels)
            [OUT] -> get_targets(gt_boxes, gt_labels, ...)
            """
            pred_outs, gt_inputs, metas = inputs
            if self.with_share_convs:
                share_feats = pred_outs['bbox'][-1]
                pred_outs['bbox'] = pred_outs['bbox'][:3]
                pred_outs['mask'] = share_feats
                pred_outs['keypoint'] = share_feats
            gt_boxes, gt_labels = gt_inputs[0], gt_inputs[3]
            return gt_boxes, gt_labels
        else:
            """
            [IN] bbox_head() -> (
                preds: 
                    bbox=(ct_hm, ct_offset, ct_wh) / (ct_hm, ct_offset, ct_wh, feats) 
                metas:
                    bbox=(scores, feat_bboxes, ct_inds, ct_xs, ct_ys)
            )
            [OUT] -> get_bboxes(ct_hm, ct_offset, ct_wh, ...)
            """
            pred_outs = inputs[0]
            if self.with_share_convs:
                share_feats = pred_outs['bbox'][-1]
                pred_outs['bbox'] = pred_outs['bbox'][:3]
                pred_outs['mask'] = share_feats
                pred_outs['keypoint'] = share_feats
            return pred_outs['bbox']

    def postprocess(self, *inputs, training=False):
        if training:
            """
            [IN] get_targets() -> (
                preds: 
                    bbox=(ct_hm, ct_offset, ct_wh)
                targets: 
                    bbox=(heatmap, offset_target, wh_target, ct_target_mask, ct_ind_target)
                metas: 
                    bbox=(feat_gt_centers_int, feat_gt_boxes_wh)
            ) -> (
                ct_hm, ct_offset, ct_wh, 
                hm_target, offset_target, wh_target, ct_target_mask, ct_ind_target
            )
            [OUT] -> loss(
                pred_hm, pred_reg, pred_wh, 
                hm_target, offset_target, wh_target, ct_target_mask, ct_ind_target
                gt_metas=(feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes), 
            )
            """
            pred_outs, targets, metas = inputs
            box_targets, box_meta = targets['bbox'], metas['bbox']
            new_inputs = pred_outs + box_targets + (box_meta,)
        else:
            """
            [IN] get_bboxes() -> (
                preds: 
                    bbox=(ct_hm, ct_offset, ct_wh)
                metas: 
                    bbox=(scores, feat_bboxes, ct_inds, ct_xs, ct_ys)
                dets: (bboxes, labels)
            ) 
            [OUT] (preds, dets) -> bbox2results(det_bboxes, det_labels)
            """
            new_inputs = inputs
        return new_inputs

    @force_fp32(apply_to=('pred_heatmap', 'pred_reg', 'pred_wh'))
    def loss(self,
             pred_hm,  # (B, #cls, H, W)
             pred_offset,  # (B, 2, H, W)
             pred_wh,  # (B, 2, H, W)
             hm_target,  # (B, #cls, H, W)
             offset_target,
             wh_target,
             ct_target_mask,
             ct_ind_target,  # (B, #max_obj)
             gt_metas,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        eps = 1e-12
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        # > class loss
        loss_cls = self.loss_heatmap(pred_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)
        ct_target_mask = ct_target_mask.unsqueeze(2)

        eps = 1e-4
        # > width/height loss
        pred_wh = self._transpose_and_gather_feat(pred_wh, ct_ind_target)
        wh_mask = ct_target_mask.expand_as(pred_wh).float()
        loss_wh = self.loss_bbox(pred_wh * wh_mask,
                                 wh_target * wh_mask,
                                 avg_factor=wh_mask.sum() + eps)

        # > offset loss
        pred_offset = self._transpose_and_gather_feat(pred_offset, ct_ind_target)
        ct_target_mask = ct_target_mask.expand_as(pred_offset).float()
        loss_reg = self.loss_offset(pred_offset * ct_target_mask,
                                    offset_target * ct_target_mask,
                                    avg_factor=ct_target_mask.sum() + eps)
        return {'bbox/loss_heatmap': loss_cls, 'bbox/loss_reg': loss_reg, 'bbox/loss_wh': loss_wh}

    def get_targets(self,
                    gt_boxes,
                    gt_labels,
                    feat_shapes,
                    img_metas,
                    **kwargs):
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
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]
        with torch.no_grad():
            # `heatmap`: (B, #cls, H, W), `wh_target`: (B, 2, H, W), `reg_target`: (B, 2, H, W)
            hm_target, offset_target, wh_target, ct_ind_target, ct_target_mask, \
            feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes = \
                multi_apply(
                    self._get_targets_single,
                    gt_boxes,  # (B, #obj, 4)
                    gt_labels,  # (B, #obj)
                    img_shape=img_shape,  # (img_H, img_W)
                    feat_shape=feat_shape  # (H, W)
                )
            hm_target = torch.stack(hm_target, dim=0)
            offset_target = torch.stack(offset_target, dim=0)
            wh_target = torch.stack(wh_target, dim=0)
            ct_ind_target = torch.stack(ct_ind_target, dim=0).to(torch.long)
            ct_target_mask = torch.stack(ct_target_mask, dim=0)
            return (hm_target, offset_target, wh_target, ct_target_mask, ct_ind_target), \
                   (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)

    def _get_targets_single(self,
                            gt_boxes,
                            gt_labels,
                            img_shape,
                            feat_shape):
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
        width_ratio = float(feat_w / img_w)  # > 0.25
        height_ratio = float(feat_h / img_h)
        num_objs = gt_boxes.size(0)
        # `H/W`: 128
        hm_target = gt_boxes.new_zeros(self.num_classes, feat_h, feat_w)  # > (#cls, H, W)
        offset_target = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        wh_target = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        ct_ind_target = gt_boxes.new_zeros(self.max_objs, dtype=torch.int64)  # > (#max_obj,)
        ct_target_mask = gt_boxes.new_zeros(self.max_objs, dtype=torch.uint8)  # > (#max_obj,)

        feat_gt_boxes = gt_boxes.clone()  # > (#obj, 4)
        feat_gt_boxes[:, [0, 2]] *= width_ratio
        feat_gt_boxes[:, [1, 3]] *= height_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(
            feat_gt_boxes[:, [0, 2]], min=0, max=feat_w - 1)  # x_min, x_max
        feat_gt_boxes[:, [1, 3]] = torch.clamp(
            feat_gt_boxes[:, [1, 3]], min=0, max=feat_h - 1)  # y_min, y_max
        feat_gt_centers = (feat_gt_boxes[:, :2] + feat_gt_boxes[:, 2:]) / 2  # > (#obj, 2)
        feat_gt_centers_int = feat_gt_centers.to(torch.int)  # > (#obj, 2)
        # > target metas
        ct_ind_target[:num_objs] = feat_gt_centers_int[..., 1] * feat_h + feat_gt_centers_int[..., 0]
        offset_target[:num_objs] = feat_gt_centers - feat_gt_centers_int
        ct_target_mask[:num_objs] = 1

        feat_gt_boxes_wh = torch.zeros_like(feat_gt_centers)  # > (#obj, 2)
        feat_gt_boxes_wh[..., 0] = feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0]
        feat_gt_boxes_wh[..., 1] = feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1]
        wh_target[:num_objs] = feat_gt_boxes_wh

        for box_id in range(num_objs):
            radius = gaussian_radius(feat_gt_boxes_wh[box_id], min_overlap=self.train_cfg.min_overlap)
            radius = max(0, int(radius))
            cls_ind = gt_labels[box_id]
            hm_target[cls_ind] = gen_gaussian_target(heatmap=hm_target[cls_ind],
                                                     center=feat_gt_centers_int[box_id],
                                                     radius=radius)
        return hm_target, offset_target, wh_target, ct_ind_target, ct_target_mask, \
               feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes


@HEADS.register_module()
class CenterPoseHead(CornerHead):

    def __init__(self,
                 *args,
                 stacked_convs=1,
                 feat_channels=256,
                 max_objs=128,
                 loss_joint=None,
                 with_limbs=False,
                 with_share_convs=False,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.max_objs = max_objs
        self.limbs = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]
        self.with_limbs = with_limbs
        self.with_share_convs = with_share_convs
        super(CenterPoseHead, self).__init__(*args, **kwargs)
        self.loss_joint = build_loss(loss_joint)
        self.loss_embedding = None
        self.fp16_enabled = False

    def _init_layers(self):
        feat_channels = self.feat_channels
        stacked_convs = self.stacked_convs
        head_in_channel = feat_channels if self.with_share_convs else self.in_channels
        self.kp_hm_head = build_head(
            head_in_channel, feat_channels, stacked_convs, self.num_classes)
        self.kp_reg_head = build_head(
            head_in_channel, feat_channels, stacked_convs, 2)
        self.ct_kp_reg_head = build_head(
            head_in_channel, feat_channels, stacked_convs, self.num_classes * 2)

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.1)
        normal_init(self.kp_hm_head[-1], std=0.01, bias=bias_cls)
        self._init_head_weights(self.ct_kp_reg_head)
        self._init_head_weights(self.kp_reg_head)

    def _init_head_weights(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, x):
        # x = feats[-1]  # (64, H, W)
        kp_hm = self.kp_hm_head(x)
        kp_reg = self.kp_reg_head(x)
        ct_kp_reg = self.ct_kp_reg_head(x)
        return kp_hm, kp_reg, ct_kp_reg

    def _assign_keypoints_to_instance(self,
                                      ct_det_kps,
                                      kp_det_kps,
                                      det_bboxes,
                                      kp_scores,
                                      score_thr,
                                      delta_weight=0.3):
        assert kp_det_kps.size() == ct_det_kps.size()
        batch, num_joints, num_topk, dim = kp_det_kps.size()
        # > (B, #kps, K, K, 2)
        ct_det_kps = ct_det_kps.unsqueeze(3).expand(batch, num_joints, num_topk, num_topk, 2)
        kp_det_kps = kp_det_kps.unsqueeze(2).expand(batch, num_joints, num_topk, num_topk, 2)

        # NOTE: `argmin(ct_kps_loc - kp_kps_loc)^2`
        #  assign each regressed location to its closest detection keypoint
        min_dist, min_dist_ind = (ct_det_kps - kp_det_kps).pow(2).sum(dim=-1).pow(0.5).min(dim=-1)  # > (B, #kps, K)
        min_dist = min_dist.unsqueeze(-1)  # (B, #kps, K, 1)
        # (B, #kps, K) ∩ (B, #kps, K) -> (B, #kps, K) -> (B, 17, K, 1)
        kp_scores = kp_scores.gather(2, min_dist_ind).unsqueeze(-1)
        # `min_ind`: (B, #kps, K, 1) -> (B, #kps, K, 1, 1) -> (B, #kps, K, 1, 2)
        min_dist_ind = min_dist_ind[..., None, None].expand(batch, num_joints, num_topk, 1, 2)
        # (B, #kps, K, K, 2) -> (B, #kps, K, 1, 2) -> (B, #kps, K, 2)
        ct_det_kps = ct_det_kps.gather(3, min_dist_ind).view(batch, num_joints, num_topk, 2)
        kp_det_kps = kp_det_kps.gather(3, min_dist_ind).view(batch, num_joints, num_topk, 2)

        # NOTE: considering only joint detections within the bbox of detected object.
        l = det_bboxes[..., 0].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        t = det_bboxes[..., 1].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        r = det_bboxes[..., 2].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        b = det_bboxes[..., 3].view(batch, 1, num_topk, 1).expand(batch, num_joints, num_topk, 1)
        mask = (kp_det_kps[..., 0:1] < l) + (kp_det_kps[..., 0:1] > r) + \
               (kp_det_kps[..., 1:2] < t) + (kp_det_kps[..., 1:2] > b) + \
               (kp_scores < score_thr) + (min_dist > (torch.max(b - t, r - l) * delta_weight))
        mask = (mask > 0).float().expand(batch, num_joints, num_topk, 2)

        kps = (1 - mask) * kp_det_kps + mask * ct_det_kps
        kps = kps.permute(0, 2, 1, 3).contiguous()  # (B, #kps, K, 2) -> (B, K, #kps, 2)
        kp_scores = kp_scores.permute(0, 2, 1, 3)  # (B, K, #kps, 1)
        return kps, kp_scores

    def get_keypoints(self,
                      keypoint_outs,
                      bbox_metas,
                      img_metas,
                      rescale=False,
                      with_nms=True):
        pred_kp_hm, pred_kp_offset, pred_ct_kps = keypoint_outs
        _, bbox_scores, feat_bboxes, ct_inds, ct_xs, ct_ys, _ = bbox_metas
        batch, _, feat_h, feat_w = pred_kp_hm.size()
        num_topk = self.test_cfg.max_per_img
        num_joints = self.num_classes
        kp_hm = pred_kp_hm.detach().sigmoid_()  # > (B, #kps, H, W)
        kp_offset = pred_kp_offset.detach()  # > (B, 2, H, W)
        ct_kps = pred_ct_kps.detach()  # > (B, #kps x 2, H, W)

        # > keypoint heatmaps
        kp_hm = self._local_maximum(kp_hm, kernel=3)  # > (B, #kps, H, W)
        kp_scores, kp_inds, kp_ys, kp_xs = self._keypoint_topk(kp_hm, k=num_topk)

        # > keypoint regressed locations
        kp_offset = self._transpose_and_gather_feat(kp_offset, kp_inds.view(batch, -1))
        kp_offset = kp_offset.view(batch, num_joints, num_topk, 2)  # (B, #kps, K, 2)
        kp_xs += kp_offset[..., 0]
        kp_ys += kp_offset[..., 1]

        # > keep positives
        scores_mask = (kp_scores > self.test_cfg.kp_score_thr).float()  # (B, #kps, K)
        kp_scores = (1 - scores_mask) * -1 + scores_mask * kp_scores
        kp_xs = (1 - scores_mask) * (-10000) + scores_mask * kp_xs
        kp_ys = (1 - scores_mask) * (-10000) + scores_mask * kp_ys
        kp_kps = torch.stack([kp_xs, kp_ys], dim=-1)  # > (B, #kps, K, 2)

        # > offsets of center to keypoints
        ct_kps = self._transpose_and_gather_feat(ct_kps, ct_inds)  # (B, K, 34)
        ct_kps[..., 0::2] += ct_xs.unsqueeze(-1)
        ct_kps[..., 1::2] += ct_ys.unsqueeze(-1)
        # > (B, K, #kps, 2) -> (B, #kps, K, 2)
        ct_kps = ct_kps.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()

        # NOTE: keypoints are on feature-map scale
        det_kps, det_scores = self._assign_keypoints_to_instance(
            ct_kps, kp_kps, feat_bboxes, kp_scores, self.test_cfg.kp_score_thr)

        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self.get_keypoints_single(
                det_kps[img_id],
                det_scores[img_id],
                bbox_scores[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale,
                with_nms=with_nms
            ))
        return result_list, None

    def get_keypoints_single(self,
                             keypoints,
                             scores,
                             bbox_scores,
                             img_meta,
                             feat_shape,
                             rescale=False,
                             with_nms=True):
        num_topk = self.test_cfg.max_per_img
        size_stride = keypoints.new_tensor([float(img_meta['pad_shape'][1] / feat_shape[1]),  # > 4
                                            float(img_meta['pad_shape'][0] / feat_shape[0])])
        keypoints *= size_stride
        if rescale:
            scale_factor = img_meta['scale_factor'][:2]
            keypoints /= keypoints.new_tensor(scale_factor)

        keypoints = torch.cat([keypoints, scores], dim=2).view(num_topk, -1)  # (K, #kp x 3)
        # TOCHECK: add concat() to (K, 56) for softnms39().
        return keypoints

    def _keypoint_topk(self, scores, k=20):
        batch, cls, height, width = scores.size()
        # `scores`: (B, #cls, H, W) -> (B, #cls, HxW) -> (B, #cls, K)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), k)

        topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
        topk_ys = (topk_inds // width).float()  # (B, #cls, topk)
        topk_xs = (topk_inds % width).float()  # (B, #cls, topk)

        return topk_scores, topk_inds, topk_ys, topk_xs

    def preprocess(self, *inputs, training=False):
        if not self.with_share_convs:
            return inputs
        feats, preds = inputs
        """
        [IN] -> (
            x: backbone features
            preds: 
                bbox=(ct_hm, ct_offset, ct_wh)
                keypoint=(share_feats,)
        ) -> (
            share_feats,
            bbox=(ct_hm, ct_offset, ct_wh)
        )
        [OUT] -> keypoint_head(share_feats)
        """
        if not self.with_share_convs:
            return inputs
        feats = preds['keypoint']
        preds['keypoint'] = None
        return feats, preds

    def interprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """
            [IN] keypoint_head() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    keypoint=(kp_hm, kp_reg, ct_kp_reg)
                gts: (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
                metas: 
                    bbox=(feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            ) -> (
                (gt_keypoints, gt_labels),
                (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            )
            [OUT] -> get_targets(
                gt_inputs: (gt_keypoints, gt_labels)
                boxes_meta: (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            )
            """
            pred_outs, gt_inputs, metas = inputs
            gt_keypoints, gt_labels = gt_inputs[2], gt_inputs[3]
            boxes_meta = metas['bbox']
            return (gt_keypoints, gt_labels), boxes_meta
        else:
            # `TEST`
            """
            [IN] keypoint_head() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    keypoint=(kp_hm, kp_reg, ct_kp_reg)
                metas:
                    bbox=(scores, feat_bboxes, ct_inds, ct_xs, ct_ys)
                    keypoint=None
            ) -> (
                (kp_hm, kp_reg, ct_kp_reg)
                (clses, scores, feat_bboxes, ct_inds, ct_xs, ct_ys, ct_wh),
            )
            [OUT] -> get_keypoints(
                keypoint_outs: (kp_hm, kp_reg, ct_kp_reg)
                bbox_metas: (clses, scores, bboxes, ct_inds, ct_xs, ct_ys, ct_wh)
            )
            """
            pred_outs, metas = inputs
            keypoint_outs = pred_outs['keypoint']
            bbox_metas = metas['bbox']
            return keypoint_outs, bbox_metas

    def postprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """
            [IN] get_targets() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    keypoint=(kp_hm, kp_reg, ct_kp_reg)
                targets:
                    bbox=(hm_target, offset_target, wh_target, ct_target_mask, ct_ind_target)
                    keypoint=(hm_target, offset_target, ct2kps_target, offset_ind_target, offset_target_mask, ck2kps_target_mask)
                metas:
                    bbox=(feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
                    keypoint=None
            ) -> (
                kp_hm, kp_reg, ct_kp_reg,
                hm_target, offset_target, ct2kps_target, offset_ind_target, offset_target_mask, ck2kps_target_mask,
                ct_ind_target
            )
            [OUT] -> loss(
                pred_hm, pred_offset, pred_ct2kps,
                hm_target, offset_target, ct2kps_target, offset_ind_target, offset_target_mask, ck2kps_target_mask,
                ct_ind_target
            ) 
            """
            pred_outs, targets, metas = inputs
            keypoint_targets = targets['keypoint']
            bbox_ind_target = targets['bbox'][-1]
            new_inputs = pred_outs + keypoint_targets + (bbox_ind_target,)
        else:
            # `TEST`
            """
            [IN] get_keypoints() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    keypoint=(kp_hm, kp_reg, ct_kp_reg)
                metas: 
                    bbox=(clses, scores, bboxes, ct_inds, ct_xs, ct_ys, ct_wh)
                dets: (keypoints,)         
            )
            [OUT] (preds, dets) -> keypoint2result(keypoints)
            """
            new_inputs = inputs
        return new_inputs

    @force_fp32(apply_to=('pred_hm', 'pred_reg', 'pred_ct2kps'))
    def loss(self,
             pred_hm,  # (B, #joints, H, W)
             pred_offset,  # (B, 2, H, W)
             pred_ct2kps,  # (B, #joints x 2, H, W)
             hm_target,
             offset_target,
             ct2kps_target,
             offset_ind_target,
             offset_target_mask,
             ck2kps_target_mask,
             ct_ind_target,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        eps = 1e-12
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        # > keypoint heatmap loss
        loss_hm = self.loss_heatmap(pred_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        eps = 1e-4
        # > center2keypoint loss
        pred_ct2kps = self._transpose_and_gather_feat(pred_ct2kps, ct_ind_target)
        ct2kps_mask = ck2kps_target_mask.float()
        loss_joint = self.loss_joint(pred_ct2kps * ct2kps_mask,
                                     ct2kps_target * ct2kps_mask,
                                     avg_factor=ct2kps_mask.sum() + eps)

        # > offset loss
        pred_offset = self._transpose_and_gather_feat(pred_offset, offset_ind_target)
        offset_mask = offset_target_mask.unsqueeze(2).expand_as(pred_offset).float()
        loss_offset = self.loss_offset(pred_offset * offset_mask,
                                       offset_target * offset_mask,
                                       avg_factor=offset_mask.sum() + eps)
        return {'keypoint/loss_heatmap': loss_hm,
                'keypoint/loss_offset': loss_offset,
                'keypoint/loss_joint': loss_joint}

    def get_targets(self,
                    gt_inputs,
                    box_metas,
                    feat_shapes,
                    img_metas,
                    **kwargs):
        """
        Args:
            gt_inputs: (gt_keypoints, gt_labels).
            box_metas: (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            feat_shapes: tuple
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        gt_keypoints, gt_labels = gt_inputs
        feat_gt_centers_int, feat_gt_boxes_wh = box_metas[0], box_metas[1]
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]  # > (#level, 2)

        with torch.no_grad():
            heatmap, offset_target, ct2kps_target, offset_ind_target, offset_target_mask, ck2kps_target_mask = multi_apply(
                self._get_targets_single,
                gt_keypoints,
                feat_gt_centers_int,
                feat_gt_boxes_wh,  # (B, #obj, 2)
                img_shape=img_shape,  # (img_H, img_W)
                feat_shape=feat_shape  # (H, W)
            )
            heatmap = torch.stack(heatmap, dim=0)
            offset_target = torch.stack(offset_target, dim=0)
            ct2kps_target = torch.stack(ct2kps_target, dim=0)
            offset_ind_target = torch.stack(offset_ind_target, dim=0).to(torch.long)
            offset_target_mask = torch.stack(offset_target_mask, dim=0)
            ck2kps_target_mask = torch.stack(ck2kps_target_mask, dim=0)
            return (heatmap, offset_target, ct2kps_target,
                    offset_ind_target, offset_target_mask, ck2kps_target_mask), None

    def _get_targets_single(self,
                            gt_keypoints,
                            feat_gt_centers_int,
                            feat_gt_boxes_wh,
                            img_shape,
                            feat_shape,
                            **kwargs):
        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_keypoints: tensor, tensor <=> img, (num_gt, 3).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        size_strides = torch.tensor([float(img_shape[1] / feat_shape[1]),
                                     float(img_shape[0] / feat_shape[0])],
                                    dtype=gt_keypoints.dtype,
                                    device=gt_keypoints.device)
        joint_target = self._get_joint_target(gt_keypoints,
                                              feat_gt_centers_int,
                                              feat_gt_boxes_wh,
                                              feat_shape=feat_shape,
                                              size_strides=size_strides)
        return joint_target

    def _get_joint_target(self,
                          gt_keypoints,
                          feat_gt_centers_int,  # (#obj, 2)
                          feat_gt_boxes_wh,
                          feat_shape,
                          size_strides,
                          **kwargs):
        feat_h, feat_w = feat_shape
        num_objs = gt_keypoints.size(0)
        num_joints = self.num_classes
        hm_target = gt_keypoints.new_zeros(num_joints, feat_h, feat_w)  # > (#joint, H, W)
        offset_target = gt_keypoints.new_zeros(self.max_objs * num_joints, 2)  # > (#max_obj x #joint, 2)
        offset_ind_target = gt_keypoints.new_zeros(self.max_objs * num_joints, dtype=torch.int64)  # > (#max_obj x #joint,)
        offset_target_mask = gt_keypoints.new_zeros(self.max_objs * num_joints, dtype=torch.int64)  # > (#max_obj,)
        ct2kps_target = gt_keypoints.new_zeros(self.max_objs, num_joints * 2)  # > (#max_obj, #joint x 2)
        ct2kps_target_mask = gt_keypoints.new_zeros(self.max_objs, num_joints * 2, dtype=torch.uint8)  # > (#max_obj,)

        feat_gt_keypoints = gt_keypoints.clone()  # > (#obj, #joints(x,y,v))
        feat_gt_keypoints[..., :2] /= size_strides

        for obj_id in range(num_objs):
            feat_obj_keypoints = feat_gt_keypoints[obj_id]
            kp_radius = gaussian_radius(feat_gt_boxes_wh[obj_id],
                                        min_overlap=self.train_cfg.min_overlap)
            kp_radius = max(0, int(kp_radius))
            for joint_id in range(num_joints):
                if feat_obj_keypoints[joint_id, 2] > 0:
                    joint_xy = feat_obj_keypoints[joint_id, :2]
                    joint_xy_int = joint_xy.to(torch.int)
                    if 0 <= joint_xy[0] < feat_w and 0 <= joint_xy[1] < feat_h:
                        start_ind, end_ind = joint_id * 2, joint_id * 2 + 2
                        obj_joint_ind = obj_id * num_joints + joint_id
                        offset_target[obj_joint_ind] = joint_xy - joint_xy_int
                        offset_ind_target[obj_joint_ind] = joint_xy_int[1] * feat_h + joint_xy_int[0]
                        offset_target_mask[obj_joint_ind] = 1
                        ct2kps_target[obj_id, start_ind:end_ind] = joint_xy - feat_gt_centers_int[obj_id]
                        ct2kps_target_mask[obj_id, start_ind:end_ind] = 1
                        hm_target[joint_id] = gen_gaussian_target(hm_target[joint_id],
                                                                  joint_xy_int,
                                                                  radius=int(kp_radius))
        return hm_target, offset_target, ct2kps_target, offset_ind_target, offset_target_mask, ct2kps_target_mask

    def _get_limb_target(self,
                         gt_keypoints,
                         feat_shape,
                         size_strides):
        num_objs = gt_keypoints.size(0)
        feat_gt_keypoints = gt_keypoints.clone()  # > (#obj, #joints(x,y,v))
        feat_gt_keypoints[..., :2] /= size_strides
        limb_target = feat_gt_keypoints.new_zeros(self.max_objs, len(self.limbs))  # > (#max_obj, #limb)
        limb_target_mask = feat_gt_keypoints.new_zeros(self.max_objs, len(self.limbs), dtype=torch.uint8)  # > (#max_obj,)
        feat_gt_limbs = feat_gt_keypoints[:, self.limbs, :2]  # (B, #limbs, (fr, to), (x, y))
        feat_gt_limbs_length = (feat_gt_limbs[:, :, 1] - feat_gt_limbs[:, :, 0]).pow(2).sum(dim=-1).sqrt()  # (#obj, #limbs)
        keep = (feat_gt_limbs[:, :, 0] > 0).all(dim=-1) & (feat_gt_limbs[:, :, 1] > 0).all(dim=-1)  # (#obj, #limbs)
        feat_gt_limbs_length[~keep] = 0  # (#obj, #limbs)
        limb_target[:num_objs] = feat_gt_limbs_length
        limb_target_mask[:num_objs] = keep.to(torch.int32)
        # TOADD: limb_target_inds
        # TOCHECK: 透過pred_kps_hm計算長度，再算loss
        # TOCHECK: 透過pred_ct2kps計算長度，再算loss
        # TOCHECK: 透過pred_kps_hm計算線中心點，再算loss
        # TOCHECK: 透過pred_ct2kps計算線中心點，再算loss
        # obj_joint_ind = obj_id * num_joints + joint_id
        # ct2kps_target[obj_id, start_ind:end_ind] = joint_xy - feat_gt_centers_int[obj_id]
        return limb_target, limb_target_mask


@HEADS.register_module()
class CenterMaskHead(CornerHead):

    def __init__(self,
                 *args,
                 stacked_convs=1,
                 feat_channels=256,
                 saliency_channels=1,
                 shape_channels=64,
                 max_objs=128,
                 resize_method='bilinear',
                 rescale_ratio=1.0,
                 weight_init='xavier',
                 upsample_cfg=dict(
                     type='bilinear',
                     scale_factor=1.0,
                 ),
                 with_share_convs=False,
                 loss_mask=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.saliency_channels = saliency_channels
        self.shape_channels = shape_channels
        self.shape_dim = int(math.sqrt(shape_channels))
        self.max_objs = max_objs
        self.resize_method = resize_method
        self.rescale_ratio = rescale_ratio
        self.weight_init = weight_init
        self.with_share_convs = with_share_convs
        self.with_upsample = upsample_cfg is not None
        self.upsample_cfg = upsample_cfg
        super(CenterMaskHead, self).__init__(*args, **kwargs)
        self.loss_mask = build_loss(loss_mask)
        self.loss_embedding = None
        self.fp16_enabled = False

    def build_upsample(self, feat_channel, upsample_cfg):
        _upsample = upsample_cfg.get('type')
        assert _upsample in [
            'nearest', 'bilinear', 'deconv', 'pixel_shuffle', 'carafe', None
        ]
        if _upsample in ['deconv', 'pixel_shuffle']:
            assert hasattr(
                upsample_cfg,
                'upsample_kernel') and upsample_cfg.upsample_kernel > 0
            upsample_kernel = upsample_cfg.pop('upsample_kernel')
        upsample_cfg_ = upsample_cfg.copy()
        if _upsample == 'deconv':
            upsample_cfg_.update(
                in_channels=feat_channel,
                out_channels=feat_channel,
                kernel_size=upsample_kernel,
                stride=2,
                padding=(upsample_kernel - 1) // 2,
                output_padding=(upsample_kernel - 1) // 2)
        elif _upsample == 'pixel_shuffle':
            upsample_cfg_.update(
                in_channels=feat_channel,
                out_channels=feat_channel,
                upsample_kernel=upsample_kernel)
        elif _upsample == 'carafe':
            upsample_cfg_.update(channels=feat_channel)
        else:
            # suppress warnings
            align_corners = (None
                             if _upsample == 'nearest' else False)
            upsample_cfg_.update(
                mode=_upsample,
                align_corners=align_corners)
        return build_upsample_layer(upsample_cfg_)

    def _init_layers(self):
        feat_channels = self.feat_channels
        stacked_convs = self.stacked_convs
        head_in_channel = feat_channels if self.with_share_convs else self.in_channels
        self.saliency_head = build_head(
            head_in_channel, feat_channels, stacked_convs, self.saliency_channels)
        self.shape_head = build_head(
            head_in_channel, feat_channels, stacked_convs, self.shape_channels)
        if self.with_upsample:
            self.salient_upsample = self.build_upsample(self.saliency_channels, self.upsample_cfg)

    def init_weights(self):
        self._init_head_weights(self.saliency_head)
        self._init_head_weights(self.shape_head)
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()

    def _init_head_weights(self, layer):
        for _, m in layer.named_modules():
            if isinstance(m, nn.Conv2d):
                if self.weight_init == 'xavier':
                    xavier_init(m, distribution='uniform')
                elif self.weight_init == 'normal':
                    normal_init(m, std=0.001)
                else:
                    raise NotImplementedError

    def forward(self, x):  # `feat_h/feat_w`: 128, `S`: 32
        global_saliency = self.saliency_head(x)  # (B, 1, feat_h, feat_w)
        local_shape = self.shape_head(x)  # (B, SxS, feat_h, feat_w)
        if self.with_upsample:
            global_saliency = self.salient_upsample(global_saliency)
        return global_saliency, local_shape

    def get_segm_masks(self,
                       mask_outs,
                       bbox_metas,
                       img_metas,
                       rescale=False,
                       with_nms=True):
        pred_ct_saliency, pred_ct_shape = mask_outs
        clses, scores, bboxes, ct_inds, ct_xs, ct_ys, ct_wh = bbox_metas
        batch, shape_dim, feat_h, feat_w = pred_ct_shape.size()
        num_topk = self.test_cfg.max_per_img
        ct_saliency = pred_ct_saliency.detach().sigmoid_()  # (B, 1, H, W)
        ct_shape = pred_ct_shape.detach().sigmoid_()  # (B, SxS, H, W)

        ct_shape = self._transpose_and_gather_feat(ct_shape, ct_inds)  # (B, SxS, H, W) -> (B, K, SxS)
        ct_shape = ct_shape.view(batch, num_topk, self.shape_dim, self.shape_dim)  # (B, K, S, S)

        result_list = []
        for img_id in range(len(img_metas)):  # > #img
            result_list.append(self.get_segm_masks_single(
                ct_saliency[img_id],
                ct_shape[img_id],
                bboxes[img_id],
                clses[img_id],
                scores[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale,
                with_nms=with_nms
            ))
        return result_list, None

    def get_segm_masks_single(self,
                              pred_saliency,  # (1, H, W)
                              pred_shape,  # (K, S, S)
                              pred_bboxes,  # (K, 4)
                              pred_labels,
                              pred_scores,
                              img_meta,
                              feat_shape,
                              rescale=False,
                              with_nms=True):
        pred_masks = self.assemble_masks(pred_saliency, pred_shape, pred_bboxes)

        img_h, img_w = ori_shape = img_meta['ori_shape'][:2]
        pad_shape = img_meta['pad_shape']
        scale_factor = img_meta['scale_factor']
        if rescale:
            pad_h, pad_w = pad_shape[:2]
        else:
            pad_h = np.round(ori_shape[0] * scale_factor[1]).astype(np.int32)
            pad_w = np.round(ori_shape[1] * scale_factor[0]).astype(np.int32)

        cls_segms = [[] for _ in range(self.num_classes)]
        if len(pred_masks) == 0:
            return cls_segms

        # upsample masks to image-scale
        pred_masks = F.interpolate(
            pred_masks.unsqueeze(0), (pad_h, pad_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > self.test_cfg.mask_score_thr
        pred_masks = pred_masks[:, :img_h, :img_w]

        if isinstance(pred_masks, torch.Tensor):
            pred_masks = pred_masks.cpu().numpy().astype(np.uint8)
            pred_labels = pred_labels.squeeze().cpu().numpy().astype(np.int32)

        for m, l in zip(pred_masks, pred_labels):
            cls_segms[l].append(m)
        return cls_segms

    def preprocess(self, *inputs, training=False):
        if not self.with_share_convs:
            return inputs
        feats, preds = inputs
        """
        [IN] -> (
            x: backbone features
            preds: 
                bbox=(ct_hm, ct_offset, ct_wh)
                mask=(share_feats,)
        ) -> (
            share_feats,
            bbox=(ct_hm, ct_offset, ct_wh)
        )
        [OUT] -> mask_head(share_feats)
        """
        if not self.with_share_convs:
            return inputs
        feats = preds['mask']
        preds['mask'] = None
        return feats, preds

    def interprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`: `with_share_convs` is enabled
            """
            [IN] mask_head() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    mask=(ct_saliency, ct_shape)
                gts: (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
                metas: 
                    bbox=(feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            ) -> (
                (gt_bboxes, gt_masks, gt_labels),
                (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            )
            [OUT] -> get_targets(
                gt_inputs: (gt_bboxes, gt_masks, gt_labels)
                boxes_meta: (feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
            )
            """
            pred_outs, gt_inputs, metas = inputs
            gt_bboxes, gt_masks, gt_labels = gt_inputs[0], gt_inputs[1], gt_inputs[3]
            boxes_meta = metas['bbox']
            return (gt_bboxes, gt_masks, gt_labels), boxes_meta
        else:
            # `TEST`
            """
            [IN] mask_head() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    mask=(ct_saliency, ct_shape)
                metas:
                    bbox=(scores, feat_bboxes, ct_inds, ct_xs, ct_ys, ct_wh)
                    mask=None
            ) -> (
                (ct_saliency, ct_shape)
                (clses, scores, feat_bboxes, ct_inds, ct_xs, ct_ys, ct_wh),
            )
            [OUT] -> get_masks(
                mask_outs: (ct_saliency, ct_shape)
                bbox_metas: (clses, scores, bboxes, ct_inds, ct_xs, ct_ys, ct_wh)
            )
            """
            pred_outs, metas = inputs
            mask_outs = pred_outs['mask']
            bbox_metas = metas['bbox']
            return mask_outs, bbox_metas

    def postprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """
            [IN] get_targets() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    mask=(ct_saliency, ct_shape)
                targets:
                    bbox=(hm_target, offset_target, wh_target, ct_target_mask, ct_ind_target)
                    mask=(boxes_target, mask_target)
                metas:
                    bbox=(feat_gt_centers_int, feat_gt_boxes_wh, feat_gt_boxes)
                    mask=None
            ) -> (
                ct_saliency, ct_shape,
                boxes_target, mask_target
                ct_ind_target
            )
            [OUT] -> loss(
                pred_ct_saliency, pred_ct_shape,
                boxes_target, mask_target
                box_target, ct_ind_target
            ) 
            """
            pred_outs, targets, metas = inputs
            mask_targets = targets['mask']
            bbox_ind_target = targets['bbox'][-1]
            new_inputs = pred_outs + mask_targets + (bbox_ind_target,)
        else:
            # `TEST`
            """
            [IN] get_segm_masks() -> (
                preds:
                    bbox=(ct_hm, ct_offset, ct_wh)
                    mask=(ct_saliency, ct_shape)
                metas: 
                    bbox=(clses, scores, bboxes, ct_inds, ct_xs, ct_ys, ct_wh)
                dets: (masks,)         
            )
            [OUT] (preds, dets) -> mask2result(masks)
            """
            new_inputs = inputs
        return new_inputs

    def _crop_masks(self, masks, boxes, pad_ratio=0.0, padding=1):
        """
        Args:
            masks(Tensor): (N, H, W)
            boxes(Tensor): bbox coords (N, 4)
            pad_ratio(float): crop by a ratio of a box
            padding(int): padding on width and height boundary
        Return:
            Tensor: the cropped masks
        """
        if masks.size(0) == boxes.size(0):
            n, h, w = masks.size()
        else:
            _, h, w = masks.size()
            n = boxes.size(0)

        box_w = boxes[..., 2] - boxes[..., 0]
        box_h = boxes[..., 3] - boxes[..., 1]
        x1 = torch.clamp(boxes[..., 0] - box_w * pad_ratio - padding, min=0).unsqueeze(1)
        x2 = torch.clamp(boxes[..., 2] + box_w * pad_ratio - padding, max=w).unsqueeze(1)
        y1 = torch.clamp(boxes[..., 1] - box_h * pad_ratio + padding, min=0).unsqueeze(1)
        y2 = torch.clamp(boxes[..., 3] + box_h * pad_ratio + padding, max=h).unsqueeze(1)

        rows = torch.arange(w, device=x1.device, dtype=x1.dtype).view(1, 1, -1).expand(n, h, w)
        cols = torch.arange(h, device=x1.device, dtype=x1.dtype).view(1, -1, 1).expand(n, h, w)

        left = rows >= x1.view(-1, 1, 1)
        right = rows < x2.view(-1, 1, 1)
        up = cols >= y1.view(-1, 1, 1)
        down = cols < y2.view(-1, 1, 1)

        crop_mask = left * right * up * down  # (N, H, W)
        return crop_mask

    def _masks2bboxes(self, masks, padding=0):
        """
        SEE: https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/utils.py
        Args:
            masks: (N, H, W)
        Return:
            boxes: (N, (x1, y1, x2, y2))
        """
        n = masks.size(0)
        boxes = masks.new_zeros((n, 4), dtype=torch.float32)
        for i in range(n):
            m = masks[i].bool()  # > (H, W)
            rows = torch.where(torch.any(m, dim=0))[0]
            cols = torch.where(torch.any(m, dim=1))[0]
            if rows.size(0):
                x1, x2 = rows[[0, -1]]
                y1, y2 = cols[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                boxes[i, 0] = x1
                boxes[i, 1] = y1
                boxes[i, 2] = x2 + padding
                boxes[i, 3] = y2 + padding
        return boxes

    def assemble_masks(self, ct_saliency, ct_shape, crop_boxes, training=False):
        """
        When `TRAINING` this is called in `loss()`:
            ct_saliency(Tensor): shape (feat_h, feat_w)
            ct_shape(Tensor): shape (#max_obj, S, S)
            crop_boxes(Tensor): shape (#obj, 4)
        When `TESTING` this is called in `get_segm_masks_single()`:
            ct_saliency(Tensor): shape (feat_h, feat_w)
            ct_shape(Tensor): shape (#num_topk, S, S)
            crop_boxes(Tensor): shape (#num_topk, 4)
        """
        num_obj = crop_boxes.size(0)  # > #obj or topk
        feat_h, feat_w = ct_saliency.size()[1:]
        if training and num_obj == 0:
            return ct_saliency.new_zeros((1, feat_h, feat_w), dtype=torch.float32)
        if self.with_upsample:
            crop_boxes = crop_boxes * self.upsample_cfg.get('scale_factor', 2.0)
        # cropping salient maps by bboxes
        crop_masks = self._crop_masks(ct_saliency, crop_boxes, padding=0)
        crop_salient_maps = ct_saliency.expand(num_obj, feat_h, feat_w) * crop_masks.float()
        crop_boxes = self._masks2bboxes(crop_masks, padding=1).long()
        if training:
            pred_masks = []
            for obj_id in range(num_obj):
                x1, y1, x2, y2 = crop_boxes[obj_id].tolist()
                min_x, min_y, max_x, max_y = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                box_w, box_h = max_x - min_x, max_y - min_y
                if box_w > 0 and box_h > 0:
                    max_x, max_y = x1 + box_w, y1 + box_h
                    # crop salient map by bbox
                    crop_saliency = crop_salient_maps[obj_id, min_y:max_y, min_x:max_x].view(box_h, box_w)
                    # resize local shape by bbox size
                    crop_shape = self._resize_masks(ct_shape[obj_id].unsqueeze(0), (box_h, box_w),
                                                    method=self.resize_method, squeeze=True)  # (box_h, box_w)
                    # NOTE: perform `Hadamard product` to form final mask, see paper 3.3 Mask Assembly
                    pred_masks.append(torch.einsum('ij,ij->ij', [crop_saliency, crop_shape]))
                else:
                    pred_masks.append(crop_boxes.new_zeros((1, 1)))
        else:
            crop_local_shapes = ct_saliency.new_zeros((num_obj, feat_h, feat_w), dtype=torch.float32)
            for obj_id in range(num_obj):
                x1, y1, x2, y2 = crop_boxes[obj_id].tolist()
                min_x, min_y, max_x, max_y = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                box_w, box_h = max_x - min_x, max_y - min_y
                if box_w > 0 and box_h > 0:
                    crop_shape = self._resize_masks(ct_shape[obj_id].unsqueeze(0), (box_h, box_w),
                                                    method=self.resize_method, squeeze=True)  # (box_h, box_w)
                    crop_local_shapes[obj_id, min_y:max_y, min_x:max_x] = crop_shape
            # NOTE: perform `Hadamard product` to form final masks, see paper 3.3 Mask Assembly
            pred_masks = torch.einsum('bij,bij->bij', [crop_salient_maps, crop_local_shapes])
        return pred_masks

    def loss(self,
             pred_ct_saliency,  # (B, 1, H, W)
             pred_ct_shape,  # (B, SxS, H, W)
             boxes_target,  # (B, #obj, 4)
             masks_target,  # (B, #obj, H, W)
             ct_ind_target,  # (B, #max_obj)
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        eps = 1e-12
        pred_ct_saliency = torch.clamp(pred_ct_saliency.sigmoid_(), min=eps, max=1 - eps)
        pred_ct_shape = torch.clamp(pred_ct_shape.sigmoid_(), min=eps, max=1 - eps)
        batch, max_obj = ct_ind_target.size()
        pred_ct_shape = self._transpose_and_gather_feat(pred_ct_shape, ct_ind_target)  # (B, SxS, H, W) -> (B, K, SxS)
        pred_ct_shape = pred_ct_shape.view(batch, max_obj, self.shape_dim, self.shape_dim)

        loss_mask, num_objs = 0, 0
        for img_id, ct_saliency, ct_shape, box_target, mask_target in \
                zip(range(batch), pred_ct_saliency, pred_ct_shape, boxes_target, masks_target):  # > #img
            num_obj = box_target.size(0)
            # > 1 salient map vs N bboxes
            pred_masks = self.assemble_masks(ct_saliency, ct_shape, box_target, training=True)  # (#obj, H, W)
            if num_obj == 0:
                loss_mask += pred_masks.sum() * 0.
            else:
                for obj_id in range(num_obj):
                    if mask_target[obj_id].max().item() != 0:
                        loss_mask += self.loss_mask(pred_masks[obj_id], mask_target[obj_id])
            num_objs += num_obj
        loss_mask /= (num_objs + eps)
        return {'mask/loss_mask': loss_mask}

    def get_targets(self,
                    gt_inputs,
                    box_metas,
                    feat_shapes,
                    img_metas,
                    **kwargs):
        gt_boxes, gt_masks, gt_labels = gt_inputs
        feat_gt_boxes = box_metas[-1]
        feat_shape = feat_shapes[0]
        with torch.no_grad():
            feat_gt_boxes, feat_gt_masks = multi_apply(
                self._get_targets_single,
                gt_boxes,
                gt_masks,
                feat_gt_boxes,
                img_metas,
                feat_shape=feat_shape
            )

            return (feat_gt_boxes, feat_gt_masks), None

    def _get_targets_single(self,
                            gt_boxes,  # (#obj, 4)
                            gt_masks,  # (#obj, H, W)
                            feat_gt_boxes,
                            img_meta,
                            feat_shape):
        num_obj = gt_boxes.size(0)
        pad_h, pad_w = img_meta['pad_shape'][:2]
        width_stride = float(pad_w / feat_shape[1])  # > 4
        height_stride = float(pad_h / feat_shape[0])
        gt_masks = gt_masks.float()

        if num_obj == 0:
            feat_gt_masks = feat_gt_boxes.new_zeros(0, feat_shape[0], feat_shape[1], dtype=torch.float32)
        else:
            feat_gt_masks = self._resize_masks(gt_masks, feat_shape, method=self.resize_method)
        # get bboxes from gt masks
        bboxes_target = self._masks2bboxes(gt_masks, padding=1)  # (#obj, 4)
        bboxes_target[:, 0::2] = bboxes_target[:, 0::2] / width_stride
        bboxes_target[:, 1::2] = bboxes_target[:, 1::2] / height_stride
        bboxes_target[:, [0, 1]] = bboxes_target[:, [0, 1]].floor()
        bboxes_target[:, [2, 3]] = bboxes_target[:, [2, 3]].ceil()
        bboxes_target[:, [0, 2]] = torch.clamp(bboxes_target[:, [0, 2]], min=0, max=feat_shape[1])
        bboxes_target[:, [1, 3]] = torch.clamp(bboxes_target[:, [1, 3]], min=0, max=feat_shape[0])
        bboxes_target = bboxes_target.long()

        feat_bboxes_target = self._masks2bboxes(feat_gt_masks, padding=1)  # (#obj, 4)
        feat_bboxes_target[:, 0::2] = torch.clamp(feat_bboxes_target[:, 0::2], max=feat_shape[1])
        feat_bboxes_target[:, 1::2] = torch.clamp(feat_bboxes_target[:, 1::2], max=feat_shape[0])
        feat_bboxes_target = feat_bboxes_target.long()

        masks_target, feat_masks_target = [], []
        for obj_id, bbox, feat_bbox, feat_mask in \
                zip(range(num_obj), bboxes_target, feat_bboxes_target, feat_gt_masks):  # > #obj
            # NOTE: [1] use `gt_masks` to get `bboxes` and then downsize to get `feat_bboxes`
            x1, y1, x2, y2 = bbox.tolist()
            box_w, box_h = max(x2 - x1, 1), max(y2 - y1, 1)
            crop_mask = feat_mask[y1:y1+box_h, x1:x1+box_w]  # (crop_h, crop_w)
            masks_target.append(crop_mask.float())

            # NOTE: [2] use `feat_gt_masks` to get `feat_bboxes`
            x1, y1, x2, y2 = feat_bbox.tolist()
            box_w, box_h = max(x2 - x1, 1), max(y2 - y1, 1)
            crop_mask = feat_mask[y1:y1+box_h, x1:x1+box_w]  # (crop_h, crop_w)
            feat_masks_target.append(crop_mask.float())
        masks_target = feat_masks_target
        bboxes_target = feat_bboxes_target
        return bboxes_target, masks_target

    def _resize_masks(self, masks, shape, method='bilinear', squeeze=False):
        assert len(masks.size()) >= 2
        if len(masks.size()) == 2:
            n, h, w = 1, masks.size(0), masks.size(1)
        else:
            n, h, w = masks.size()[-3:]
        if len(masks.size()) != 4:
            masks = masks.view(1, n, h, w)
        if method == 'gap':
            masks = F.adaptive_avg_pool2d(masks, shape)
        elif method == 'gmp':
            masks = F.adaptive_max_pool2d(masks, shape)
        elif method == 'bilinear':
            masks = F.interpolate(
                masks, shape, mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError

        masks = masks.view(n, shape[0], shape[1])
        if squeeze:
            return masks.view(shape[0], shape[1])
        return masks