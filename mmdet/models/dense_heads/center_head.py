import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.runner import force_fp32
import numpy as np
import math
from mmdet.core import multi_apply, calc_region
from ..builder import HEADS, build_loss
from .corner_head import CornerHead
from ..utils import gaussian_radius, gen_gaussian_target


@HEADS.register_module
class CenterHead(CornerHead):

    def __init__(self,
                 *args,
                 stacked_convs=1,
                 feat_channels=256,
                 max_objs=128,
                 loss_bbox=dict(type='L1Loss', loss_weight=0.1),
                 **kwargs):
        self.max_objs = max_objs
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        super(CenterHead, self).__init__(*args, **kwargs)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_embedding = None
        self.fp16_enabled = False

    def build_head(self, in_channel, feat_channel, stacked_convs, out_channel):
        head_convs = [ConvModule(
            in_channel, feat_channel, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True))]
        for i in range(1, stacked_convs):
            head_convs.append(ConvModule(
                feat_channel, feat_channel, 3,
                padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))

        head_convs.append(nn.Conv2d(feat_channel, out_channel, 1))
        return nn.Sequential(*head_convs)

    def _init_layers(self):
        self.ct_hm_head = self.build_head(
            self.in_channels, self.feat_channels, self.stacked_convs, self.num_classes)
        self.ct_reg_head = self.build_head(
            self.in_channels, self.feat_channels, self.stacked_convs, 2)
        self.ct_wh_head = self.build_head(
            self.in_channels, self.feat_channels, self.stacked_convs, 2)

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

    def forward(self, x):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        ct_hm = self.ct_hm_head(x)  # > (B, #cls, 128, 128)
        ct_offset = self.ct_reg_head(x)  # > (B, 2, 128, 128)
        ct_wh = self.ct_wh_head(x)  # > (B, 2, 128, 128)
        return ct_hm, ct_offset, ct_wh

    def get_bboxes(self,
                   pred_ct_cls, pred_ct_offset, pred_ct_wh,
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

        # > `bboxes`: (B, topk, 4)
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
        return result_list, (scores, feat_bboxes, ct_inds, ct_xs, ct_ys)

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_size,
                          rescale=False,
                          with_nms=True):
        # `bboxes`: (B, K, 4), `bbox_scores`: (B, K, 1)
        pad_h, pad_w = img_meta['pad_shape'][:2]
        width_ratio = float(feat_size[1] / pad_w)
        height_ratio = float(feat_size[0] / pad_h)
        bboxes[:, 0::2] = (bboxes[:, 0::2] / width_ratio).clamp(min=0, max=pad_w - 1)
        bboxes[:, 1::2] = (bboxes[:, 1::2] / height_ratio).clamp(min=0, max=pad_h - 1)
        if rescale:
            scale_factor = img_meta['scale_factor']
            bboxes /= bboxes.new_tensor(scale_factor)
        bboxes = torch.cat([bboxes, scores], dim=1)  # (K, 4 + 1)
        labels = labels.squeeze(-1)  # (K, 1) -> (K,)
        # TOCHECK: `self._bboxes_nms` in `Cornerhead`
        return bboxes, labels

    def postprocess(self, *inputs, return_loss=False):
        if return_loss:
            pred_outs, gt_inputs, gt_metas, img_metas = inputs
            gt_box_input, gt_box_meta = gt_inputs['bbox'], gt_metas['bbox']
            new_inputs = pred_outs + gt_box_input + (gt_box_meta, img_metas,)
        else:
            head_outs, bbox_list = inputs
            new_bbox_list, bbox_metas = bbox_list
            head_outs['bbox'] = head_outs['bbox'] + bbox_metas
            new_inputs = (head_outs, new_bbox_list)
        return new_inputs

    @force_fp32(apply_to=('pred_heatmap', 'pred_reg', 'pred_wh'))
    def loss(self,
             pred_hm,  # (B, #cls, H, W)
             pred_reg,  # (B, 2, H, W)
             pred_wh,  # (B, 2, H, W)
             gt_hm,
             gt_reg,
             gt_wh,
             gt_inds,
             gt_mask,
             gt_metas,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        eps = 1e-12
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = gt_hm.eq(1).float().sum()
        # > class loss
        loss_cls = self.loss_heatmap(pred_hm, gt_hm, avg_factor=1 if num_pos == 0 else num_pos)
        gt_mask = gt_mask.unsqueeze(2)

        eps = 1e-4
        # > width/height loss
        pred_wh = self._transpose_and_gather_feat(pred_wh, gt_inds)
        wh_mask = gt_mask.expand_as(pred_wh).float()
        loss_wh = self.loss_bbox(pred_wh * wh_mask,
                                 gt_wh * wh_mask,
                                 avg_factor=wh_mask.sum() + eps)

        # > offset loss
        pred_reg = self._transpose_and_gather_feat(pred_reg, gt_inds)
        reg_mask = gt_mask.expand_as(pred_reg).float()
        loss_reg = self.loss_offset(pred_reg * reg_mask,
                                    gt_reg * reg_mask,
                                    avg_factor=reg_mask.sum() + eps)
        return {'bbox/loss_heatmap': loss_cls, 'bbox/loss_reg': loss_reg, 'bbox/loss_wh': loss_wh}

    def get_targets(self,
                    gt_inputs,
                    gt_metas,
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
        gt_boxes, gt_labels = gt_inputs[0], gt_inputs[3]
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]
        with torch.no_grad():
            # `heatmap`: (B, #cls, H, W), `wh_target`: (B, 2, H, W), `reg_target`: (B, 2, H, W)
            heatmap, reg_target, wh_target, ind_target, target_mask, feat_gt_centers_int, gt_radius = \
                multi_apply(
                    self.target_single_image,
                    gt_boxes,  # (B, #obj, 4)
                    gt_labels,  # (B, #obj)
                    img_shape=img_shape,  # (img_H, img_W)
                    feat_shape=feat_shape  # (H, W)
                )
            heatmap = torch.stack(heatmap, dim=0)
            reg_target = torch.stack(reg_target, dim=0)
            wh_target = torch.stack(wh_target, dim=0)
            ind_target = torch.stack(ind_target, dim=0).to(torch.long)
            target_mask = torch.stack(target_mask, dim=0)
            return (heatmap, reg_target, wh_target, ind_target, target_mask), (feat_gt_centers_int, gt_radius)

    def target_single_image(self,
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
        gt_heatmap = gt_boxes.new_zeros(self.num_classes, feat_h, feat_w)  # > (#cls, H, W)
        gt_reg = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        gt_wh = gt_boxes.new_zeros(self.max_objs, 2)  # > (#max_obj, 2)
        gt_inds = gt_boxes.new_zeros(self.max_objs, dtype=torch.int64)  # > (#max_obj,)
        gt_mask = gt_boxes.new_zeros(self.max_objs, dtype=torch.uint8)  # > (#max_obj,)

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
        gt_radius = gt_boxes.new_zeros(num_objs, dtype=torch.int32)
        for box_id in range(num_objs):
            radius = gaussian_radius(feat_gt_wh[box_id], min_overlap=self.train_cfg.min_overlap)
            radius = max(0, int(radius))
            cls_ind = gt_labels[box_id]
            gt_radius[box_id] = radius
            gt_heatmap[cls_ind] = gen_gaussian_target(heatmap=gt_heatmap[cls_ind],
                                                      center=feat_gt_centers_int[box_id],
                                                      radius=radius)
        return gt_heatmap, gt_reg, gt_wh, gt_inds, gt_mask, feat_gt_centers_int, gt_radius


@HEADS.register_module()
class CenterPoseHead(CornerHead):

    def __init__(self,
                 *args,
                 feat_channels=256,
                 max_objs=128,
                 upsample_cfg=None,
                 loss_joint=None,
                 **kwargs):
        self.feat_channels = feat_channels
        self.upsample_cfg = upsample_cfg
        self.max_objs = max_objs
        super(CenterPoseHead, self).__init__(*args, **kwargs)
        self.loss_joint = build_loss(loss_joint)
        self.loss_embedding = None
        self.fp16_enabled = False
        self.limbs = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

    def build_head(self, out_channel):
        head_convs = [ConvModule(
            self.in_channels, self.feat_channels, 3,
            padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True))]
        for i in range(1, self.num_feat_levels):
            head_convs.append(ConvModule(
                self.feat_channels, self.feat_channels, 3,
                padding=1, bias=True, act_cfg=dict(type='ReLU', inplace=True)))

        head_convs.append(nn.Conv2d(self.feat_channels, out_channel, 1))
        return nn.Sequential(*head_convs)

    def _init_layers(self):
        self.kp_hm_head = self.build_head(self.num_classes)
        self.kp_reg_head = self.build_head(2)
        self.ct_kp_reg_head = self.build_head(self.num_classes * 2)

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

    def get_keypoints(self,
                      bbox_outs,
                      keypoint_outs,
                      img_metas,
                      rescale=False,
                      with_nms=True):
        bbox_scores, feat_bboxes, ct_inds, ct_xs, ct_ys = bbox_outs
        pred_kp_hm, pred_kp_reg, pred_ct_kp_reg = keypoint_outs
        batch, _, feat_h, feat_w = pred_kp_hm.size()
        num_topk = self.test_cfg.max_per_img
        num_joints = self.num_classes
        kp_hm = pred_kp_hm.detach().sigmoid_()  # > (B, #kp, H, W)
        kp_offset = pred_kp_reg.detach()  # > (B, 2, H, W)
        ct_kp_loc = pred_ct_kp_reg.detach()  # > (B, #kp x 2, H, W)

        # > offsets of center to keypoints
        ct_kp_loc = self._transpose_and_gather_feat(ct_kp_loc, ct_inds)
        ct_kp_loc = ct_kp_loc.view(batch, num_topk, num_joints * 2)
        ct_kp_loc[..., 0::2] += ct_xs.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct_kp_loc[..., 1::2] += ct_ys.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct_kp_loc = ct_kp_loc.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()
        ct_det_kps = ct_kp_loc.unsqueeze(3).expand(batch, num_joints, num_topk, num_topk, 2)

        # > keypoint heatmaps
        kp_hm = self._local_maximum(kp_hm, kernel=3)
        kp_scores, kp_inds, kp_ys, kp_xs = self._kepoint_topk(kp_hm, k=num_topk)

        # > keypoint regressed locations
        kp_offset = self._transpose_and_gather_feat(kp_offset, kp_inds.view(batch, -1))
        kp_offset = kp_offset.view(batch, num_joints, num_topk, 2)
        kp_xs += kp_offset[..., 0]
        kp_ys += kp_offset[..., 1]

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

        kps = (1 - mask) * kp_det_kps + mask * ct_kp_loc
        kps = kps.permute(0, 2, 1, 3).contiguous().view(batch, num_topk, num_joints * 2)
        kp_scores = torch.transpose(kp_scores.squeeze(dim=3), 1, 2)  # (B, K, #kp)

        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self.get_keypoints_single(
                kps[img_id],
                kp_scores[img_id],
                bbox_scores[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale,
                with_nms=True
            ))
        return result_list

    def get_keypoints_single(self,
                             keypoints,
                             scores,
                             bbox_scores,
                             img_meta,
                             feat_size,
                             rescale=False,
                             with_nms=True):
        pad_h, pad_w = img_meta['pad_shape'][:2]
        num_topk = self.test_cfg.max_per_img
        width_ratio = float(feat_size[1] / pad_w)
        height_ratio = float(feat_size[0] / pad_h)
        keypoints = keypoints.view(keypoints.size(0), self.num_classes, 2)
        keypoints[..., 0] /= width_ratio
        keypoints[..., 1] /= height_ratio
        if rescale:
            scale_factor = img_meta['scale_factor'][:2]
            keypoints /= keypoints.new_tensor(scale_factor)
        scores = (scores > self.test_cfg.kp_score_thr).int().unsqueeze(-1)

        keypoints = torch.cat([keypoints, scores], dim=2).view(num_topk, -1)  # (K, #kp x 3)
        keypoints = torch.cat([keypoints, bbox_scores], dim=1)  # (K, #kp x 3 + 1)
        # TOCHECK: add concat() to (K, 56) for softnms39().
        return keypoints

    def _kepoint_topk(self, scores, k=20):
        # TOCHECK: use `CornerHead._topk()`
        batch, cls, height, width = scores.size()
        # `scores`: (B, #cls, H, W) -> (B, #cls, HxW) -> (B, #cls, K)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cls, -1), k)

        topk_inds = topk_inds % (height * width)  # (B, #cls, topk)
        topk_ys = (topk_inds // width).float()  # (B, #cls, topk)
        topk_xs = (topk_inds % width).float()  # (B, #cls, topk)

        return topk_scores, topk_inds, topk_ys, topk_xs

    def feat_process(self, bbox_outs, keypoint_outs, return_loss=False):
        new_bbox_outs = bbox_outs[3:]
        return new_bbox_outs, keypoint_outs

    def postprocess(self, *inputs, return_loss=False):
        if return_loss:
            pred_outs, gt_inputs, gt_metas, img_metas = inputs
            # > heatmap, reg_target, ct2kps_target, reg_ind_target, target_reg_mask, target_ck2kps_mask
            gt_keypoint_input = gt_inputs['keypoint']
            bbox_ind_target = gt_inputs['bbox'][3]
            new_inputs = pred_outs + gt_keypoint_input + (bbox_ind_target, img_metas,)
        else:
            new_inputs = inputs
        return new_inputs

    @force_fp32(apply_to=('pred_hm', 'pred_reg', 'pred_ct2kps'))
    def loss(self,
             pred_hm,  # (B, #joints, H, W)
             pred_reg,  # (B, 2, H, W)
             pred_ct2kps,  # (B, #joints x 2, H, W)
             gt_hm,
             gt_reg,
             gt_ct2kps,
             gt_reg_inds,
             gt_reg_mask,
             gt_ct2kps_mask,
             gt_ct_inds,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks_ignore=None,
             gt_keypoints_ignore=None):
        eps = 1e-12
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = gt_hm.eq(1).float().sum()
        # > keypoint heatmap loss
        loss_hm = self.loss_heatmap(pred_hm, gt_hm, avg_factor=1 if num_pos == 0 else num_pos)

        eps = 1e-4
        # > center2keypoint loss
        pred_ct2kps = self._transpose_and_gather_feat(pred_ct2kps, gt_ct_inds)
        ct2kps_mask = gt_ct2kps_mask.float()
        loss_joint = self.loss_joint(pred_ct2kps * ct2kps_mask,
                                     gt_ct2kps * ct2kps_mask,
                                     avg_factor=ct2kps_mask.sum() + eps)

        # > offset loss
        pred_reg = self._transpose_and_gather_feat(pred_reg, gt_reg_inds)   # TOCHECK: index out of bounds
        reg_mask = gt_reg_mask.unsqueeze(2).expand_as(pred_reg).float()
        loss_reg = self.loss_offset(pred_reg * reg_mask,
                                    gt_reg * reg_mask,
                                    avg_factor=reg_mask.sum() + eps)
        return {'keypoint/loss_heatmap': loss_hm, 'keypoint/loss_reg': loss_reg, 'keypoint/loss_joint': loss_joint}

    def get_targets(self,
                    gt_inputs,
                    gt_metas,
                    feat_shapes,
                    img_metas,
                    **kwargs):
        """
        Args:
            gt_boxes: list(tensor). tensor <=> image, (gt_num, 4).
            gt_keypoints: list(tensor). tensor <=> image, (gt_num, 3).
            gt_labels: list(tensor). tensor <=> image, (gt_num,).
            img_metas: list(dict).

        Returns:
            heatmap: tensor, (batch, 80, h, w).
            box_target: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
            reg_weight: tensor, same as box_target.
        """
        gt_keypoints, gt_labels = gt_inputs[2], gt_inputs[3]
        gt_boxes_meta = gt_metas['bbox']
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]  # > (#level, 2)
        feat_gt_centers_int, gt_radius = gt_boxes_meta
        with torch.no_grad():
            heatmap, reg_target, ct2kps_target, reg_ind_target, target_reg_mask, target_ck2kps_mask = multi_apply(
                self.target_single_image,
                gt_keypoints,
                gt_radius,  # (B, #obj, 4)
                feat_gt_centers_int,
                img_shape=img_shape,  # (img_H, img_W)
                feat_shape=feat_shape  # (H, W)
            )
            heatmap = torch.stack(heatmap, dim=0)
            reg_target = torch.stack(reg_target, dim=0)
            ct2kps_target = torch.stack(ct2kps_target, dim=0)
            reg_ind_target = torch.stack(reg_ind_target, dim=0).to(torch.long)
            target_reg_mask = torch.stack(target_reg_mask, dim=0)
            target_ck2kps_mask = torch.stack(target_ck2kps_mask, dim=0)
            return (heatmap, reg_target, ct2kps_target, reg_ind_target, target_reg_mask, target_ck2kps_mask), None

    def target_single_image(self,
                            gt_keypoints,
                            gt_radius,
                            feat_gt_centers_int,
                            img_shape,
                            feat_shape):
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
        img_h, img_w = img_shape  # > input size: (512, 512)
        feat_h, feat_w = feat_shape  # > feat size: (128, 128)
        width_ratio = float(feat_w / img_w)  # > 0.25
        height_ratio = float(feat_h / img_h)
        num_objs = gt_keypoints.size(0)
        num_joints = self.num_classes
        # `H/W`: 128
        gt_heatmap = gt_keypoints.new_zeros(num_joints, feat_h, feat_w)  # > (#joint, H, W)
        gt_reg = gt_keypoints.new_zeros(self.max_objs * num_joints, 2)  # > (#max_obj x #joint, 2)
        gt_ct2kps = gt_keypoints.new_zeros(self.max_objs, num_joints * 2)  # > (#max_obj, #joint x 2)
        gt_reg_inds = gt_keypoints.new_zeros(self.max_objs * num_joints, dtype=torch.int64)  # > (#max_obj x #joint,)
        gt_reg_mask = gt_keypoints.new_zeros(self.max_objs * num_joints, dtype=torch.int64)  # > (#max_obj,)
        gt_ct2kps_mask = gt_keypoints.new_zeros(self.max_objs, num_joints * 2, dtype=torch.uint8)  # > (#max_obj,)

        feat_gt_keypints = gt_keypoints.clone()  # > (#obj, #joints(x,y,v))
        feat_gt_keypints[..., 0] *= width_ratio
        feat_gt_keypints[..., 1] *= height_ratio

        for obj_id in range(num_objs):
            feat_obj_keypoints = feat_gt_keypints[obj_id]
            radius = gt_radius[obj_id]
            for joint_id in range(num_joints):
                if feat_obj_keypoints[joint_id, 2] > 0:
                    joint_xy = feat_obj_keypoints[joint_id, :2]
                    joint_xy_int = joint_xy.to(torch.int)
                    if 0 <= joint_xy[0] < feat_w and 0 <= joint_xy[1] < feat_h:
                        start_ind, end_ind = joint_id * 2, joint_id * 2 + 2
                        obj_joint_ind = obj_id * num_joints + joint_id
                        gt_reg[obj_joint_ind] = joint_xy - joint_xy_int
                        gt_reg_inds[obj_joint_ind] = joint_xy_int[1] * feat_h + joint_xy_int[0]
                        gt_reg_mask[obj_joint_ind] = 1
                        gt_ct2kps[obj_id, start_ind:end_ind] = joint_xy - feat_gt_centers_int[obj_id]
                        gt_ct2kps_mask[obj_id, start_ind:end_ind] = 1
                        gt_heatmap[joint_id] = gen_gaussian_target(heatmap=gt_heatmap[joint_id],
                                                                   center=joint_xy_int,
                                                                   radius=int(radius))
        return gt_heatmap, gt_reg, gt_ct2kps, gt_reg_inds, gt_reg_mask, gt_ct2kps_mask