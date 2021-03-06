import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, calc_region
from . import CenterHead, CenterPoseHead
from ..builder import HEADS, build_loss
from ..utils import gaussian_radius, gen_gaussian_target
from ...core.bbox.iou_calculators.iou2d_calculator import bbox_areas


@HEADS.register_module
class TTFHead(CenterHead):

    def __init__(self,
                 *args,
                 offset_base=16.,
                 area_cfg=dict(
                     type='log',
                     agnostic=True,
                     gaussian=True,
                     alpha=0.54,
                     beta=0.54
                 ),
                 with_centerpose=False,
                 **kwargs):
        assert area_cfg.type in [None, 'norm', 'log', 'sqrt']
        self.offset_base = offset_base
        self.area_cfg = area_cfg
        self.wh_channels = 4 if area_cfg.agnostic else 4 * self.num_classes
        self.base_loc = None
        self.with_centerpose = with_centerpose
        super(TTFHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        if isinstance(self.stacked_convs, tuple) and isinstance(self.feat_channels, tuple) and \
                len(self.stacked_convs) == len(self.feat_channels):
            self.ct_hm_head = self.build_head(
                self.in_channels, self.feat_channels[0], self.stacked_convs[0], self.num_classes)
            self.ct_wh_head = self.build_head(
                self.in_channels, self.feat_channels[1], self.stacked_convs[1], self.wh_channels)
        else:
            self.ct_hm_head = self.build_head(
                self.in_channels, self.feat_channels, self.stacked_convs, self.num_classes)
            self.ct_wh_head = self.build_head(
                self.in_channels, self.feat_channels, self.stacked_convs, self.wh_channels)

    def init_weights(self):
        for _, m in self.ct_hm_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.ct_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.ct_wh_head)

    def forward(self, x):
        """
        Args:
            x: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        hm = self.ct_hm_head(x)  # > (64, H, W) -> (128, H, W) -> (#cls, H, W)
        # > SEE: https://github.com/ZJULearning/ttfnet/issues/17
        wh = F.relu(self.ct_wh_head(x)) * self.offset_base  # > (64, H, W) -> (64, H, W) -> (4, H, W)
        return hm, wh

    def get_bboxes(self,
                   ct_hm,
                   ct_wh,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        batch, cls, feat_h, feat_w = ct_hm.size()
        ct_hm = ct_hm.detach().sigmoid_()  # > (B, #cls, H, W)
        ct_wh = ct_wh.detach()  # > (B, 4, H, W)
        num_topk = self.test_cfg.max_per_img

        pad_h, pad_w = img_metas[0]['pad_shape'][:2]
        width_stride = float(pad_w / feat_w)  # > 4
        height_stride = float(pad_h / feat_h)

        # > perform nms & topk over center points
        ct_hm = self._local_maximum(ct_hm, kernel=3)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_hm, k=num_topk)
        refined_ct_xs = ct_xs.unsqueeze(-1) * width_stride
        refined_ct_ys = ct_ys.unsqueeze(-1) * height_stride

        # > `wh`: (B, 4, H, W) -> (B, H, W, 4) -> (B, K, 4)
        ct_wh = self._transpose_and_gather_feat(ct_wh, ct_inds)
        if not self.area_cfg.agnostic:
            ct_wh = ct_wh.view(-1, num_topk, self.num_classes, 4)
            ct_wh = torch.gather(ct_wh, 2, ct_clses[..., None, None].expand(
                ct_clses.size(0), ct_clses.size(1), 1, 4).long())
        ct_wh = ct_wh.view(batch, num_topk, 4)  # (B, K, 4)

        clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)
        # (B, K, 4)
        bboxes = torch.cat([refined_ct_xs - ct_wh[..., [0]], refined_ct_ys - ct_wh[..., [1]],
                            refined_ct_xs + ct_wh[..., [2]], refined_ct_ys + ct_wh[..., [3]]], dim=2)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self.get_bboxes_single(
                bboxes[img_id],
                scores[img_id],
                clses[img_id],
                img_metas[img_id],
                (feat_h, feat_w),
                rescale=rescale,
                with_nms=with_nms
            ))
        if not self.with_centerpose:
            ct_xs = refined_ct_xs
            ct_ys = refined_ct_ys
        return result_list, (scores, bboxes, ct_inds, ct_xs, ct_ys)

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_shape,
                          rescale=False,
                          with_nms=True):
        scores_keep = (scores > self.test_cfg.score_thr).squeeze(-1)  # (K, 1) -> (K,)
        scores = scores[scores_keep]  # (K, 1)
        bboxes = bboxes[scores_keep]  # (B, K, 4) -> (K, 4)
        labels = labels[scores_keep]  # (B, K, 1) -> (K, 1)
        img_shape = img_meta['pad_shape']
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

        if rescale:
            scale_factor = img_meta['scale_factor']
            bboxes /= bboxes.new_tensor(scale_factor)

        bboxes = torch.cat([bboxes, scores], dim=1)  # (K, 4 + 1)
        labels = labels.squeeze(-1)  # (K, 1) -> (K,)
        return bboxes, labels

    def interprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """ 
            [IN] bbox_head() -> (
                preds: 
                    bbox=(ct_hm, ct_wh), 
                gts: (gt_boxes, gt_masks, gt_keypoints, gt_labels), 
                metas: 
                    bbox=None
            ) -> (gt_boxes, gt_labels)
            [OUT] -> get_targets(gt_boxes, gt_labels, ...)
            """
            pred_outs, gt_inputs, metas = inputs
            gt_boxes, gt_labels = gt_inputs[0], gt_inputs[3]
            return gt_boxes, gt_labels
        else:
            # `TEST`
            """ 
            [IN] bbox_head() -> (
                preds: 
                    bbox=(ct_hm, ct_wh)
                metas:
                    bbox=None
            )
            [OUT] -> get_bboxes(ct_hm, ct_wh, ...)
            """
            pred_outs = inputs[0]
            return pred_outs['bbox']

    def postprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """
            [IN] get_targets() -> (
                preds: 
                    bbox=(ct_hm, ct_wh), 
                targets:
                    bbox=(hm_target, box_target, wh_weight, ct_ind_target)
                metas: 
                    bbox=(feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log)
            ) -> (ct_hm, ct_wh, hm_target, box_target, wh_weight, ct_ind_target)
            [OUT] -> loss(pred_hm, pred_wh, hm_target, box_target, wh_weight, ct_ind_target)
            """
            pred_outs, targets, metas = inputs
            bbox_inputs = targets['bbox']
            new_inputs = pred_outs + bbox_inputs
        else:
            # `TEST`
            """ 
            [IN] get_bboxes() -> (
                preds: 
                    bbox=(ct_hm, ct_wh), 
                metas: 
                    bbox=None
                dets: (bboxes, labels)
            ) 
            [OUT] (preds, dets) -> bbox2result(det_bboxes, det_labels)
            """
            new_inputs = inputs
        return new_inputs

    def get_points(self,
                   featmap_size,
                   stride,
                   dtype,
                   device):
        assert isinstance(featmap_size, tuple)
        assert len(featmap_size) == len(stride)
        h, w = featmap_size
        h_stride, w_stride = stride
        x_range = torch.arange(0, (w - 1) * w_stride + 1, w_stride,
                               dtype=dtype, device=device)
        y_range = torch.arange(0, (h - 1) * h_stride + 1, h_stride,
                               dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x, y), dim=0)  # (2, h, w)
        return points

    @force_fp32(apply_to=('pred_hm', 'pred_wh'))
    def loss(self,
             ct_hm,  # (B, #cls, H, W)
             ct_wh,  # (B, 4, H, W)
             hm_target,
             box_target,
             wh_weight,
             ct_ind_target,
             img_metas,
             **kwargs):
        """

        Args:
            ct_hm: tensor, (batch, #cls, h, w).
            ct_wh: tensor, (batch, 4, h, w) or (batch, #cls * 4, h, w).
            heatmap_target: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        eps = 1e-4
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        feat_h, feat_w = ct_hm.shape[2:]
        width_stride = float(img_w / feat_w)  # > 4
        height_stride = float(img_h / feat_h)
        ct_hm = torch.clamp(ct_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        hm_loss = self.loss_heatmap(ct_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        # `base_loc`: None
        if self.base_loc is None or \
                feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            self.base_loc = self.get_points((feat_h, feat_w),
                                            (height_stride, width_stride),
                                            dtype=torch.float32,
                                            device=hm_target.device)

        # (B, 4, H, W) -> (B, H, W, 4)
        pred_boxes = torch.cat((self.base_loc - ct_wh[:, [0, 1]],
                                self.base_loc + ct_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        box_target = box_target.permute(0, 2, 3, 1)

        wh_weight = wh_weight.view(-1, feat_h, feat_w)  # (B, 1, H, W) -> (B, H, W)
        keep = wh_weight > 0  # (B, H, W)
        weight = wh_weight[keep].float()  # (#pos,)
        pred_boxes = pred_boxes[keep].view(-1, 4)  # (B, H, W, 4) ∩ (#pos,) -> (#pos, 4)
        box_target = box_target[keep].view(-1, 4)  # (B, H, W, 4) ∩ (#pos,) -> (#pos, 4)

        wh_loss = self.loss_bbox(pred_boxes, box_target, weight, avg_factor=wh_weight.sum() + eps)
        return {'bbox/loss_heatmap': hm_loss, 'bbox/loss_wh': wh_loss}

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
            # `heatmap`: (B, #cls, H, W), `box_target`: (B, 4, H, W), `reg_weight`: (B, 1, H, W)
            hm_target, box_target, wh_weight, ct_ind_target, \
            feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log = multi_apply(
                self._get_targets_single,
                gt_boxes,  # (B, #obj, 4)
                gt_labels,  # (B, #obj)
                img_shape=img_shape,
                feat_shape=feat_shape  # (H, W)
            )
            # `hm`: (B, #cls, H, W), `box`: (B, 4, H, W)
            hm_target, box_target = [torch.stack(t, dim=0).detach() for t in [hm_target, box_target]]
            # (B, 1, H, W)
            wh_weight = torch.stack(wh_weight, dim=0).detach()
            ct_ind_target = torch.stack(ct_ind_target, dim=0).to(torch.long)
            return (hm_target, box_target, wh_weight, ct_ind_target), \
                   (feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log)

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
        img_h, img_w = img_shape
        feat_h, feat_w = feat_shape
        width_ratio = float(feat_w / img_w)  # > 0.25
        height_ratio = float(feat_h / img_h)
        num_objs = gt_boxes.size(0)
        hm_target = gt_boxes.new_zeros((self.num_classes, feat_h, feat_w))  # > (#cls, H, W)
        fake_heatmap = gt_boxes.new_zeros((feat_h, feat_w))  # > (H, W)

        box_target = gt_boxes.new_ones((self.wh_channels, feat_h, feat_w)) * -1  # > (4, 128, 128)
        reg_weight = gt_boxes.new_zeros((self.wh_channels // 4, feat_h, feat_w))  # > (1, 128, 128)

        if self.area_cfg.type == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()  # See Eq(7): get `base` for all small and big objects
        elif self.area_cfg.type == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_topk_ind = torch.topk(boxes_areas_log, num_objs)  # order: from big to small

        if self.area_cfg.type == 'norm':
            boxes_area_topk_log[:] = 1.
        # change order to `big → small`
        gt_boxes = gt_boxes[boxes_topk_ind]
        gt_labels = gt_labels[boxes_topk_ind]

        feat_gt_boxes = gt_boxes.clone()  # > (#obj, 4)
        feat_gt_boxes[:, [0, 2]] *= width_ratio
        feat_gt_boxes[:, [1, 3]] *= height_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]],
                                               min=0, max=feat_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]],
                                               min=0, max=feat_h - 1)
        feat_gt_hs, feat_gt_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                  feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
        feat_boxes_wh = torch.stack([feat_gt_ws, feat_gt_hs], dim=1)
        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        feat_gt_centers = torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                       (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], dim=1)
        feat_gt_centers_int = torch.zeros_like(feat_gt_centers)
        feat_gt_centers_int[:, 0] = (feat_gt_centers[:, 0] * width_ratio)  # (#obj, 2)
        feat_gt_centers_int[:, 1] = (feat_gt_centers[:, 1] * height_ratio)
        feat_gt_centers_int = feat_gt_centers_int.to(torch.int)  # (#obj, 2)

        ct_ind_target = gt_boxes.new_zeros(128, dtype=torch.int64)
        ct_ind_target[:num_objs] = feat_gt_centers_int[..., 1] * feat_h + feat_gt_centers_int[..., 0]

        # `wh_gaussian`: True, `alpha`: 0.54, `beta`: 0.54
        h_radiuses_alpha = (feat_gt_hs / 2. * self.area_cfg.alpha).int()
        w_radiuses_alpha = (feat_gt_ws / 2. * self.area_cfg.alpha).int()
        if self.area_cfg.gaussian and self.area_cfg.alpha != self.area_cfg.beta:
            h_radiuses_beta = (feat_gt_hs / 2. * self.area_cfg.beta).int()
            w_radiuses_beta = (feat_gt_ws / 2. * self.area_cfg.beta).int()

        if not self.area_cfg.gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() * width_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=feat_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=feat_h - 1) for y in [ctr_y1s, ctr_y2s]]
        feat_boxes_radius = []
        # larger boxes have lower priority than small boxes.
        for obj_id in range(num_objs):  # > #obj: large -> small
            cls_id = gt_labels[obj_id]
            alpha_radius = (h_radiuses_alpha[obj_id].item(), w_radiuses_alpha[obj_id].item())
            # > `heatmap`: (#cls, H, W), `ct_ints`: (#obj, 2)
            fake_heatmap = fake_heatmap.zero_()  # (H, W)
            fake_heatmap = gen_gaussian_target(fake_heatmap, feat_gt_centers_int[obj_id], alpha_radius)
            hm_target[cls_id] = torch.max(hm_target[cls_id], fake_heatmap)
            feat_boxes_radius.append(alpha_radius)
            if self.area_cfg.gaussian:  # <-
                if self.area_cfg.alpha != self.area_cfg.beta:
                    beta_radius = (h_radiuses_beta[obj_id].item(), w_radiuses_beta[obj_id].item())
                    gen_gaussian_target(fake_heatmap, feat_gt_centers_int[obj_id], beta_radius)
                box_target_inds = fake_heatmap > 0  # > (H, W)
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[obj_id], ctr_y1s[obj_id], ctr_x2s[obj_id], ctr_y2s[obj_id]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.area_cfg.agnostic:  # <-
                box_target[:, box_target_inds] = gt_boxes[obj_id][:, None]  # > (#obj, 4) -> (4, 1) -> (4, H, W)
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[obj_id][:, None]

            if self.area_cfg.gaussian:  # <-
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[obj_id]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = boxes_area_topk_log[obj_id] / box_target_inds.sum().float()

        return hm_target, box_target, reg_weight, ct_ind_target, \
               feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log


@HEADS.register_module
class TTFPoseHead(CenterPoseHead):

    def __init__(self,
                 *args,
                 offset_base=16.,
                 **kwargs):
        super(TTFPoseHead, self).__init__(*args, **kwargs)
        self.offset_base = offset_base
        self.base_loc = None

    def _init_layers(self):
        if isinstance(self.stacked_convs, tuple) and isinstance(self.feat_channels, tuple) and \
                len(self.stacked_convs) == len(self.feat_channels):
            self.kp_hm_head = self.build_head(
                self.in_channels, self.feat_channels[0], self.stacked_convs[0], self.num_classes)
            self.ct_kps_head = self.build_head(
                self.in_channels, self.feat_channels[1], self.stacked_convs[1], self.num_classes * 2)
        else:
            self.kp_hm_head = self.build_head(
                self.in_channels, self.feat_channels, self.stacked_convs, self.num_classes)
            self.ct_kps_head = self.build_head(
                self.in_channels, self.feat_channels, self.stacked_convs, self.num_classes * 2)

    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.kp_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.ct_kps_head)

    def forward(self, x):
        kp_hm = self.kp_hm_head(x)
        ct_kps = F.relu(self.ct_kps_head(x)) * self.offset_base
        return kp_hm, ct_kps

    def get_keypoints(self,
                      keypoint_outs,
                      bbox_metas,
                      img_metas,
                      rescale=False,
                      with_nms=True):
        pred_hm, pred_ct_kps = keypoint_outs
        bbox_scores, bboxes, ct_inds, ct_xs, ct_ys = bbox_metas
        batch, num_joints, feat_h, feat_w = pred_hm.size()
        kp_hm = pred_hm.detach().sigmoid_()
        ct_kps = pred_ct_kps.detach()
        num_topk = self.test_cfg.max_per_img

        pad_h, pad_w = img_metas[0]['pad_shape'][:2]
        width_stride = float(pad_w / feat_w)  # > 4
        height_stride = float(pad_h / feat_h)

        # > perform nms & topk over center points
        kp_hm = self._local_maximum(kp_hm, kernel=3)  # used maxpool to filter the max score
        kp_scores, kp_inds, kp_ys, kp_xs = self._keypoint_topk(kp_hm, k=num_topk)
        kp_xs *= width_stride
        kp_ys *= height_stride

        scores_mask = (kp_scores > self.test_cfg.kp_score_thr).float()
        kp_scores = (1 - scores_mask) * -1 + scores_mask * kp_scores
        kp_xs = (1 - scores_mask) * (-10000) + scores_mask * kp_xs
        kp_ys = (1 - scores_mask) * (-10000) + scores_mask * kp_ys
        kp_kps = torch.stack([kp_xs, kp_ys], dim=-1)  # > (B, #kps, K, 2)

        # > (B, #kps x 2, H, W) -> (B, H, W, #kps x 2) -> (B, K, #kps x 2)
        ct_kps = self._transpose_and_gather_feat(ct_kps, ct_inds)
        ct_kps[..., 0::2] += ct_xs
        ct_kps[..., 1::2] += ct_ys
        # > (B, K, #kps, 2) -> (B, #kps, K, 2)
        ct_kps = ct_kps.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()

        # NOTE: keypoints are on image scale
        det_kps, det_scores = self._assign_keypoints_to_instance(
            ct_kps, kp_kps, bboxes, kp_scores, self.test_cfg.kp_score_thr)

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
        # > gt_inputs, gt_metas
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
        kp_score_thr = self.test_cfg.kp_score_thr
        if rescale:
            scale_factor = img_meta['scale_factor'][:2]
            keypoints /= keypoints.new_tensor(scale_factor)
        scores = (scores > kp_score_thr).int()

        keypoints = torch.cat([keypoints, scores], dim=2).view(num_topk, -1)  # (K, #kp x 3)
        keypoints = torch.cat([keypoints, bbox_scores], dim=1)  # (K, #kp x 3 + 1)
        return keypoints

    def interprocess(self, *inputs, training=False):
        if training:
            # `TRAIN`
            """
            [IN] keypoint_head() -> (
                preds:
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                gts: (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
                metas:
                    bbox=(feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log)
            ) -> (
                (gt_keypoints, gt_labels),
                (feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log)
            )
            [OUT] -> get_targets(
                gt_inputs: (gt_keypoints, gt_labels)
                boxes_meta: (feat_gt_centers_int, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log)
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
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                metas:
                    bbox=(scores, bboxes, ct_inds, ct_xs, ct_ys),
                    keypoint=None
            ) -> (
                (kp_hm, ct_kps),
                (scores, bboxes, ct_inds, ct_xs, ct_ys),
            )
            [OUT] -> get_keypoints(
                keypoint_outs: (kp_hm, ct_kps)
                bbox_metas: (scores, bboxes, ct_inds, ct_xs, ct_ys)
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
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                targets:
                    bbox=(hm_target, box_target, wh_weight, ct_ind_target)
                    keypoint=(hm_target, ct_kps_target, reg_weight)
                metas:
                    bbox=(feat_gt_centers_int, feat_boxes_wh, boxes_topk_ind, boxes_area_topk_log)
                    keypoint=None
            ) -> (
                kp_hm, ct_kps,
                hm_target, ct_kps_target, reg_weight, ct_ind_target, 
            )
            [OUT] -> loss(
                kp_hm, ct_kps, hm_target, ct_kps_target, reg_weight, ct_ind_target
            )
            """
            pred_outs, targets, metas = inputs
            keypoint_targets = targets['keypoint']
            ct_ind_target = targets['bbox'][-1]
            new_inputs = pred_outs + keypoint_targets + (ct_ind_target,)
        else:
            # `TEST`
            """
            [IN] keypoint_head() -> (
                preds:
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                metas:
                    bbox=(scores, bboxes, ct_inds, ct_xs, ct_ys)
                    keypoint=None
                dets: (keypoints,)
            ) 
            [OUT] (preds, dets) -> keypoint2result(keypoints
            """
            new_inputs = inputs
        return new_inputs

    def get_points(self,
                   featmap_size,
                   stride,
                   dtype,
                   device):
        assert isinstance(featmap_size, tuple)
        assert len(featmap_size) == len(stride)
        h, w = featmap_size
        h_stride, w_stride = stride
        x_range = torch.arange(0, (w - 1) * w_stride + 1, w_stride,
                               dtype=dtype, device=device)
        y_range = torch.arange(0, (h - 1) * h_stride + 1, h_stride,
                               dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x, y), dim=0)  # (2, h, w)
        return points

    def loss(self,
             pred_kp_hm,
             pred_ct_kps,
             hm_target,
             ct_kps_target,
             reg_weight,
             ct_ind_target,
             img_metas,
             **kwargs):
        eps = 1e-4
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        batch, num_joints, feat_h, feat_w = pred_kp_hm.shape
        size_strides = torch.tensor([float(img_w / feat_w),  # > 4
                                     float(img_h / feat_h)],
                                    dtype=pred_kp_hm.dtype,
                                    device=pred_kp_hm.device)
        pred_kp_hm = torch.clamp(pred_kp_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        hm_loss = self.loss_heatmap(pred_kp_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        if self.base_loc is None or \
                feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            self.base_loc = self.get_points((feat_h, feat_w),
                                            stride=size_strides,
                                            dtype=torch.float32,
                                            device=hm_target.device)
        pred_ct_kps[:, 0::2] += self.base_loc[0]
        pred_ct_kps[:, 1::2] += self.base_loc[1]
        pred_ct_kps = pred_ct_kps.permute(0, 2, 3, 1)  # (B, H, W, #kps x 2)
        ct_kps_target = ct_kps_target.permute(0, 2, 3, 1)  # (B, H, W, #kps x 2)
        reg_weight = reg_weight.permute(0, 2, 3, 1)  # (B, H, W, #kps x 2)
        keep = reg_weight > 0  # (B, H, W)
        weight = reg_weight[keep].float()  # (#pos,)
        pred_ct_kps = pred_ct_kps[keep]  # (B, H, W, #kps x 2) -> (#pos,)
        ct_kps_target = ct_kps_target[keep]  # (B, H, W, #kps x 2) -> (#pos,)
        reg_loss = self.loss_joint(pred_ct_kps,
                                   ct_kps_target,
                                   weight,
                                   avg_factor=reg_weight.sum() + eps)
        return {'keypoint/loss_heatmap': hm_loss, 'keypoint/loss_reg': reg_loss}

    def get_targets(self,
                    gt_inputs,
                    box_metas,
                    feat_shapes,
                    img_metas,
                    **kwargs):
        gt_keypoints, gt_labels = gt_inputs
        feat_gt_centers, feat_boxes_wh, feat_boxes_radius, boxes_topk_ind, boxes_area_topk_log = box_metas
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]  # > (#level, 2)
        with torch.no_grad():
            hm_target, ct_kps_target, reg_weight = multi_apply(
                self._get_targets_single,
                gt_keypoints,
                feat_gt_centers,
                feat_boxes_wh,
                feat_boxes_radius,
                boxes_topk_ind,
                boxes_area_topk_log,
                img_shape=img_shape,
                feat_shape=feat_shape
            )
            hm_target = torch.stack(hm_target, dim=0).detach()
            ct_kps_target = torch.stack(ct_kps_target, dim=0).detach()
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            return (hm_target, ct_kps_target, reg_weight), None

    def _get_targets_single(self,
                            gt_keypoints,  # (#obj, #kps, 3)
                            feat_gt_centers_int,  # (#obj, 2)
                            feat_boxes_wh,  # (#obj, 2)
                            feat_boxes_radius,
                            boxes_topk_ind,  # order: big → small
                            boxes_area_topk_log,
                            img_shape,
                            feat_shape,
                            **kwargs):
        size_strides = torch.tensor([float(img_shape[1] / feat_shape[1]),
                                     float(img_shape[0] / feat_shape[0])],
                                    dtype=gt_keypoints.dtype,
                                    device=gt_keypoints.device)
        gt_keypoints = gt_keypoints[boxes_topk_ind]  # sort `big → small`
        # > kp_hm_target, ct_kps_target, reg_weight
        joint_target = self._get_joint_target(gt_keypoints,
                                              feat_gt_centers_int,
                                              feat_boxes_wh,
                                              feat_boxes_radius,
                                              boxes_area_topk_log,
                                              feat_shape=feat_shape,
                                              size_strides=size_strides)
        return joint_target

    def _get_joint_target(self,
                          gt_keypoints,
                          feat_gt_centers_int,  # (#obj, 2)
                          feat_gt_boxes_wh,  # (#obj, 2)
                          feat_boxes_radius,
                          boxes_area_topk_log,
                          feat_shape,
                          size_strides,
                          **kwargs):
        feat_h, feat_w = feat_shape
        num_objs = gt_keypoints.size(0)
        num_joints = self.num_classes

        kp_hm_target = gt_keypoints.new_zeros(num_joints, feat_h, feat_w)  # > (#joint, H, W)
        reg_weight = gt_keypoints.new_zeros((num_joints * 2, feat_h, feat_w))  # > (#joint, H, W)
        ct_kps_target = gt_keypoints.new_zeros((num_joints * 2, feat_h, feat_w)) * -1  # (#joint x 2, H, W)
        box_fake_heatmap = gt_keypoints.new_zeros((feat_h, feat_w))  # > (H, W)

        feat_gt_keypoints = gt_keypoints.clone()  # > (#obj, #joints(x,y,v))
        feat_gt_keypoints[..., :2] /= size_strides

        for obj_id in range(num_objs):
            feat_obj_keypoints = feat_gt_keypoints[obj_id]
            kp_radius = gaussian_radius(feat_gt_boxes_wh[obj_id],
                                        min_overlap=self.train_cfg.min_overlap)
            kp_radius = max(0, int(kp_radius))
            for joint_id in range(num_joints):
                if feat_obj_keypoints[joint_id, 2] > 0:  # this keypoint is labeled
                    feat_joint_xy = feat_obj_keypoints[joint_id, :2]
                    feat_joint_xy_int = feat_joint_xy.to(torch.int)
                    if 0 <= feat_joint_xy[0] < feat_w and 0 <= feat_joint_xy[1] < feat_h:
                        start_ind, end_ind = joint_id * 2, joint_id * 2 + 2
                        # > heatmaps for each object
                        box_fake_heatmap = box_fake_heatmap.zero_()
                        box_fake_heatmap = gen_gaussian_target(box_fake_heatmap,
                                                               feat_gt_centers_int[obj_id],
                                                               radius=feat_boxes_radius[obj_id])
                        # TOCHECK: combine `kp_heatmap` and `bbox_heatmap` to gen `inds`
                        box_target_inds = box_fake_heatmap > 0

                        # > heatmaps for each joint
                        kp_hm_target[joint_id] = gen_gaussian_target(kp_hm_target[joint_id],
                                                                     feat_joint_xy_int,
                                                                     radius=int(kp_radius))

                        ct_kps_target[start_ind:end_ind, box_target_inds] = \
                            gt_keypoints[obj_id, joint_id, :2][:, None]

                        local_heatmap = box_fake_heatmap[box_target_inds]
                        ct_div = local_heatmap.sum()
                        local_heatmap *= boxes_area_topk_log[obj_id]
                        reg_weight[start_ind:end_ind, box_target_inds] = local_heatmap / ct_div
        return kp_hm_target, ct_kps_target, reg_weight
