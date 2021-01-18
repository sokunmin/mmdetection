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
                 **kwargs):
        assert area_cfg.type in [None, 'norm', 'log', 'sqrt']
        self.offset_base = offset_base
        self.area_cfg = area_cfg
        self.wh_channels = 4 if area_cfg.agnostic else 4 * self.num_classes
        self.base_loc = None
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
        # pred_hm, pred_wh = bbox_outs
        batch, cls, feat_h, feat_w = ct_hm.size()
        ct_hm = ct_hm.detach().sigmoid_()  # > (B, #cls, H, W)
        ct_wh = ct_wh.detach()  # > (B, 4, H, W)
        num_topk = self.test_cfg.max_per_img

        pad_h, pad_w = img_metas[0]['pad_shape'][:2]
        width_ratio = float(feat_w / pad_w)
        height_ratio = float(feat_h / pad_h)

        # > `hm`: perform nms & topk over center points
        ct_hm = self._local_maximum(ct_hm, kernel=3)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_hm, k=num_topk)
        ct_xs = ct_xs.view(batch, num_topk, 1) / width_ratio
        ct_ys = ct_ys.view(batch, num_topk, 1) / height_ratio

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
        bboxes = torch.cat([ct_xs - ct_wh[..., [0]], ct_ys - ct_wh[..., [1]],
                            ct_xs + ct_wh[..., [2]], ct_ys + ct_wh[..., [3]]], dim=2)
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
        return result_list, (scores, bboxes, ct_inds, ct_xs, ct_ys)

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_size,
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

    def interprocess(self, *inputs, return_loss=False):
        if return_loss:
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

    def postprocess(self, *inputs, return_loss=False):
        if return_loss:
            # `TRAIN`
            """
            [IN] get_targets() -> (
                preds: 
                    bbox=(ct_hm, ct_wh), 
                targets:
                    bbox=(hm_target, box_target, wh_weight)
                metas: 
                    bbox=(gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log)
            ) -> (ct_hm, ct_wh, hm_target, box_target, wh_weight)
            [OUT] -> loss(pred_hm, pred_wh, hm_target, box_target, wh_weight, ...)
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
        assert isinstance(stride, tuple)
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
        width_ratio = float(img_w / feat_w)  # > 4
        height_ratio = float(img_h / feat_h)
        ct_hm = torch.clamp(ct_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        hm_loss = self.loss_heatmap(ct_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        mask = wh_weight.view(-1, feat_h, feat_w)  # (B, 1, H, W) -> (B, H, W)
        avg_factor = mask.sum() + eps
        # `base_loc`: None
        if self.base_loc is None or \
                feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            self.base_loc = self.get_points((feat_h, feat_w),
                                            (height_ratio, width_ratio),
                                            dtype=torch.float32,
                                            device=hm_target.device)

        # (B, 4, H, W) -> (B, H, W, 4)
        pred_boxes = torch.cat((self.base_loc - ct_wh[:, [0, 1]],
                                self.base_loc + ct_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        gt_boxes = box_target.permute(0, 2, 3, 1)
        wh_loss = self.loss_bbox(pred_boxes, gt_boxes, mask, avg_factor=avg_factor)
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
            hm_target, box_target, wh_weight, gt_centers, feat_box_whs, boxes_topk_ind, boxes_area_topk_log = multi_apply(
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
            return (hm_target, box_target, wh_weight), (gt_centers, feat_box_whs, boxes_topk_ind, boxes_area_topk_log)

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
        num_obj = gt_boxes.size(0)
        gt_heatmap = gt_boxes.new_zeros((self.num_classes, feat_h, feat_w))  # > (#cls, H, W)
        fake_heatmap = gt_boxes.new_zeros((feat_h, feat_w))  # > (H, W)

        box_target = gt_boxes.new_ones((self.wh_channels, feat_h, feat_w)) * -1  # > (4, 128, 128)
        reg_weight = gt_boxes.new_zeros((self.wh_channels // 4, feat_h, feat_w))  # > (1, 128, 128)

        if self.area_cfg.type == 'log':
            boxes_areas_log = bbox_areas(gt_boxes).log()  # See Eq(7): get `base` for all small and big objects
        elif self.area_cfg.type == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_topk_ind = torch.topk(boxes_areas_log, num_obj)  # order: from big to small

        if self.area_cfg.type == 'norm':
            boxes_area_topk_log[:] = 1.
        # change order to `big → small`
        gt_boxes = gt_boxes[boxes_topk_ind]
        gt_labels = gt_labels[boxes_topk_ind]

        feat_gt_boxes = gt_boxes.clone()  # > (#obj, 4)
        feat_gt_boxes[:, [0, 2]] *= width_ratio
        feat_gt_boxes[:, [1, 3]] *= height_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(
            feat_gt_boxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(
            feat_gt_boxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_gt_hs, feat_gt_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                  feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])
        feat_gt_whs = torch.stack([feat_gt_ws, feat_gt_hs], dim=1)
        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        gt_centers = torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                  (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], dim=1)
        feat_gt_centers_int = torch.zeros_like(gt_centers)
        feat_gt_centers_int[:, 0] = (gt_centers[:, 0] * width_ratio)  # (#obj, 2)
        feat_gt_centers_int[:, 1] = (gt_centers[:, 1] * height_ratio)
        feat_gt_centers_int = feat_gt_centers_int.to(torch.int)  # (#obj, 2)
        # `wh_gaussian`: True, `alpha`: 0.54, `beta`: 0.54
        h_radiuses_alpha = (feat_gt_hs / 2. * self.area_cfg.alpha).int()  # > TOCHECK: why?
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

        # larger boxes have lower priority than small boxes.
        for obj_id in range(num_obj):  # > #obj: large -> small
            cls_id = gt_labels[obj_id]
            alpha_radius = (h_radiuses_alpha[obj_id].item(), w_radiuses_alpha[obj_id].item())
            # > `heatmap`: (#cls, H, W), `ct_ints`: (#obj, 2)
            fake_heatmap = fake_heatmap.zero_()  # (H, W)
            fake_heatmap = gen_gaussian_target(fake_heatmap, feat_gt_centers_int[obj_id], alpha_radius)
            gt_heatmap[cls_id] = torch.max(gt_heatmap[cls_id], fake_heatmap)

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
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div  # TOCHECK: why?
            else:
                reg_weight[cls_id, box_target_inds] = boxes_area_topk_log[obj_id] / box_target_inds.sum().float()

        return gt_heatmap, box_target, reg_weight, gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log


@HEADS.register_module
class TTFPoseHead(CenterPoseHead):

    def __init__(self,
                 *args,
                 offset_base=16.,
                 with_base_loc=True,
                 **kwargs):
        super(TTFPoseHead, self).__init__(*args, **kwargs)
        self.offset_base = offset_base
        self.with_base_loc = with_base_loc
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
        width_ratio = float(feat_w / pad_w)
        height_ratio = float(feat_h / pad_h)

        # > perform nms & topk over center points
        kp_hm = self._local_maximum(kp_hm, kernel=3)  # used maxpool to filter the max score
        kp_scores, kp_inds, kp_ys, kp_xs = self._keypoint_topk(kp_hm, k=num_topk)
        kp_xs = kp_xs.view(batch, num_topk, 1) / width_ratio
        kp_ys = kp_ys.view(batch, num_topk, 1) / height_ratio

        # NOTE: centers(ct_xs, ct_ys) already mapped back to original image size,
        # > (B, #kps, H, W) -> (B, H, W, #kps) -> (B, K, #kps)
        ct_kps = self._transpose_and_gather_feat(ct_kps, ct_inds)
        ct_kps[..., 0::2] += ct_xs.expand(batch, num_topk, num_joints)
        ct_kps[..., 1::2] += ct_ys.expand(batch, num_topk, num_joints)
        # > (B, K, #kps, 2) -> (B, #kps, K, 2)
        ct_kps = ct_kps.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()

        scores_mask = (kp_scores > self.test_cfg.kp_score_thr).float()
        kp_scores = (1 - scores_mask) * -1 + scores_mask * kp_scores
        kp_xs = (1 - scores_mask) * (-10000) + scores_mask * kp_xs
        kp_ys = (1 - scores_mask) * (-10000) + scores_mask * kp_ys
        kp_det_kps = torch.stack([kp_xs, kp_ys], dim=-1)

        det_kps, det_scores = self._assign_keypoints_to_instance(
            ct_kps, kp_det_kps, bboxes, kp_scores, self.test_cfg.kp_score_thr)

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
                             feat_size,
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

    def interprocess(self, *inputs, return_loss=False):
        if return_loss:
            # `TRAIN`
            """
            [IN] keypoint_head() -> (
                preds:
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                gts: (gt_bboxes, gt_masks, gt_keypoints, gt_labels)
                metas:
                    bbox=(gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log)
            ) -> (
                (gt_keypoints, gt_labels),
                (gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log)
            )
            [OUT] -> get_targets(
                gt_inputs: (gt_keypoints, gt_labels)
                boxes_meta: (gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log)
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

    def postprocess(self, *inputs, return_loss=False):
        if return_loss:
            # `TRAIN`
            """
            [IN] get_targets() -> (
                preds:
                    bbox=(ct_hm, ct_wh),
                    keypoint=(kp_hm, ct_kps)
                targets:
                    bbox=(hm_target, box_target, wh_weight)
                    keypoint=(hm_target, ct_kps_target, ct_ind_target, ct_target_mask, reg_weight)
                metas:
                    bbox=(gt_centers, feat_gt_whs, boxes_topk_ind, boxes_area_topk_log)
                    keypoint=None
            ) -> (
                kp_hm, ct_kps,
                hm_target, ct_kps_target, ct_ind_target, ct_target_mask, reg_weight
            )
            [OUT] -> loss(
                kp_hm, ct_kps, hm_target, ct_kps_target, ct_ind_target, ct_target_mask, reg_weight
            )
            """
            pred_outs, targets, metas = inputs
            keypoint_targets = targets['keypoint']
            new_inputs = pred_outs + keypoint_targets
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
        assert isinstance(stride, tuple)
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
             kp_hm,
             ct_kps,
             hm_target,
             ct_kps_target,
             ct_ind_target,
             ct_target_mask,
             reg_weight,
             img_metas,
             **kwargs):
        eps = 1e-4
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        num_joints, feat_h, feat_w = kp_hm.shape[1:]
        width_ratio = float(img_w / feat_w)  # > 4
        height_ratio = float(img_h / feat_h)
        kp_hm = torch.clamp(kp_hm.sigmoid_(), min=eps, max=1 - eps)
        num_pos = hm_target.eq(1).float().sum()
        hm_loss = self.loss_heatmap(kp_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        mask = reg_weight.view(-1, feat_h, feat_w)

        if self.base_loc is None or \
                feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            self.base_loc = self.get_points((feat_h, feat_w),
                                            (height_ratio, width_ratio),
                                            dtype=torch.float32,
                                            device=hm_target.device)
        pred_ct_kps = torch.cat((self.base_loc[0] + ct_kps[:, 0::2],
                                 self.base_loc[1] + ct_kps[:, 1::2]), dim=1)
        if self.with_base_loc:
            pred_ct_kps = pred_ct_kps.permute(0, 2, 3, 1)
            ct_kps_target = ct_kps_target.permute(0, 2, 3, 1)
            keep = mask > 0  # (B, H, W)
            mask = mask[keep].unsqueeze(-1).float()  # (#pos,)
            pred_ct_kps = pred_ct_kps[keep].view(-1, num_joints * 2)  # (B, H, W, 34) ∩ (#pos,) -> (#pos, 34)
            ct_kps_target = ct_kps_target[keep].view(-1, num_joints * 2)  # (B, H, W, 34) ∩ (#pos,) -> (#pos, 34)
            reg_loss = self.loss_joint(pred_ct_kps, ct_kps_target, mask, avg_factor=mask.sum() + eps)
        else:
            ct_target_mask = ct_target_mask.float()
            pred_ct_kps = self._transpose_and_gather_feat(pred_ct_kps, ct_ind_target)   # TOCHECK: index out of bounds
            reg_loss = self.loss_joint(pred_ct_kps * ct_target_mask,
                                       ct_kps_target * ct_target_mask,
                                       avg_factor=ct_target_mask.sum() + eps)
        return {'keypoint/loss_heatmap': hm_loss, 'keypoint/loss_reg': reg_loss}

    def get_targets(self,
                    gt_inputs,
                    box_metas,
                    feat_shapes,
                    img_metas,
                    **kwargs):
        gt_keypoints, gt_labels = gt_inputs
        gt_centers, feat_box_whs, boxes_topk_ind, boxes_area_topk_log = box_metas
        img_shape = img_metas[0]['img_shape'][:2]
        feat_shape = feat_shapes[0]  # > (#level, 2)
        with torch.no_grad():
            hm_target, ct_kps_target, ct_ind_target, ct_target_mask, reg_weight = multi_apply(
                self._get_targets_single,
                gt_keypoints,
                gt_centers,
                feat_box_whs,
                boxes_topk_ind,
                boxes_area_topk_log,
                img_shape=img_shape,
                feat_shape=feat_shape
            )
            hm_target = torch.stack(hm_target, dim=0).detach()
            ct_kps_target = torch.stack(ct_kps_target, dim=0).detach()
            ct_ind_target = torch.stack(ct_ind_target, dim=0).to(torch.long)
            ct_target_mask = torch.stack(ct_target_mask, dim=0)
            reg_weight = torch.stack(reg_weight, dim=0).detach()
            return (hm_target, ct_kps_target, ct_ind_target, ct_target_mask, reg_weight), None

    def _get_targets_single(self,
                            gt_keypoints,  # (#obj, #kps, 3)
                            gt_centers,  # (#obj, 2)
                            feat_box_whs,  # (#obj, 2)
                            boxes_topk_ind,  # order: big → small
                            boxes_area_topk_log,
                            img_shape,
                            feat_shape):
        img_h, img_w = img_shape
        feat_h, feat_w = feat_shape
        width_ratio = float(feat_w / img_w)  # > 0.25
        height_ratio = float(feat_h / img_h)
        num_obj = gt_keypoints.size(0)
        num_joints = self.num_classes

        gt_keypoints = gt_keypoints[boxes_topk_ind]  # sort `big → small`

        gt_heatmap = gt_keypoints.new_zeros(num_joints, feat_h, feat_w)  # > (#joint, H, W)
        fake_heatmap = gt_keypoints.new_zeros((feat_h, feat_w))  # > (H, W)
        reg_weight = gt_keypoints.new_zeros((1, feat_h, feat_w))  # > (#joint, H, W)
        if self.with_base_loc:
            ct_kps_target = gt_keypoints.new_zeros((num_joints * 2, feat_h, feat_w)) * -1  # (#joint x 2, H, W)
        else:
            ct_kps_target = gt_keypoints.new_zeros(128, num_joints * 2)

        feat_gt_keypoints = gt_keypoints.clone()  # > (#obj, #joints(x,y,v))
        feat_gt_keypoints[..., 0] *= width_ratio
        feat_gt_keypoints[..., 1] *= height_ratio

        feat_gt_centers = gt_centers.clone()
        feat_gt_centers[:, 0] *= width_ratio
        feat_gt_centers[:, 1] *= height_ratio

        ct_ind_target = gt_keypoints.new_zeros(128, dtype=torch.int64)
        ct_ind_target[:num_obj] = (feat_gt_centers[..., 1] * feat_h + feat_gt_centers[..., 0]).to(torch.int)
        ct_kps_target_mask = gt_keypoints.new_zeros(128, num_joints * 2, dtype=torch.uint8)

        for obj_id in range(num_obj):
            feat_obj_keypoints = feat_gt_keypoints[obj_id]
            radius = gaussian_radius(feat_box_whs[obj_id],
                                     min_overlap=self.train_cfg.min_overlap)
            radius = max(0, int(radius))
            for joint_id in range(num_joints):
                if feat_obj_keypoints[joint_id, 2] > 0:  # this keypoint is labeled
                    joint_xy = feat_obj_keypoints[joint_id, :2]
                    joint_xy_int = joint_xy.to(torch.int)
                    if 0 <= joint_xy_int[0] < feat_w and 0 <= joint_xy_int[1] < feat_h:
                        # > heatmaps for each joint
                        fake_heatmap = fake_heatmap.zero_()
                        fake_heatmap = gen_gaussian_target(fake_heatmap,
                                                           joint_xy_int,
                                                           radius=int(radius))
                        gt_heatmap[joint_id] = torch.max(gt_heatmap[joint_id], fake_heatmap)

                        # > offsets to centers: (#obj, #kps, 3) -> (3,)
                        kp_target_inds = fake_heatmap > 0
                        start_ind, end_ind = joint_id * 2, joint_id * 2 + 2

                        if self.with_base_loc:
                            # TOCHECK: ttfnet's way
                            ct_kps_target[start_ind:end_ind, kp_target_inds] = \
                                gt_keypoints[obj_id, joint_id, :2][:, None]
                        else:
                            # TOCHECK: centernet's way [1]
                            # ct_kps_target[joint_id:joint_id+2, kp_target_inds] = \
                            #     (gt_keypoints[obj_id, joint_id, :2] - gt_centers[obj_id]).unsqueeze(-1)
                            # TOCHECK: centernet's way [2]
                            ct_kps_target[obj_id, start_ind:end_ind] = \
                                gt_keypoints[obj_id, joint_id, :2] - feat_gt_centers[obj_id]
                            ct_kps_target_mask[obj_id, start_ind:end_ind] = 1

                        local_heatmap = fake_heatmap[kp_target_inds]
                        ct_div = local_heatmap.sum()
                        local_heatmap *= boxes_area_topk_log[obj_id]
                        reg_weight[0, kp_target_inds] = local_heatmap / ct_div

        return gt_heatmap, ct_kps_target, ct_ind_target, ct_kps_target_mask, reg_weight