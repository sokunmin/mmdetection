import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, calc_region
from . import CenterHead
from ..builder import HEADS
from ..utils import gen_gaussian_target
from ...core.bbox.iou_calculators.iou2d_calculator import bbox_areas


@HEADS.register_module
class TTFHead(CenterHead):

    def __init__(self,
                 *args,
                 wh_offset_base=16.,
                 area_cfg=dict(
                     type='log',
                     agnostic=True,
                     gaussian=True,
                     alpha=0.54,
                     beta=0.54
                 ),
                 **kwargs):
        assert area_cfg.type in [None, 'norm', 'log', 'sqrt']
        self.wh_offset_base = wh_offset_base
        self.area_cfg = area_cfg
        self.wh_channels = 4 if area_cfg.agnostic else 4 * self.num_classes
        self.base_loc = None
        super(TTFHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        if self.with_upsample:
            self.upsamples = self.build_upsample(self.upsample_cfg, with_sequential=False)

        if self.with_shortcut:
            self.shortcuts = self.build_shortcut(self.shortcut_cfg)

        # heads
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
        if self.with_shortcut:
            for _, m in self.shortcuts.named_modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        for _, m in self.ct_hm_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.ct_hm_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.ct_wh_head)

    def forward(self, feats):
        """
        Args:
            feats: list(tensor).
        Returns:
            hm: tensor, (batch, 80, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, 80 * 4, h, w).
        """
        feats = feats[::-1]  # > reverse order
        x = feats[0]  # > [512, 256, 128, 64]
        if self.with_shortcut:
            assert len(self.upsamples) == len(self.shortcuts)
            for i, upsample, shortcut in \
                    zip(range(len(self.upsamples)), self.upsamples, self.shortcuts):
                x = upsample(x)
                residual = shortcut(feats[i+1])
                x = x + residual

        hm = self.ct_hm_head(x)  # > (64, H, W) -> (128, H, W) -> (#cls, H, W)
        # > TOCHECK `wh_offset_base`: https://github.com/ZJULearning/ttfnet/issues/17
        wh = F.relu(self.ct_wh_head(x)) * self.wh_offset_base  # > (64, H, W) -> (64, H, W) -> (4, H, W)

        return hm, wh

    def get_bboxes(self,
                   bbox_outs,
                   img_metas,
                   rescale=False):
        pred_hm, pred_wh = bbox_outs
        batch, cls, feat_h, feat_w = pred_hm.size()
        ct_hm = pred_hm.detach().sigmoid_()  # > (#cls, 128, 128)
        ct_wh = pred_wh.detach()
        num_topk = self.test_cfg.max_per_img

        pad_h, pad_w = img_metas[0]['pad_shape'][:2]
        width_ratio = float(feat_w / pad_w)
        height_ratio = float(feat_h / pad_h)

        # > `hm`: perform nms & topk over center points
        ct_hm = self._local_maximum(ct_hm, kernel=3)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self._topk(ct_hm, k=num_topk)
        ct_xs = ct_xs.view(batch, num_topk, 1) / width_ratio
        ct_ys = ct_ys.view(batch, num_topk, 1) / height_ratio
        # > `wh`: (B, 4, H, W) -> (B, H, W, 4)
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
                rescale
            ))
        # > gt_inputs, gt_metas
        return result_list, None

    def get_bboxes_single(self,
                          bboxes,
                          scores,
                          labels,
                          img_meta,
                          feat_size,
                          rescale=False):
        scores_keep = (scores > self.test_cfg.score_thr).squeeze(-1)

        scores = scores[scores_keep]  # (topk, 1)
        bboxes = bboxes[scores_keep]  # (B, topk, 4) -> (topk, 4)
        labels = labels[scores_keep]  # (B, topk, 1) -> (topk, 1)
        img_shape = img_meta['pad_shape']
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

        if rescale:
            scale_factor = img_meta['scale_factor']
            bboxes /= bboxes.new_tensor(scale_factor)

        bboxes = torch.cat([bboxes, scores], dim=1)  # (topk, 4 + 1)
        labels = labels.squeeze(-1)  # (topk, 1) -> (topk,)
        return bboxes, labels

    def postprocess(self, *inputs, return_loss=False):
        if return_loss:
            pred_outs, gt_inputs, _, img_metas = inputs
            new_inputs = pred_outs + gt_inputs['bbox'] + (img_metas,)
        else:
            head_outs, bbox_list = inputs
            new_inputs = (head_outs['bbox'], bbox_list[0])
        return new_inputs

    @force_fp32(apply_to=('pred_hm', 'pred_wh'))
    def loss(self,
             pred_hm,  # (B, #cls, H, W)
             pred_wh,  # (B, 4, H, W)
             hm_target,
             box_target,
             wh_weight,
             img_metas,
             **kwargs):
        """

        Args:
            pred_hm: tensor, (batch, #cls, h, w).
            pred_wh: tensor, (batch, 4, h, w) or (batch, #cls * 4, h, w).
            heatmap_target: tensor, same as pred_hm.
            box_target: tensor, same as pred_wh.
            wh_weight: tensor, same as pred_wh.

        Returns:
            hm_loss
            wh_loss
        """
        img_h, img_w = img_metas[0]['pad_shape'][:2]
        feat_h, feat_w = pred_hm.shape[2:]
        width_ratio = float(img_w / feat_w)  # > 4
        height_ratio = float(img_h / feat_h)
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        num_pos = hm_target.eq(1).float().sum()
        hm_loss = self.loss_heatmap(pred_hm, hm_target, avg_factor=1 if num_pos == 0 else num_pos)

        mask = wh_weight.view(-1, feat_h, feat_w)  # (B, 1, H, W) -> (B, H, W)
        avg_factor = mask.sum() + 1e-4
        # `base_loc`: None
        if self.base_loc is None or \
                feat_h != self.base_loc.shape[1] or \
                feat_w != self.base_loc.shape[2]:
            h_base_step, w_base_step = height_ratio, width_ratio
            shifts_x = torch.arange(0, (feat_w - 1) * w_base_step + 1, w_base_step,
                                    dtype=torch.float32, device=hm_target.device)
            shifts_y = torch.arange(0, (feat_h - 1) * h_base_step + 1, h_base_step,
                                    dtype=torch.float32, device=hm_target.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (B, H, W, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)

        gt_boxes = box_target.permute(0, 2, 3, 1)  # (B, 4, H, W) -> (B, H, W, 4)
        wh_loss = self.loss_bbox(pred_boxes, gt_boxes, mask, avg_factor=avg_factor)
        return {'bbox/loss_heatmap': hm_loss, 'bbox/loss_wh': wh_loss}

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
            # `heatmap`: (B, #cls, H, W), `box_target`: (B, 4, H, W), `reg_weight`: (B, 1, H, W)
            heatmap, box_target, wh_weight = multi_apply(
                self.target_single_image,
                gt_boxes,  # (B, #obj, 4)
                gt_labels,  # (B, #obj)
                img_shape=img_shape,
                feat_shape=feat_shape  # (H, W)
            )
            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            wh_weight = torch.stack(wh_weight, dim=0).detach()

            return (heatmap, box_target, wh_weight), None

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
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, num_obj)  # order: from big to small

        if self.area_cfg.type == 'norm':
            boxes_area_topk_log[:] = 1.
        # change order to `big → small`
        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes.clone()  # > (#obj, 4)
        feat_gt_boxes[:, [0, 2]] *= width_ratio
        feat_gt_boxes[:, [1, 3]] *= height_ratio
        feat_gt_boxes[:, [0, 2]] = torch.clamp(
            feat_gt_boxes[:, [0, 2]], min=0, max=feat_w - 1)
        feat_gt_boxes[:, [1, 3]] = torch.clamp(
            feat_gt_boxes[:, [1, 3]], min=0, max=feat_h - 1)
        feat_gt_hs, feat_gt_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                  feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        feat_gt_centers = torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                       (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], dim=1)
        feat_gt_centers_int = torch.zeros_like(feat_gt_centers)
        feat_gt_centers_int[:, 0] = (feat_gt_centers[:, 0] * width_ratio)  # (#obj, 2)
        feat_gt_centers_int[:, 1] = (feat_gt_centers[:, 1] * height_ratio)
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
        for obj_id in range(boxes_ind.shape[0]):  # > #obj: large -> small
            cls_id = gt_labels[obj_id]
            alpha_radius = (h_radiuses_alpha[obj_id].item(), w_radiuses_alpha[obj_id].item())
            # > `heatmap`: (#cls, H, W), `ct_ints`: (#obj, 2)
            fake_heatmap = fake_heatmap.zero_()  # (H, W)
            fake_heatmap = gen_gaussian_target(fake_heatmap, feat_gt_centers_int[obj_id], alpha_radius)
            gt_heatmap[cls_id] = torch.max(gt_heatmap[cls_id], fake_heatmap)
            if self.area_cfg.gaussian:
                if self.area_cfg.alpha != self.area_cfg.beta:
                    beta_radius = (h_radiuses_beta[obj_id].item(), w_radiuses_beta[obj_id].item())
                    gen_gaussian_target(fake_heatmap, feat_gt_centers_int[obj_id], beta_radius)
                box_target_inds = fake_heatmap > 0  # > (H, W)
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[obj_id], ctr_y1s[obj_id], ctr_x2s[obj_id], ctr_y2s[obj_id]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.area_cfg.agnostic:  # <-
                box_target[:, box_target_inds] = gt_boxes[obj_id][:, None]
                cls_id = 0  # TOCHECK: why assign `0` here?
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[obj_id][:, None]

            if self.area_cfg.gaussian:  # <-
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[obj_id]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = boxes_area_topk_log[obj_id] / box_target_inds.sum().float()

        return gt_heatmap, box_target, reg_weight