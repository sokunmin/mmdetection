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
                 ct_head_cfg=dict(
                     wh_out_channels=2,
                     reg_out_channels=2,
                     offset_base=16.,
                     area_process='log',
                     with_agnostic=True,
                     with_gaussian=True
                 ),
                 kp_head_cfg=dict(
                     ct_out_channels=17,
                     reg_out_channels=34,
                     offset_out_channels=2
                 ),
                 alpha=0.54,
                 beta=0.54,
                 max_objs=128,
                 conv_cfg=None,
                 conv_bias='auto',
                 norm_cfg=None,
                 loss_cls=dict(
                     type='CenterFocalLoss',
                     gamma=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0),
                 train_cfg=None,
                 test_cfg=None):
        super(CenterPoseHead, self).__init__()
        self.ct_head_cfg = ct_head_cfg
        self.kp_head_cfg = kp_head_cfg
        assert ct_head_cfg.area_process in [None, 'norm', 'log', 'sqrt']
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_dla = use_dla
        self.num_classes = num_classes
        self.num_keypoints = kp_head_cfg.ct_out_channels
        self.alpha = alpha
        self.beta = beta
        self.max_objs = max_objs
        self.stacked_convs = stacked_convs

        self.limbs = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_norm = norm_cfg is not None
        self.fp16_enabled = False
        self.down_ratio = down_ratio
        self.base_loc = None

        # > ct heads
        self.hm = self.build_head(self.num_classes)
        self.wh = self.build_head(self.ct_head_cfg.wh_out_channels)
        self.reg = self.build_head(self.ct_head_cfg.reg_out_channels)

        # > #TODO: split up joint heads
        self.hm_hp = self.build_head(self.kp_head_cfg.ct_out_channels)
        self.hps = self.build_head(self.kp_head_cfg.reg_out_channels)
        self.hp_offset = self.build_head(self.kp_head_cfg.offset_out_channels)

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
        for _, m in self.hm.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        # > centernet
        bias_cls = bias_init_with_prob(0.1)
        # > ttfnet
        # bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)
        normal_init(self.hm_hp[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.wh)
        self._init_head_weights(self.hps)
        self._init_head_weights(self.reg)
        self._init_head_weights(self.hp_offset)

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
        x = feats[-1]  # (64, H, W)

        hm_ct = self.hm(x)  # > (64, H, W) -> (128, H, W) -> (#cls, H, W)
        ct_wh = self.wh(x)
        ct_offset = self.reg(x)
        ct_kp = self.hps(x)

        kp_ct = self.hm_hp(x)
        kp_offset = self.hp_offset(x)
        # `ct`: 1, `ct_offset`:2, `ct_wh`: 2, `ct_kp`:34, `kp_ct`: 17, `kp_offset`: 2
        return hm_ct, ct_offset, ct_wh, ct_kp, kp_ct, kp_offset

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
        ct = pred_ct.detach().sigmoid_()  # > (#cls, H, W)
        ct_offset = pred_ct_offset.detach()  # > (B, 2, H, W)
        ct2wh_reg = pred_ct_wh.detach()  # > (B, 2, H, W)
        ct2kp_loc = pred_ct_kp.detach()  # > (B, #kp x 2, H, W)
        kp_ct = pred_kp_ct.detach().sigmoid_()  # > (B, #kp, H, W)
        kp_ct_offset = pred_kp_ct_offset.detach()  # > (B, 2, H, W)
        num_topk = self.test_cfg.max_per_img
        num_joints = self.num_keypoints

        # > perform nms & topk over center points
        ct = simple_nms(ct)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = topk(ct, topk=num_topk)

        # [BBox + Scores]
        # > refine center points by offsets
        # > [1] TTFNet's way
        # ct_xs = ct_xs.view(batch, num_topk, 1) * self.down_ratio  # TOCHECK: multiply `down_ratio`: 4 first ?
        # ct_ys = ct_ys.view(batch, num_topk, 1) * self.down_ratio  # (B, K, 1)
        # > [2] CenterNet's way
        ct_offset = transpose_and_gather_feat(ct_offset, ct_inds)
        ct_offset = ct_offset.view(batch, num_topk, self.ct_head_cfg.reg_out_channels)  # (B, K, 2)
        ct_xs = ct_xs.view(batch, num_topk, 1) + ct_offset[:, :, 0:1]  # (B, K, 1)
        ct_ys = ct_ys.view(batch, num_topk, 1) + ct_offset[:, :, 1:2]  # (B, K, 1)

        # > `width/height`: (B, 2, H, W) ∩ (B, K) -> (B, HxW, 2) ∩ (B, K, 2) -> (B, K, 2)
        ct2wh_reg = transpose_and_gather_feat(ct2wh_reg, ct_inds)
        ct2wh_reg = ct2wh_reg.view(batch, num_topk, self.ct_head_cfg.wh_out_channels)  # (B, K, 2)

        # > `classes & scores`
        bbox_clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        bbox_scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)

        # > `bboxe`s: (B, topk, 4)
        bboxes = torch.cat([ct_xs - ct2wh_reg[..., 0:1] / 2,
                            ct_ys - ct2wh_reg[..., 1:2] / 2,
                            ct_xs + ct2wh_reg[..., 0:1] / 2,
                            ct_ys + ct2wh_reg[..., 1:2] / 2], dim=2)

        # > center to joint offsets (out: 34)
        # `ct2kp_loc` ∩ `ct_inds`: (B, 34, H, W) ∩ (B, K) -> (B, HxW, 34) ∩ (B, K, 34) -> (B, K, 34)
        ct2kp_loc = transpose_and_gather_feat(ct2kp_loc, ct_inds)
        ct2kp_loc = ct2kp_loc.view(batch, num_topk, num_joints * 2)
        # (B, K) -> (B, K, 1) -> (B, K, #kp)
        ct2kp_loc[..., ::2] += ct_xs.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_loc[..., 1::2] += ct_ys.view(batch, num_topk, 1).expand(batch, num_topk, num_joints)
        ct2kp_loc = ct2kp_loc.view(batch, num_topk, num_joints, 2).permute(0, 2, 1, 3).contiguous()

        # > NOTE: `ct_det_kps`: regressed joint locations, (B, #kp, K, K, 2)
        ct_det_kps = ct2kp_loc.unsqueeze(3).expand(batch, num_joints, num_topk, num_topk, 2)

        # [Pose]
        # > centers as joints (out:17)
        kp_ct = simple_nms(kp_ct)  # (B, #kp, H, W)
        # `kp_ct`: (B, #kp, H, W) -> (B, #kp, HxW) -> `kp_ct_score/kp_ct_inds`: (B, 17, K)
        kp_ct_scores, kp_ct_inds, kp_ct_ys, kp_ct_xs = topk_channel(kp_ct, topk=num_topk)

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
        kps = (1 - mask) * kp_det_kps + mask * ct2kp_loc
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
        all_targets = self.target_generator(gt_bboxes, gt_labels, img_metas)  # heatmap, box_target, reg_weight
        hm_loss, wh_loss = self.loss_calc(pred_heatmap, pred_wh, *all_targets)
        return {'loss_heatmap': hm_loss, 'loss_wh': wh_loss}

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom,
                          w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def target_single_image(self, gt_boxes, gt_labels, feat_shape):
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
        output_h, output_w = feat_shape  # > downsampled size
        heatmap_channel = self.num_classes  # > #cls

        heatmap = gt_boxes.new_zeros((heatmap_channel, output_h, output_w))  # > (#cls, H, W)
        fake_heatmap = gt_boxes.new_zeros((output_h, output_w))  # > (H, W)
        # > `wh_planes`: 4
        box_target = gt_boxes.new_ones((self.ct_head_cfg.plane, output_h, output_w)) * -1  # > (4, 128, 128)
        reg_weight = gt_boxes.new_zeros((self.ct_head_cfg.plane // 4, output_h, output_w))  # > (1, 128, 128)

        if self.ct_head_cfg.area_process == 'log':  # <-
            boxes_areas_log = bbox_areas(gt_boxes).log()  # TOCHECK: why use `log`? See Eq(7)
        elif self.ct_head_cfg.area_process == 'sqrt':
            boxes_areas_log = bbox_areas(gt_boxes).sqrt()
        else:
            boxes_areas_log = bbox_areas(gt_boxes)
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log, boxes_areas_log.size(0))  # sort descending

        if self.ct_head_cfg.area_process == 'norm':
            boxes_area_topk_log[:] = 1.

        gt_boxes = gt_boxes[boxes_ind]
        gt_labels = gt_labels[boxes_ind]

        feat_gt_boxes = gt_boxes / self.down_ratio  # > `down_ratio`: 4
        feat_gt_boxes[:, [0, 2]] = torch.clamp(feat_gt_boxes[:, [0, 2]], min=0,
                                               max=output_w - 1)  # x_min, x_max
        feat_gt_boxes[:, [1, 3]] = torch.clamp(feat_gt_boxes[:, [1, 3]], min=0,
                                               max=output_h - 1)  # y_min, y_max
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        # we calc the center and ignore area based on the gt-boxes of the origin scale
        # no peak will fall between pixels
        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)  # (#obj, 2)
        # `wh_gaussian`: True, `alpha`: 0.54, `beta`: 0.54
        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()  # > TOCHECK: why?
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.ct_head_cfg.with_gaussian and self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        if not self.ct_head_cfg.with_gaussian:
            # calculate positive (center) regions
            r1 = (1 - self.beta) / 2
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
            ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [torch.round(x.float() / self.down_ratio).int()
                                                  for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
            ctr_x1s, ctr_x2s = [torch.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
            ctr_y1s, ctr_y2s = [torch.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):  # > #bbox: large -> small
            cls_id = gt_labels[k] - 1  # (#obj,) -> scalar

            fake_heatmap = fake_heatmap.zero_()  # (H, W)
            self.draw_truncate_gaussian(fake_heatmap,  # (H, W)
                                        ct_ints[k],  # (#obj, 2) -> (2,)
                                        h_radiuses_alpha[k].item(),  # (#obj,) -> scalar
                                        w_radiuses_alpha[k].item())  # (#obj,) -> scalar
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.ct_head_cfg.with_gaussian:  # <-
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap.zero_()
                    self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                                h_radiuses_beta[k].item(),
                                                w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = torch.zeros_like(fake_heatmap, dtype=torch.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.ct_head_cfg.with_agnostic:  # <-
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            if self.ct_head_cfg.with_gaussian:  # <-
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]  # TOCHECK: why multiply this?  normalized?
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum().float()

        return heatmap, box_target, reg_weight

    def target_generator(self, gt_boxes, gt_labels, img_metas):
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
            feat_shape = (img_metas[0]['pad_shape'][0] // self.down_ratio,
                          img_metas[0]['pad_shape'][1] // self.down_ratio)  # (128, 128)
            # `heatmap`: (B, #cls, H, W), `box_target`: (B, 4, H, W), `reg_weight`: (B, 1, H, W)
            heatmap, box_target, reg_weight = multi_apply(
                self.target_single_image,
                gt_boxes,  # (B, #obj, 4)
                gt_labels,  # (B, #obj)
                feat_shape=feat_shape  # (H, W)
            )
            heatmap, box_target = [torch.stack(t, dim=0).detach() for t in [heatmap, box_target]]
            reg_weight = torch.stack(reg_weight, dim=0).detach()

            return heatmap, box_target, reg_weight

    def loss_calc(self,
                  pred_hm,
                  pred_wh,
                  heatmap_target,
                  box_target,
                  wh_weight):
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
        H, W = pred_hm.shape[2:]
        pred_hm = torch.clamp(pred_hm.sigmoid_(), min=1e-4, max=1 - 1e-4)
        num_pos = heatmap_target.eq(1).float().sum()
        hm_loss = self.loss_cls(pred_hm, heatmap_target, avg_factor=1 if num_pos == 0 else num_pos)

        mask = wh_weight.view(-1, H, W)  # (B, 1, H, W) -> (B, H, W)
        avg_factor = mask.sum() + 1e-4
        # `base_loc`: None
        if self.base_loc is None or H != self.base_loc.shape[1] or W != self.base_loc.shape[2]:
            base_step = self.down_ratio  # > 4
            shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap_target.device)
            shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                    dtype=torch.float32, device=heatmap_target.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (B, H, W, 4)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)

        gt_boxes = box_target.permute(0, 2, 3, 1)  # (B, 4, H, W) -> (B, H, W, 4)
        wh_loss = self.loss_bbox(pred_boxes, gt_boxes, mask, avg_factor=avg_factor)
        return hm_loss, wh_loss


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
