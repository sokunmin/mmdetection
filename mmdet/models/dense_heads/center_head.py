import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, kaiming_init, build_upsample_layer, build_norm_layer
from mmcv.cnn import ConvModule, bias_init_with_prob
from mmcv.ops import ModulatedDeformConv2dPack
from mmcv.runner import force_fp32
import numpy as np
import math
from mmdet.core import multi_apply, calc_region, simple_nms
from .base_dense_head import BaseDenseHead
from ..builder import HEADS, build_loss
from ..utils.center_ops import gather_feature, transpose_and_gather_feat, topk, topk_channel
from ...core.bbox.iou_calculators.iou2d_calculator import bbox_areas


@HEADS.register_module
class CenterHead(BaseDenseHead):

    def __init__(self,
                 in_channels=64,
                 feat_channels=256,
                 use_dla=False,
                 down_ratio=4,
                 num_classes=1,
                 stacked_convs=1,
                 wh_out_channels=2,
                 reg_out_channels=2,
                 max_objs=128,
                 conv_cfg=None,
                 conv_bias='auto',
                 norm_cfg=None,
                 dcn_cfg=dict(
                     in_channels=(512, 256, 128, 64),
                     kernels=(4, 4, 4),
                     strides=(2, 2, 2),
                     paddings=(1, 1, 1),
                     out_paddings=(0, 0, 0)
                 ),
                 loss_cls=dict(
                     type='CenterFocalLoss',
                     gamma=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='CenterGIoULoss', loss_weight=5.0),
                 train_cfg=None,
                 test_cfg=None):
        super(CenterHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_dla = use_dla
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.stacked_convs = stacked_convs
        self.wh_out_channels = wh_out_channels
        self.reg_out_channels = reg_out_channels
        self.dcn_cfg = dcn_cfg

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

        # > upsample
        self.upsamples = self.build_upsample(self.dcn_cfg.in_channels,
                                             self.dcn_cfg.kernels,
                                             self.dcn_cfg.strides,
                                             self.dcn_cfg.paddings,
                                             self.dcn_cfg.out_paddings,
                                             self.norm_cfg)

        # > ct heads
        self.cls_head = self.build_head(self.num_classes)
        self.wh_head = self.build_head(self.wh_out_channels)
        self.reg_head = self.build_head(self.reg_out_channels)

    def build_upsample(self, in_channels, kernels, strides, paddings, out_paddings, norm_cfg=None):
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
        for _, m in self.cls_head.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        # > centernet
        bias_cls = bias_init_with_prob(0.1)
        # > ttfnet
        # bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_head[-1], std=0.01, bias=bias_cls)

        self._init_head_weights(self.wh_head)
        self._init_head_weights(self.reg_head)

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
        ct_cls = self.cls_head(x)  # > (B, #cls, 128, 128)
        ct_wh = self.wh_head(x)  # > (B, 2, 128, 128)
        ct_offset = self.reg_head(x)  # > (B, 2, 128, 128)
        return ct_cls, ct_offset, ct_wh

    def get_bboxes(self,
                   pred_ct_cls,
                   pred_ct_offset,
                   pred_ct_wh,
                   img_metas,
                   rescale=False):
        # `ct`: #cls, `ct_offset`: 2, `ct_wh`: 2
        batch, cls, height, width = pred_ct_cls.size()
        ct_cls = pred_ct_cls.detach().sigmoid_()  # > (#cls, H, W)
        ct_offset = pred_ct_offset.detach()  # > (B, 2, H, W)
        ct_wh = pred_ct_wh.detach()  # > (B, 2, H, W)
        num_topk = self.test_cfg.max_per_img

        # > perform nms & topk over center points
        ct_cls = simple_nms(ct_cls)  # used maxpool to filter the max score
        ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = self.topk_score(ct_cls, K=num_topk)

        if ct_offset is not None:
            ct_offset = gather_feature(ct_offset, ct_inds, use_transform=True)  # (B, 2, H, W) -> (B, K, 2)
            ct_offset = ct_offset.reshape(batch, num_topk, self.reg_out_channels)
            ct_xs = ct_xs.view(batch, num_topk, 1) + ct_offset[:, :, 0:1]
            ct_ys = ct_ys.view(batch, num_topk, 1) + ct_offset[:, :, 1:2]
        else:
            ct_xs = ct_xs.view(batch, num_topk, 1) + 0.5
            ct_ys = ct_ys.view(batch, num_topk, 1) + 0.5

        ct_wh = gather_feature(ct_wh, ct_inds, use_transform=True)  # (B, 2, H, W) -> (B, K, 2)
        ct_wh = ct_wh.reshape(batch, num_topk, self.wh_out_channels)  # (B, K, 2)

        # > `classes & scores`
        clses = ct_clses.view(batch, num_topk, 1).float()  # (B, K, 1)
        scores = ct_scores.view(batch, num_topk, 1)  # (B, K, 1)

        # > `bboxes`: (B, topk, 4)
        half_w, half_h = ct_wh[..., 0:1] / 2, ct_wh[..., 1:2] / 2
        bboxes = torch.cat([ct_xs - half_w, ct_ys - half_h,
                            ct_xs + half_w, ct_ys + half_h],
                           dim=2)  # (B, K, 4)
        bboxes *= self.down_ratio

        # `bboxes`: (B, K, 4), `bbox_scores`: (B, K, 1)
        bbox_result_list = []
        for batch_i in range(bboxes.shape[0]):  # > #imgs
            img_shape = img_metas[batch_i]['pad_shape']  # (512, 512, 3)
            bboxes_per_img = bboxes[batch_i]
            bbox_scores_per_img = scores[batch_i]  # (B, K, 1) -> (K, 1)
            labels_per_img = clses[batch_i]  # (B, K, 1) -> (K, 1)
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0] - 1)
            if rescale:
                scale_factor = img_metas[batch_i]['scale_factor']
                bboxes_per_img /= bboxes_per_img.new_tensor(scale_factor)
            bboxes_per_img = torch.cat([bboxes_per_img, bbox_scores_per_img], dim=1)  # (K, 4 + 1)
            labels_per_img = labels_per_img.squeeze(-1)  # (K, 1) -> (K,)
            bbox_result_list.append((bboxes_per_img, labels_per_img))

        return bbox_result_list

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

    def pseudo_nms(self, fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    def topk_score(self, scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = torch.floor_divide(topk_inds, width).float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = torch.floor_divide(index, K)
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


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
