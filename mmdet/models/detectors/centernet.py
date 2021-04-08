import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from mmcv import color_val, imshow, tensor2imgs
import cv2
import mmcv
import os
# from mmcv.runner import HOOKS, LoggerHook, master_only, TensorboardLoggerHook
from mmcv.runner import TensorboardLoggerHook, HOOKS, master_only

from mmdet.core import bbox2result, bbox_mapping_back
from ..builder import DETECTORS
from .single_stage_mt import SingleStageMultiDetector

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use('WXAgg')
except ImportError:
    matplotlib = None


@DETECTORS.register_module()
class CenterNet(SingleStageMultiDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_balance=None,
                 show_tb_debug=False):
        super(CenterNet, self).__init__(backbone, neck, bbox_head, mask_head, keypoint_head,
                                        train_cfg, test_cfg, pretrained, loss_balance, show_tb_debug)

    def merge_aug_results(self, aug_results, img_metas):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes, aug_labels = [], []
        for bboxes_labels, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']  # using shape before padding
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes, labels = bboxes_labels
            bboxes, scores = bboxes[:, :4], bboxes[:, -1:]
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            aug_labels.append(labels)

        bboxes = torch.cat(recovered_bboxes, dim=0)
        labels = torch.cat(aug_labels)

        if bboxes.shape[0] > 0:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=False):
        """Augment testing of CornerNet.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))

        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, [img_metas[ind], img_metas[flip_ind]], False, False)
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])

        bboxes, labels = self.merge_aug_results(aug_results, img_metas)
        bbox_results = bbox2result(bboxes, labels, self.bbox_head.num_classes)

        return [bbox_results]

    def train_step(self, data, optimizer):
        """
            SEE: https://github.com/open-mmlab/mmdetection/issues/231
        """
        if self.show_tb_debug:
            losses, debug_results = self(**data)
        else:
            losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        if self.show_tb_debug:
            # `debug_results` is return by `debug_process()`
            pred_feats, target_feats, filenames = debug_results
            outputs['pred_imgs'] = pred_feats
            outputs['target_imgs'] = target_feats
            outputs['filenames'] = filenames
        return outputs

    def debug_train(self, imgs, all_preds, all_targets, all_metas, gt_inputs, show_results=False):
        """
        This method is called in SingleStageMultiDetector.forward_train()
        SEE:
            [1] https://stackoverflow.com/questions/44535068/opencv-python-cover-a-colored-mask-over-a-image
            [2] https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
            [3] https://www.tensorflow.org/tensorboard/image_summaries
        """
        eps = 1e-12
        pred_ct_saliency, pred_ct_shape = all_preds['mask']
        boxes_target, masks_target = all_targets['mask']
        ct_ind_target = all_targets['bbox'][-1]
        pred_ct_saliency = torch.clamp(pred_ct_saliency, min=eps, max=1 - eps).detach()
        img_metas = all_metas['img']
        mean, std, to_rgb, norm_rgb = img_metas[0]['img_norm_cfg'].values()

        gt_masks = gt_inputs[1]

        # preprocess predictions
        batch, max_obj = ct_ind_target.size()
        shape_dim = self.mask_head.shape_dim
        pred_ct_saliency = torch.clamp(pred_ct_saliency, min=eps, max=1 - eps)
        pred_ct_shape = torch.clamp(pred_ct_shape, min=eps, max=1 - eps)
        pred_ct_shape = self.mask_head._transpose_and_gather_feat(pred_ct_shape, ct_ind_target)  # (B, SxS, H, W) -> (B, K, SxS)
        pred_ct_shape = pred_ct_shape.view(batch, max_obj, shape_dim, shape_dim)

        # resize images to feature-scale
        img_shape = tuple(imgs[0].size()[1:])
        imgs = tensor2imgs(imgs.detach(), mean, std, to_rgb=False)  # (#img, feat_h, feat_w, 3)

        # resize saliency maps to image-scale
        img_salient_maps = F.interpolate(pred_ct_saliency, img_shape,
                                         mode='bilinear', align_corners=False).squeeze()
        img_salient_maps = img_salient_maps.cpu().numpy() * 255  # (#img, img_h, img_w)

        # overlays
        color_overlays = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                          (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        num_color = len(color_overlays)
        num_img = len(imgs)
        filenames, blend_preds, blend_targets = [], [], []
        for img_id, img, gt_mask, img_saliency, meta in \
                zip(range(num_img), imgs, gt_masks, img_salient_maps, img_metas):  # > #imgs: 16
            num_obj = gt_mask.size(0)
            gt_mask = (gt_mask.cpu().numpy() * 255).astype(np.uint8)
            filename = os.path.basename(meta['filename'])
            filenames.append(filename)
            text = "[#obj=" + str(num_obj) + "] " + filename

            # [1] blend image-scale image and saliency map
            img_blend = self.overlay_heatmap_to_image(img, img_saliency, dtype=np.uint8)
            self.put_text(img_blend, text)
            blend_preds.append(img_blend)
            if show_results:
                plt.imshow(img_blend)
                plt.show()

            # [2] overlay gt masks to image
            blend_target = img
            if num_obj == 0:
                overlay_mask = np.zeros((img_shape[1], img_shape[0]), dtype=np.uint8)
                blend_target = self.overlay_colormask_to_image(blend_target, overlay_mask, color=color_overlays[0])
            else:
                for obj_id in range(num_obj):  # > #objs
                    blend_target = self.overlay_colormask_to_image(
                        blend_target, gt_mask[obj_id], color=color_overlays[obj_id % num_color])
            self.put_text(blend_target, text)
            blend_targets.append(blend_target)
            if show_results:
                plt.imshow(blend_target)
                plt.show()

        # (B, H, W, 3)
        blend_preds = np.stack(blend_preds)
        blend_targets = np.stack(blend_targets)
        return blend_preds, blend_targets, filenames

    def overlay_heatmap_to_image(self, image, mask, dtype=np.int32, alpha=0.3):
        if 0.0 <= mask.max() <= 1.0:
            mask = (mask * 255)
        if issubclass(image.dtype.type, np.floating):
            image = image.astype(dtype)
        if issubclass(mask.dtype.type, np.floating):
            mask = (mask * 255).astype(np.uint8)

        mask = cv2.normalize(~mask, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET).astype(dtype)
        blend = cv2.addWeighted(image, 1.0, mask, alpha, 0)
        return blend

    def overlay_colormask_to_image(self, image, mask, color, alpha=0.45):
        assert isinstance(color, tuple) and len(color) == 3
        # create a color overlay mask
        color_overlay = np.zeros(image.shape, np.uint8)
        color_overlay[:, :] = color
        overlay_mask = cv2.bitwise_and(color_overlay, color_overlay, mask=mask)#.astype(np.int32)
        # blend image mask with a overlay mask
        blend = cv2.addWeighted(image, 1.0, overlay_mask, alpha, 0)
        return blend

    def put_text(self, image, text, pos=(10, 35), font_size=0.8, thickness=2, color=(255, 87, 255)):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, color, thickness, cv2.LINE_AA)


# SEE https://github.com/open-mmlab/mmdetection/issues/231
@HOOKS.register_module()
class TensorboardImageHook(TensorboardLoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardImageHook, self).__init__(log_dir, interval, ignore_last,
                                                   reset_flag, by_epoch)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_step(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_step(runner))
        # > for example: `log_images = runner.outputs.get('log_images')`

        filenames = runner.outputs.get('filenames', None)
        pred_imgs = runner.outputs.get('pred_imgs', None)
        target_imgs = runner.outputs.get('target_imgs', None)

        if pred_imgs is not None:
            self.writer.add_image(
                'image/saliency', pred_imgs, runner.iter, dataformats='NHWC'
            )
        if target_imgs is not None:
            self.writer.add_image(
                'image/mask_targets', target_imgs, runner.iter, dataformats='NHWC'
            )

    def show(self, img, img_shape, dpi=80):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        depth, height, width = img_shape

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Display the image.
        ax.imshow(npimg, cmap='gray')
        plt.show()

    @master_only
    def after_run(self, runner):
        self.writer.close()