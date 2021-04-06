import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO

from . import CocoDataset
from .builder import DATASETS
from ..core.bbox.transforms import mask2polybox

try:
    import pycocotools
    assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module
class CocoPersonDataset(CocoDataset):

    CLASSES = ('person')
    KEYPOINTS = ('nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder',
                 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist',
                 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle')

    def load_annotations(self, ann_file):
        """Load annotation of 'person' class only from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids(cat_ids=self.cat_ids)
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)

        # check `skeleton` key existed for `person` class
        cats = self.coco.load_cats(self.cat_ids)[0]
        if 'skeleton' in cats:
            self.skeleton = np.array(cats['skeleton']) - 1
        else:
            print("The dataset json does not contain `skeleton` key.")
            self.skeleton = None
        self.num_keypoints = len(self.KEYPOINTS)
        self.flip_pairs = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                                    [11, 12], [13, 14], [15, 16]])
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_keypoints = []
        gt_affine_centers = []
        gt_affine_scales = []
        width, height = img_info['width'], img_info['height']
        gt_mask_all = np.zeros((height, width), dtype=np.uint8)
        gt_mask_miss = np.zeros((height, width), dtype=np.uint8)
        mask_crowd = None
        img_info, ann_info = self.correct_mislabeled(img_info, ann_info)
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False) or ann['num_keypoints'] == 0:
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            # > used for affine transform
            center = [x1 + w / 2, y1 + h / 2]
            scale = h / height
            mask = self.coco.annToMask(ann)
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                mask_temp = np.bitwise_and(mask, gt_mask_all)
                mask_crowd = mask - mask_temp
            else:
                # > bboxes and labels
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                # > affine center & scale
                gt_affine_centers.append(center)
                gt_affine_scales.append(scale)
                # > masks
                gt_masks.append(ann.get('segmentation', None))
                gt_mask_all = np.bitwise_or(mask, gt_mask_all)
                if ann['num_keypoints'] <= 0:
                    gt_mask_miss = np.bitwise_or(mask, gt_mask_miss)
                # > keypoints
                gt_keypoint = np.zeros((self.num_keypoints, 3), dtype=np.float32)
                kp = np.array(ann['keypoints'])
                gt_keypoint[:, 0] = kp[0::3]
                gt_keypoint[:, 1] = kp[1::3]
                # `0`: not labeled, `1`: labeled but invisible, `2`: labeled and visible
                gt_keypoint[:, 2] = (kp[2::3] > 0).astype(np.int32)
                gt_keypoints.append(gt_keypoint)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if mask_crowd is not None:
            gt_mask_miss = np.logical_not((np.bitwise_or(gt_mask_miss, mask_crowd)))
            gt_mask_all = np.bitwise_or(gt_mask_all, mask_crowd)
        else:
            gt_mask_miss = np.logical_not(gt_mask_miss)

        gt_mask_miss = gt_mask_miss.astype(np.uint8)
        gt_mask_all = gt_mask_all.astype(np.uint8)
        gt_mask_miss *= 255
        gt_mask_all *= 255
        gt_mask_ignore = np.concatenate(
            (gt_mask_all[None, :], gt_mask_miss[None, :]), axis=0)

        if gt_keypoints:
            gt_affine_centers = np.array(gt_affine_centers, dtype=np.float32)
            gt_affine_scales = np.array(gt_affine_scales, dtype=np.float32)
            gt_keypoints = np.array(gt_keypoints, dtype=np.float32)
        else:
            gt_affine_centers = np.zeros((0, 2), dtype=np.float32)
            gt_affine_scales = np.zeros((0, 1), dtype=np.float32)
            gt_keypoints = np.zeros((0, self.num_keypoints, 3), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks,
            masks_ignore=gt_mask_ignore,
            seg_map=seg_map,
            keypoints=gt_keypoints,
            affine_centers=gt_affine_centers,
            affine_scales=gt_affine_scales,
            flip_pairs=self.flip_pairs,
        )

        return ann

    def _poly2mask(self, mask_ann, img_h, img_w):
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
        return rle

    def correct_mislabeled(self, img_info, ann_info):
        img_name = img_info['file_name']
        if img_info.get('corrected', False):  # corrected, return it
            return img_info, ann_info

        if '000000223299' in img_name:
            for i, p in enumerate(ann_info):
                num_keypoints = ann_info[i]['num_keypoints']
                if num_keypoints == 16:
                    bbox = ann_info[i]['bbox']
                    bbox[2] += 145
                    keypoints = ann_info[i]['keypoints']
                    keypoints[15*3] += 37
                    ann_info[i]['keypoins'] = keypoints
            img_info['corrected'] = True
        if '000000251523' in img_name:
            obj1_masks = ann_info[0]['segmentation']
            obj2_keypoints = ann_info[1]['keypoints']
            obj2_masks = ann_info[1]['segmentation']

            obj1_masks = maskUtils.decode(self._poly2mask(obj1_masks, img_info['height'], img_info['width']))
            obj2_masks = maskUtils.decode(self._poly2mask(obj2_masks, img_info['height'], img_info['width']))
            new_mask = np.maximum(obj1_masks, obj2_masks)
            new_keypoints = obj2_keypoints
            new_bbox = mask2polybox(new_mask, box2xywh=False)[1]
            new_mask = maskUtils.encode(new_mask)

            ann_info[0]['bbox'] = new_bbox
            ann_info[0]['keypoints'] = new_keypoints
            ann_info[0]['segmentation'] = new_mask
            ann_info[0]['num_keypoints'] = (np.array(new_keypoints)[2::3] > 0).sum()
            ann_info[0]['area'] = maskUtils.area(new_mask)
            del ann_info[1]
            img_info['corrected'] = True
        if '000000001732' in img_name:
            obj_masks = ann_info[0]['segmentation']
            obj_masks = maskUtils.decode(self._poly2mask(obj_masks, img_info['height'], img_info['width']))  # (h, w)
            new_polygon, new_bbox = mask2polybox(obj_masks, flatten=False)
            new_polygon = new_polygon[0]
            for i, poly in enumerate(new_polygon):
                if poly[1] > 390:
                    new_polygon[i, 1] = poly[1] + 70
                    if new_polygon[i, 1] > 430:
                        new_polygon[i, 0] = poly[0] - 10
            new_polygon = [new_polygon.reshape(-1)]
            new_mask = maskUtils.decode(self._poly2mask(new_polygon, img_info['height'], img_info['width']))
            new_polygon, new_bbox = mask2polybox(new_mask, box2xywh=False, flatten=True)
            new_mask = maskUtils.encode(new_mask)
            ann_info[0]['bbox'] = new_bbox
            ann_info[0]['segmentation'] = new_mask
            ann_info[0]['area'] = maskUtils.area(new_mask)
            # self._showAnns(img_info, ann_info)  # set box2xywh=True for showing
            img_info['corrected'] = True
        if '000000044170' in img_name:
            obj_masks = ann_info[0]['segmentation']
            obj_masks = maskUtils.decode(self._poly2mask(obj_masks, img_info['height'], img_info['width']))  # (h, w)
            new_polygon, new_bbox = mask2polybox(obj_masks, flatten=False)
            new_polygon = new_polygon[0]
            for i, poly in enumerate(new_polygon):
                if poly[0] > 300 and (poly[1] < 200 or poly[1] > 330):
                    new_polygon[i, 0] = poly[0] - 15
                if poly[0] < 300 and poly[1] > 150 and poly[1] < 180:
                    new_polygon[i, 1] = poly[1] + 10
                if poly[0] < 200 and poly[1] > 180:
                    new_polygon[i, 0] = poly[0] - 10
            new_polygon = [new_polygon.reshape(-1)]
            new_mask = maskUtils.decode(self._poly2mask(new_polygon, img_info['height'], img_info['width']))
            new_polygon, new_bbox = mask2polybox(new_mask, box2xywh=False, flatten=True)
            new_mask = maskUtils.encode(new_mask)
            ann_info[0]['bbox'] = new_bbox
            ann_info[0]['segmentation'] = new_mask
            ann_info[0]['area'] = maskUtils.area(new_mask)
            img_info['corrected'] = True
        if '000000257336' in img_name:
            obj_masks = ann_info[0]['segmentation']
            obj_masks = maskUtils.decode(self._poly2mask(obj_masks, img_info['height'], img_info['width']))  # (h, w)
            new_polygon, new_bbox = mask2polybox(obj_masks, flatten=False)
            new_polygon = new_polygon[0]
            for i, poly in enumerate(new_polygon):
                if poly[1] > 360:
                    new_polygon[i, 1] = poly[1] + 20
            new_polygon = [new_polygon.reshape(-1)]
            new_mask = maskUtils.decode(self._poly2mask(new_polygon, img_info['height'], img_info['width']))
            new_polygon, new_bbox = mask2polybox(new_mask, box2xywh=False, flatten=True)
            new_mask = maskUtils.encode(new_mask)
            ann_info[0]['bbox'] = new_bbox
            ann_info[0]['segmentation'] = new_mask
            ann_info[0]['area'] = maskUtils.area(new_mask)
            img_info['corrected'] = True
        return img_info, ann_info

    def _showAnns(self, img_info, ann_info):
        import matplotlib.pyplot as plt
        import skimage.io as io
        img = io.imread(img_info['coco_url'])
        plt.imshow(img)
        self.coco.showAnns(ann_info, draw_bbox=True)
        plt.show()