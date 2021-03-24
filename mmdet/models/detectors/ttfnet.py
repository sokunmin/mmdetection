from .single_stage_mt import SingleStageMultiDetector
from ..builder import DETECTORS


@DETECTORS.register_module()
class TTFNet(SingleStageMultiDetector):

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
        super(TTFNet, self).__init__(backbone, neck, bbox_head, mask_head, keypoint_head,
                                        train_cfg, test_cfg, pretrained, loss_balance, show_tb_debug)