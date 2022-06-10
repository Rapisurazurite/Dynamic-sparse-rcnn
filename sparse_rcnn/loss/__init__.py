from torch import nn

from .SetCriterion import SetCriterion
from ..utils.HungarianMatcher import HungarianMatcher
from ..utils.OTA import OtaMatcher


class SparseRcnnLoss(nn.Module):
    def __init__(self, cfg):
        super(SparseRcnnLoss, self).__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        class_weight = cfg.MODEL.LOSS.CLASS_WEIGHT
        giou_weight = cfg.MODEL.LOSS.GIOU_WEIGHT
        l1_weight = cfg.MODEL.LOSS.L1_WEIGHT
        no_object_weight = cfg.MODEL.LOSS.NO_OBJECT_WEIGHT
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        losses = ["labels", "boxes"]

        if cfg.MODEL.LOSS.MATCHER.NAME == "HungarianMatcher":
            matcher = HungarianMatcher(cfg=cfg,
                                       cost_class=class_weight,
                                       cost_bbox=l1_weight,
                                       cost_giou=giou_weight,
                                       use_focal=self.use_focal)
        elif cfg.MODEL.LOSS.MATCHER.NAME == "OtaMatcher":
            matcher = OtaMatcher(cfg=cfg,
                                 cost_class=class_weight,
                                 cost_bbox=l1_weight,
                                 cost_giou=giou_weight,
                                 cost_bg=2,
                                 use_focal=self.use_focal)
        else:
            raise NotImplementedError

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal)

    def forward(self, output, targets):
        loss_dict = self.criterion(output, targets)
        return loss_dict
