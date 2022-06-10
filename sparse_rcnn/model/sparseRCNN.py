from collections import namedtuple
from typing import Dict

import timm
import torch
from torch import nn

from .head import DynamicHead
from ..utils.box_ops import box_cxcywh_to_xyxy
from .available_backbones import _available_backbones


class FPN(nn.Module):
    def __init__(self, c2, c3, c4, c5, inner_channel=256, bias=False):
        super(FPN, self).__init__()
        self.c2_to_f2 = nn.Conv2d(c2, inner_channel, 1, 1, 0, bias=bias)
        self.c3_to_f3 = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_to_f4 = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_f5 = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)

        self.p2_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p3_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)

    def forward(self, c2, c3, c4, c5):
        latent_2 = self.c2_to_f2(c2)
        latent_3 = self.c3_to_f3(c3)
        latent_4 = self.c4_to_f4(c4)
        latent_5 = self.c5_to_f5(c5)

        f4 = latent_4 + nn.UpsamplingBilinear2d(size=(latent_4.shape[2:]))(latent_5)
        f3 = latent_3 + nn.UpsamplingBilinear2d(size=(latent_3.shape[2:]))(f4)
        f2 = latent_2 + nn.UpsamplingBilinear2d(size=(latent_2.shape[2:]))(f3)
        p2 = self.p2_out(f2)
        p3 = self.p3_out(f3)
        p4 = self.p4_out(f4)
        p5 = self.p5_out(latent_5)
        return [p2, p3, p4, p5]


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """

    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class SparseRCNN(torch.nn.Module):
    def __init__(self, cfg, num_classes, backbone,
                 raw_outputs=False):
        super(SparseRCNN, self).__init__()
        assert backbone in _available_backbones.keys(), f"{backbone} is not available, currently we only support {_available_backbones.keys()}"
        self.cfg = cfg
        self.in_channels = 256
        self.raw_outputs = raw_outputs
        # model components
        self.backbone: nn.Module = timm.create_model(**_available_backbones[backbone])
        self.fpn = FPN(*self.backbone.feature_info.channels(), inner_channel=cfg.MODEL.FPN.OUT_CHANNELS)
        input_shape = self.get_input_shape(self.backbone.feature_info, new_channels=cfg.MODEL.FPN.OUT_CHANNELS)
        self.dynamic_head = DynamicHead(cfg, input_shape)

        # embedding parameters
        self.init_proposal_features = nn.Embedding(self.cfg.MODEL.NUM_PROPOSALS, self.in_channels)
        self.init_proposal_boxes = nn.Embedding(self.cfg.MODEL.NUM_PROPOSALS, 4)  # cx, cy, w, h
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)  # center
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)  # size

    @staticmethod
    def get_input_shape(feature_info: timm.models.features.FeatureInfo, new_channels: int):
        src = {}
        for i, info in enumerate(feature_info.info):
            src[f"p{i + 1}"] = ShapeSpec(
                channels=new_channels,
                stride=info["reduction"],
            )
        return src

    def forward(self, x: torch.Tensor, img_whwh: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _, *image_wh_pad = x.shape
        features = self.backbone(x)
        features = self.fpn(*features)

        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes_xyxy = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes_xyxy = proposal_boxes_xyxy[None] * img_whwh[:, None, :]

        # outputs_class: [(N, NUM_PROPOSALS, NUM_CLASSES)*NUM_HEADS]
        # outputs_coord: [(N, NUM_PROPOSALS, 4)*NUM_HEADS]
        outputs_class, outputs_coord = self.dynamic_head(features, proposal_boxes_xyxy,
                                                         self.init_proposal_features.weight)

        if not self.training and not self.raw_outputs:
            scores = torch.sigmoid(outputs_class[-1])
            scores, labels = torch.max(scores, -1)
            output = {
                "scores": scores,
                "labels": labels,
                "boxes": outputs_coord[-1]
            }
            return output
        else:
            output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1],
                      'aux_outputs': [{'pred_logits': a, 'pred_boxes': b}
                                      for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]}
            return output
