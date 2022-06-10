from collections import namedtuple
from typing import Dict

import timm
import torch
from torch import nn
from torch.nn import functional as F

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


class StaircaseStructure(nn.Module):
    def __init__(self, c2, c3, c4, c5, num_experts, num_proposals, bias=True):
        super(StaircaseStructure, self).__init__()
        self.interpolate_size = 30
        self.out_channels = num_experts * num_proposals
        self.num_experts = num_experts
        self.num_proposals = num_proposals

        start_channel = 0
        for i, c in enumerate([c2, c3, c4, c5]):
            start_channel += c
            dw = nn.Sequential(
                nn.Conv2d(start_channel, start_channel, kernel_size=3, stride=2, padding=1, bias=bias,
                          groups=start_channel),
                nn.BatchNorm2d(start_channel),
                nn.ReLU(inplace=True)
            )
            setattr(self, f"dw{i+1}", dw)

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.interpolate_size ** 2, out_features=1500),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1500, out_features=self.out_channels)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, c2, c3, c4, c5):
        latent_2 = self.dw1(c2)
        latent_3 = self.dw2(torch.cat([latent_2, c3], dim=1))
        latent_4 = self.dw3(torch.cat([latent_3, c4], dim=1))
        latent_5 = self.dw4(torch.cat([latent_4, c5], dim=1))
        out = F.interpolate(input=latent_5, size=(self.interpolate_size, self.interpolate_size))

        #  You should reshape the tensor first, then use softmax in num_expert dimension. that cause
        #  [batch_size, num_expert, num_props] tensor, and then consider num_props dimension ,has the weight of length num_expert.
        #  then abatain the num_prop proposal, each proposal is the weight sum of initial proposal features.
        out = out.sum(dim=1).flatten(1) #[batch_size, 900]
        out = self.linear(out) #[batch_size, num_experts * num_proposals]
        out = out.reshape([-1, self.num_proposals, self.num_experts])
        out = self.softmax(out)
        return out


class DynamicProposalGenerator(torch.nn.Module):
    def __init__(self, cfg, fpn_feature_channels):
        super(DynamicProposalGenerator, self).__init__()
        self.cfg = cfg
        self.num_experts = cfg.MODEL.NUM_EXPERTS
        self.num_proposals = cfg.MODEL.NUM_PROPOSALS
        self.init_proposal_features = nn.Embedding(self.num_experts, 256)
        self.init_proposal_boxes = nn.Embedding(self.num_experts, 4)  # cx, cy, w, h
        nn.init.uniform_(self.init_proposal_boxes.weight[:, :2], 0, 1)
        nn.init.uniform_(self.init_proposal_boxes.weight[:, 2:], 0, 1)

        # nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)  # size


        self.expert_weight_layer = StaircaseStructure(*fpn_feature_channels,
                                                      num_experts=self.num_experts, num_proposals=self.num_proposals)

    def forward(self, features):
        expert_weight = self.expert_weight_layer(*features) # [batch_size, num_proposals, num_experts]

        proposal_boxes = torch.matmul(expert_weight, self.init_proposal_boxes.weight.clone())  # [batch_size, num_proposal, 4]
        proposal_features = torch.matmul(expert_weight, self.init_proposal_features.weight.clone()) # [batch_size, num_proposal, 256]

        return proposal_boxes, proposal_features


class DynamicSparseRCNN(torch.nn.Module):
    def __init__(self, cfg, num_classes, backbone,
                 raw_outputs=False):
        super(DynamicSparseRCNN, self).__init__()
        assert backbone in _available_backbones.keys(), f"{backbone} is not available, currently we only support {_available_backbones.keys()}"
        self.cfg = cfg
        self.in_channels = 256
        self.raw_outputs = raw_outputs
        # model components
        self.backbone: nn.Module = timm.create_model(**_available_backbones[backbone])
        self.fpn = FPN(*self.backbone.feature_info.channels(), inner_channel=cfg.MODEL.FPN.OUT_CHANNELS)
        self.dynamic_proposal_generator = DynamicProposalGenerator(cfg, self.backbone.feature_info.channels())
        input_shape = self.get_input_shape(self.backbone.feature_info, new_channels=cfg.MODEL.FPN.OUT_CHANNELS)
        self.dynamic_head = DynamicHead(cfg, input_shape)


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

        proposal_boxes, proposal_features = self.dynamic_proposal_generator(features)
        features = self.fpn(*features)

        proposal_boxes_xyxy = proposal_boxes.new_zeros((batch_size, self.cfg.MODEL.NUM_PROPOSALS, 4))
        for i in range(batch_size):
            proposal_boxes_xyxy[i] = box_cxcywh_to_xyxy(proposal_boxes[i])
        proposal_boxes_xyxy = proposal_boxes_xyxy * img_whwh[:, None, :]

        # outputs_class: [(N, NUM_PROPOSALS, NUM_CLASSES)*NUM_HEADS]
        # outputs_coord: [(N, NUM_PROPOSALS, 4)*NUM_HEADS]
        outputs_class, outputs_coord = self.dynamic_head(features, proposal_boxes_xyxy,
                                                         proposal_features)

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
