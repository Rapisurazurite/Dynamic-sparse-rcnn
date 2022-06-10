import numpy as np
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


class OtaMatcher(nn.Module):
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_bg: float = 1,
                 use_focal: bool = False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.cost_bg = cost_bg
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES

        self.k = cfg.MODEL.LOSS.MATCHER.K
        self.k = self.k if self.k > 0 else -1

        eps = float(cfg.MODEL.LOSS.MATCHER.EPS)
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=100)

        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.LOSS.FOCAL_LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.LOSS.FOCAL_LOSS_GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, T=6, t=6):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            T: Number of iterations of rcnn head.
            t: index of current iteration of rcnn head. range [1, T].

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.k > 0:
            assert self.k - 0.5 * (T - 1) > 0, "k should be greater than 0.5*(T-1) to ensure the min(k) > 0"

        bs, num_queries = outputs["pred_logits"].shape[:2]
        indice = []

        for index in range(bs):
            tgt_ids = targets[index]["gt_classes"].reshape(-1)  # [num_target_boxes]
            tgt_bbox = targets[index]["gt_boxes"].reshape(-1, 4)  # [num_target_boxes, 4]
            num_gt = len(tgt_ids)
            out_prob = outputs["pred_logits"][index]  # [num_queries, num_classes]
            out_bbox = outputs["pred_boxes"][index]  # [num_queries, 4]

            # Compute the focal loss for each image
            if self.use_focal:
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                tgt_ids_onehot = F.one_hot(tgt_ids, num_classes=self.num_classes).float()
                loss_cls = sigmoid_focal_loss_jit(
                    out_prob.unsqueeze(0).expand(num_gt, num_queries, -1),
                    tgt_ids_onehot.unsqueeze(1).expand(num_gt, num_queries, -1),
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none"
                ).sum(dim=-1)  # [num_gt, num_queries]
                loss_bg = sigmoid_focal_loss_jit(
                    out_prob,
                    torch.zeros_like(out_prob),
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none"
                ).sum(dim=-1)  # [num_queries]
            else:
                raise NotImplementedError

            # Compute the L1 loss for each image
            image_size_out = targets[index]["image_size_xyxy"]
            out_bbox_ = out_bbox / image_size_out
            image_size_tgt = targets[index]["image_size_xyxy_tgt"]
            tgt_bbox_ = tgt_bbox / image_size_tgt
            cost_bbox = torch.cdist(tgt_bbox_, out_bbox_, p=1)

            # Compute the GIoU loss for each image
            cost_giou = -generalized_box_iou(tgt_bbox, out_bbox)

            loss = self.cost_class * loss_cls + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            loss = torch.cat([loss, self.cost_bg * loss_bg.unsqueeze(0)], dim=0)  # [num_gt + 1, num_queries]

            mu = cost_giou.new_ones(num_gt + 1)
            # static K
            if self.k > 0:
                mu[:-1] = self.k - 0.5*(T-t)
            else:
                raise NotImplementedError
            mu[:-1] = num_queries - mu[:-1].sum()
            nu = cost_giou.new_ones(num_queries)
            _, pi = self.sinkhorn(mu, nu, loss)
            rescale_factor, _ = pi.max(dim=1)
            pi = pi / rescale_factor.unsqueeze(1)
            max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)

            # dude !
            prop_indice = (matched_gt_inds != num_gt).nonzero().squeeze(-1)
            gt_indice = matched_gt_inds[prop_indice]
            indice.append((prop_indice, gt_indice))
        # print(f"\n in stage {t}, k = {self.k - 0.5*(T-t)}, matched: {[len(i[0]) for i in indice]}, n_gt: {[len(x['gt_classes']) for x in targets]}\n")
        return indice
