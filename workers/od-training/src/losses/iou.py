"""
IoU-based Loss Functions for Bounding Box Regression.

Implements:
- IoU Loss: Basic intersection over union
- GIoU Loss: Generalized IoU (handles non-overlapping boxes)
- DIoU Loss: Distance IoU (considers center distance)
- CIoU Loss: Complete IoU (considers aspect ratio) - RECOMMENDED
- SIoU Loss: Scylla IoU (angle-aware)

CIoU is generally the best choice for object detection as it
optimizes overlap, center distance, and aspect ratio simultaneously.

Formula:
    IoU  = Intersection / Union
    GIoU = IoU - (C - Union) / C
    DIoU = IoU - d²/c²
    CIoU = DIoU - αv

Where:
    C = enclosing box area
    d = center distance
    c = enclosing box diagonal
    v = (4/π²)(arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
    α = v / (1 - IoU + v)

References:
    - GIoU: Rezatofighi et al., CVPR 2019
    - DIoU/CIoU: Zheng et al., AAAI 2020
    - SIoU: Gevorgyan, arXiv 2022
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.

    Args:
        boxes: [N, 4] in xyxy format

    Returns:
        Areas [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format to corner format.

    Args:
        boxes: [N, 4] as (cx, cy, w, h)

    Returns:
        [N, 4] as (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from corner format to center format.

    Args:
        boxes: [N, 4] as (x1, y1, x2, y2)

    Returns:
        [N, 4] as (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def bbox_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    iou_type: str = "iou",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in xyxy format
        boxes2: [N, 4] boxes in xyxy format (same N as boxes1)
        iou_type: "iou", "giou", "diou", "ciou", or "siou"
        eps: Small value for numerical stability

    Returns:
        IoU values [N]
    """
    # Ensure same device and dtype
    boxes2 = boxes2.to(boxes1.device)

    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area + eps

    # IoU
    iou = inter_area / union_area

    if iou_type == "iou":
        return iou

    # Enclosing box
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)

    if iou_type == "giou":
        # GIoU
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + eps
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou

    # Center points
    b1_cx = (b1_x1 + b1_x2) / 2
    b1_cy = (b1_y1 + b1_y2) / 2
    b2_cx = (b2_x1 + b2_x2) / 2
    b2_cy = (b2_y1 + b2_y2) / 2

    # Center distance squared
    center_dist_sq = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Enclosing box diagonal squared
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    if iou_type == "diou":
        # DIoU
        diou = iou - center_dist_sq / enclose_diag_sq
        return diou

    if iou_type == "ciou":
        # CIoU
        # Width and height
        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1
        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1

        # Aspect ratio consistency term
        v = (4 / math.pi ** 2) * torch.pow(
            torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
        )

        # Trade-off parameter
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        # CIoU
        ciou = iou - center_dist_sq / enclose_diag_sq - alpha * v
        return ciou

    if iou_type == "siou":
        # SIoU (Scylla IoU) - angle-aware
        # Width and height
        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1
        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1

        # Angle cost
        sigma = torch.sqrt(center_dist_sq) + eps
        sin_alpha = torch.abs(b2_cy - b1_cy) / sigma
        sin_beta = torch.abs(b2_cx - b1_cx) / sigma

        threshold = torch.pow(torch.tensor(2.0, device=boxes1.device), 0.5) / 2
        sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)

        angle_cost = torch.cos(2 * torch.asin(sin_alpha) - math.pi / 2)

        # Distance cost
        rho_x = ((b2_cx - b1_cx) / (enclose_x2 - enclose_x1 + eps)) ** 2
        rho_y = ((b2_cy - b1_cy) / (enclose_y2 - enclose_y1 + eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

        # Shape cost
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)

        siou = iou - 0.5 * (distance_cost + shape_cost)
        return siou

    raise ValueError(f"Unknown IoU type: {iou_type}")


def bbox_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    iou_type: str = "iou",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute pairwise IoU matrix between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in xyxy format
        boxes2: [M, 4] boxes in xyxy format
        iou_type: "iou", "giou", "diou", or "ciou"
        eps: Small value for numerical stability

    Returns:
        IoU matrix [N, M]
    """
    N = boxes1.size(0)
    M = boxes2.size(0)

    # Expand for broadcasting: [N, 1, 4] and [1, M, 4]
    boxes1_exp = boxes1.unsqueeze(1).expand(N, M, 4)
    boxes2_exp = boxes2.unsqueeze(0).expand(N, M, 4)

    # Flatten to [N*M, 4] for bbox_iou
    boxes1_flat = boxes1_exp.reshape(-1, 4)
    boxes2_flat = boxes2_exp.reshape(-1, 4)

    # Compute IoU
    iou_flat = bbox_iou(boxes1_flat, boxes2_flat, iou_type=iou_type, eps=eps)

    # Reshape back to [N, M]
    return iou_flat.reshape(N, M)


class IoULoss(nn.Module):
    """
    Basic IoU Loss.

    Loss = 1 - IoU

    Args:
        reduction: 'none', 'mean', or 'sum'
        eps: Small value for numerical stability
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute IoU loss.

        Args:
            pred_boxes: [N, 4] predicted boxes (xyxy)
            target_boxes: [N, 4] target boxes (xyxy)
            weights: Optional [N] sample weights

        Returns:
            IoU loss
        """
        iou = bbox_iou(pred_boxes, target_boxes, iou_type="iou", eps=self.eps)
        loss = 1 - iou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss.

    Loss = 1 - GIoU

    GIoU handles non-overlapping boxes better than IoU by considering
    the enclosing box.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        giou = bbox_iou(pred_boxes, target_boxes, iou_type="giou", eps=self.eps)
        loss = 1 - giou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DIoULoss(nn.Module):
    """
    Distance IoU Loss.

    Loss = 1 - DIoU

    DIoU considers the distance between box centers, leading to
    faster convergence than GIoU.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diou = bbox_iou(pred_boxes, target_boxes, iou_type="diou", eps=self.eps)
        loss = 1 - diou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss. RECOMMENDED for object detection.

    Loss = 1 - CIoU

    CIoU additionally considers aspect ratio consistency, making it
    the most comprehensive IoU-based loss for bounding box regression.

    Optimizes:
    1. Overlap area (IoU)
    2. Center point distance (DIoU)
    3. Aspect ratio consistency (v term)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CIoU loss.

        Args:
            pred_boxes: [N, 4] predicted boxes (xyxy format)
            target_boxes: [N, 4] target boxes (xyxy format)
            weights: Optional [N] sample weights

        Returns:
            CIoU loss (scalar if reduction != 'none')
        """
        ciou = bbox_iou(pred_boxes, target_boxes, iou_type="ciou", eps=self.eps)
        loss = 1 - ciou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SIoULoss(nn.Module):
    """
    Scylla IoU Loss.

    SIoU considers the angle between box centers for better convergence
    when boxes are misaligned.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        siou = bbox_iou(pred_boxes, target_boxes, iou_type="siou", eps=self.eps)
        loss = 1 - siou

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_iou_loss(iou_type: str = "ciou", reduction: str = "mean") -> nn.Module:
    """
    Factory function to build IoU loss.

    Args:
        iou_type: "iou", "giou", "diou", "ciou", or "siou"
        reduction: Loss reduction mode

    Returns:
        IoU loss module
    """
    losses = {
        "iou": IoULoss,
        "giou": GIoULoss,
        "diou": DIoULoss,
        "ciou": CIoULoss,
        "siou": SIoULoss,
    }

    if iou_type not in losses:
        raise ValueError(f"Unknown IoU type: {iou_type}. Choose from {list(losses.keys())}")

    return losses[iou_type](reduction=reduction)
