"""
Classification Loss Functions - SOTA Losses

Includes:
- CrossEntropyLoss (standard)
- LabelSmoothingCrossEntropy
- FocalLoss (for class imbalance)
- CircleLoss (metric learning)
- ArcFaceLoss (angular margin)
- CosFaceLoss (additive margin)
- PolyLoss (ICCV 2022)
- CombinedLoss (multi-objective)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.

    Prevents overconfident predictions by softening labels:
    - True class: 1 - smoothing + smoothing/num_classes
    - Other classes: smoothing/num_classes
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            targets_one_hot = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1
            )
            targets_smoothed = (
                targets_one_hot * (1 - self.smoothing)
                + self.smoothing / num_classes
            )

        # Compute loss
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = -(targets_smoothed * log_probs * weight).sum(dim=-1)
        else:
            loss = -(targets_smoothed * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - Reduces loss for well-classified examples
    - Focuses training on hard examples

    Reference: "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            gamma: Focusing parameter (0 = standard CE, higher = more focus on hard)
            alpha: Class weights [C] or None for uniform
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Optional label smoothing
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        num_classes = logits.size(-1)
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )

        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CircleLoss(nn.Module):
    """
    Circle Loss for unified metric learning.

    Provides flexible optimization by treating positive and negative
    similarities differently with separate margins.

    Reference: "Circle Loss: A Unified Perspective of Pair Similarity Optimization" (CVPR 2020)
    """

    def __init__(
        self,
        m: float = 0.25,  # margin
        gamma: float = 256,  # scale
    ):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            targets: [B] class labels

        Returns:
            Scalar loss
        """
        # Compute similarity matrix
        sim_mat = torch.matmul(embeddings, embeddings.t())

        # Create positive and negative masks
        targets = targets.view(-1, 1)
        pos_mask = (targets == targets.t()).float()
        neg_mask = (targets != targets.t()).float()

        # Remove diagonal
        pos_mask.fill_diagonal_(0)

        # Compute positive and negative pairs
        pos_sim = sim_mat * pos_mask
        neg_sim = sim_mat * neg_mask

        # Circle loss formula
        pos_margin = 1 - self.m
        neg_margin = self.m

        # Detach for weight computation
        alpha_p = torch.clamp_min(1 + self.m - pos_sim.detach(), 0)
        alpha_n = torch.clamp_min(neg_sim.detach() + self.m, 0)

        # Weighted similarities
        logit_p = -self.gamma * alpha_p * (pos_sim - pos_margin)
        logit_n = self.gamma * alpha_n * (neg_sim - neg_margin)

        # Mask out invalid pairs
        logit_p = torch.where(pos_mask > 0, logit_p, torch.zeros_like(logit_p))
        logit_n = torch.where(neg_mask > 0, logit_n, torch.zeros_like(logit_n))

        # LogSumExp for stability
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        )

        return loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss for face/fine-grained recognition.

    Adds angular margin penalty to enhance intra-class compactness
    and inter-class discrepancy.

    Reference: "ArcFace: Additive Angular Margin Loss" (CVPR 2019)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            num_classes: Number of classes
            margin: Angular margin in radians
            scale: Feature scale
            easy_margin: Use easy margin formula
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # Learnable class centers
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            targets: [B] class labels

        Returns:
            Scalar loss
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # Apply margin to target class only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return F.cross_entropy(output, targets)


class CosFaceLoss(nn.Module):
    """
    CosFace Loss (Large Margin Cosine Loss).

    Simpler than ArcFace - adds margin directly to cosine.

    Reference: "CosFace: Large Margin Cosine Loss" (CVPR 2018)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.35,
        scale: float = 64.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D] L2-normalized embeddings
            targets: [B] class labels
        """
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight_norm)

        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)

        # Subtract margin from target class
        output = cosine - one_hot * self.margin
        output = output * self.scale

        return F.cross_entropy(output, targets)


class PolyLoss(nn.Module):
    """
    Poly Loss - adjusts CE loss by adding polynomial terms.

    Better than CE for class imbalanced data.

    Reference: "PolyLoss: A Polynomial Expansion Perspective of Classification Loss" (ICLR 2022)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            epsilon: Polynomial coefficient (default: 1.0)
            weight: Class weights
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        PolyLoss = CE + epsilon * (1 - p_t)
        """
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )

        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        poly_loss = ce_loss + self.epsilon * (1 - p_t)

        if self.reduction == "mean":
            return poly_loss.mean()
        elif self.reduction == "sum":
            return poly_loss.sum()
        return poly_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss for multi-objective training.

    Supports combining:
    - Classification loss (CE/Focal/LabelSmoothing)
    - Metric loss (Circle/ArcFace/CosFace)
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        metric_loss: Optional[nn.Module] = None,
        cls_weight: float = 1.0,
        metric_weight: float = 0.5,
    ):
        super().__init__()
        self.cls_loss = cls_loss
        self.metric_loss = metric_loss
        self.cls_weight = cls_weight
        self.metric_weight = metric_weight

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: Optional[torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with individual losses and total.
        """
        losses = {}

        # Classification loss
        cls_loss = self.cls_loss(logits, targets)
        losses["cls_loss"] = cls_loss

        total = self.cls_weight * cls_loss

        # Metric loss (if provided)
        if self.metric_loss is not None and embeddings is not None:
            metric_loss = self.metric_loss(embeddings, targets)
            losses["metric_loss"] = metric_loss
            total = total + self.metric_weight * metric_loss

        losses["total"] = total
        return losses


def get_loss(
    loss_name: str,
    num_classes: int,
    embedding_dim: Optional[int] = None,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss by name.

    Args:
        loss_name: One of 'ce', 'label_smoothing', 'focal', 'poly',
                   'circle', 'arcface', 'cosface'
        num_classes: Number of classes
        embedding_dim: Required for metric losses
        class_weights: Optional class weights for imbalanced data
        **kwargs: Additional loss-specific parameters

    Returns:
        Loss module
    """
    loss_name = loss_name.lower()

    if loss_name == "ce" or loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == "label_smoothing":
        smoothing = kwargs.get("smoothing", 0.1)
        return LabelSmoothingCrossEntropy(
            smoothing=smoothing, weight=class_weights
        )

    elif loss_name == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)

    elif loss_name == "poly":
        epsilon = kwargs.get("epsilon", 1.0)
        return PolyLoss(epsilon=epsilon, weight=class_weights)

    elif loss_name == "circle":
        m = kwargs.get("margin", 0.25)
        gamma = kwargs.get("gamma", 256)
        return CircleLoss(m=m, gamma=gamma)

    elif loss_name == "arcface":
        if embedding_dim is None:
            raise ValueError("embedding_dim required for ArcFace")
        margin = kwargs.get("margin", 0.5)
        scale = kwargs.get("scale", 64.0)
        return ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    elif loss_name == "cosface":
        if embedding_dim is None:
            raise ValueError("embedding_dim required for CosFace")
        margin = kwargs.get("margin", 0.35)
        scale = kwargs.get("scale", 64.0)
        return CosFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )

    else:
        raise ValueError(f"Unknown loss: {loss_name}")


AVAILABLE_LOSSES = [
    "ce", "label_smoothing", "focal", "poly", "circle", "arcface", "cosface"
]


if __name__ == "__main__":
    # Test losses
    batch_size, num_classes, embed_dim = 32, 10, 256

    logits = torch.randn(batch_size, num_classes)
    embeddings = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    targets = torch.randint(0, num_classes, (batch_size,))

    print("Testing losses:")

    for loss_name in AVAILABLE_LOSSES:
        try:
            if loss_name in ["arcface", "cosface", "circle"]:
                loss_fn = get_loss(loss_name, num_classes, embed_dim)
                loss = loss_fn(embeddings, targets)
            else:
                loss_fn = get_loss(loss_name, num_classes)
                loss = loss_fn(logits, targets)
            print(f"  {loss_name}: {loss.item():.4f}")
        except Exception as e:
            print(f"  {loss_name}: ERROR - {e}")
