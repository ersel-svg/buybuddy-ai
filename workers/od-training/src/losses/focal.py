"""
Focal Loss for Object Detection.

Focal Loss addresses class imbalance by down-weighting easy examples
and focusing on hard ones.

Formula:
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

Where:
    p_t = p if y=1 else (1-p)
    α = class balance weight (default: 0.25)
    γ = focusing parameter (default: 2.0)

The focusing parameter γ smoothly adjusts the rate at which easy
examples are down-weighted:
    - γ = 0: equivalent to standard cross-entropy
    - γ = 2: good default for object detection
    - γ > 2: more focus on hard examples

Reference:
    Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces the loss contribution from easy examples and focuses
    training on hard negatives.

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class balance weight. Can be:
            - float: Balance weight for positive class (default: 0.25)
            - Tensor: Per-class weights
        reduction: 'none', 'mean', or 'sum' (default: 'mean')
        label_smoothing: Label smoothing factor (default: 0.0)

    Example:
        >>> criterion = FocalLoss(gamma=2.0, alpha=0.25)
        >>> logits = torch.randn(100, 10)  # 100 samples, 10 classes
        >>> targets = torch.randint(0, 10, (100,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [N, C] or [N, C, H, W]
            targets: Target class indices [N] or [N, H, W]
            weight: Optional per-sample weights [N]

        Returns:
            Focal loss (scalar if reduction != 'none')
        """
        # Handle different input shapes
        if inputs.dim() > 2:
            # [N, C, H, W] -> [N, H, W, C] -> [N*H*W, C]
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            inputs = inputs.view(-1, inputs.size(-1))
            targets = targets.view(-1)

        num_classes = inputs.size(-1)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of correct class
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # Get p_t (probability of target class)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Binary case or uniform alpha
                alpha_t = torch.where(
                    targets == 1,
                    torch.tensor(self.alpha, device=inputs.device),
                    torch.tensor(1 - self.alpha, device=inputs.device),
                )
            else:
                # Per-class alpha
                alpha_t = self.alpha[targets]
            focal_weight = focal_weight * alpha_t

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply sample weights if provided
        if weight is not None:
            focal_loss = focal_loss * weight

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for single-class detection.

    Useful when treating detection as binary classification
    (object vs background).

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Positive class weight (default: 0.25)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary focal loss.

        Args:
            inputs: Predicted logits [N] or [N, 1]
            targets: Binary targets [N] (0 or 1)

        Returns:
            Binary focal loss
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute probabilities
        p = torch.sigmoid(inputs)

        # Compute ce loss
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) for joint classification and localization.

    QFL is designed for object detection where the target is not a
    one-hot label but a continuous IoU score.

    Formula:
        QFL(σ) = -|y - σ|^β × ((1-y) × log(1-σ) + y × log(σ))

    Where:
        σ = predicted quality score (0 to 1)
        y = target IoU score (0 to 1)
        β = focusing parameter

    Reference:
        Li et al., "Generalized Focal Loss" (NeurIPS 2020)
    """

    def __init__(
        self,
        beta: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute quality focal loss.

        Args:
            inputs: Predicted quality scores [N] (after sigmoid)
            targets: Target IoU scores [N] (0 to 1)
            weights: Optional sample weights [N]

        Returns:
            Quality focal loss
        """
        # Ensure inputs are probabilities
        probs = torch.sigmoid(inputs) if inputs.max() > 1 else inputs

        # Compute scaling factor
        scale_factor = torch.abs(targets - probs) ** self.beta

        # Binary cross entropy
        bce = F.binary_cross_entropy(
            probs, targets, reduction="none"
        )

        loss = scale_factor * bce

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss for dense object detection.

    Different from Focal Loss, VFL treats positive and negative
    samples asymmetrically:
    - Negative: Standard focal loss
    - Positive: Weighted by target IoU score

    Reference:
        Zhang et al., "VarifocalNet" (CVPR 2021)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute varifocal loss.

        Args:
            inputs: Predicted scores [N, C]
            targets: Target scores [N, C] (IoU for positives, 0 for negatives)

        Returns:
            Varifocal loss
        """
        probs = torch.sigmoid(inputs)

        # Focal weight for negative samples
        focal_weight = self.alpha * probs ** self.gamma

        # For positive samples, use target IoU as weight
        pos_mask = targets > 0
        focal_weight = torch.where(
            pos_mask,
            targets,
            focal_weight,
        )

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional version of sigmoid focal loss.

    Useful for quick testing or when you don't want a module.

    Args:
        inputs: Predicted logits [N, C]
        targets: One-hot targets [N, C]
        alpha: Positive class weight
        gamma: Focusing parameter
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Focal loss
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
