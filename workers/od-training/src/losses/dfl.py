"""
Distribution Focal Loss (DFL) for Bounding Box Regression.

DFL predicts box boundaries as probability distributions over discrete
bins rather than single values. This provides:
1. Better uncertainty modeling
2. More stable gradients
3. Improved localization accuracy

Instead of directly predicting x1, y1, x2, y2, the model predicts
a distribution over possible values and takes the expected value.

Formula:
    E[x] = Σ(i × softmax(logits[i]))  # Expected value from distribution
    DFL = -(y_{i+1} - y) × log(p_i) - (y - y_i) × log(p_{i+1})

Where:
    y = ground truth value
    y_i, y_{i+1} = neighboring discrete bins
    p_i, p_{i+1} = predicted probabilities for these bins

Reference:
    Li et al., "Generalized Focal Loss" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DFLoss(nn.Module):
    """
    Distribution Focal Loss for bounding box regression.

    Predicts box coordinates as distributions and computes loss
    based on the difference from target distribution.

    Args:
        num_bins: Number of discrete bins (default: 16)
        reduction: 'none', 'mean', or 'sum'

    Example:
        >>> dfl = DFLoss(num_bins=16)
        >>> # pred_dist: [N, 4, num_bins] - distribution for each of 4 box coords
        >>> # target: [N, 4] - target box coordinates (normalized 0-1)
        >>> loss = dfl(pred_dist, target)
    """

    def __init__(
        self,
        num_bins: int = 16,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction

        # Register bins as buffer (0 to num_bins-1)
        self.register_buffer(
            "bins",
            torch.arange(num_bins, dtype=torch.float32),
        )

    def forward(
        self,
        pred_dist: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute DFL loss.

        Args:
            pred_dist: [N, 4, num_bins] predicted distributions
                       OR [N, 4*num_bins] flattened
            target: [N, 4] target box coordinates (scaled to bin range)
            weights: Optional [N] or [N, 4] sample weights

        Returns:
            DFL loss
        """
        # Handle flattened input
        if pred_dist.dim() == 2:
            # [N, 4*num_bins] -> [N, 4, num_bins]
            pred_dist = pred_dist.view(-1, 4, self.num_bins)

        batch_size = pred_dist.size(0)

        # Clamp target to valid bin range
        target = target.clamp(0, self.num_bins - 1 - 1e-6)

        # Get left and right bin indices
        target_left = target.long()  # Floor
        target_right = target_left + 1  # Ceil

        # Get interpolation weights
        weight_right = target - target_left.float()
        weight_left = 1 - weight_right

        # Compute log probabilities
        log_probs = F.log_softmax(pred_dist, dim=-1)  # [N, 4, num_bins]

        # Gather log probs for left and right bins
        # [N, 4, 1] -> [N, 4]
        log_prob_left = log_probs.gather(
            -1, target_left.unsqueeze(-1)
        ).squeeze(-1)
        log_prob_right = log_probs.gather(
            -1, target_right.clamp(max=self.num_bins - 1).unsqueeze(-1)
        ).squeeze(-1)

        # DFL = -w_left * log(p_left) - w_right * log(p_right)
        loss = -(weight_left * log_prob_left + weight_right * log_prob_right)

        # Apply weights
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(-1)  # [N] -> [N, 1]
            loss = loss * weights

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def distribution_to_bbox(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """
        Convert predicted distribution to bbox coordinates.

        Args:
            pred_dist: [N, 4, num_bins] or [N, 4*num_bins]

        Returns:
            [N, 4] bbox coordinates
        """
        if pred_dist.dim() == 2:
            pred_dist = pred_dist.view(-1, 4, self.num_bins)

        # Softmax to get probabilities
        probs = F.softmax(pred_dist, dim=-1)  # [N, 4, num_bins]

        # Expected value: E[x] = Σ(i × p_i)
        bins = self.bins.to(pred_dist.device)
        bbox = (probs * bins).sum(dim=-1)  # [N, 4]

        return bbox


class QualityDFLoss(nn.Module):
    """
    Quality-aware Distribution Focal Loss.

    Combines DFL with quality estimation for better box regression.
    The loss is weighted by the predicted IoU quality.

    Args:
        num_bins: Number of discrete bins
        reduction: Loss reduction mode
    """

    def __init__(
        self,
        num_bins: int = 16,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_bins = num_bins
        self.reduction = reduction
        self.dfl = DFLoss(num_bins=num_bins, reduction="none")

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_quality: torch.Tensor,
        target: torch.Tensor,
        target_quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quality-weighted DFL.

        Args:
            pred_dist: [N, 4, num_bins] predicted distributions
            pred_quality: [N] predicted IoU quality
            target: [N, 4] target coordinates
            target_quality: [N] target IoU (from matching)

        Returns:
            Quality-weighted DFL
        """
        # Base DFL
        dfl_loss = self.dfl(pred_dist, target)  # [N, 4]

        # Weight by target quality (higher IoU = more important)
        weights = target_quality.unsqueeze(-1)  # [N, 1]
        weighted_loss = dfl_loss * weights

        if self.reduction == "mean":
            return weighted_loss.sum() / weights.sum().clamp(min=1e-6)
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


def distribution_focal_loss(
    pred_dist: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 16,
) -> torch.Tensor:
    """
    Functional interface for distribution focal loss.

    Args:
        pred_dist: [N, 4, num_bins] predicted distributions
        target: [N, 4] target coordinates (scaled to bin range)
        num_bins: Number of bins

    Returns:
        DFL loss (scalar)
    """
    if pred_dist.dim() == 2:
        pred_dist = pred_dist.view(-1, 4, num_bins)

    target = target.clamp(0, num_bins - 1 - 1e-6)

    target_left = target.long()
    target_right = target_left + 1

    weight_right = target - target_left.float()
    weight_left = 1 - weight_right

    log_probs = F.log_softmax(pred_dist, dim=-1)

    log_prob_left = log_probs.gather(-1, target_left.unsqueeze(-1)).squeeze(-1)
    log_prob_right = log_probs.gather(
        -1, target_right.clamp(max=num_bins - 1).unsqueeze(-1)
    ).squeeze(-1)

    loss = -(weight_left * log_prob_left + weight_right * log_prob_right)

    return loss.mean()


def integral(pred_dist: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
    """
    Compute integral (expected value) from distribution predictions.

    This converts the discrete distribution back to continuous coordinates.

    Args:
        pred_dist: [N, 4, num_bins] or [N, 4*num_bins]
        num_bins: Number of bins

    Returns:
        [N, 4] integrated coordinates
    """
    if pred_dist.dim() == 2:
        pred_dist = pred_dist.view(-1, 4, num_bins)

    probs = F.softmax(pred_dist, dim=-1)
    bins = torch.arange(num_bins, dtype=pred_dist.dtype, device=pred_dist.device)

    return (probs * bins).sum(dim=-1)


class IntegralHead(nn.Module):
    """
    Integral head for converting distribution to coordinates.

    Used in detection heads that predict distributions.

    Args:
        num_bins: Number of bins for distribution
    """

    def __init__(self, num_bins: int = 16):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer(
            "bins",
            torch.arange(num_bins, dtype=torch.float32),
        )

    def forward(self, pred_dist: torch.Tensor) -> torch.Tensor:
        """
        Convert distribution to coordinates.

        Args:
            pred_dist: [..., num_bins] distribution logits

        Returns:
            [...] integrated coordinates
        """
        probs = F.softmax(pred_dist, dim=-1)
        return (probs * self.bins.to(pred_dist.device)).sum(dim=-1)
