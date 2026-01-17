"""
ArcFace (Additive Angular Margin Loss) implementation.

ArcFace is a state-of-the-art loss function for face recognition
and metric learning that enforces a large angular margin.

Reference:
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/abs/1801.07698
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head with additive angular margin.

    This head is used for training embeddings with strong discriminative power.
    It adds an angular margin to the softmax loss, pushing different classes
    further apart in the embedding space.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.30,
        easy_margin: bool = False,
    ):
        """
        Initialize ArcFace head.

        Args:
            in_features: Input embedding dimension.
            num_classes: Number of classes.
            scale: Feature scale (temperature). Higher = more confident predictions.
            margin: Angular margin in radians. Higher = harder training.
            easy_margin: Use easier margin formulation for stability.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Class weight matrix (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute angular values for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ArcFace logits.

        Args:
            embeddings: Input embeddings of shape (B, in_features)
            labels: Class labels of shape (B,). Required for training.

        Returns:
            Logits of shape (B, num_classes)
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        if labels is None:
            # Inference mode - just return scaled cosine
            return cosine * self.scale

        # Training mode - apply angular margin
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return output


class EnhancedArcFaceLoss(nn.Module):
    """
    Enhanced ArcFace Loss with integrated label smoothing and loss computation.

    This module combines the ArcFace head with cross-entropy loss and
    optional label smoothing for improved training stability.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.30,
        label_smoothing: float = 0.0,
        easy_margin: bool = False,
    ):
        """
        Initialize enhanced ArcFace loss.

        Args:
            in_features: Input embedding dimension.
            num_classes: Number of classes.
            scale: Feature scale (temperature).
            margin: Angular margin in radians.
            label_smoothing: Label smoothing factor (0.0 = no smoothing).
            easy_margin: Use easier margin formulation.
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.label_smoothing = label_smoothing

        # Class weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute angular values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.easy_margin = easy_margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: Input embeddings of shape (B, in_features)
            labels: Class labels of shape (B,)

        Returns:
            Loss scalar
        """
        # Normalize embeddings and weights for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)

        # Compute sine (for angular margin)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # Apply angular margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding
        one_hot = torch.zeros_like(cosine, device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        # Compute loss with optional label smoothing
        if self.training and self.label_smoothing > 0:
            n_classes = output.size(1)
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    output, self.label_smoothing / (n_classes - 1)
                )
                smooth_targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

            log_probs = F.log_softmax(output, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1).mean()
            return loss

        return F.cross_entropy(output, labels)


class SubCenterArcFace(nn.Module):
    """
    Sub-center ArcFace for handling noisy labels.

    Uses multiple sub-centers per class to handle within-class variance.

    Reference:
        Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy
        Web Faces https://arxiv.org/abs/2010.05089
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_subcenters: int = 3,
        scale: float = 30.0,
        margin: float = 0.30,
    ):
        """
        Initialize sub-center ArcFace.

        Args:
            in_features: Input embedding dimension.
            num_classes: Number of classes.
            num_subcenters: Number of sub-centers per class.
            scale: Feature scale.
            margin: Angular margin.
        """
        super().__init__()
        self.num_subcenters = num_subcenters

        # Weight matrix with sub-centers: (num_classes * num_subcenters, in_features)
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * num_subcenters, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # Angular values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sub-center ArcFace logits.

        Args:
            embeddings: Input embeddings of shape (B, in_features)
            labels: Class labels of shape (B,)

        Returns:
            Logits of shape (B, num_classes)
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity with all sub-centers
        cosine = F.linear(embeddings, weight)  # (B, num_classes * num_subcenters)

        # Reshape to (B, num_classes, num_subcenters)
        cosine = cosine.view(-1, self.num_classes, self.num_subcenters)

        # Take max over sub-centers
        cosine, _ = cosine.max(dim=2)  # (B, num_classes)

        if labels is None:
            return cosine * self.scale

        # Apply angular margin (same as standard ArcFace)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale
