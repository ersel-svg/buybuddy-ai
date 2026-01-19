"""
Loss Functions for SOTA Product Recognition Training.

Includes:
- TripletLoss with online hard mining
- CombinedProductLoss (ArcFace + Triplet + Domain)
- Gradient Reversal for Domain Adversarial Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TripletLoss(nn.Module):
    """
    Basic Triplet Loss with margin.

    L = max(0, d(a,p) - d(a,n) + margin)

    Where:
    - d(a,p) = distance between anchor and positive
    - d(a,n) = distance between anchor and negative
    """

    def __init__(self, margin: float = 0.3, distance: str = "euclidean"):
        """
        Args:
            margin: Margin for triplet loss
            distance: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: [B, D] anchor embeddings
            positive: [B, D] positive embeddings
            negative: [B, D] negative embeddings

        Returns:
            Scalar loss
        """
        if self.distance == "euclidean":
            ap_dist = F.pairwise_distance(anchor, positive)
            an_dist = F.pairwise_distance(anchor, negative)
        else:  # cosine
            ap_dist = 1 - F.cosine_similarity(anchor, positive)
            an_dist = 1 - F.cosine_similarity(anchor, negative)

        loss = F.relu(ap_dist - an_dist + self.margin)
        return loss.mean()


class OnlineHardTripletLoss(nn.Module):
    """
    Triplet Loss with Online Hard Negative Mining (Vectorized).

    Efficiently mines hard and semi-hard triplets using matrix operations.
    O(B²) complexity instead of O(B³).

    Mining strategies:
    - Hard negatives: an_dist < ap_dist (violating triplets)
    - Semi-hard negatives: ap_dist < an_dist < ap_dist + margin

    Reference: "In Defense of the Triplet Loss for Person Re-Identification"
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining_type: str = "hard",  # "hard", "semi_hard", "all"
        squared: bool = False,
    ):
        """
        Args:
            margin: Triplet margin
            mining_type: Type of mining - "hard", "semi_hard", or "all"
            squared: Use squared Euclidean distance
        """
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.squared = squared

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with vectorized online mining.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] class labels

        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute pairwise distance matrix [B, B]
        dist_mat = self._pairwise_distances(embeddings)

        # Create masks [B, B]
        labels_col = labels.view(-1, 1)
        labels_row = labels.view(1, -1)
        positive_mask = (labels_col == labels_row)
        negative_mask = (labels_col != labels_row)

        # Remove diagonal (self-similarity)
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~eye

        # For each anchor, get hardest positive distance [B]
        # Set non-positives to 0, then take max
        ap_dist_mat = dist_mat * positive_mask.float()
        hardest_positive_dist, _ = ap_dist_mat.max(dim=1)  # [B]

        # For each anchor, get hardest negative distance [B]
        # Set positives to large value, then take min
        large_value = dist_mat.max() + 1
        an_dist_mat = dist_mat + (~negative_mask).float() * large_value
        hardest_negative_dist, _ = an_dist_mat.min(dim=1)  # [B]

        if self.mining_type == "hard":
            # Batch hard: hardest positive, hardest negative per anchor
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

            # Only count anchors that have at least one positive
            valid_anchors = positive_mask.sum(dim=1) > 0
            if valid_anchors.sum() > 0:
                triplet_loss = triplet_loss[valid_anchors].mean()
            else:
                triplet_loss = torch.tensor(0.0, device=device)

        elif self.mining_type == "semi_hard":
            # Semi-hard mining: negatives within margin
            # For each (anchor, positive) pair, find semi-hard negatives
            triplet_loss = self._semi_hard_mining(dist_mat, positive_mask, negative_mask)

        else:  # "all"
            # All valid triplets
            triplet_loss = self._all_triplets(dist_mat, positive_mask, negative_mask)

        return triplet_loss

    def _semi_hard_mining(
        self,
        dist_mat: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mine semi-hard triplets vectorized."""
        device = dist_mat.device
        batch_size = dist_mat.size(0)

        # Get anchor-positive distances for all valid pairs
        # Shape: [B, B] where (i,j) is dist(anchor_i, positive_j)
        ap_distances = dist_mat.unsqueeze(2)  # [B, B, 1]
        an_distances = dist_mat.unsqueeze(1)  # [B, 1, B]

        # Triplet loss: ap - an + margin for all (a, p, n) combinations
        # Shape: [B, B, B]
        loss_tensor = ap_distances - an_distances + self.margin

        # Valid triplets mask [B, B, B]
        # (i, j, k) is valid if j is positive of i and k is negative of i
        pos_mask_3d = positive_mask.unsqueeze(2)  # [B, B, 1]
        neg_mask_3d = negative_mask.unsqueeze(1)  # [B, 1, B]
        valid_triplets = pos_mask_3d & neg_mask_3d  # [B, B, B]

        # Semi-hard: an > ap and an < ap + margin
        semi_hard_mask = (an_distances > ap_distances) & (an_distances < ap_distances + self.margin)
        semi_hard_mask = semi_hard_mask.squeeze() if semi_hard_mask.dim() > 3 else semi_hard_mask

        # Combine masks
        final_mask = valid_triplets & semi_hard_mask

        # Apply mask and compute mean loss
        valid_losses = loss_tensor[final_mask]
        if valid_losses.numel() > 0:
            return F.relu(valid_losses).mean()
        else:
            # Fallback to batch hard if no semi-hard triplets
            return torch.tensor(0.0, device=device)

    def _all_triplets(
        self,
        dist_mat: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss over all valid triplets."""
        device = dist_mat.device

        ap_distances = dist_mat.unsqueeze(2)  # [B, B, 1]
        an_distances = dist_mat.unsqueeze(1)  # [B, 1, B]

        loss_tensor = ap_distances - an_distances + self.margin  # [B, B, B]

        pos_mask_3d = positive_mask.unsqueeze(2)
        neg_mask_3d = negative_mask.unsqueeze(1)
        valid_triplets = pos_mask_3d & neg_mask_3d

        valid_losses = loss_tensor[valid_triplets]
        if valid_losses.numel() > 0:
            return F.relu(valid_losses).mean()
        else:
            return torch.tensor(0.0, device=device)

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances efficiently."""
        # Using torch.cdist for cleaner code
        distances = torch.cdist(embeddings, embeddings, p=2)

        if self.squared:
            distances = distances ** 2

        return distances


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.

    For each anchor, selects:
    - Hardest positive (furthest same-class sample)
    - Hardest negative (closest different-class sample)

    This is more aggressive than semi-hard mining.
    """

    def __init__(self, margin: float = 0.3, soft: bool = False):
        """
        Args:
            margin: Triplet margin
            soft: Use soft margin (log(1 + exp(loss)))
        """
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute batch hard triplet loss."""
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Pairwise distances
        dist_mat = self._pairwise_distances(embeddings)

        # Masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove diagonal
        eye = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye

        # Hardest positive: max distance among positives
        # Set negatives to 0 so they don't affect max
        ap_dist = dist_mat * positive_mask
        hardest_positive, _ = ap_dist.max(dim=1)

        # Hardest negative: min distance among negatives
        # Set positives to large value so they don't affect min
        an_dist = dist_mat + positive_mask * 1e9 + eye * 1e9
        hardest_negative, _ = an_dist.min(dim=1)

        # Triplet loss
        if self.soft:
            loss = torch.log1p(torch.exp(hardest_positive - hardest_negative + self.margin))
        else:
            loss = F.relu(hardest_positive - hardest_negative + self.margin)

        # Only count valid triplets (anchors with at least one positive)
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        return torch.sqrt(F.relu(distances) + 1e-16)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for Domain Adversarial Training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer.

    Used for domain adversarial training to make features domain-invariant.
    During forward pass: identity function
    During backward pass: negates and scales gradients
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """Update alpha (can increase during training)."""
        self.alpha = alpha


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial training.

    Predicts domain type of embeddings.
    Domain types: 0=synthetic, 1=real, 2=augmented, 3=cutout, 4=unknown
    Used with gradient reversal to encourage domain-invariant features.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        num_domains: int = 5,  # synthetic, real, augmented, cutout, unknown
        dropout: float = 0.5,
    ):
        super().__init__()
        self.grl = GradientReversal(alpha=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient reversal.

        Args:
            x: [B, D] embeddings

        Returns:
            [B, num_domains] domain logits
        """
        x = self.grl(x)
        return self.classifier(x)

    def set_alpha(self, alpha: float):
        """Set gradient reversal strength."""
        self.grl.set_alpha(alpha)


class CombinedProductLoss(nn.Module):
    """
    Combined Loss for SOTA Product Recognition.

    Combines:
    1. ArcFace Loss - Angular margin classification
    2. Triplet Loss - Metric learning with hard mining
    3. Circle Loss - Unified similarity optimization (CVPR 2020)
    4. Domain Adversarial Loss - Domain adaptation (optional)

    Total Loss = w1 * ArcFace + w2 * Triplet + w3 * Circle + w4 * Domain
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        # ArcFace params
        arcface_weight: float = 1.0,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        # Triplet params
        triplet_weight: float = 0.5,
        triplet_margin: float = 0.3,
        triplet_mining: str = "batch_hard",  # "online", "batch_hard", "batch_all"
        # Circle Loss params (CVPR 2020)
        circle_weight: float = 0.0,  # Default disabled, enable for fine-grained products
        circle_margin: float = 0.25,
        circle_gamma: float = 256.0,
        # Domain params
        domain_weight: float = 0.1,
        use_domain_adaptation: bool = False,
        # Label smoothing
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            num_classes: Number of product classes
            embedding_dim: Embedding dimension
            arcface_weight: Weight for ArcFace loss
            arcface_margin: Angular margin for ArcFace
            arcface_scale: Scale for ArcFace
            triplet_weight: Weight for triplet loss
            triplet_margin: Margin for triplet loss
            triplet_mining: Mining strategy
            circle_weight: Weight for Circle Loss (0 = disabled)
            circle_margin: Margin for Circle Loss
            circle_gamma: Scale factor for Circle Loss
            domain_weight: Weight for domain loss
            use_domain_adaptation: Enable domain adversarial training
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
        self.circle_weight = circle_weight
        self.domain_weight = domain_weight
        self.use_domain_adaptation = use_domain_adaptation

        # ArcFace head
        self.arcface_fc = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.arcface_fc.weight)
        self.arcface_margin = arcface_margin
        self.arcface_scale = arcface_scale
        self.cos_m = math.cos(arcface_margin)
        self.sin_m = math.sin(arcface_margin)
        self.th = math.cos(math.pi - arcface_margin)
        self.mm = math.sin(math.pi - arcface_margin) * arcface_margin

        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Triplet loss
        if triplet_weight > 0:
            if triplet_mining == "batch_hard":
                self.triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
            else:
                self.triplet_loss = OnlineHardTripletLoss(margin=triplet_margin)
        else:
            self.triplet_loss = None

        # Circle Loss (CVPR 2020) - ideal for fine-grained product recognition
        if circle_weight > 0:
            self.circle_loss = CircleLoss(margin=circle_margin, gamma=circle_gamma)
        else:
            self.circle_loss = None

        # Domain classifier (5 domains: synthetic, real, augmented, cutout, unknown)
        if use_domain_adaptation:
            self.domain_classifier = DomainClassifier(embedding_dim, num_domains=5)
            self.domain_loss_fn = nn.CrossEntropyLoss()
        else:
            self.domain_classifier = None

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] class labels (product_ids mapped to indices)
            domains: [B] domain labels (0=synthetic, 1=real) - optional

        Returns:
            Dictionary with 'total' loss and component losses
        """
        device = embeddings.device
        losses = {}

        # 1. ArcFace Loss
        if self.arcface_weight > 0:
            arcface_loss = self._arcface_loss(embeddings, labels)
            losses["arcface"] = arcface_loss
        else:
            arcface_loss = torch.tensor(0.0, device=device)
            losses["arcface"] = arcface_loss

        # 2. Triplet Loss
        if self.triplet_loss is not None and self.triplet_weight > 0:
            triplet_loss = self.triplet_loss(embeddings, labels)
            losses["triplet"] = triplet_loss
        else:
            triplet_loss = torch.tensor(0.0, device=device)
            losses["triplet"] = triplet_loss

        # 3. Circle Loss (CVPR 2020)
        if self.circle_loss is not None and self.circle_weight > 0:
            circle_loss = self.circle_loss(embeddings, labels)
            losses["circle"] = circle_loss
        else:
            circle_loss = torch.tensor(0.0, device=device)
            losses["circle"] = circle_loss

        # 4. Domain Adversarial Loss
        if self.domain_classifier is not None and domains is not None and self.domain_weight > 0:
            domain_logits = self.domain_classifier(embeddings)
            domain_loss = self.domain_loss_fn(domain_logits, domains)
            losses["domain"] = domain_loss
        else:
            domain_loss = torch.tensor(0.0, device=device)
            losses["domain"] = domain_loss

        # Total loss
        total_loss = (
            self.arcface_weight * arcface_loss +
            self.triplet_weight * triplet_loss +
            self.circle_weight * circle_loss +
            self.domain_weight * domain_loss
        )
        losses["total"] = total_loss

        return losses

    def _arcface_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ArcFace loss."""
        # Normalize weights
        weight = F.normalize(self.arcface_fc.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Boundary condition
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to correct class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.arcface_scale

        return self.ce_loss(output, labels)

    def set_domain_alpha(self, alpha: float):
        """Set domain adversarial strength (increase during training)."""
        if self.domain_classifier is not None:
            self.domain_classifier.set_alpha(alpha)


class CircleLoss(nn.Module):
    """
    Circle Loss for flexible similarity learning.

    Reference: "Circle Loss: A Unified Perspective of Pair Similarity Optimization"

    More flexible than triplet loss as it handles positive and negative
    pairs with different optimization goals.
    """

    def __init__(
        self,
        margin: float = 0.25,
        gamma: float = 256,
    ):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Circle Loss."""
        # Cosine similarity matrix
        sim_mat = torch.mm(embeddings, embeddings.t())

        # Masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove diagonal
        eye = torch.eye(embeddings.size(0), device=embeddings.device)
        positive_mask = positive_mask - eye

        # Positive similarities
        sp = sim_mat * positive_mask
        # Negative similarities
        sn = sim_mat * negative_mask

        # Circle loss computation
        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        # Weighted log-sum-exp
        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        # Mask invalid pairs
        logit_p = logit_p * positive_mask
        logit_n = logit_n * negative_mask

        # Replace zeros with large negative for stable softplus
        logit_p = torch.where(positive_mask > 0, logit_p, torch.tensor(-1e9, device=embeddings.device))
        logit_n = torch.where(negative_mask > 0, logit_n, torch.tensor(-1e9, device=embeddings.device))

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)
        ).mean()

        return loss
