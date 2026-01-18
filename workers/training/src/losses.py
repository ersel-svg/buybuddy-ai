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
    Triplet Loss with Online Hard Negative Mining.

    Mines hard triplets from the batch:
    - Hard negatives: Closest negatives to anchor
    - Semi-hard negatives: Negatives within margin
    - Random negatives: For diversity

    Reference: "In Defense of the Triplet Loss for Person Re-Identification"
    """

    def __init__(
        self,
        margin: float = 0.3,
        hard_ratio: float = 0.5,
        semi_hard_ratio: float = 0.3,
        random_ratio: float = 0.2,
        squared: bool = False,
    ):
        """
        Args:
            margin: Triplet margin
            hard_ratio: Ratio of hard negatives to mine
            semi_hard_ratio: Ratio of semi-hard negatives
            random_ratio: Ratio of random negatives
            squared: Use squared Euclidean distance
        """
        super().__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.random_ratio = random_ratio
        self.squared = squared

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss with online mining.

        Args:
            embeddings: [B, D] normalized embeddings
            labels: [B] class labels

        Returns:
            Scalar loss
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Compute pairwise distance matrix
        dist_mat = self._pairwise_distances(embeddings)

        # Create masks
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()

        # Remove diagonal (self-similarity)
        eye = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - eye

        # Mine triplets
        triplet_loss = torch.tensor(0.0, device=device)
        num_valid_triplets = 0

        for i in range(batch_size):
            # Get positive indices for anchor i
            pos_indices = torch.where(positive_mask[i] > 0)[0]
            neg_indices = torch.where(negative_mask[i] > 0)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Anchor-positive distances
            ap_dists = dist_mat[i, pos_indices]

            # Anchor-negative distances
            an_dists = dist_mat[i, neg_indices]

            # For each positive, mine negatives
            for pos_idx in pos_indices:
                ap_dist = dist_mat[i, pos_idx]

                # Classify negatives
                # Hard: an_dist < ap_dist
                hard_mask = an_dists < ap_dist
                # Semi-hard: ap_dist < an_dist < ap_dist + margin
                semi_hard_mask = (an_dists > ap_dist) & (an_dists < ap_dist + self.margin)

                # Sample negatives based on ratios
                num_neg = len(neg_indices)
                num_hard = max(1, int(num_neg * self.hard_ratio))
                num_semi = max(1, int(num_neg * self.semi_hard_ratio))

                # Get hard negatives (smallest distance)
                if hard_mask.any():
                    hard_neg_dists = an_dists[hard_mask]
                    hard_neg_dists = hard_neg_dists[:num_hard]
                    for an_dist in hard_neg_dists:
                        loss = F.relu(ap_dist - an_dist + self.margin)
                        triplet_loss += loss
                        num_valid_triplets += 1

                # Get semi-hard negatives
                if semi_hard_mask.any():
                    semi_hard_neg_dists = an_dists[semi_hard_mask]
                    semi_hard_neg_dists = semi_hard_neg_dists[:num_semi]
                    for an_dist in semi_hard_neg_dists:
                        loss = F.relu(ap_dist - an_dist + self.margin)
                        triplet_loss += loss
                        num_valid_triplets += 1

        if num_valid_triplets > 0:
            triplet_loss = triplet_loss / num_valid_triplets

        return triplet_loss

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Numerical stability

        if not self.squared:
            distances = torch.sqrt(distances + 1e-16)

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
    3. Domain Adversarial Loss - Domain adaptation (optional)

    Total Loss = w1 * ArcFace + w2 * Triplet + w3 * Domain
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
            domain_weight: Weight for domain loss
            use_domain_adaptation: Enable domain adversarial training
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight
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

        # 3. Domain Adversarial Loss
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
