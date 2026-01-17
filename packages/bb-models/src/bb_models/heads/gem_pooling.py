"""
Generalized Mean (GeM) Pooling implementation.

GeM pooling learns a parameter p that controls the pooling behavior:
- p=1: Average pooling
- p=inf: Max pooling
- p>1: Weighted average favoring larger values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling with learnable parameter.

    GeM pooling provides a learnable trade-off between average and max pooling.
    It's particularly effective for image retrieval and fine-grained recognition.

    Reference:
        Fine-tuning CNN Image Retrieval with No Human Annotation
        https://arxiv.org/abs/1711.02512
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        """
        Initialize GeM pooling.

        Args:
            p: Initial value for the pooling parameter. Higher values
               give more weight to larger activations.
            eps: Small constant for numerical stability.
            learnable: If True, p is a learnable parameter.
        """
        super().__init__()
        self.eps = eps

        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GeM pooling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Pooled tensor of shape (B, C)
        """
        # Clamp to ensure numerical stability with large p values
        x_clamped = x.clamp(min=self.eps)

        # GeM pooling: (1/N * sum(x^p))^(1/p)
        pooled = F.avg_pool2d(
            x_clamped.pow(self.p),
            kernel_size=(x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)

        return pooled.flatten(1)

    def __repr__(self) -> str:
        p_val = self.p.item() if isinstance(self.p, nn.Parameter) else self.p.item()
        return f"GeMPooling(p={p_val:.2f}, eps={self.eps})"


class AdaptiveGeMPooling(nn.Module):
    """
    Adaptive GeM pooling with separate learned p for each channel.

    This variant allows different channels to have different pooling behaviors.
    """

    def __init__(self, channels: int, p: float = 3.0, eps: float = 1e-6):
        """
        Initialize adaptive GeM pooling.

        Args:
            channels: Number of input channels.
            p: Initial value for the pooling parameter.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.channels = channels
        self.p = nn.Parameter(torch.ones(1, channels, 1, 1) * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise GeM pooling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Pooled tensor of shape (B, C)
        """
        x_clamped = x.clamp(min=self.eps)

        # Channel-wise GeM pooling
        pooled = (
            x_clamped.pow(self.p).mean(dim=(-2, -1), keepdim=True).pow(1.0 / self.p)
        )

        return pooled.flatten(1)


class MixedPooling(nn.Module):
    """
    Mixed pooling combining GeM and average pooling.

    Useful when you want to combine different pooling strategies.
    """

    def __init__(
        self,
        p: float = 3.0,
        gem_weight: float = 0.7,
        avg_weight: float = 0.3,
        eps: float = 1e-6,
    ):
        """
        Initialize mixed pooling.

        Args:
            p: GeM pooling parameter.
            gem_weight: Weight for GeM pooling output.
            avg_weight: Weight for average pooling output.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.gem = GeMPooling(p=p, eps=eps)
        self.gem_weight = gem_weight
        self.avg_weight = avg_weight

        # Normalize weights
        total = gem_weight + avg_weight
        self.gem_weight /= total
        self.avg_weight /= total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mixed pooling.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Pooled tensor of shape (B, C)
        """
        gem_out = self.gem(x)
        avg_out = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return self.gem_weight * gem_out + self.avg_weight * avg_out
