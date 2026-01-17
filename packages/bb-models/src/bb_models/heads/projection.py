"""
Projection head implementations for embedding transformation.

Projection heads transform backbone features into a different embedding space,
typically with regularization and normalization.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Simple linear projection head with optional normalization.

    Projects embeddings from backbone dimension to target dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalize: bool = True,
        bias: bool = True,
    ):
        """
        Initialize projection head.

        Args:
            in_features: Input dimension (from backbone).
            out_features: Output dimension.
            normalize: Whether to L2-normalize output.
            bias: Whether to include bias in linear layer.
        """
        super().__init__()
        self.normalize = normalize

        self.proj = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights
        nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.

        Args:
            x: Input tensor of shape (B, in_features)

        Returns:
            Projected tensor of shape (B, out_features)
        """
        x = self.proj(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


class MLPProjectionHead(nn.Module):
    """
    Multi-layer projection head with dropout and activation.

    More expressive than linear projection, useful for complex transformations.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.15,
        normalize: bool = True,
        activation: str = "gelu",
    ):
        """
        Initialize MLP projection head.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden layer dimension. Defaults to in_features.
            out_features: Output dimension. Defaults to in_features.
            dropout: Dropout probability.
            normalize: Whether to L2-normalize output.
            activation: Activation function ("gelu", "relu", "silu").
        """
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.normalize = normalize

        # Activation function
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(inplace=True),
            "silu": nn.SiLU(inplace=True),
        }
        self.activation = activations.get(activation, nn.GELU())

        # MLP layers
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings through MLP.

        Args:
            x: Input tensor of shape (B, in_features)

        Returns:
            Projected tensor of shape (B, out_features)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


class BottleneckProjectionHead(nn.Module):
    """
    Bottleneck projection head that compresses then expands.

    Useful for dimensionality reduction or creating compact embeddings.
    """

    def __init__(
        self,
        in_features: int,
        bottleneck_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        """
        Initialize bottleneck projection head.

        Args:
            in_features: Input dimension.
            bottleneck_features: Compressed dimension.
            out_features: Output dimension. Defaults to in_features.
            dropout: Dropout probability.
            normalize: Whether to L2-normalize output.
        """
        super().__init__()

        out_features = out_features or in_features
        self.normalize = normalize

        self.compress = nn.Linear(in_features, bottleneck_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.expand = nn.Linear(bottleneck_features, out_features)

        # Initialize
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)
        nn.init.xavier_uniform_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project through bottleneck.

        Args:
            x: Input tensor of shape (B, in_features)

        Returns:
            Projected tensor of shape (B, out_features)
        """
        x = self.compress(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.expand(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


class BNProjectionHead(nn.Module):
    """
    Projection head with Batch Normalization.

    Often used in self-supervised learning methods like SimCLR.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        use_bn: bool = True,
    ):
        """
        Initialize BN projection head.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden layer dimension.
            out_features: Output dimension.
            use_bn: Whether to use batch normalization.
        """
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features) if use_bn else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)

        # SimCLR-style BN on output
        self.bn2 = nn.BatchNorm1d(out_features, affine=False) if use_bn else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project with batch normalization.

        Args:
            x: Input tensor of shape (B, in_features)

        Returns:
            Projected tensor of shape (B, out_features)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.bn2(x)

        return x
