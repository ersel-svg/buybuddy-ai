"""
Base classes for model backbones.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """
    Abstract base class for all model backbones.

    All backbones must implement:
    - forward(): Extract features from images
    - get_embedding_dim(): Return the output embedding dimension
    - load_pretrained(): Load pretrained weights
    """

    def __init__(self, model_id: str):
        super().__init__()
        self.model_id = model_id
        self._embedding_dim: Optional[int] = None
        self._image_size: int = 224
        self._is_loaded: bool = False

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        if self._embedding_dim is None:
            raise ValueError("Embedding dimension not set. Call load_pretrained() first.")
        return self._embedding_dim

    @property
    def image_size(self) -> int:
        """Return the expected input image size."""
        return self._image_size

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        return self._is_loaded

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W)

        Returns:
            Feature tensor of shape (B, embedding_dim)
        """
        pass

    @abstractmethod
    def load_pretrained(self, checkpoint_url: Optional[str] = None) -> None:
        """
        Load pretrained weights.

        Args:
            checkpoint_url: Optional URL to custom checkpoint.
                          If None, loads default HuggingFace weights.
        """
        pass

    @abstractmethod
    def get_layer_groups(self) -> list:
        """
        Get parameter groups for layer-wise learning rate decay (LLRD).

        Returns:
            List of parameter groups, from earliest to latest layers.
        """
        pass

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration for this model.

        Returns:
            Dict with keys: image_mean, image_std, image_size
        """
        return {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "image_size": self._image_size,
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first N layers of the backbone.

        Args:
            num_layers: Number of layers to freeze from the beginning.
        """
        layer_groups = self.get_layer_groups()
        for i, group in enumerate(layer_groups):
            if i < num_layers:
                for param in group:
                    param.requires_grad = False
            else:
                for param in group:
                    param.requires_grad = True

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"model_id='{self.model_id}', "
            f"embedding_dim={self._embedding_dim}, "
            f"image_size={self._image_size}, "
            f"params={total:,}, "
            f"trainable={trainable:,})"
        )
