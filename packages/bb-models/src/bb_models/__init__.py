"""
BuyBuddy AI Shared Model Package (bb-models)

Supports:
- DINOv2 (Small, Base, Large)
- DINOv3 (Small, Base, Large)
- CLIP (ViT-B/16, ViT-B/32, ViT-L/14)
- Custom fine-tuned models

Usage:
    from bb_models import get_backbone, get_model_config
    from bb_models.heads import GeMPooling, ArcFaceHead
"""

from bb_models.registry import (
    MODEL_CONFIGS,
    get_model_config,
    get_backbone,
    list_available_models,
    is_model_supported,
)
from bb_models.base import BaseBackbone

__version__ = "0.1.2"
__all__ = [
    "MODEL_CONFIGS",
    "get_model_config",
    "get_backbone",
    "list_available_models",
    "is_model_supported",
    "BaseBackbone",
]
