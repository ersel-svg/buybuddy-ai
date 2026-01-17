"""
Backbone implementations for various model families.
"""

from bb_models.backbones.dinov2 import DINOv2Backbone
from bb_models.backbones.dinov3 import DINOv3Backbone
from bb_models.backbones.clip import CLIPBackbone

__all__ = [
    "DINOv2Backbone",
    "DINOv3Backbone",
    "CLIPBackbone",
]
