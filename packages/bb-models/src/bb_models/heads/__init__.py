"""
Training heads for fine-tuning models.
"""

from bb_models.heads.gem_pooling import GeMPooling
from bb_models.heads.arcface import ArcFaceHead, EnhancedArcFaceLoss
from bb_models.heads.projection import ProjectionHead

__all__ = [
    "GeMPooling",
    "ArcFaceHead",
    "EnhancedArcFaceLoss",
    "ProjectionHead",
]
