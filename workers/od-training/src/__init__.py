"""
SOTA Object Detection Training - Core Modules

This package contains SOTA training utilities:
- augmentations: Mosaic, MixUp, CopyPaste, Albumentations integration
- losses: Focal, CIoU, DFL
- training: EMA, LLRD optimizer, LR scheduler, SOTABaseTrainer
- evaluation: COCO mAP evaluator
- data: Dataset and DataLoader utilities
- trainers: RT-DETR, D-FINE trainers with SOTA features
"""

from . import augmentations
from . import losses
from . import training
from . import evaluation
from . import data
from . import trainers

__all__ = [
    "augmentations",
    "losses",
    "training",
    "evaluation",
    "data",
    "trainers",
]
