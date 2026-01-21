"""
Loss functions for SOTA OD training.

- Focal Loss: For class imbalance
- IoU Losses: GIoU, DIoU, CIoU, SIoU
- DFL: Distribution Focal Loss
"""

from .focal import FocalLoss, QualityFocalLoss
from .iou import IoULoss, GIoULoss, DIoULoss, CIoULoss, bbox_iou
from .dfl import DFLoss, distribution_focal_loss
