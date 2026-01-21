"""
Evaluation module for Object Detection.

Provides:
- COCO mAP evaluation (mAP@50, mAP@50:95)
- Per-class metrics
- Confusion matrix generation
"""

from .coco_eval import (
    COCOEvaluator,
    compute_iou,
    evaluate_predictions,
    calculate_map,
)

__all__ = [
    "COCOEvaluator",
    "compute_iou",
    "evaluate_predictions",
    "calculate_map",
]
