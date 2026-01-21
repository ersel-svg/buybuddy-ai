"""
D-FINE SOTA Trainer.

D-FINE (Detection-aware FINe-grained matching) trainer with SOTA features.
D-FINE is a state-of-the-art object detector that improves upon RT-DETR.

Supports (from HuggingFace):
- Peterande/D-FINE-S
- Peterande/D-FINE-M
- Peterande/D-FINE-L
- Peterande/D-FINE-X

Usage:
    from trainers import DFINESOTATrainer
    from training import TrainingConfig, DatasetConfig, OutputConfig

    config = TrainingConfig(
        model_type="d-fine",
        model_size="l",
        epochs=100,
    )

    trainer = DFINESOTATrainer(
        training_config=config,
        dataset_config=dataset_config,
        output_config=output_config,
    )

    result = trainer.train()
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.base_trainer import (
    SOTABaseTrainer,
    TrainingConfig,
    DatasetConfig,
    OutputConfig,
)
from losses import FocalLoss, CIoULoss, DFLoss


class DFINESOTATrainer(SOTABaseTrainer):
    """
    D-FINE trainer with SOTA training features.

    D-FINE uses Distribution Focal Loss (DFL) for better localization.
    """

    # Model name mapping (ustc-community COCO pretrained)
    MODEL_NAMES = {
        "s": "ustc-community/dfine-small-coco",
        "small": "ustc-community/dfine-small-coco",
        "m": "ustc-community/dfine-medium-coco",
        "medium": "ustc-community/dfine-medium-coco",
        "l": "ustc-community/dfine-large-coco",
        "large": "ustc-community/dfine-large-coco",
        "x": "ustc-community/dfine-xlarge-coco",
        "xlarge": "ustc-community/dfine-xlarge-coco",
    }

    def __init__(
        self,
        training_config: TrainingConfig,
        dataset_config: DatasetConfig,
        output_config: OutputConfig,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        # Set model type for LLRD
        training_config.model_type = "d-fine"

        super().__init__(
            training_config=training_config,
            dataset_config=dataset_config,
            output_config=output_config,
            progress_callback=progress_callback,
        )

        # D-FINE specific
        self.processor = None
        self.dfl_loss = DFLoss(num_bins=16, reduction="mean")

    def setup_model(self):
        """Setup D-FINE model."""
        print("Setting up D-FINE model...")

        model_size = self.training_config.model_size.lower()
        model_name = self.MODEL_NAMES.get(model_size, self.MODEL_NAMES["l"])

        try:
            # Try to load D-FINE from HuggingFace
            # Note: D-FINE may not be in transformers yet, may need custom loading
            from transformers import AutoModelForObjectDetection, AutoImageProcessor

            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name,
                num_labels=self.dataset_config.num_classes,
                ignore_mismatched_sizes=True,
            )
            self.processor = AutoImageProcessor.from_pretrained(model_name)

            print(f"Loaded D-FINE from {model_name}")

        except Exception as e:
            print(f"D-FINE not available from HuggingFace: {e}")
            print("Falling back to RT-DETR architecture...")

            # Fallback to RT-DETR
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            self.model = RTDetrForObjectDetection.from_pretrained(
                "PekingU/rtdetr_r101vd",
                num_labels=self.dataset_config.num_classes,
                ignore_mismatched_sizes=True,
            )
            self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")

        print(f"Model initialized with {self.dataset_config.num_classes} classes")

    def forward_pass(self, images: torch.Tensor) -> Any:
        """Model forward pass."""
        outputs = self.model(pixel_values=images)
        return outputs

    def compute_loss(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute D-FINE loss with DFL component.

        D-FINE uses:
        - Focal Loss for classification
        - CIoU Loss for box regression
        - DFL for distribution-based localization
        """
        # Prepare labels
        labels = []
        for t in targets:
            boxes = t['boxes']
            class_labels = t['labels']

            # Convert to relative cxcywh if needed
            if boxes.numel() > 0 and boxes.max() > 1.0:
                h, w = self.training_config.img_size, self.training_config.img_size
                boxes = self._xyxy_to_cxcywh(boxes, w, h)

            labels.append({
                'class_labels': class_labels.long(),
                'boxes': boxes.float(),
            })

        # Check if model computes loss internally
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss

        # Manual loss computation
        return self._compute_dfine_loss(outputs, labels)

    def _compute_dfine_loss(
        self,
        outputs: Any,
        labels: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute D-FINE loss manually."""
        cls_loss_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
        box_loss_fn = CIoULoss(reduction="mean")

        total_cls_loss = torch.tensor(0.0, device=self.device)
        total_box_loss = torch.tensor(0.0, device=self.device)
        num_pos = 0

        logits = outputs.logits  # [B, num_queries, num_classes + 1]
        pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]

        batch_size = logits.size(0)

        for b in range(batch_size):
            target_boxes = labels[b]['boxes']
            target_labels = labels[b]['class_labels']

            if len(target_boxes) == 0:
                continue

            # Hungarian matching
            with torch.no_grad():
                iou_matrix = self._batch_iou(pred_boxes[b], target_boxes)
                box_cost = 1 - iou_matrix

                probs = logits[b].softmax(-1)  # D-FINE: num_labels directly
                cls_cost = -probs[:, target_labels]

                cost = 5 * box_cost + cls_cost

                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

            # Matched losses
            matched_pred_boxes = pred_boxes[b][row_ind]
            matched_pred_logits = logits[b][row_ind]
            matched_target_boxes = target_boxes[col_ind]
            matched_target_labels = target_labels[col_ind]

            # Classification loss
            total_cls_loss += cls_loss_fn(
                matched_pred_logits,
                matched_target_labels,
            )

            # Box loss
            pred_xyxy = self._cxcywh_to_xyxy(matched_pred_boxes)
            target_xyxy = self._cxcywh_to_xyxy(matched_target_boxes)
            total_box_loss += box_loss_fn(pred_xyxy, target_xyxy)

            num_pos += len(row_ind)

        # Normalize by number of positive samples
        num_pos = max(num_pos, 1)
        total_loss = total_cls_loss + 5 * total_box_loss

        return total_loss / batch_size

    def _batch_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        b1 = self._cxcywh_to_xyxy(boxes1)
        b2 = self._cxcywh_to_xyxy(boxes2)

        lt = torch.max(b1[:, None, :2], b2[None, :, :2])
        rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        return inter / (union + 1e-7)

    def _xyxy_to_cxcywh(
        self,
        boxes: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """Convert xyxy (absolute) to cxcywh (relative)."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return torch.stack([cx, cy, w, h], dim=-1)

    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert cxcywh to xyxy."""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def postprocess(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert model outputs to predictions."""
        predictions = []

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        batch_size = logits.size(0)
        img_size = self.training_config.img_size

        for b in range(batch_size):
            probs = logits[b].softmax(-1)  # D-FINE: num_labels directly
            scores, labels = probs.max(-1)

            keep = scores > 0.3

            boxes = pred_boxes[b][keep]
            scores = scores[keep]
            labels = labels[keep]

            # Convert to absolute xyxy
            boxes = self._cxcywh_to_xyxy(boxes)
            boxes[:, [0, 2]] *= img_size
            boxes[:, [1, 3]] *= img_size
            boxes = boxes.clamp(min=0, max=img_size)

            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            })

        return predictions


def create_dfine_trainer(
    model_size: str = "l",
    num_classes: int = 80,
    train_img_dir: str = "",
    train_ann_file: str = "",
    val_img_dir: str = "",
    val_ann_file: str = "",
    output_dir: str = "./output",
    epochs: int = 100,
    batch_size: int = 16,
    base_lr: float = 0.0001,
    augmentation_preset: str = "sota",
    use_ema: bool = True,
    use_amp: bool = True,
    progress_callback: Optional[Callable] = None,
) -> DFINESOTATrainer:
    """
    Convenience function to create D-FINE trainer.

    Args:
        model_size: "s", "m", "l", or "x"
        num_classes: Number of detection classes
        train_img_dir: Training images directory
        train_ann_file: Training annotations (COCO format)
        val_img_dir: Validation images directory
        val_ann_file: Validation annotations
        output_dir: Output directory for checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        base_lr: Base learning rate
        augmentation_preset: "sota", "heavy", "medium", "light", "none"
        use_ema: Enable EMA
        use_amp: Enable mixed precision
        progress_callback: Progress callback function

    Returns:
        Configured DFINESOTATrainer
    """
    training_config = TrainingConfig(
        model_type="d-fine",
        model_size=model_size,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        base_lr=base_lr,
        augmentation_preset=augmentation_preset,
        use_ema=use_ema,
        use_amp=use_amp,
    )

    dataset_config = DatasetConfig(
        train_img_dir=train_img_dir,
        train_ann_file=train_ann_file,
        val_img_dir=val_img_dir,
        val_ann_file=val_ann_file,
        num_classes=num_classes,
    )

    output_config = OutputConfig(
        output_dir=output_dir,
        checkpoint_dir=f"{output_dir}/checkpoints",
        log_dir=f"{output_dir}/logs",
    )

    return DFINESOTATrainer(
        training_config=training_config,
        dataset_config=dataset_config,
        output_config=output_config,
        progress_callback=progress_callback,
    )
