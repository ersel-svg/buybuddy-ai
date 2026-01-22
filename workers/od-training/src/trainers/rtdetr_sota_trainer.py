"""
RT-DETR SOTA Trainer.

Real-Time DEtection TRansformer trainer with SOTA features:
- HuggingFace transformers integration
- EMA, LLRD, Warmup+Cosine LR
- Mixed precision training
- COCO mAP evaluation

Supports:
- PekingU/rtdetr_r18vd (small)
- PekingU/rtdetr_r50vd (medium)
- PekingU/rtdetr_r101vd (large)

Usage:
    from trainers import RTDETRSOTATrainer
    from training import TrainingConfig, DatasetConfig, OutputConfig

    config = TrainingConfig(
        model_type="rt-detr",
        model_size="l",
        epochs=100,
    )

    trainer = RTDETRSOTATrainer(
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
from losses import FocalLoss, CIoULoss


class RTDETRSOTATrainer(SOTABaseTrainer):
    """
    RT-DETR trainer with SOTA training features.

    Inherits all SOTA features from SOTABaseTrainer:
    - EMA
    - LLRD
    - Mixed Precision
    - Multi-scale training
    - COCO evaluation
    """

    # Model name mapping
    MODEL_NAMES = {
        "s": "PekingU/rtdetr_r18vd",
        "small": "PekingU/rtdetr_r18vd",
        "m": "PekingU/rtdetr_r50vd",
        "medium": "PekingU/rtdetr_r50vd",
        "l": "PekingU/rtdetr_r101vd",
        "large": "PekingU/rtdetr_r101vd",
    }

    def __init__(
        self,
        training_config: TrainingConfig,
        dataset_config: DatasetConfig,
        output_config: OutputConfig,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        # Set model type for LLRD
        training_config.model_type = "rt-detr"

        super().__init__(
            training_config=training_config,
            dataset_config=dataset_config,
            output_config=output_config,
            progress_callback=progress_callback,
        )

        # RT-DETR specific components
        self.processor = None

    def setup_model(self):
        """Setup RT-DETR model from HuggingFace."""
        print("Setting up RT-DETR model...")

        model_size = self.training_config.model_size.lower()
        model_name = self.MODEL_NAMES.get(model_size, self.MODEL_NAMES["m"])

        try:
            from transformers import (
                RTDetrForObjectDetection,
                RTDetrImageProcessor,
            )

            # Load model config first, then set proper prior_prob to avoid math domain error
            from transformers import RTDetrConfig

            config = RTDetrConfig.from_pretrained(model_name)
            config.num_labels = self.dataset_config.num_classes
            # Set valid prior_prob (must be between 0 and 1, exclusive)
            # Default is usually 0.01, which works for any num_labels
            if not hasattr(config, 'prior_prob') or config.prior_prob <= 0 or config.prior_prob >= 1:
                config.prior_prob = 0.01

            self.model = RTDetrForObjectDetection.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True,
            )

            # Load processor for post-processing
            self.processor = RTDetrImageProcessor.from_pretrained(model_name)

            print(f"Loaded RT-DETR from {model_name}")

        except ImportError as e:
            print(f"RT-DETR not available: {e}")
            print("Falling back to DETR...")

            from transformers import DetrForObjectDetection, DetrImageProcessor

            self.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=self.dataset_config.num_classes,
                ignore_mismatched_sizes=True,
            )
            self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Freeze backbone initially if using heavy augmentation
        if self.training_config.augmentation_preset == "heavy":
            self._freeze_backbone()

        print(f"Model initialized with {self.dataset_config.num_classes} classes")

    def _freeze_backbone(self, unfreeze_layers: int = 0):
        """Freeze backbone layers for initial training stability."""
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                # Optionally unfreeze last N layers
                layer_num = -1
                for i in range(4, 0, -1):
                    if f"layer{i}" in name:
                        layer_num = i
                        break

                if layer_num < (4 - unfreeze_layers + 1):
                    param.requires_grad = False

        frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in self.model.parameters())
        print(f"Frozen {frozen_count}/{total_count} parameters")

    def forward_pass(self, images: torch.Tensor) -> Any:
        """
        Model forward pass.

        Args:
            images: [B, C, H, W] normalized images

        Returns:
            Model outputs (RTDetrObjectDetectionOutput)
        """
        outputs = self.model(pixel_values=images)
        return outputs

    def compute_loss(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute training loss using Hungarian matching.

        Since forward_pass doesn't provide labels, we always use
        manual loss computation with bipartite matching.

        Args:
            outputs: RTDetrObjectDetectionOutput
            targets: List of target dicts

        Returns:
            Total loss
        """
        # Prepare labels for loss computation
        labels = []
        for t in targets:
            boxes = t['boxes']
            class_labels = t['labels']

            # Convert boxes to relative coordinates if needed
            # RT-DETR expects [cx, cy, w, h] in [0, 1]
            if boxes.numel() > 0 and boxes.max() > 1.0:
                # Assume boxes are in absolute xyxy, convert to relative cxcywh
                h, w = self.training_config.img_size, self.training_config.img_size
                boxes = self._xyxy_to_cxcywh(boxes, w, h)

            labels.append({
                'class_labels': class_labels.long(),
                'boxes': boxes.float(),
            })

        # Always use manual loss computation since we've already done forward pass
        return self._compute_manual_loss(outputs, labels)

    def _compute_manual_loss(
        self,
        outputs: Any,
        labels: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute loss manually using our loss functions.

        Used when model doesn't compute loss internally.
        """
        # Classification loss
        cls_loss = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
        box_loss = CIoULoss(reduction="mean")

        total_loss = torch.tensor(0.0, device=self.device)

        # Get predictions
        logits = outputs.logits  # [B, num_queries, num_classes + 1]
        pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]

        batch_size = logits.size(0)
        num_queries = logits.size(1)

        for b in range(batch_size):
            target_boxes = labels[b]['boxes']
            target_labels = labels[b]['class_labels']

            if len(target_boxes) == 0:
                continue

            # Simple matching: assign each query to nearest GT
            pred_b = pred_boxes[b]  # [num_queries, 4]
            logits_b = logits[b]  # [num_queries, num_classes + 1]

            # Compute cost matrix
            from scipy.optimize import linear_sum_assignment
            import numpy as np

            with torch.no_grad():
                # Box IoU cost
                iou_matrix = self._batch_iou(pred_b, target_boxes)
                box_cost = 1 - iou_matrix

                # Classification cost
                probs = logits_b.softmax(-1)  # RT-DETR: num_labels directly (no separate no-object)
                cls_cost = -probs[:, target_labels]

                # Combined cost
                cost = 5 * box_cost + cls_cost
                cost = cost.cpu().numpy()

                # Hungarian matching
                row_ind, col_ind = linear_sum_assignment(cost)

            # Compute loss for matched pairs
            matched_pred_boxes = pred_b[row_ind]
            matched_pred_logits = logits_b[row_ind]
            matched_target_boxes = target_boxes[col_ind]
            matched_target_labels = target_labels[col_ind]

            # Box loss (convert both to xyxy for CIoU)
            pred_xyxy = self._cxcywh_to_xyxy(matched_pred_boxes)
            target_xyxy = self._cxcywh_to_xyxy(matched_target_boxes)
            total_loss += box_loss(pred_xyxy, target_xyxy)

            # Classification loss
            total_loss += cls_loss(matched_pred_logits, matched_target_labels)

        return total_loss / max(batch_size, 1)

    def _batch_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU between two sets of boxes (cxcywh format)."""
        # Convert to xyxy
        b1 = self._cxcywh_to_xyxy(boxes1)
        b2 = self._cxcywh_to_xyxy(boxes2)

        # Compute intersection
        lt = torch.max(b1[:, None, :2], b2[None, :, :2])
        rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        # Compute union
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
        """Convert cxcywh (relative or absolute) to xyxy."""
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
        """
        Convert model outputs to predictions for evaluation.

        Args:
            outputs: RTDetrObjectDetectionOutput
            targets: List of targets (for image size info)

        Returns:
            List of prediction dicts with 'boxes', 'scores', 'labels'
        """
        predictions = []

        logits = outputs.logits  # [B, num_queries, num_classes + 1]
        pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]

        batch_size = logits.size(0)
        img_size = self.training_config.img_size

        for b in range(batch_size):
            # Get probabilities (RT-DETR outputs num_labels directly)
            probs = logits[b].softmax(-1)  # [num_queries, num_classes]
            scores, labels = probs.max(-1)

            # Filter by confidence
            keep = scores > 0.3

            # Get boxes and convert to xyxy absolute
            boxes = pred_boxes[b][keep]  # [N, 4] in cxcywh relative
            scores = scores[keep]
            labels = labels[keep]

            # Convert to xyxy absolute
            boxes = self._cxcywh_to_xyxy(boxes)
            boxes[:, [0, 2]] *= img_size
            boxes[:, [1, 3]] *= img_size

            # Clip to image bounds
            boxes = boxes.clamp(min=0, max=img_size)

            predictions.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            })

        return predictions


def create_rtdetr_trainer(
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
) -> RTDETRSOTATrainer:
    """
    Convenience function to create RT-DETR trainer.

    Args:
        model_size: "s", "m", or "l"
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
        Configured RTDETRSOTATrainer
    """
    training_config = TrainingConfig(
        model_type="rt-detr",
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

    return RTDETRSOTATrainer(
        training_config=training_config,
        dataset_config=dataset_config,
        output_config=output_config,
        progress_callback=progress_callback,
    )
