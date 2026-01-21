"""
OD Training Worker - Configuration

Configuration for training object detection models with SOTA features.

Supports:
- RT-DETR (Apache 2.0)
- D-FINE (Apache 2.0)

SOTA Features:
- EMA (Exponential Moving Average)
- LLRD (Layer-wise Learning Rate Decay)
- Warmup + Cosine LR Scheduler
- Mixed Precision (FP16)
- Augmentation Presets (SOTA, Heavy, Medium, Light, None)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TrainingConfig:
    """Training configuration with SOTA features."""

    # Model settings
    model_type: str = "rt-detr"  # rt-detr, d-fine
    model_size: str = "l"  # s, m, l, x (for d-fine)
    pretrained: bool = True

    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 16
    accumulation_steps: int = 1  # Gradient accumulation

    # Optimizer (LLRD)
    optimizer: str = "adamw"
    learning_rate: float = 0.0001  # Base LR
    weight_decay: float = 0.0001
    llrd_decay: float = 0.9  # Layer-wise LR decay (1.0 = no decay)
    head_lr_factor: float = 10.0  # Detection head LR multiplier

    # Scheduler
    scheduler: str = "cosine"  # cosine, step, linear
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01  # Final LR = base_lr * min_lr_ratio

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000

    # Mixed Precision
    mixed_precision: bool = True

    # Regularization
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0

    # Image settings
    image_size: int = 640

    # Multi-scale Training
    multi_scale: bool = False
    multi_scale_range: tuple = (0.5, 1.5)

    # Augmentation (NEW: Preset-based)
    augmentation_preset: str = "sota"  # sota, heavy, medium, light, none
    augmentation_overrides: Optional[Dict[str, Any]] = None

    # Legacy (for backward compatibility)
    use_augmentation: bool = True
    mosaic: bool = True
    mixup: float = 0.3

    # Early stopping
    patience: int = 20
    min_delta: float = 0.001

    # Checkpointing
    save_best_only: bool = True
    save_freq: int = 5  # Save every N epochs

    # Device
    device: str = "cuda"

    def to_sota_config(self):
        """Convert to SOTA TrainingConfig."""
        from src.training import TrainingConfig as SOTATrainingConfig

        return SOTATrainingConfig(
            model_type=self.model_type,
            model_size=self.model_size,
            pretrained=self.pretrained,
            epochs=self.epochs,
            batch_size=self.batch_size,
            accumulation_steps=self.accumulation_steps,
            optimizer=self.optimizer,
            base_lr=self.learning_rate,
            weight_decay=self.weight_decay,
            llrd_decay=self.llrd_decay,
            head_lr_factor=self.head_lr_factor,
            scheduler=self.scheduler,
            warmup_epochs=self.warmup_epochs,
            min_lr_ratio=self.min_lr_ratio,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            ema_warmup_steps=self.ema_warmup_steps,
            use_amp=self.mixed_precision,
            gradient_clip=self.gradient_clip,
            label_smoothing=self.label_smoothing,
            img_size=self.image_size,
            multi_scale=self.multi_scale,
            multi_scale_range=self.multi_scale_range,
            augmentation_preset=self.augmentation_preset,
            augmentation_overrides=self.augmentation_overrides,
            save_freq=self.save_freq,
            patience=self.patience,
            device=self.device,
        )


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    # Dataset paths (will be set from job input)
    dataset_path: str = ""
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""

    # COCO format annotation files
    train_ann_file: str = ""
    val_ann_file: str = ""

    # Dataset format
    format: str = "coco"  # coco, yolo

    # Class info
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)

    def to_sota_config(self):
        """Convert to SOTA DatasetConfig."""
        from src.training import DatasetConfig as SOTADatasetConfig

        return SOTADatasetConfig(
            train_img_dir=self.train_path,
            train_ann_file=self.train_ann_file,
            val_img_dir=self.val_path,
            val_ann_file=self.val_ann_file,
            num_classes=self.num_classes,
            class_names=self.class_names if self.class_names else None,
        )


@dataclass
class OutputConfig:
    """Output configuration."""

    # Output paths
    output_dir: str = "/tmp/training_output"
    checkpoint_dir: str = "/tmp/checkpoints"
    logs_dir: str = "/tmp/logs"

    # Upload settings
    upload_to_supabase: bool = True
    supabase_bucket: str = "od-models"

    def to_sota_config(self):
        """Convert to SOTA OutputConfig."""
        from src.training import OutputConfig as SOTAOutputConfig

        return SOTAOutputConfig(
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            log_dir=self.logs_dir,
        )


# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# Model configurations (Apache 2.0 licensed only)
MODEL_CONFIGS = {
    "rt-detr": {
        "s": {"model_name": "PekingU/rtdetr_r18vd", "description": "RT-DETR Small (ResNet-18)"},
        "small": {"model_name": "PekingU/rtdetr_r18vd", "description": "RT-DETR Small"},
        "m": {"model_name": "PekingU/rtdetr_r50vd", "description": "RT-DETR Medium (ResNet-50)"},
        "medium": {"model_name": "PekingU/rtdetr_r50vd", "description": "RT-DETR Medium"},
        "l": {"model_name": "PekingU/rtdetr_r101vd", "description": "RT-DETR Large (ResNet-101)"},
        "large": {"model_name": "PekingU/rtdetr_r101vd", "description": "RT-DETR Large"},
    },
    "d-fine": {
        "s": {"model_name": "ustc-community/dfine-small-coco", "description": "D-FINE Small"},
        "small": {"model_name": "ustc-community/dfine-small-coco", "description": "D-FINE Small"},
        "m": {"model_name": "ustc-community/dfine-medium-coco", "description": "D-FINE Medium"},
        "medium": {"model_name": "ustc-community/dfine-medium-coco", "description": "D-FINE Medium"},
        "l": {"model_name": "ustc-community/dfine-large-coco", "description": "D-FINE Large"},
        "large": {"model_name": "ustc-community/dfine-large-coco", "description": "D-FINE Large"},
        "x": {"model_name": "ustc-community/dfine-xlarge-coco", "description": "D-FINE XLarge"},
        "xlarge": {"model_name": "ustc-community/dfine-xlarge-coco", "description": "D-FINE XLarge"},
    },
}

# Augmentation preset info (for API documentation)
AUGMENTATION_PRESETS = {
    "sota": {
        "name": "SOTA (Recommended)",
        "description": "State-of-the-art: Mosaic, MixUp, CopyPaste + standard augmentations",
        "training_time_factor": 1.5,
        "accuracy_boost": "+3-5% mAP",
    },
    "heavy": {
        "name": "Heavy",
        "description": "All augmentations enabled, ideal for small datasets",
        "training_time_factor": 2.0,
        "accuracy_boost": "+5-8% mAP",
    },
    "medium": {
        "name": "Medium",
        "description": "Balanced augmentations for general use",
        "training_time_factor": 1.2,
        "accuracy_boost": "+2-3% mAP",
    },
    "light": {
        "name": "Light",
        "description": "Basic augmentations only, fast training",
        "training_time_factor": 1.05,
        "accuracy_boost": "+1% mAP",
    },
    "none": {
        "name": "None",
        "description": "No augmentation, for baseline comparison",
        "training_time_factor": 1.0,
        "accuracy_boost": "Baseline",
    },
}
