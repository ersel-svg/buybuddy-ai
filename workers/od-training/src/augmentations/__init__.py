"""
SOTA Augmentations for Object Detection Training.

This module provides a complete augmentation system with:
- Preset-based configuration (SOTA, Heavy, Medium, Light, None)
- Multi-image augmentations (Mosaic, MixUp, CutMix, CopyPaste)
- Single-image augmentations via Albumentations
- UI-friendly category organization

Quick Start:
    from augmentations import AugmentationPipeline

    # Using SOTA preset (recommended)
    pipeline = AugmentationPipeline.from_preset("sota", img_size=640)

    # In dataset:
    image, target = pipeline(image, target_dict)

For UI integration:
    from augmentations import get_preset_info, get_augmentation_categories

    presets = get_preset_info()  # For preset dropdown
    categories = get_augmentation_categories()  # For augmentation checkboxes
"""

# Main pipeline interface
from .pipeline import (
    AugmentationPipeline,
    create_train_pipeline,
    create_val_pipeline,
)

# Presets and configuration
from .presets import (
    AugmentationPreset,
    AugmentationConfig,
    PresetLevel,
    get_preset,
    get_preset_info,
    get_augmentation_categories,
    PRESETS,
)

# Multi-image augmentations (custom implementations)
from .mosaic import MosaicAugmentation, Mosaic9Augmentation
from .mixup import MixUpAugmentation, CutMixAugmentation
from .copypaste import CopyPasteAugmentation, SimpleCopyPaste

# Albumentations wrapper
from .albumentations_wrapper import (
    build_albumentations_pipeline,
    build_val_pipeline as build_albu_val_pipeline,
    AlbumentationsTransform,
    get_augmentation_list_from_preset,
)

__all__ = [
    # Main interface
    "AugmentationPipeline",
    "create_train_pipeline",
    "create_val_pipeline",
    # Presets
    "AugmentationPreset",
    "AugmentationConfig",
    "PresetLevel",
    "get_preset",
    "get_preset_info",
    "get_augmentation_categories",
    "PRESETS",
    # Multi-image
    "MosaicAugmentation",
    "Mosaic9Augmentation",
    "MixUpAugmentation",
    "CutMixAugmentation",
    "CopyPasteAugmentation",
    "SimpleCopyPaste",
    # Albumentations
    "build_albumentations_pipeline",
    "build_albu_val_pipeline",
    "AlbumentationsTransform",
    "get_augmentation_list_from_preset",
]
