"""
Unified Augmentation Pipeline for Object Detection Training.

This module combines:
1. Multi-image augmentations (Mosaic, MixUp, CopyPaste) - Custom implementations
2. Single-image augmentations - Albumentations library

The pipeline is designed to be:
- UI-configurable via presets or custom settings
- Easy to use with datasets
- Production-ready with proper error handling

Usage:
    from augmentations.pipeline import AugmentationPipeline
    from augmentations.presets import get_preset

    # Using preset
    pipeline = AugmentationPipeline.from_preset("sota", img_size=640)

    # Or custom config
    preset = get_preset("custom", overrides={
        "mosaic": {"enabled": True, "prob": 0.5},
        "horizontal_flip": {"enabled": True, "prob": 0.5}
    })
    pipeline = AugmentationPipeline(preset, img_size=640)

    # In dataset:
    image, target = pipeline(
        image=img,
        target=target_dict,
        dataset=dataset_ref,  # For multi-image augmentations
    )
"""

import random
from typing import Tuple, Dict, Any, Optional, Callable, List
import numpy as np

from .presets import (
    AugmentationPreset,
    get_preset,
    PresetLevel,
)
from .mosaic import MosaicAugmentation, Mosaic9Augmentation
from .mixup import MixUpAugmentation, CutMixAugmentation
from .copypaste import CopyPasteAugmentation, SimpleCopyPaste
from .albumentations_wrapper import (
    build_albumentations_pipeline,
    build_val_pipeline,
    get_augmentation_list_from_preset,
)


class AugmentationPipeline:
    """
    Main augmentation pipeline combining multi-image and single-image transforms.

    The pipeline applies augmentations in this order:
    1. Multi-image augmentations (Mosaic, MixUp, CopyPaste)
    2. Single-image augmentations via Albumentations

    Args:
        preset: AugmentationPreset configuration
        img_size: Target image size (default: 640)
        is_training: Whether this is for training (default: True)
        sample_fn: Function to sample random images from dataset
                   Required for multi-image augmentations
                   Signature: sample_fn(exclude_idx: int) -> Tuple[image, target]
    """

    def __init__(
        self,
        preset: AugmentationPreset,
        img_size: int = 640,
        is_training: bool = True,
        sample_fn: Optional[Callable] = None,
    ):
        self.preset = preset
        self.img_size = img_size
        self.is_training = is_training
        self.sample_fn = sample_fn

        # Initialize multi-image augmentations
        self._init_multi_image_augmentations()

        # Initialize Albumentations pipeline
        if is_training:
            self.albu_transform = build_albumentations_pipeline(
                preset,
                img_size=img_size,
                include_normalize=True,
                include_to_tensor=True,
            )
        else:
            self.albu_transform = build_val_pipeline(
                img_size=img_size,
                include_normalize=True,
                include_to_tensor=True,
            )

        # Log enabled augmentations
        self.enabled_augmentations = get_augmentation_list_from_preset(preset)

    def _init_multi_image_augmentations(self):
        """Initialize multi-image augmentation instances."""
        # Mosaic
        if self.preset.mosaic.enabled:
            self.mosaic = MosaicAugmentation(
                img_size=self.preset.mosaic.params.get("img_size", self.img_size),
                center_ratio_range=self.preset.mosaic.params.get(
                    "center_ratio_range", (0.5, 1.5)
                ),
                min_bbox_size=self.preset.mosaic.params.get("min_bbox_size", 2),
            )
            self.mosaic_prob = self.preset.mosaic.prob
        else:
            self.mosaic = None
            self.mosaic_prob = 0.0

        # Mosaic-9
        if self.preset.mosaic9.enabled:
            self.mosaic9 = Mosaic9Augmentation(
                img_size=self.preset.mosaic9.params.get("img_size", self.img_size),
                min_bbox_size=self.preset.mosaic9.params.get("min_bbox_size", 2),
            )
            self.mosaic9_prob = self.preset.mosaic9.prob
        else:
            self.mosaic9 = None
            self.mosaic9_prob = 0.0

        # MixUp
        if self.preset.mixup.enabled:
            self.mixup = MixUpAugmentation(
                alpha=self.preset.mixup.params.get("alpha", 8.0),
                min_ratio=self.preset.mixup.params.get("min_ratio", 0.0),
                max_ratio=self.preset.mixup.params.get("max_ratio", 1.0),
            )
            self.mixup_prob = self.preset.mixup.prob
        else:
            self.mixup = None
            self.mixup_prob = 0.0

        # CutMix
        if self.preset.cutmix.enabled:
            self.cutmix = CutMixAugmentation(
                alpha=self.preset.cutmix.params.get("alpha", 1.0),
                min_area_ratio=self.preset.cutmix.params.get("min_area_ratio", 0.1),
                max_area_ratio=self.preset.cutmix.params.get("max_area_ratio", 0.5),
            )
            self.cutmix_prob = self.preset.cutmix.prob
        else:
            self.cutmix = None
            self.cutmix_prob = 0.0

        # CopyPaste
        if self.preset.copypaste.enabled:
            self.copypaste = CopyPasteAugmentation(
                prob=1.0,  # Prob handled externally
                max_paste=self.preset.copypaste.params.get("max_paste", 3),
                blend_ratio=self.preset.copypaste.params.get("blend_ratio", 0.0),
            )
            self.copypaste_prob = self.preset.copypaste.prob
        else:
            self.copypaste = None
            self.copypaste_prob = 0.0

    def __call__(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        idx: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply full augmentation pipeline.

        Args:
            image: Input image [H, W, C] (numpy array, BGR or RGB)
            target: Target dict with 'boxes' [N, 4] and 'labels' [N]
            idx: Current sample index (for excluding in random sampling)

        Returns:
            Tuple of (augmented_image, augmented_target)
            - image is a torch tensor if ToTensorV2 is included
            - target dict contains 'boxes' and 'labels'
        """
        if not self.is_training:
            # Validation: only apply resize and normalize
            return self._apply_albumentations(image, target)

        # Step 1: Multi-image augmentations
        image, target = self._apply_multi_image_augmentations(image, target, idx)

        # Step 2: Single-image augmentations via Albumentations
        return self._apply_albumentations(image, target)

    def _apply_multi_image_augmentations(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply multi-image augmentations if enabled and sample_fn is available."""
        if self.sample_fn is None:
            return image, target

        # Mosaic-9 (exclusive with Mosaic)
        if self.mosaic9 and random.random() < self.mosaic9_prob:
            try:
                images = [image]
                targets = [target]
                for _ in range(8):  # Need 8 more images for mosaic-9
                    img, tgt = self.sample_fn(idx)
                    images.append(img)
                    targets.append(tgt)
                image, target = self.mosaic9(images, targets)
                return image, target  # Skip regular mosaic
            except Exception:
                pass  # Fall through to regular mosaic or skip

        # Mosaic (4 images)
        if self.mosaic and random.random() < self.mosaic_prob:
            try:
                images = [image]
                targets = [target]
                for _ in range(3):  # Need 3 more images
                    img, tgt = self.sample_fn(idx)
                    images.append(img)
                    targets.append(tgt)
                image, target = self.mosaic(images, targets)
            except Exception:
                pass  # Continue with single image

        # MixUp (2 images) - can be combined with Mosaic
        if self.mixup and random.random() < self.mixup_prob:
            try:
                img2, tgt2 = self.sample_fn(idx)
                image, target = self.mixup(image, target, img2, tgt2)
            except Exception:
                pass

        # CutMix (alternative to MixUp)
        elif self.cutmix and random.random() < self.cutmix_prob:
            try:
                img2, tgt2 = self.sample_fn(idx)
                image, target = self.cutmix(image, target, img2, tgt2)
            except Exception:
                pass

        # CopyPaste
        if self.copypaste and random.random() < self.copypaste_prob:
            try:
                img2, tgt2 = self.sample_fn(idx)
                image, target = self.copypaste(image, target, img2, tgt2)
            except Exception:
                pass

        return image, target

    def _apply_albumentations(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply Albumentations transforms."""
        boxes = target.get('boxes', np.zeros((0, 4)))
        labels = target.get('labels', np.array([]))

        # Convert to list format for Albumentations
        boxes_list = [list(b) for b in boxes] if len(boxes) > 0 else []
        labels_list = list(labels) if len(labels) > 0 else []

        # Apply transforms
        try:
            result = self.albu_transform(
                image=image,
                bboxes=boxes_list,
                labels=labels_list,
            )

            # Convert back to numpy arrays
            transformed_boxes = np.array(result['bboxes']) if result['bboxes'] else np.zeros((0, 4))
            transformed_labels = np.array(result['labels']) if result['labels'] else np.array([])

            transformed_target = {
                'boxes': transformed_boxes,
                'labels': transformed_labels,
            }

            return result['image'], transformed_target

        except Exception as e:
            # Fallback: return original with basic resize and tensor conversion
            import warnings
            import torch
            import cv2
            warnings.warn(f"Augmentation failed: {e}. Returning original.")
            
            # Resize image to target size
            h, w = image.shape[:2]
            if h != self.img_size or w != self.img_size:
                image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Normalize and convert to tensor
            image = image.astype('float32') / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            # Convert target boxes and labels to tensor
            target = {
                'boxes': torch.from_numpy(target.get('boxes', np.zeros((0, 4))).astype('float32')),
                'labels': torch.from_numpy(target.get('labels', np.array([])).astype('int64')),
            }
            
            return image, target

    def set_sample_fn(self, sample_fn: Callable):
        """Set the sample function for multi-image augmentations."""
        self.sample_fn = sample_fn

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        img_size: int = 640,
        is_training: bool = True,
        sample_fn: Optional[Callable] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "AugmentationPipeline":
        """
        Create pipeline from preset name.

        Args:
            preset_name: One of "sota", "heavy", "medium", "light", "none", "custom"
            img_size: Target image size
            is_training: Whether for training
            sample_fn: Sample function for multi-image augmentations
            overrides: Optional preset overrides

        Returns:
            Configured AugmentationPipeline
        """
        preset = get_preset(preset_name, overrides=overrides)
        return cls(
            preset=preset,
            img_size=img_size,
            is_training=is_training,
            sample_fn=sample_fn,
        )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        sample_fn: Optional[Callable] = None,
    ) -> "AugmentationPipeline":
        """
        Create pipeline from configuration dictionary.

        Expected config format (from API):
        {
            "preset": "sota",
            "img_size": 640,
            "overrides": {
                "mosaic": {"enabled": True, "prob": 0.5},
                ...
            }
        }

        Args:
            config: Configuration dictionary
            sample_fn: Sample function

        Returns:
            Configured AugmentationPipeline
        """
        preset_name = config.get("preset", "sota")
        img_size = config.get("img_size", 640)
        overrides = config.get("overrides", None)
        is_training = config.get("is_training", True)

        return cls.from_preset(
            preset_name=preset_name,
            img_size=img_size,
            is_training=is_training,
            sample_fn=sample_fn,
            overrides=overrides,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information for logging."""
        return {
            "img_size": self.img_size,
            "is_training": self.is_training,
            "enabled_augmentations": self.enabled_augmentations,
            "multi_image": {
                "mosaic": self.mosaic is not None,
                "mosaic_prob": self.mosaic_prob,
                "mosaic9": self.mosaic9 is not None,
                "mosaic9_prob": self.mosaic9_prob,
                "mixup": self.mixup is not None,
                "mixup_prob": self.mixup_prob,
                "cutmix": self.cutmix is not None,
                "cutmix_prob": self.cutmix_prob,
                "copypaste": self.copypaste is not None,
                "copypaste_prob": self.copypaste_prob,
            },
        }


def create_train_pipeline(
    preset_name: str = "sota",
    img_size: int = 640,
    sample_fn: Optional[Callable] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AugmentationPipeline:
    """
    Convenience function to create training pipeline.

    Args:
        preset_name: Preset name
        img_size: Image size
        sample_fn: Sample function
        overrides: Preset overrides

    Returns:
        Training augmentation pipeline
    """
    return AugmentationPipeline.from_preset(
        preset_name=preset_name,
        img_size=img_size,
        is_training=True,
        sample_fn=sample_fn,
        overrides=overrides,
    )


def create_val_pipeline(
    img_size: int = 640,
) -> AugmentationPipeline:
    """
    Convenience function to create validation pipeline.

    Args:
        img_size: Image size

    Returns:
        Validation augmentation pipeline (no augmentations)
    """
    preset = get_preset("none")
    return AugmentationPipeline(
        preset=preset,
        img_size=img_size,
        is_training=False,
        sample_fn=None,
    )
