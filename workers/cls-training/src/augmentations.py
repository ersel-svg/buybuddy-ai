"""
Classification Augmentations - SOTA Data Augmentation

Presets:
- sota: AutoAugment + MixUp + CutMix + RandErasing
- heavy: All augmentations enabled
- medium: Standard augmentations
- light: Basic flip/crop only
- none: Just normalization
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple, List
import random
import numpy as np
from PIL import Image
from enum import Enum

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    from timm.data.mixup import Mixup
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class AugmentationPreset(str, Enum):
    SOTA = "sota"
    HEAVY = "heavy"
    MEDIUM = "medium"
    LIGHT = "light"
    NONE = "none"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: int = 224,
    preset: str = "medium",
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
    use_albumentations: bool = True,
) -> Any:
    """Get training transforms based on preset."""
    preset = preset.lower()

    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        return _get_albumentations_train(img_size, preset, mean, std)
    else:
        return _get_torchvision_train(img_size, preset, mean, std)


def _get_albumentations_train(
    img_size: int,
    preset: str,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> Any:
    """Albumentations-based training transforms."""

    transforms_list = []

    # Resize - use newer API
    scale = (0.8, 1.0) if preset in ["light", "none"] else (0.5, 1.0)
    transforms_list.append(
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=scale,
            ratio=(0.75, 1.33),
        )
    )

    if preset == "none":
        pass

    elif preset == "light":
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
        ])

    elif preset == "medium":
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        ])

    elif preset == "heavy":
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.3, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.5,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.02, 0.15), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size // 8,
                max_width=img_size // 8,
                min_holes=1,
                fill_value=0,
                p=0.3,
            ),
        ])

    elif preset == "sota":
        transforms_list.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4,
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.2),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=img_size // 8,
                max_width=img_size // 8,
                min_holes=1,
                fill_value=0,
                p=0.25,
            ),
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
        ])

    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def _get_torchvision_train(
    img_size: int,
    preset: str,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> transforms.Compose:
    """Torchvision-based training transforms (fallback)."""

    transforms_list = []

    if preset == "none":
        transforms_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif preset == "light":
        transforms_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif preset == "medium":
        transforms_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif preset in ["heavy", "sota"]:
        transforms_list.extend([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])

    return transforms.Compose(transforms_list)


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
    use_albumentations: bool = True,
) -> Any:
    """Get validation/test transforms (no augmentation)."""
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class MixUpCutMix:
    """MixUp and CutMix augmentation for batch-level mixing."""

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
        num_classes: int = 1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

        if TIMM_AVAILABLE:
            self._mixup = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=mixup_prob + cutmix_prob,
                switch_prob=switch_prob,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )
        else:
            self._mixup = None

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._mixup is not None:
            return self._mixup(images, targets)
        return self._mixup_cutmix_fallback(images, targets)

    def _mixup_cutmix_fallback(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)

        targets_one_hot = torch.zeros(batch_size, self.num_classes, device=images.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        r = random.random()

        if r < self.mixup_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(batch_size)
            mixed_images = lam * images + (1 - lam) * images[rand_index]
            mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[rand_index]
        elif r < self.mixup_prob + self.cutmix_prob:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(batch_size)
            _, _, H, W = images.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[rand_index]
        else:
            mixed_images = images
            mixed_targets = targets_one_hot

        return mixed_images, mixed_targets


AUGMENTATION_PRESETS = {
    "sota": {"description": "SOTA with MixUp, CutMix", "mixup": True, "cutmix": True},
    "heavy": {"description": "Heavy augmentation", "mixup": False, "cutmix": False},
    "medium": {"description": "Balanced augmentation", "mixup": False, "cutmix": False},
    "light": {"description": "Minimal augmentation", "mixup": False, "cutmix": False},
    "none": {"description": "No augmentation", "mixup": False, "cutmix": False},
}
