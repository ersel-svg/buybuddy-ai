"""
Image preprocessing utilities for training and inference.

Provides:
- Training augmentations (domain-aware)
- Validation/inference transforms
- Preprocessing pipeline creation
"""

from typing import Optional, Tuple, List, Dict, Any
from PIL import Image

import torch
from torchvision import transforms


def get_preprocessing_transforms(
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    is_training: bool = False,
    augmentation_strength: str = "moderate",
) -> transforms.Compose:
    """
    Get preprocessing transforms for a model.

    Args:
        image_size: Target image size.
        image_mean: Normalization mean. Defaults to ImageNet.
        image_std: Normalization std. Defaults to ImageNet.
        is_training: Whether to include training augmentations.
        augmentation_strength: One of "light", "moderate", "strong".

    Returns:
        Composed transforms.
    """
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    if is_training:
        return create_train_transforms(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            strength=augmentation_strength,
        )
    else:
        return create_val_transforms(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )


def create_train_transforms(
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    strength: str = "moderate",
) -> transforms.Compose:
    """
    Create training transforms with augmentations.

    Augmentation strategies:
    - light: Minimal augmentation for fine-grained tasks
    - moderate: Balanced augmentation (recommended)
    - strong: Aggressive augmentation for regularization

    Args:
        image_size: Target image size.
        image_mean: Normalization mean.
        image_std: Normalization std.
        strength: Augmentation strength level.

    Returns:
        Training transforms.
    """
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    if strength == "light":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])

    elif strength == "moderate":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

    elif strength == "strong":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=(-10, 10),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])

    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")


def create_val_transforms(
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """
    Create validation/inference transforms.

    No augmentation, just resize, crop, and normalize.

    Args:
        image_size: Target image size.
        image_mean: Normalization mean.
        image_std: Normalization std.

    Returns:
        Validation transforms.
    """
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])


def preprocess_image(
    image: Image.Image,
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    is_training: bool = False,
    augmentation_strength: str = "moderate",
) -> torch.Tensor:
    """
    Preprocess a single PIL image.

    Args:
        image: PIL Image.
        image_size: Target size.
        image_mean: Normalization mean.
        image_std: Normalization std.
        is_training: Whether to apply training augmentations.
        augmentation_strength: Augmentation level.

    Returns:
        Preprocessed tensor of shape (C, H, W).
    """
    transform = get_preprocessing_transforms(
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        is_training=is_training,
        augmentation_strength=augmentation_strength,
    )

    return transform(image)


def denormalize(
    tensor: torch.Tensor,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W).
        image_mean: Normalization mean used.
        image_std: Normalization std used.

    Returns:
        Denormalized tensor with values in [0, 1].
    """
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std.to(tensor.device) + mean.to(tensor.device)


class DomainAwareTransform:
    """
    Domain-aware augmentation that applies different transforms
    based on image domain (real vs synthetic).

    Synthetic images often need less augmentation since they're
    already generated with variation.
    """

    def __init__(
        self,
        image_size: int = 384,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        real_strength: str = "moderate",
        synthetic_strength: str = "light",
    ):
        """
        Initialize domain-aware transform.

        Args:
            image_size: Target image size.
            image_mean: Normalization mean.
            image_std: Normalization std.
            real_strength: Augmentation strength for real images.
            synthetic_strength: Augmentation strength for synthetic images.
        """
        self.real_transform = create_train_transforms(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            strength=real_strength,
        )
        self.synthetic_transform = create_train_transforms(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            strength=synthetic_strength,
        )

    def __call__(self, image: Image.Image, domain: str = "real") -> torch.Tensor:
        """
        Apply domain-appropriate transform.

        Args:
            image: PIL Image.
            domain: "real" or "synthetic".

        Returns:
            Transformed tensor.
        """
        if domain == "synthetic":
            return self.synthetic_transform(image)
        return self.real_transform(image)


# Aliases for backward compatibility
def get_augmentation_transform(
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    strength: str = "moderate",
) -> transforms.Compose:
    """Alias for create_train_transforms (backward compatibility)."""
    return create_train_transforms(
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        strength=strength,
    )


def get_eval_transform(
    image_size: int = 384,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Alias for create_val_transforms (backward compatibility)."""
    return create_val_transforms(
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    )
