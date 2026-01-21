"""
Albumentations Wrapper for Object Detection.

Converts our AugmentationPreset config into Albumentations Compose pipeline.
All single-image augmentations use Albumentations with proper bbox support.

Key features:
- BboxParams with 'pascal_voc' format (x1, y1, x2, y2)
- Automatic filtering of invalid bboxes
- Integration with our preset system

Usage:
    from augmentations.albumentations_wrapper import build_albumentations_pipeline
    from augmentations.presets import get_preset

    preset = get_preset("sota")
    transform = build_albumentations_pipeline(preset, img_size=640)

    # Apply to image and bboxes
    result = transform(
        image=image,
        bboxes=boxes,  # [[x1, y1, x2, y2], ...]
        labels=labels   # [class_id, ...]
    )
    transformed_image = result['image']
    transformed_boxes = result['bboxes']
    transformed_labels = result['labels']
"""

from typing import Optional, List, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .presets import AugmentationPreset, AugmentationConfig


def build_albumentations_pipeline(
    preset: AugmentationPreset,
    img_size: int = 640,
    include_normalize: bool = True,
    include_to_tensor: bool = True,
    min_visibility: float = 0.3,
    min_area: int = 100,
) -> A.Compose:
    """
    Build Albumentations pipeline from preset configuration.

    Args:
        preset: AugmentationPreset with enabled augmentations
        img_size: Target image size for resize
        include_normalize: Whether to include normalization
        include_to_tensor: Whether to include PyTorch tensor conversion
        min_visibility: Minimum bbox visibility after transform (0-1)
        min_area: Minimum bbox area in pixels

    Returns:
        Albumentations Compose pipeline with bbox support
    """
    transforms = []

    # ==== GEOMETRIC ====
    if preset.horizontal_flip.enabled:
        transforms.append(
            A.HorizontalFlip(p=preset.horizontal_flip.prob)
        )

    if preset.vertical_flip.enabled:
        transforms.append(
            A.VerticalFlip(p=preset.vertical_flip.prob)
        )

    if preset.rotate90.enabled:
        transforms.append(
            A.RandomRotate90(p=preset.rotate90.prob)
        )

    if preset.random_rotate.enabled:
        limit = preset.random_rotate.params.get("limit", 15)
        transforms.append(
            A.Rotate(
                limit=limit,
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=preset.random_rotate.prob,
            )
        )

    if preset.affine.enabled:
        params = preset.affine.params
        transforms.append(
            A.Affine(
                scale=params.get("scale", (0.9, 1.1)),
                translate_percent=params.get("translate_percent", 0.1),
                shear=params.get("shear", 10),
                mode=0,  # cv2.BORDER_CONSTANT
                p=preset.affine.prob,
            )
        )

    if preset.perspective.enabled:
        transforms.append(
            A.Perspective(
                scale=(0.05, 0.1),
                p=preset.perspective.prob,
            )
        )

    if preset.random_scale.enabled:
        scale_limit = preset.random_scale.params.get("scale_limit", 0.2)
        transforms.append(
            A.RandomScale(
                scale_limit=scale_limit,
                p=preset.random_scale.prob,
            )
        )

    if preset.random_crop.enabled:
        # Random crop with minimum size
        transforms.append(
            A.RandomCrop(
                height=int(img_size * 0.8),
                width=int(img_size * 0.8),
                p=preset.random_crop.prob,
            )
        )

    # ==== COLOR / LIGHT ====
    if preset.brightness_contrast.enabled:
        params = preset.brightness_contrast.params
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=params.get("brightness_limit", 0.2),
                contrast_limit=params.get("contrast_limit", 0.2),
                p=preset.brightness_contrast.prob,
            )
        )

    if preset.hue_saturation.enabled:
        params = preset.hue_saturation.params
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=params.get("hue_shift_limit", 20),
                sat_shift_limit=params.get("sat_shift_limit", 30),
                val_shift_limit=params.get("val_shift_limit", 20),
                p=preset.hue_saturation.prob,
            )
        )

    if preset.rgb_shift.enabled:
        params = preset.rgb_shift.params
        r_limit = params.get("r_shift_limit", 20)
        transforms.append(
            A.RGBShift(
                r_shift_limit=r_limit,
                g_shift_limit=r_limit,
                b_shift_limit=r_limit,
                p=preset.rgb_shift.prob,
            )
        )

    if preset.channel_shuffle.enabled:
        transforms.append(
            A.ChannelShuffle(p=preset.channel_shuffle.prob)
        )

    if preset.clahe.enabled:
        params = preset.clahe.params
        transforms.append(
            A.CLAHE(
                clip_limit=params.get("clip_limit", 4.0),
                p=preset.clahe.prob,
            )
        )

    if preset.equalize.enabled:
        transforms.append(
            A.Equalize(p=preset.equalize.prob)
        )

    if preset.to_gray.enabled:
        transforms.append(
            A.ToGray(p=preset.to_gray.prob)
        )

    # ==== BLUR / NOISE ====
    if preset.gaussian_blur.enabled:
        params = preset.gaussian_blur.params
        transforms.append(
            A.GaussianBlur(
                blur_limit=params.get("blur_limit", 7),
                p=preset.gaussian_blur.prob,
            )
        )

    if preset.motion_blur.enabled:
        params = preset.motion_blur.params
        transforms.append(
            A.MotionBlur(
                blur_limit=params.get("blur_limit", 7),
                p=preset.motion_blur.prob,
            )
        )

    if preset.median_blur.enabled:
        params = preset.median_blur.params
        transforms.append(
            A.MedianBlur(
                blur_limit=params.get("blur_limit", 7),
                p=preset.median_blur.prob,
            )
        )

    if preset.gaussian_noise.enabled:
        params = preset.gaussian_noise.params
        var_limit = params.get("var_limit", (10.0, 50.0))
        transforms.append(
            A.GaussNoise(
                var_limit=var_limit,
                p=preset.gaussian_noise.prob,
            )
        )

    if preset.iso_noise.enabled:
        transforms.append(
            A.ISONoise(p=preset.iso_noise.prob)
        )

    # ==== WEATHER / OCCLUSION ====
    if preset.random_rain.enabled:
        transforms.append(
            A.RandomRain(p=preset.random_rain.prob)
        )

    if preset.random_fog.enabled:
        transforms.append(
            A.RandomFog(p=preset.random_fog.prob)
        )

    if preset.random_shadow.enabled:
        transforms.append(
            A.RandomShadow(p=preset.random_shadow.prob)
        )

    if preset.coarse_dropout.enabled:
        params = preset.coarse_dropout.params
        transforms.append(
            A.CoarseDropout(
                max_holes=params.get("max_holes", 8),
                max_height=params.get("max_height", 32),
                max_width=params.get("max_width", 32),
                fill_value=114,  # Gray fill
                p=preset.coarse_dropout.prob,
            )
        )

    # ==== FINAL TRANSFORMS ====
    # Resize to target size (always applied)
    transforms.append(
        A.LongestMaxSize(max_size=img_size)
    )
    transforms.append(
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,  # BORDER_CONSTANT
            value=(114, 114, 114),  # Gray padding
        )
    )

    # Normalization (ImageNet stats)
    if include_normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    # Convert to PyTorch tensor
    if include_to_tensor:
        transforms.append(ToTensorV2())

    # Build Compose with bbox support
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # x1, y1, x2, y2
            label_fields=['labels'],
            min_visibility=min_visibility,
            min_area=min_area,
        ),
    )


def build_val_pipeline(
    img_size: int = 640,
    include_normalize: bool = True,
    include_to_tensor: bool = True,
) -> A.Compose:
    """
    Build validation/inference pipeline (no augmentations).

    Args:
        img_size: Target image size
        include_normalize: Whether to normalize
        include_to_tensor: Whether to convert to tensor

    Returns:
        Albumentations Compose for validation
    """
    transforms = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            value=(114, 114, 114),
        ),
    ]

    if include_normalize:
        transforms.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )

    if include_to_tensor:
        transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.0,  # Keep all boxes in validation
            min_area=0,
        ),
    )


class AlbumentationsTransform:
    """
    Wrapper class for easier usage with datasets.

    Provides a simple interface that matches common dataset expectations.
    """

    def __init__(
        self,
        preset: AugmentationPreset,
        img_size: int = 640,
        is_training: bool = True,
    ):
        """
        Initialize transform wrapper.

        Args:
            preset: AugmentationPreset configuration
            img_size: Target image size
            is_training: Whether this is for training (with augmentations)
        """
        self.img_size = img_size
        self.is_training = is_training

        if is_training:
            self.transform = build_albumentations_pipeline(
                preset,
                img_size=img_size,
                include_normalize=True,
                include_to_tensor=True,
            )
        else:
            self.transform = build_val_pipeline(
                img_size=img_size,
                include_normalize=True,
                include_to_tensor=True,
            )

    def __call__(
        self,
        image,
        boxes=None,
        labels=None,
    ):
        """
        Apply transforms.

        Args:
            image: numpy array [H, W, C] (BGR or RGB)
            boxes: list of [x1, y1, x2, y2] or numpy array [N, 4]
            labels: list of class ids or numpy array [N]

        Returns:
            Dict with 'image', 'boxes', 'labels'
        """
        import numpy as np

        # Handle empty boxes
        if boxes is None or len(boxes) == 0:
            boxes = []
            labels = []
        else:
            boxes = [list(b) for b in boxes]
            labels = list(labels) if labels is not None else [0] * len(boxes)

        # Apply transform
        result = self.transform(
            image=image,
            bboxes=boxes,
            labels=labels,
        )

        # Convert boxes back to numpy
        transformed_boxes = np.array(result['bboxes']) if result['bboxes'] else np.zeros((0, 4))
        transformed_labels = np.array(result['labels']) if result['labels'] else np.array([])

        return {
            'image': result['image'],
            'boxes': transformed_boxes,
            'labels': transformed_labels,
        }


def get_augmentation_list_from_preset(preset: AugmentationPreset) -> List[str]:
    """
    Get list of enabled augmentation names from preset.

    Useful for logging and debugging.

    Args:
        preset: AugmentationPreset

    Returns:
        List of enabled augmentation names
    """
    enabled = []

    # Check all attributes
    for attr_name in dir(preset):
        if attr_name.startswith('_'):
            continue
        attr = getattr(preset, attr_name)
        if isinstance(attr, AugmentationConfig) and attr.enabled:
            enabled.append(attr_name)

    return enabled
