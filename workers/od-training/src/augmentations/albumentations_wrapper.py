"""
Albumentations Wrapper for Object Detection.

Converts our AugmentationPreset config into Albumentations Compose pipeline.
All single-image augmentations use Albumentations with proper bbox support.

Key features:
- BboxParams with 'pascal_voc' format (x1, y1, x2, y2)
- Automatic filtering of invalid bboxes
- Integration with our preset system
- Support for 40+ augmentation types

Usage:
    from augmentations.albumentations_wrapper import build_albumentations_pipeline
    from augmentations.presets import get_preset

    preset = get_preset("sota-v2")
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
import cv2

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

    # ==========================================================================
    # GEOMETRIC AUGMENTATIONS
    # ==========================================================================

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
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.random_rotate.prob,
            )
        )

    if preset.safe_rotate.enabled:
        limit = preset.safe_rotate.params.get("limit", 15)
        transforms.append(
            A.SafeRotate(
                limit=limit,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.safe_rotate.prob,
            )
        )

    if preset.shift_scale_rotate.enabled:
        params = preset.shift_scale_rotate.params
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=params.get("shift_limit", 0.1),
                scale_limit=params.get("scale_limit", 0.2),
                rotate_limit=params.get("rotate_limit", 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.shift_scale_rotate.prob,
            )
        )

    if preset.affine.enabled:
        params = preset.affine.params
        transforms.append(
            A.Affine(
                scale=params.get("scale", (0.9, 1.1)),
                translate_percent=params.get("translate_percent", 0.1),
                shear=params.get("shear", 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.affine.prob,
            )
        )

    if preset.perspective.enabled:
        params = preset.perspective.params
        scale = params.get("scale", (0.02, 0.05))
        transforms.append(
            A.Perspective(
                scale=scale,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
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
        params = preset.random_crop.params
        scale = params.get("scale", (0.8, 1.0))
        transforms.append(
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=scale,
                ratio=params.get("ratio", (0.9, 1.1)),
                p=preset.random_crop.prob,
            )
        )

    if preset.grid_distortion.enabled:
        params = preset.grid_distortion.params
        transforms.append(
            A.GridDistortion(
                num_steps=params.get("num_steps", 5),
                distort_limit=params.get("distort_limit", 0.3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.grid_distortion.prob,
            )
        )

    if preset.elastic_transform.enabled:
        params = preset.elastic_transform.params
        transforms.append(
            A.ElasticTransform(
                alpha=params.get("alpha", 50),
                sigma=params.get("sigma", 5),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.elastic_transform.prob,
            )
        )

    if preset.optical_distortion.enabled:
        params = preset.optical_distortion.params
        transforms.append(
            A.OpticalDistortion(
                distort_limit=params.get("distort_limit", 0.3),
                shift_limit=params.get("shift_limit", 0.1),
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=preset.optical_distortion.prob,
            )
        )

    if preset.piecewise_affine.enabled:
        params = preset.piecewise_affine.params
        transforms.append(
            A.PiecewiseAffine(
                scale=params.get("scale", 0.03),
                nb_rows=params.get("nb_rows", 4),
                nb_cols=params.get("nb_cols", 4),
                p=preset.piecewise_affine.prob,
            )
        )

    # ==========================================================================
    # COLOR / LIGHT AUGMENTATIONS
    # ==========================================================================

    if preset.brightness_contrast.enabled:
        params = preset.brightness_contrast.params
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=params.get("brightness_limit", 0.2),
                contrast_limit=params.get("contrast_limit", 0.2),
                p=preset.brightness_contrast.prob,
            )
        )

    if preset.color_jitter.enabled:
        params = preset.color_jitter.params
        transforms.append(
            A.ColorJitter(
                brightness=params.get("brightness", 0.2),
                contrast=params.get("contrast", 0.2),
                saturation=params.get("saturation", 0.3),
                hue=params.get("hue", 0.015),
                p=preset.color_jitter.prob,
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

    if preset.random_gamma.enabled:
        params = preset.random_gamma.params
        gamma_limit = params.get("gamma_limit", (80, 120))
        transforms.append(
            A.RandomGamma(
                gamma_limit=gamma_limit,
                p=preset.random_gamma.prob,
            )
        )

    if preset.rgb_shift.enabled:
        params = preset.rgb_shift.params
        transforms.append(
            A.RGBShift(
                r_shift_limit=params.get("r_shift_limit", 20),
                g_shift_limit=params.get("g_shift_limit", 20),
                b_shift_limit=params.get("b_shift_limit", 20),
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
                tile_grid_size=(
                    params.get("tile_grid_size", 8),
                    params.get("tile_grid_size", 8),
                ),
                p=preset.clahe.prob,
            )
        )

    if preset.equalize.enabled:
        transforms.append(
            A.Equalize(p=preset.equalize.prob)
        )

    if preset.random_tone_curve.enabled:
        params = preset.random_tone_curve.params
        transforms.append(
            A.RandomToneCurve(
                scale=params.get("scale", 0.1),
                p=preset.random_tone_curve.prob,
            )
        )

    if preset.posterize.enabled:
        params = preset.posterize.params
        transforms.append(
            A.Posterize(
                num_bits=params.get("num_bits", 4),
                p=preset.posterize.prob,
            )
        )

    if preset.solarize.enabled:
        params = preset.solarize.params
        transforms.append(
            A.Solarize(
                threshold=params.get("threshold", 128),
                p=preset.solarize.prob,
            )
        )

    if preset.sharpen.enabled:
        params = preset.sharpen.params
        transforms.append(
            A.Sharpen(
                alpha=params.get("alpha", (0.2, 0.5)),
                lightness=params.get("lightness", (0.5, 1.0)),
                p=preset.sharpen.prob,
            )
        )

    if preset.unsharp_mask.enabled:
        params = preset.unsharp_mask.params
        transforms.append(
            A.UnsharpMask(
                blur_limit=params.get("blur_limit", (3, 7)),
                alpha=params.get("alpha", (0.2, 0.5)),
                p=preset.unsharp_mask.prob,
            )
        )

    if preset.fancy_pca.enabled:
        params = preset.fancy_pca.params
        transforms.append(
            A.FancyPCA(
                alpha=params.get("alpha", 0.1),
                p=preset.fancy_pca.prob,
            )
        )

    if preset.invert_img.enabled:
        transforms.append(
            A.InvertImg(p=preset.invert_img.prob)
        )

    if preset.to_gray.enabled:
        transforms.append(
            A.ToGray(p=preset.to_gray.prob)
        )

    # ==========================================================================
    # BLUR AUGMENTATIONS
    # ==========================================================================

    if preset.gaussian_blur.enabled:
        params = preset.gaussian_blur.params
        blur_limit = params.get("blur_limit", (3, 7))
        if isinstance(blur_limit, int):
            blur_limit = (3, blur_limit)
        transforms.append(
            A.GaussianBlur(
                blur_limit=blur_limit,
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
                blur_limit=params.get("blur_limit", 5),
                p=preset.median_blur.prob,
            )
        )

    if preset.defocus.enabled:
        params = preset.defocus.params
        transforms.append(
            A.Defocus(
                radius=params.get("radius", (3, 7)),
                alias_blur=params.get("alias_blur", (0.1, 0.3)),
                p=preset.defocus.prob,
            )
        )

    if preset.zoom_blur.enabled:
        params = preset.zoom_blur.params
        transforms.append(
            A.ZoomBlur(
                max_factor=params.get("max_factor", 1.1),
                p=preset.zoom_blur.prob,
            )
        )

    if preset.glass_blur.enabled:
        params = preset.glass_blur.params
        transforms.append(
            A.GlassBlur(
                sigma=params.get("sigma", 0.7),
                max_delta=params.get("max_delta", 4),
                iterations=params.get("iterations", 2),
                p=preset.glass_blur.prob,
            )
        )

    if preset.advanced_blur.enabled:
        params = preset.advanced_blur.params
        transforms.append(
            A.AdvancedBlur(
                blur_limit=params.get("blur_limit", (3, 7)),
                p=preset.advanced_blur.prob,
            )
        )

    # ==========================================================================
    # NOISE AUGMENTATIONS
    # ==========================================================================

    if preset.gaussian_noise.enabled:
        params = preset.gaussian_noise.params
        # Convert var_limit to std_range (sqrt of variance)
        var_limit = params.get("var_limit", (10.0, 50.0))
        if isinstance(var_limit, (list, tuple)):
            std_range = (var_limit[0] ** 0.5 / 255.0, var_limit[1] ** 0.5 / 255.0)
        else:
            std_range = (0.0, (var_limit ** 0.5) / 255.0)
        transforms.append(
            A.GaussNoise(
                std_range=std_range,
                p=preset.gaussian_noise.prob,
            )
        )

    if preset.iso_noise.enabled:
        params = preset.iso_noise.params
        transforms.append(
            A.ISONoise(
                color_shift=params.get("color_shift", (0.01, 0.05)),
                intensity=params.get("intensity", (0.1, 0.5)),
                p=preset.iso_noise.prob,
            )
        )

    if preset.multiplicative_noise.enabled:
        params = preset.multiplicative_noise.params
        transforms.append(
            A.MultiplicativeNoise(
                multiplier=params.get("multiplier", (0.9, 1.1)),
                p=preset.multiplicative_noise.prob,
            )
        )

    # ==========================================================================
    # QUALITY DEGRADATION AUGMENTATIONS
    # ==========================================================================

    if preset.image_compression.enabled:
        params = preset.image_compression.params
        quality_lower = params.get("quality_lower", 70)
        quality_upper = params.get("quality_upper", 95)
        transforms.append(
            A.ImageCompression(
                quality_range=(quality_lower, quality_upper),
                p=preset.image_compression.prob,
            )
        )

    if preset.downscale.enabled:
        params = preset.downscale.params
        scale_min = params.get("scale_min", 0.5)
        scale_max = params.get("scale_max", 0.9)
        transforms.append(
            A.Downscale(
                scale_range=(scale_min, scale_max),
                p=preset.downscale.prob,
            )
        )

    # ==========================================================================
    # DROPOUT / OCCLUSION AUGMENTATIONS
    # ==========================================================================

    if preset.coarse_dropout.enabled:
        params = preset.coarse_dropout.params
        max_holes = params.get("max_holes", 8)
        max_height = params.get("max_height", 32)
        max_width = params.get("max_width", 32)
        fill_val = params.get("fill_value", 114)
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(8, max_height),
                hole_width_range=(8, max_width),
                fill=(fill_val, fill_val, fill_val) if isinstance(fill_val, int) else fill_val,
                p=preset.coarse_dropout.prob,
            )
        )

    if preset.grid_dropout.enabled:
        params = preset.grid_dropout.params
        unit_min = params.get("unit_size_min", 10)
        unit_max = params.get("unit_size_max", 40)
        transforms.append(
            A.GridDropout(
                ratio=params.get("ratio", 0.3),
                unit_size_range=(unit_min, unit_max),
                fill=(114, 114, 114),
                p=preset.grid_dropout.prob,
            )
        )

    if preset.pixel_dropout.enabled:
        params = preset.pixel_dropout.params
        transforms.append(
            A.PixelDropout(
                dropout_prob=params.get("dropout_prob", 0.01),
                p=preset.pixel_dropout.prob,
            )
        )

    # Note: mask_dropout is for instance segmentation, skipped for bbox-only

    # ==========================================================================
    # WEATHER AUGMENTATIONS
    # ==========================================================================

    if preset.random_rain.enabled:
        params = preset.random_rain.params
        slant_lower = params.get("slant_lower", -10)
        slant_upper = params.get("slant_upper", 10)
        transforms.append(
            A.RandomRain(
                slant_range=(slant_lower, slant_upper),
                drop_length=params.get("drop_length", 20),
                drop_width=params.get("drop_width", 1),
                blur_value=params.get("blur_value", 5),
                p=preset.random_rain.prob,
            )
        )

    if preset.random_fog.enabled:
        params = preset.random_fog.params
        fog_lower = params.get("fog_coef_lower", 0.1)
        fog_upper = params.get("fog_coef_upper", 0.3)
        transforms.append(
            A.RandomFog(
                fog_coef_range=(fog_lower, fog_upper),
                alpha_coef=params.get("alpha_coef", 0.08),
                p=preset.random_fog.prob,
            )
        )

    if preset.random_shadow.enabled:
        params = preset.random_shadow.params
        num_lower = params.get("num_shadows_lower", 1)
        num_upper = params.get("num_shadows_upper", 2)
        transforms.append(
            A.RandomShadow(
                num_shadows_limit=(num_lower, num_upper),
                p=preset.random_shadow.prob,
            )
        )

    if preset.random_sun_flare.enabled:
        params = preset.random_sun_flare.params
        num_lower = params.get("num_flare_circles_lower", 3)
        num_upper = params.get("num_flare_circles_upper", 7)
        transforms.append(
            A.RandomSunFlare(
                src_radius=params.get("src_radius", 100),
                num_flare_circles_range=(num_lower, num_upper),
                p=preset.random_sun_flare.prob,
            )
        )

    if preset.random_snow.enabled:
        params = preset.random_snow.params
        transforms.append(
            A.RandomSnow(
                snow_point_lower=params.get("snow_point_lower", 0.1),
                snow_point_upper=params.get("snow_point_upper", 0.3),
                brightness_coeff=params.get("brightness_coeff", 2.5),
                p=preset.random_snow.prob,
            )
        )

    if preset.spatter.enabled:
        params = preset.spatter.params
        transforms.append(
            A.Spatter(
                mode=params.get("mode", "rain"),
                p=preset.spatter.prob,
            )
        )

    # Note: plasma_brightness is not in standard albumentations, skipped

    # ==========================================================================
    # FINAL TRANSFORMS
    # ==========================================================================

    # Resize to target size (always applied)
    transforms.append(
        A.LongestMaxSize(max_size=img_size)
    )
    transforms.append(
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),  # Gray padding
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
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
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
        self.preset = preset

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

    def get_enabled_augmentations(self) -> List[str]:
        """Get list of enabled augmentation names."""
        return self.preset.get_enabled_augmentations() if self.is_training else []


def get_augmentation_list_from_preset(preset: AugmentationPreset) -> List[str]:
    """
    Get list of enabled augmentation names from preset.

    Useful for logging and debugging.

    Args:
        preset: AugmentationPreset

    Returns:
        List of enabled augmentation names
    """
    return preset.get_enabled_augmentations()


def build_augmentation_oneof_blocks(preset: AugmentationPreset) -> List[A.OneOf]:
    """
    Build grouped OneOf blocks for augmentations.

    This allows mutually exclusive augmentations within categories.
    Useful for advanced pipelines that want to limit augmentation overlap.

    Args:
        preset: AugmentationPreset

    Returns:
        List of OneOf blocks
    """
    blocks = []

    # Blur OneOf block
    blur_transforms = []
    if preset.gaussian_blur.enabled:
        blur_transforms.append(A.GaussianBlur(blur_limit=7))
    if preset.motion_blur.enabled:
        blur_transforms.append(A.MotionBlur(blur_limit=7))
    if preset.median_blur.enabled:
        blur_transforms.append(A.MedianBlur(blur_limit=5))
    if preset.defocus.enabled:
        blur_transforms.append(A.Defocus())
    if blur_transforms:
        blocks.append(A.OneOf(blur_transforms, p=0.3))

    # Color OneOf block
    color_transforms = []
    if preset.brightness_contrast.enabled:
        color_transforms.append(A.RandomBrightnessContrast())
    if preset.hue_saturation.enabled:
        color_transforms.append(A.HueSaturationValue())
    if preset.rgb_shift.enabled:
        color_transforms.append(A.RGBShift())
    if preset.random_gamma.enabled:
        color_transforms.append(A.RandomGamma())
    if color_transforms:
        blocks.append(A.OneOf(color_transforms, p=0.5))

    return blocks
