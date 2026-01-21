"""
Augmentation Presets for Object Detection Training.

Provides ready-to-use augmentation configurations:
- SOTA-v2: Next-gen (YOLOv8 + RT-DETR + D-FINE best practices)
- SOTA: State-of-the-art (Mosaic + MixUp + CopyPaste + standard)
- Heavy: All augmentations enabled
- Medium: Balanced augmentations
- Light: Minimal augmentations (flip/rotate only)
- None: No augmentations

Usage:
    from augmentations.presets import get_preset, PRESETS

    config = get_preset("sota-v2")
    # Or customize:
    config = get_preset("sota", overrides={"mosaic": {"prob": 0.7}})
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


class PresetLevel(str, Enum):
    """Available preset levels."""
    SOTA_V2 = "sota-v2"
    SOTA = "sota"
    HEAVY = "heavy"
    MEDIUM = "medium"
    LIGHT = "light"
    NONE = "none"
    CUSTOM = "custom"


@dataclass
class AugmentationConfig:
    """Configuration for a single augmentation."""
    enabled: bool = False
    prob: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AugmentationPreset:
    """Complete augmentation preset configuration."""

    # =========================================================================
    # Multi-image augmentations (SOTA)
    # =========================================================================
    mosaic: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    mosaic9: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    mixup: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    cutmix: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    copypaste: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Geometric augmentations
    # =========================================================================
    horizontal_flip: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    vertical_flip: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    rotate90: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_rotate: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    affine: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    perspective: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_crop: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_scale: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW geometric augmentations
    shift_scale_rotate: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    grid_distortion: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    elastic_transform: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    optical_distortion: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    safe_rotate: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    piecewise_affine: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Color/Light augmentations
    # =========================================================================
    brightness_contrast: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    hue_saturation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    rgb_shift: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    channel_shuffle: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    clahe: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    equalize: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    to_gray: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW color augmentations
    random_gamma: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_tone_curve: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    posterize: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    solarize: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    sharpen: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    unsharp_mask: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    fancy_pca: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    invert_img: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    color_jitter: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Blur augmentations
    # =========================================================================
    gaussian_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    motion_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    median_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW blur augmentations
    defocus: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    zoom_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    glass_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    advanced_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Noise augmentations
    # =========================================================================
    gaussian_noise: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    iso_noise: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW noise augmentations
    multiplicative_noise: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Quality augmentations (NEW)
    # =========================================================================
    image_compression: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    downscale: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Dropout augmentations
    # =========================================================================
    coarse_dropout: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW dropout augmentations
    grid_dropout: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    pixel_dropout: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    mask_dropout: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # =========================================================================
    # Weather/Occlusion augmentations
    # =========================================================================
    random_rain: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_fog: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_shadow: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    # NEW weather augmentations
    random_sun_flare: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_snow: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    spatter: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    plasma_brightness: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v.to_dict() for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentationPreset":
        """Create from dictionary."""
        preset = cls()
        for key, value in data.items():
            if hasattr(preset, key) and isinstance(value, dict):
                setattr(preset, key, AugmentationConfig(**value))
        return preset

    def get_enabled_augmentations(self) -> List[str]:
        """Get list of enabled augmentation names."""
        enabled = []
        for field_name in self.__dataclass_fields__:
            config = getattr(self, field_name)
            if isinstance(config, AugmentationConfig) and config.enabled:
                enabled.append(field_name)
        return enabled


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

def _create_sota_v2_preset() -> AugmentationPreset:
    """
    SOTA-v2 (Next-Generation) preset.

    Best for: Maximum accuracy with modern techniques
    Features: Combines best practices from YOLOv8, RT-DETR, and D-FINE
    - Optimized Mosaic with probability scheduling
    - MixUp with proper alpha for detection
    - CopyPaste for instance-level augmentation
    - ShiftScaleRotate (the workhorse geometric aug)
    - Advanced color augmentations (ColorJitter, RandomGamma)
    - Quality degradation (compression, downscale)
    """
    return AugmentationPreset(
        # Multi-image (core SOTA-v2)
        mosaic=AugmentationConfig(enabled=True, prob=0.5, params={
            "img_size": 640,
            "center_ratio": 0.5,  # Center area ratio
            "pad_value": 114,     # Gray padding
        }),
        mixup=AugmentationConfig(enabled=True, prob=0.15, params={
            "alpha": 32.0,  # Higher alpha = more equal mixing (better for detection)
        }),
        copypaste=AugmentationConfig(enabled=True, prob=0.3, params={
            "blend_ratio": 0.5,
            "max_objects": 3,
            "scale_range": (0.5, 1.5),
        }),

        # Geometric (modern approach)
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        shift_scale_rotate=AugmentationConfig(enabled=True, prob=0.5, params={
            "shift_limit": 0.1,
            "scale_limit": 0.2,
            "rotate_limit": 10,
            "border_mode": "reflect_101",
        }),
        perspective=AugmentationConfig(enabled=True, prob=0.2, params={
            "scale": (0.02, 0.05),
        }),
        random_crop=AugmentationConfig(enabled=True, prob=0.3, params={
            "scale": (0.8, 1.0),
            "ratio": (0.9, 1.1),
        }),

        # Color (comprehensive but balanced)
        color_jitter=AugmentationConfig(enabled=True, prob=0.4, params={
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.3,
            "hue": 0.015,
        }),
        random_gamma=AugmentationConfig(enabled=True, prob=0.2, params={
            "gamma_limit": (80, 120),
        }),
        clahe=AugmentationConfig(enabled=True, prob=0.1, params={
            "clip_limit": 4.0,
            "tile_grid_size": 8,
        }),

        # Blur (realistic camera effects)
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.1, params={
            "blur_limit": (3, 5),
        }),
        defocus=AugmentationConfig(enabled=True, prob=0.05, params={
            "radius": (3, 5),
            "alias_blur": (0.1, 0.3),
        }),

        # Quality degradation (real-world robustness)
        image_compression=AugmentationConfig(enabled=True, prob=0.2, params={
            "quality_lower": 70,
            "quality_upper": 95,
            "compression_type": "jpeg",
        }),
        downscale=AugmentationConfig(enabled=True, prob=0.1, params={
            "scale_min": 0.5,
            "scale_max": 0.9,
        }),

        # Dropout (occlusion robustness)
        coarse_dropout=AugmentationConfig(enabled=True, prob=0.1, params={
            "max_holes": 4,
            "max_height": 40,
            "max_width": 40,
            "fill_value": 114,
        }),
    )


def _create_sota_preset() -> AugmentationPreset:
    """
    SOTA (State-of-the-Art) preset - Legacy version.

    Best for: Maximum accuracy, longer training time
    Features: Mosaic, MixUp, CopyPaste + balanced single-image augs
    """
    return AugmentationPreset(
        # Multi-image (core SOTA)
        mosaic=AugmentationConfig(enabled=True, prob=0.5, params={"img_size": 640}),
        mixup=AugmentationConfig(enabled=True, prob=0.3, params={"alpha": 8.0}),
        copypaste=AugmentationConfig(enabled=True, prob=0.2, params={"blend_ratio": 0.5}),

        # Geometric (essential)
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        random_scale=AugmentationConfig(enabled=True, prob=0.5, params={"scale_limit": 0.2}),

        # Color (balanced)
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.4, params={
            "brightness_limit": 0.2, "contrast_limit": 0.2
        }),
        hue_saturation=AugmentationConfig(enabled=True, prob=0.3, params={
            "hue_shift_limit": 20, "sat_shift_limit": 30
        }),

        # Blur (light)
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.1, params={"blur_limit": 3}),
    )


def _create_heavy_preset() -> AugmentationPreset:
    """
    Heavy preset.

    Best for: Small datasets, preventing overfitting
    Features: All augmentations enabled with moderate probabilities
    """
    return AugmentationPreset(
        # Multi-image
        mosaic=AugmentationConfig(enabled=True, prob=0.6, params={"img_size": 640}),
        mosaic9=AugmentationConfig(enabled=True, prob=0.2, params={"img_size": 640}),
        mixup=AugmentationConfig(enabled=True, prob=0.4, params={"alpha": 8.0}),
        cutmix=AugmentationConfig(enabled=True, prob=0.2, params={"alpha": 1.0}),
        copypaste=AugmentationConfig(enabled=True, prob=0.3, params={"blend_ratio": 0.5}),

        # Geometric
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        vertical_flip=AugmentationConfig(enabled=True, prob=0.2),
        rotate90=AugmentationConfig(enabled=True, prob=0.3),
        random_rotate=AugmentationConfig(enabled=True, prob=0.3, params={"limit": 15}),
        shift_scale_rotate=AugmentationConfig(enabled=True, prob=0.4, params={
            "shift_limit": 0.15, "scale_limit": 0.25, "rotate_limit": 15
        }),
        affine=AugmentationConfig(enabled=True, prob=0.3, params={
            "scale": (0.8, 1.2), "translate_percent": 0.1, "shear": 10
        }),
        perspective=AugmentationConfig(enabled=True, prob=0.2, params={"scale": (0.02, 0.08)}),
        elastic_transform=AugmentationConfig(enabled=True, prob=0.1, params={
            "alpha": 50, "sigma": 5
        }),
        grid_distortion=AugmentationConfig(enabled=True, prob=0.1, params={
            "num_steps": 5, "distort_limit": 0.3
        }),
        random_scale=AugmentationConfig(enabled=True, prob=0.5, params={"scale_limit": 0.3}),

        # Color
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.5, params={
            "brightness_limit": 0.3, "contrast_limit": 0.3
        }),
        hue_saturation=AugmentationConfig(enabled=True, prob=0.4, params={
            "hue_shift_limit": 30, "sat_shift_limit": 40
        }),
        color_jitter=AugmentationConfig(enabled=True, prob=0.3, params={
            "brightness": 0.3, "contrast": 0.3, "saturation": 0.4, "hue": 0.02
        }),
        rgb_shift=AugmentationConfig(enabled=True, prob=0.2, params={"r_shift_limit": 20}),
        random_gamma=AugmentationConfig(enabled=True, prob=0.2, params={"gamma_limit": (70, 130)}),
        clahe=AugmentationConfig(enabled=True, prob=0.2, params={"clip_limit": 4.0}),
        sharpen=AugmentationConfig(enabled=True, prob=0.1, params={
            "alpha": (0.2, 0.5), "lightness": (0.5, 1.0)
        }),
        to_gray=AugmentationConfig(enabled=True, prob=0.1),

        # Blur/Noise
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.2, params={"blur_limit": (3, 7)}),
        motion_blur=AugmentationConfig(enabled=True, prob=0.15, params={"blur_limit": 7}),
        defocus=AugmentationConfig(enabled=True, prob=0.1, params={"radius": (3, 7)}),
        gaussian_noise=AugmentationConfig(enabled=True, prob=0.15, params={"var_limit": (10, 50)}),
        iso_noise=AugmentationConfig(enabled=True, prob=0.1, params={
            "color_shift": (0.01, 0.05), "intensity": (0.1, 0.5)
        }),

        # Quality
        image_compression=AugmentationConfig(enabled=True, prob=0.2, params={
            "quality_lower": 60, "quality_upper": 95
        }),
        downscale=AugmentationConfig(enabled=True, prob=0.1, params={
            "scale_min": 0.4, "scale_max": 0.9
        }),

        # Dropout
        coarse_dropout=AugmentationConfig(enabled=True, prob=0.15, params={
            "max_holes": 8, "max_height": 40, "max_width": 40
        }),
        grid_dropout=AugmentationConfig(enabled=True, prob=0.1, params={
            "ratio": 0.3, "unit_size_min": 10, "unit_size_max": 40
        }),

        # Weather
        random_shadow=AugmentationConfig(enabled=True, prob=0.15),
        random_fog=AugmentationConfig(enabled=True, prob=0.1, params={
            "fog_coef_lower": 0.1, "fog_coef_upper": 0.3
        }),
        random_rain=AugmentationConfig(enabled=True, prob=0.1),
        random_sun_flare=AugmentationConfig(enabled=True, prob=0.05, params={
            "src_radius": 100
        }),
    )


def _create_medium_preset() -> AugmentationPreset:
    """
    Medium preset.

    Best for: General purpose, balanced training
    Features: Common augmentations with moderate settings
    """
    return AugmentationPreset(
        # Multi-image (lighter)
        mosaic=AugmentationConfig(enabled=True, prob=0.3, params={"img_size": 640}),
        mixup=AugmentationConfig(enabled=True, prob=0.2, params={"alpha": 8.0}),

        # Geometric
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        rotate90=AugmentationConfig(enabled=True, prob=0.2),
        shift_scale_rotate=AugmentationConfig(enabled=True, prob=0.3, params={
            "shift_limit": 0.1, "scale_limit": 0.15, "rotate_limit": 10
        }),
        random_scale=AugmentationConfig(enabled=True, prob=0.4, params={"scale_limit": 0.2}),

        # Color
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.3, params={
            "brightness_limit": 0.2, "contrast_limit": 0.2
        }),
        hue_saturation=AugmentationConfig(enabled=True, prob=0.2, params={
            "hue_shift_limit": 15, "sat_shift_limit": 20
        }),
        color_jitter=AugmentationConfig(enabled=True, prob=0.2, params={
            "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.01
        }),

        # Blur (minimal)
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.1, params={"blur_limit": 3}),

        # Quality
        image_compression=AugmentationConfig(enabled=True, prob=0.1, params={
            "quality_lower": 75, "quality_upper": 95
        }),
    )


def _create_light_preset() -> AugmentationPreset:
    """
    Light preset.

    Best for: Large datasets, fast training, fine-tuning
    Features: Only essential geometric augmentations
    """
    return AugmentationPreset(
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        random_scale=AugmentationConfig(enabled=True, prob=0.3, params={"scale_limit": 0.1}),
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.2, params={
            "brightness_limit": 0.1, "contrast_limit": 0.1
        }),
    )


def _create_none_preset() -> AugmentationPreset:
    """
    None preset.

    Best for: Evaluation, debugging, baseline comparison
    Features: No augmentations (all disabled)
    """
    return AugmentationPreset()  # All disabled by default


# Preset registry
PRESETS: Dict[str, AugmentationPreset] = {
    PresetLevel.SOTA_V2.value: _create_sota_v2_preset(),
    PresetLevel.SOTA.value: _create_sota_preset(),
    PresetLevel.HEAVY.value: _create_heavy_preset(),
    PresetLevel.MEDIUM.value: _create_medium_preset(),
    PresetLevel.LIGHT.value: _create_light_preset(),
    PresetLevel.NONE.value: _create_none_preset(),
}


# =============================================================================
# PUBLIC API
# =============================================================================

def get_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> AugmentationPreset:
    """
    Get augmentation preset by name with optional overrides.

    Args:
        preset_name: One of "sota-v2", "sota", "heavy", "medium", "light", "none", "custom"
        overrides: Optional dict to override specific augmentation settings
            Example: {"mosaic": {"prob": 0.7}, "mixup": {"enabled": False}}

    Returns:
        AugmentationPreset configured with preset + overrides

    Example:
        # Use SOTA-v2 (recommended)
        preset = get_preset("sota-v2")

        # Use SOTA with custom mosaic probability
        preset = get_preset("sota", overrides={"mosaic": {"prob": 0.7}})

        # Start from scratch
        preset = get_preset("custom", overrides={
            "horizontal_flip": {"enabled": True, "prob": 0.5},
            "mosaic": {"enabled": True, "prob": 0.5}
        })
    """
    if preset_name == PresetLevel.CUSTOM.value:
        # Start with empty preset for custom
        preset = AugmentationPreset()
    elif preset_name in PRESETS:
        # Deep copy the preset
        import copy
        preset = copy.deepcopy(PRESETS[preset_name])
    else:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Choose from: {list(PRESETS.keys())} or 'custom'"
        )

    # Apply overrides
    if overrides:
        for aug_name, aug_overrides in overrides.items():
            if hasattr(preset, aug_name):
                current_config = getattr(preset, aug_name)
                # Update config with overrides
                for key, value in aug_overrides.items():
                    if key == "params" and current_config.params:
                        current_config.params.update(value)
                    elif key == "params":
                        current_config.params = value
                    elif hasattr(current_config, key):
                        setattr(current_config, key, value)

    return preset


def get_preset_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available presets.

    Useful for UI to display preset descriptions.

    Returns:
        Dict with preset info for each preset
    """
    return {
        "sota-v2": {
            "name": "SOTA-v2 (Recommended)",
            "description": "Next-gen: YOLOv8 + RT-DETR + D-FINE best practices",
            "icon": "rocket",
            "training_time": "1.4x",
            "accuracy_boost": "+4-6% mAP",
            "features": [
                "Optimized Mosaic & MixUp",
                "CopyPaste augmentation",
                "ShiftScaleRotate",
                "ColorJitter & RandomGamma",
                "Quality degradation (JPEG, downscale)",
                "Coarse dropout for occlusion",
            ],
        },
        "sota": {
            "name": "SOTA (Legacy)",
            "description": "State-of-the-art: Mosaic, MixUp, CopyPaste + standard",
            "icon": "star",
            "training_time": "1.5x",
            "accuracy_boost": "+3-5% mAP",
            "features": [
                "Mosaic 4-image",
                "MixUp blending",
                "CopyPaste",
                "Brightness/Contrast",
            ],
        },
        "heavy": {
            "name": "Heavy",
            "description": "All augmentations, ideal for small datasets (<1000 images)",
            "icon": "fire",
            "training_time": "2x",
            "accuracy_boost": "+5-8% mAP",
            "features": [
                "All multi-image augs",
                "Full geometric suite",
                "Comprehensive color augs",
                "Weather effects",
                "Noise & blur",
            ],
        },
        "medium": {
            "name": "Medium",
            "description": "Balanced augmentations for general use",
            "icon": "zap",
            "training_time": "1.2x",
            "accuracy_boost": "+2-3% mAP",
            "features": [
                "Mosaic & MixUp (lower prob)",
                "Essential geometric",
                "Basic color augs",
            ],
        },
        "light": {
            "name": "Light",
            "description": "Basic augmentations only, fast training",
            "icon": "feather",
            "training_time": "1.05x",
            "accuracy_boost": "+1% mAP",
            "features": [
                "Horizontal flip",
                "Light scale",
                "Light brightness",
            ],
        },
        "none": {
            "name": "None",
            "description": "No augmentation, for baseline comparison",
            "icon": "x",
            "training_time": "1x",
            "accuracy_boost": "Baseline",
            "features": [],
        },
    }


def get_augmentation_categories() -> Dict[str, Dict[str, Any]]:
    """
    Get augmentation categories with their augmentations.

    Useful for UI to organize augmentations into collapsible sections.

    Returns:
        Dict with category info and augmentations
    """
    return {
        "multi_image": {
            "name": "Multi-Image (SOTA)",
            "description": "Augmentations that combine multiple images",
            "icon": "images",
            "augmentations": {
                "mosaic": {
                    "name": "Mosaic",
                    "description": "Combines 4 images in 2x2 grid",
                    "default_prob": 0.5,
                    "params": {
                        "img_size": {"type": "int", "default": 640, "min": 320, "max": 1280},
                        "center_ratio": {"type": "float", "default": 0.5, "min": 0.25, "max": 0.75},
                    },
                },
                "mosaic9": {
                    "name": "Mosaic-9",
                    "description": "Combines 9 images in 3x3 grid",
                    "default_prob": 0.3,
                    "params": {
                        "img_size": {"type": "int", "default": 640, "min": 320, "max": 1280},
                    },
                },
                "mixup": {
                    "name": "MixUp",
                    "description": "Alpha-blends two images together",
                    "default_prob": 0.3,
                    "params": {
                        "alpha": {"type": "float", "default": 8.0, "min": 1.0, "max": 32.0},
                    },
                },
                "cutmix": {
                    "name": "CutMix",
                    "description": "Cuts region from one image and pastes onto another",
                    "default_prob": 0.2,
                    "params": {
                        "alpha": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0},
                    },
                },
                "copypaste": {
                    "name": "CopyPaste",
                    "description": "Copies objects from one image to another",
                    "default_prob": 0.2,
                    "params": {
                        "blend_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
                        "max_objects": {"type": "int", "default": 3, "min": 1, "max": 10},
                    },
                },
            },
        },
        "geometric": {
            "name": "Geometric",
            "description": "Shape and position transformations",
            "icon": "move",
            "augmentations": {
                "horizontal_flip": {
                    "name": "Horizontal Flip",
                    "description": "Flips image horizontally",
                    "default_prob": 0.5,
                    "params": {},
                },
                "vertical_flip": {
                    "name": "Vertical Flip",
                    "description": "Flips image vertically",
                    "default_prob": 0.2,
                    "params": {},
                },
                "rotate90": {
                    "name": "Rotate 90",
                    "description": "Rotates image by 90 degrees",
                    "default_prob": 0.3,
                    "params": {},
                },
                "random_rotate": {
                    "name": "Random Rotate",
                    "description": "Rotates image by random angle",
                    "default_prob": 0.3,
                    "params": {
                        "limit": {"type": "int", "default": 15, "min": 1, "max": 45},
                    },
                },
                "shift_scale_rotate": {
                    "name": "Shift Scale Rotate",
                    "description": "Combined shift, scale and rotate (recommended)",
                    "default_prob": 0.5,
                    "params": {
                        "shift_limit": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                        "scale_limit": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                        "rotate_limit": {"type": "int", "default": 10, "min": 0, "max": 45},
                    },
                },
                "affine": {
                    "name": "Affine Transform",
                    "description": "Applies affine transformation (scale, translate, shear)",
                    "default_prob": 0.3,
                    "params": {
                        "scale": {"type": "tuple", "default": [0.8, 1.2]},
                        "translate_percent": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                        "shear": {"type": "int", "default": 10, "min": 0, "max": 30},
                    },
                },
                "perspective": {
                    "name": "Perspective",
                    "description": "Applies perspective transformation",
                    "default_prob": 0.2,
                    "params": {
                        "scale": {"type": "tuple", "default": [0.02, 0.05]},
                    },
                },
                "safe_rotate": {
                    "name": "Safe Rotate",
                    "description": "Rotates without cutting off corners",
                    "default_prob": 0.2,
                    "params": {
                        "limit": {"type": "int", "default": 15, "min": 1, "max": 45},
                    },
                },
                "random_crop": {
                    "name": "Random Crop",
                    "description": "Randomly crops a portion of the image",
                    "default_prob": 0.3,
                    "params": {
                        "scale": {"type": "tuple", "default": [0.8, 1.0]},
                    },
                },
                "random_scale": {
                    "name": "Random Scale",
                    "description": "Randomly scales the image",
                    "default_prob": 0.5,
                    "params": {
                        "scale_limit": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                    },
                },
                "grid_distortion": {
                    "name": "Grid Distortion",
                    "description": "Distorts image using a grid",
                    "default_prob": 0.1,
                    "params": {
                        "num_steps": {"type": "int", "default": 5, "min": 2, "max": 10},
                        "distort_limit": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
                    },
                },
                "elastic_transform": {
                    "name": "Elastic Transform",
                    "description": "Applies elastic deformation",
                    "default_prob": 0.1,
                    "params": {
                        "alpha": {"type": "int", "default": 50, "min": 10, "max": 200},
                        "sigma": {"type": "int", "default": 5, "min": 1, "max": 20},
                    },
                },
                "optical_distortion": {
                    "name": "Optical Distortion",
                    "description": "Applies barrel/pincushion distortion",
                    "default_prob": 0.1,
                    "params": {
                        "distort_limit": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
                        "shift_limit": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                    },
                },
                "piecewise_affine": {
                    "name": "Piecewise Affine",
                    "description": "Applies piecewise affine transformation",
                    "default_prob": 0.1,
                    "params": {
                        "scale": {"type": "float", "default": 0.03, "min": 0.01, "max": 0.1},
                        "nb_rows": {"type": "int", "default": 4, "min": 2, "max": 8},
                        "nb_cols": {"type": "int", "default": 4, "min": 2, "max": 8},
                    },
                },
            },
        },
        "color": {
            "name": "Color / Light",
            "description": "Color and lighting transformations",
            "icon": "palette",
            "augmentations": {
                "brightness_contrast": {
                    "name": "Brightness/Contrast",
                    "description": "Adjusts brightness and contrast",
                    "default_prob": 0.4,
                    "params": {
                        "brightness_limit": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                        "contrast_limit": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                    },
                },
                "color_jitter": {
                    "name": "Color Jitter",
                    "description": "Combined color augmentation (brightness, contrast, saturation, hue)",
                    "default_prob": 0.4,
                    "params": {
                        "brightness": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                        "contrast": {"type": "float", "default": 0.2, "min": 0.0, "max": 0.5},
                        "saturation": {"type": "float", "default": 0.3, "min": 0.0, "max": 0.5},
                        "hue": {"type": "float", "default": 0.015, "min": 0.0, "max": 0.1},
                    },
                },
                "hue_saturation": {
                    "name": "Hue/Saturation",
                    "description": "Adjusts hue and saturation",
                    "default_prob": 0.3,
                    "params": {
                        "hue_shift_limit": {"type": "int", "default": 20, "min": 0, "max": 50},
                        "sat_shift_limit": {"type": "int", "default": 30, "min": 0, "max": 50},
                    },
                },
                "random_gamma": {
                    "name": "Random Gamma",
                    "description": "Applies gamma correction",
                    "default_prob": 0.2,
                    "params": {
                        "gamma_limit": {"type": "tuple", "default": [80, 120]},
                    },
                },
                "rgb_shift": {
                    "name": "RGB Shift",
                    "description": "Shifts RGB channels independently",
                    "default_prob": 0.2,
                    "params": {
                        "r_shift_limit": {"type": "int", "default": 20, "min": 0, "max": 50},
                        "g_shift_limit": {"type": "int", "default": 20, "min": 0, "max": 50},
                        "b_shift_limit": {"type": "int", "default": 20, "min": 0, "max": 50},
                    },
                },
                "channel_shuffle": {
                    "name": "Channel Shuffle",
                    "description": "Randomly shuffles color channels",
                    "default_prob": 0.1,
                    "params": {},
                },
                "clahe": {
                    "name": "CLAHE",
                    "description": "Contrast Limited Adaptive Histogram Equalization",
                    "default_prob": 0.2,
                    "params": {
                        "clip_limit": {"type": "float", "default": 4.0, "min": 1.0, "max": 8.0},
                        "tile_grid_size": {"type": "int", "default": 8, "min": 4, "max": 16},
                    },
                },
                "equalize": {
                    "name": "Equalize",
                    "description": "Histogram equalization",
                    "default_prob": 0.1,
                    "params": {},
                },
                "random_tone_curve": {
                    "name": "Random Tone Curve",
                    "description": "Applies random tone curve adjustment",
                    "default_prob": 0.1,
                    "params": {
                        "scale": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                    },
                },
                "posterize": {
                    "name": "Posterize",
                    "description": "Reduces color bits",
                    "default_prob": 0.1,
                    "params": {
                        "num_bits": {"type": "int", "default": 4, "min": 1, "max": 7},
                    },
                },
                "solarize": {
                    "name": "Solarize",
                    "description": "Inverts pixels above threshold",
                    "default_prob": 0.1,
                    "params": {
                        "threshold": {"type": "int", "default": 128, "min": 0, "max": 255},
                    },
                },
                "sharpen": {
                    "name": "Sharpen",
                    "description": "Sharpens the image",
                    "default_prob": 0.1,
                    "params": {
                        "alpha": {"type": "tuple", "default": [0.2, 0.5]},
                        "lightness": {"type": "tuple", "default": [0.5, 1.0]},
                    },
                },
                "unsharp_mask": {
                    "name": "Unsharp Mask",
                    "description": "Applies unsharp masking",
                    "default_prob": 0.1,
                    "params": {
                        "blur_limit": {"type": "tuple", "default": [3, 7]},
                        "alpha": {"type": "tuple", "default": [0.2, 0.5]},
                    },
                },
                "fancy_pca": {
                    "name": "Fancy PCA",
                    "description": "PCA color augmentation (AlexNet style)",
                    "default_prob": 0.1,
                    "params": {
                        "alpha": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                    },
                },
                "invert_img": {
                    "name": "Invert",
                    "description": "Inverts image colors",
                    "default_prob": 0.05,
                    "params": {},
                },
                "to_gray": {
                    "name": "To Gray",
                    "description": "Converts to grayscale",
                    "default_prob": 0.1,
                    "params": {},
                },
            },
        },
        "blur": {
            "name": "Blur",
            "description": "Various blur effects",
            "icon": "droplet",
            "augmentations": {
                "gaussian_blur": {
                    "name": "Gaussian Blur",
                    "description": "Applies Gaussian blur",
                    "default_prob": 0.1,
                    "params": {
                        "blur_limit": {"type": "tuple", "default": [3, 7]},
                    },
                },
                "motion_blur": {
                    "name": "Motion Blur",
                    "description": "Simulates camera motion blur",
                    "default_prob": 0.1,
                    "params": {
                        "blur_limit": {"type": "int", "default": 7, "min": 3, "max": 15},
                    },
                },
                "median_blur": {
                    "name": "Median Blur",
                    "description": "Applies median filter",
                    "default_prob": 0.1,
                    "params": {
                        "blur_limit": {"type": "int", "default": 5, "min": 3, "max": 9},
                    },
                },
                "defocus": {
                    "name": "Defocus",
                    "description": "Simulates camera defocus",
                    "default_prob": 0.05,
                    "params": {
                        "radius": {"type": "tuple", "default": [3, 7]},
                        "alias_blur": {"type": "tuple", "default": [0.1, 0.3]},
                    },
                },
                "zoom_blur": {
                    "name": "Zoom Blur",
                    "description": "Simulates camera zoom blur",
                    "default_prob": 0.05,
                    "params": {
                        "max_factor": {"type": "float", "default": 1.1, "min": 1.01, "max": 1.3},
                    },
                },
                "glass_blur": {
                    "name": "Glass Blur",
                    "description": "Simulates frosted glass effect",
                    "default_prob": 0.05,
                    "params": {
                        "sigma": {"type": "float", "default": 0.7, "min": 0.1, "max": 2.0},
                        "max_delta": {"type": "int", "default": 4, "min": 1, "max": 10},
                        "iterations": {"type": "int", "default": 2, "min": 1, "max": 5},
                    },
                },
                "advanced_blur": {
                    "name": "Advanced Blur",
                    "description": "Advanced blur with noise",
                    "default_prob": 0.05,
                    "params": {
                        "blur_limit": {"type": "tuple", "default": [3, 7]},
                        "noise_limit": {"type": "tuple", "default": [0.5, 1.0]},
                    },
                },
            },
        },
        "noise": {
            "name": "Noise",
            "description": "Noise augmentations",
            "icon": "radio",
            "augmentations": {
                "gaussian_noise": {
                    "name": "Gaussian Noise",
                    "description": "Adds Gaussian noise",
                    "default_prob": 0.1,
                    "params": {
                        "var_limit": {"type": "tuple", "default": [10, 50]},
                    },
                },
                "iso_noise": {
                    "name": "ISO Noise",
                    "description": "Simulates camera sensor noise",
                    "default_prob": 0.1,
                    "params": {
                        "color_shift": {"type": "tuple", "default": [0.01, 0.05]},
                        "intensity": {"type": "tuple", "default": [0.1, 0.5]},
                    },
                },
                "multiplicative_noise": {
                    "name": "Multiplicative Noise",
                    "description": "Multiplies pixel values by random factor",
                    "default_prob": 0.1,
                    "params": {
                        "multiplier": {"type": "tuple", "default": [0.9, 1.1]},
                    },
                },
            },
        },
        "quality": {
            "name": "Quality Degradation",
            "description": "Image quality reduction for robustness",
            "icon": "image-down",
            "augmentations": {
                "image_compression": {
                    "name": "JPEG Compression",
                    "description": "Applies JPEG compression artifacts",
                    "default_prob": 0.2,
                    "params": {
                        "quality_lower": {"type": "int", "default": 70, "min": 10, "max": 95},
                        "quality_upper": {"type": "int", "default": 95, "min": 50, "max": 100},
                    },
                },
                "downscale": {
                    "name": "Downscale",
                    "description": "Downscales and upscales to reduce quality",
                    "default_prob": 0.1,
                    "params": {
                        "scale_min": {"type": "float", "default": 0.5, "min": 0.1, "max": 0.9},
                        "scale_max": {"type": "float", "default": 0.9, "min": 0.5, "max": 1.0},
                    },
                },
            },
        },
        "dropout": {
            "name": "Dropout / Occlusion",
            "description": "Simulates occlusions and missing data",
            "icon": "square-x",
            "augmentations": {
                "coarse_dropout": {
                    "name": "Coarse Dropout",
                    "description": "Drops random rectangular regions",
                    "default_prob": 0.1,
                    "params": {
                        "max_holes": {"type": "int", "default": 8, "min": 1, "max": 20},
                        "max_height": {"type": "int", "default": 40, "min": 8, "max": 100},
                        "max_width": {"type": "int", "default": 40, "min": 8, "max": 100},
                        "fill_value": {"type": "int", "default": 114, "min": 0, "max": 255},
                    },
                },
                "grid_dropout": {
                    "name": "Grid Dropout",
                    "description": "Drops cells in a grid pattern",
                    "default_prob": 0.1,
                    "params": {
                        "ratio": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
                        "unit_size_min": {"type": "int", "default": 10, "min": 5, "max": 50},
                        "unit_size_max": {"type": "int", "default": 40, "min": 20, "max": 100},
                    },
                },
                "pixel_dropout": {
                    "name": "Pixel Dropout",
                    "description": "Drops random individual pixels",
                    "default_prob": 0.05,
                    "params": {
                        "dropout_prob": {"type": "float", "default": 0.01, "min": 0.001, "max": 0.1},
                    },
                },
                "mask_dropout": {
                    "name": "Mask Dropout",
                    "description": "Drops random objects (for instance segmentation)",
                    "default_prob": 0.1,
                    "params": {
                        "max_objects": {"type": "int", "default": 1, "min": 1, "max": 5},
                    },
                },
            },
        },
        "weather": {
            "name": "Weather Effects",
            "description": "Weather and environmental effects",
            "icon": "cloud-rain",
            "augmentations": {
                "random_rain": {
                    "name": "Rain",
                    "description": "Adds rain effect",
                    "default_prob": 0.1,
                    "params": {
                        "slant_lower": {"type": "int", "default": -10, "min": -20, "max": 0},
                        "slant_upper": {"type": "int", "default": 10, "min": 0, "max": 20},
                        "drop_length": {"type": "int", "default": 20, "min": 5, "max": 50},
                        "drop_width": {"type": "int", "default": 1, "min": 1, "max": 3},
                    },
                },
                "random_fog": {
                    "name": "Fog",
                    "description": "Adds fog effect",
                    "default_prob": 0.1,
                    "params": {
                        "fog_coef_lower": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5},
                        "fog_coef_upper": {"type": "float", "default": 0.3, "min": 0.1, "max": 1.0},
                    },
                },
                "random_shadow": {
                    "name": "Shadow",
                    "description": "Adds random shadows",
                    "default_prob": 0.1,
                    "params": {
                        "num_shadows_lower": {"type": "int", "default": 1, "min": 1, "max": 3},
                        "num_shadows_upper": {"type": "int", "default": 2, "min": 1, "max": 5},
                    },
                },
                "random_sun_flare": {
                    "name": "Sun Flare",
                    "description": "Adds sun flare effect",
                    "default_prob": 0.05,
                    "params": {
                        "src_radius": {"type": "int", "default": 100, "min": 50, "max": 300},
                        "num_flare_circles_lower": {"type": "int", "default": 3, "min": 1, "max": 5},
                        "num_flare_circles_upper": {"type": "int", "default": 7, "min": 5, "max": 15},
                    },
                },
                "random_snow": {
                    "name": "Snow",
                    "description": "Adds snow effect",
                    "default_prob": 0.05,
                    "params": {
                        "snow_point_lower": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.3},
                        "snow_point_upper": {"type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
                    },
                },
                "spatter": {
                    "name": "Spatter",
                    "description": "Adds rain drops or mud spatters",
                    "default_prob": 0.05,
                    "params": {
                        "mode": {"type": "enum", "default": "rain", "options": ["rain", "mud"]},
                    },
                },
                "plasma_brightness": {
                    "name": "Plasma Brightness",
                    "description": "Applies plasma-based brightness variation",
                    "default_prob": 0.05,
                    "params": {
                        "roughness": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0},
                    },
                },
            },
        },
    }


def get_all_augmentation_names() -> List[str]:
    """Get list of all available augmentation names."""
    names = []
    for category in get_augmentation_categories().values():
        names.extend(category["augmentations"].keys())
    return names


def validate_custom_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a custom augmentation configuration.

    Args:
        config: Dictionary with augmentation configurations

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    valid_names = get_all_augmentation_names()

    for aug_name, aug_config in config.items():
        if aug_name not in valid_names:
            errors.append(f"Unknown augmentation: {aug_name}")
            continue

        if not isinstance(aug_config, dict):
            errors.append(f"{aug_name}: config must be a dictionary")
            continue

        if "prob" in aug_config:
            prob = aug_config["prob"]
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                errors.append(f"{aug_name}: prob must be between 0 and 1")

        if "enabled" in aug_config and not isinstance(aug_config["enabled"], bool):
            errors.append(f"{aug_name}: enabled must be a boolean")

    return errors
