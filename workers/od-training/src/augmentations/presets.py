"""
Augmentation Presets for Object Detection Training.

Provides ready-to-use augmentation configurations:
- SOTA: State-of-the-art (Mosaic + MixUp + CopyPaste + standard)
- Heavy: All augmentations enabled
- Medium: Balanced augmentations
- Light: Minimal augmentations (flip/rotate only)
- None: No augmentations

Usage:
    from augmentations.presets import get_preset, PRESETS

    config = get_preset("sota")
    # Or customize:
    config = get_preset("sota", overrides={"mosaic": {"prob": 0.7}})
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class PresetLevel(str, Enum):
    """Available preset levels."""
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

    # Multi-image augmentations (SOTA)
    mosaic: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    mosaic9: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    mixup: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    cutmix: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    copypaste: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # Geometric augmentations
    horizontal_flip: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    vertical_flip: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    rotate90: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_rotate: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    affine: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    perspective: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_crop: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_scale: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # Color/Light augmentations
    brightness_contrast: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    hue_saturation: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    rgb_shift: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    channel_shuffle: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    clahe: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    equalize: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    to_gray: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # Blur/Noise augmentations
    gaussian_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    motion_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    median_blur: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    gaussian_noise: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    iso_noise: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # Weather/Occlusion augmentations
    random_rain: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_fog: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    random_shadow: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())
    coarse_dropout: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

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


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

def _create_sota_preset() -> AugmentationPreset:
    """
    SOTA (State-of-the-Art) preset.

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
        mixup=AugmentationConfig(enabled=True, prob=0.4, params={"alpha": 8.0}),
        copypaste=AugmentationConfig(enabled=True, prob=0.3, params={"blend_ratio": 0.5}),

        # Geometric
        horizontal_flip=AugmentationConfig(enabled=True, prob=0.5),
        vertical_flip=AugmentationConfig(enabled=True, prob=0.2),
        rotate90=AugmentationConfig(enabled=True, prob=0.3),
        random_rotate=AugmentationConfig(enabled=True, prob=0.3, params={"limit": 15}),
        affine=AugmentationConfig(enabled=True, prob=0.3, params={
            "scale": (0.8, 1.2), "translate_percent": 0.1, "shear": 10
        }),
        random_scale=AugmentationConfig(enabled=True, prob=0.5, params={"scale_limit": 0.3}),

        # Color
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.5, params={
            "brightness_limit": 0.3, "contrast_limit": 0.3
        }),
        hue_saturation=AugmentationConfig(enabled=True, prob=0.4, params={
            "hue_shift_limit": 30, "sat_shift_limit": 40
        }),
        rgb_shift=AugmentationConfig(enabled=True, prob=0.2, params={"r_shift_limit": 20}),
        clahe=AugmentationConfig(enabled=True, prob=0.2, params={"clip_limit": 4.0}),
        to_gray=AugmentationConfig(enabled=True, prob=0.1),

        # Blur/Noise
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.2, params={"blur_limit": 5}),
        motion_blur=AugmentationConfig(enabled=True, prob=0.1, params={"blur_limit": 5}),
        gaussian_noise=AugmentationConfig(enabled=True, prob=0.1, params={"var_limit": (10, 50)}),

        # Weather
        random_shadow=AugmentationConfig(enabled=True, prob=0.1),
        coarse_dropout=AugmentationConfig(enabled=True, prob=0.1, params={
            "max_holes": 8, "max_height": 32, "max_width": 32
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
        random_scale=AugmentationConfig(enabled=True, prob=0.4, params={"scale_limit": 0.2}),

        # Color
        brightness_contrast=AugmentationConfig(enabled=True, prob=0.3, params={
            "brightness_limit": 0.2, "contrast_limit": 0.2
        }),
        hue_saturation=AugmentationConfig(enabled=True, prob=0.2, params={
            "hue_shift_limit": 15, "sat_shift_limit": 20
        }),

        # Blur (minimal)
        gaussian_blur=AugmentationConfig(enabled=True, prob=0.1, params={"blur_limit": 3}),
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
        preset_name: One of "sota", "heavy", "medium", "light", "none", "custom"
        overrides: Optional dict to override specific augmentation settings
            Example: {"mosaic": {"prob": 0.7}, "mixup": {"enabled": False}}

    Returns:
        AugmentationPreset configured with preset + overrides

    Example:
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
        "sota": {
            "name": "SOTA (Önerilen)",
            "description": "State-of-the-art: Mosaic, MixUp, CopyPaste + standart augmentationlar",
            "icon": "⭐",
            "training_time": "1.5x",
            "accuracy_boost": "+3-5% mAP",
        },
        "heavy": {
            "name": "Güçlü",
            "description": "Tüm augmentationlar aktif, küçük datasetler için ideal",
            "icon": "🔥",
            "training_time": "2x",
            "accuracy_boost": "+5-8% mAP",
        },
        "medium": {
            "name": "Orta",
            "description": "Dengeli augmentationlar, genel kullanım için",
            "icon": "⚡",
            "training_time": "1.2x",
            "accuracy_boost": "+2-3% mAP",
        },
        "light": {
            "name": "Hafif",
            "description": "Sadece temel augmentationlar, hızlı eğitim",
            "icon": "💨",
            "training_time": "1.05x",
            "accuracy_boost": "+1% mAP",
        },
        "none": {
            "name": "Kapalı",
            "description": "Augmentation yok, baseline karşılaştırma için",
            "icon": "❌",
            "training_time": "1x",
            "accuracy_boost": "Baseline",
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
            "description": "Birden fazla resmi birleştiren augmentationlar",
            "icon": "🖼️",
            "augmentations": {
                "mosaic": {
                    "name": "Mosaic",
                    "description": "4 resmi 2x2 grid'de birleştirir",
                    "default_prob": 0.5,
                    "params": ["img_size"],
                },
                "mosaic9": {
                    "name": "Mosaic-9",
                    "description": "9 resmi 3x3 grid'de birleştirir",
                    "default_prob": 0.3,
                    "params": ["img_size"],
                },
                "mixup": {
                    "name": "MixUp",
                    "description": "2 resmi alpha-blend ile karıştırır",
                    "default_prob": 0.3,
                    "params": ["alpha"],
                },
                "cutmix": {
                    "name": "CutMix",
                    "description": "Bir resimden kesip diğerine yapıştırır",
                    "default_prob": 0.2,
                    "params": ["alpha"],
                },
                "copypaste": {
                    "name": "CopyPaste",
                    "description": "Objeleri bir resimden diğerine kopyalar",
                    "default_prob": 0.2,
                    "params": ["blend_ratio"],
                },
            },
        },
        "geometric": {
            "name": "Geometrik",
            "description": "Şekil ve konum değiştiren augmentationlar",
            "icon": "📐",
            "augmentations": {
                "horizontal_flip": {
                    "name": "Yatay Çevir",
                    "description": "Resmi yatay eksende çevirir",
                    "default_prob": 0.5,
                    "params": [],
                },
                "vertical_flip": {
                    "name": "Dikey Çevir",
                    "description": "Resmi dikey eksende çevirir",
                    "default_prob": 0.2,
                    "params": [],
                },
                "rotate90": {
                    "name": "90° Döndür",
                    "description": "Resmi 90° döndürür",
                    "default_prob": 0.3,
                    "params": [],
                },
                "random_rotate": {
                    "name": "Rastgele Döndür",
                    "description": "Resmi rastgele açıda döndürür",
                    "default_prob": 0.3,
                    "params": ["limit"],
                },
                "affine": {
                    "name": "Affine Transform",
                    "description": "Scale, translate, shear uygular",
                    "default_prob": 0.3,
                    "params": ["scale", "translate_percent", "shear"],
                },
                "random_scale": {
                    "name": "Rastgele Ölçekle",
                    "description": "Resmi rastgele ölçeklendirir",
                    "default_prob": 0.5,
                    "params": ["scale_limit"],
                },
            },
        },
        "color": {
            "name": "Renk / Işık",
            "description": "Renk ve parlaklık değiştiren augmentationlar",
            "icon": "🎨",
            "augmentations": {
                "brightness_contrast": {
                    "name": "Parlaklık/Kontrast",
                    "description": "Parlaklık ve kontrastı değiştirir",
                    "default_prob": 0.4,
                    "params": ["brightness_limit", "contrast_limit"],
                },
                "hue_saturation": {
                    "name": "Hue/Saturation",
                    "description": "Renk tonu ve doygunluğu değiştirir",
                    "default_prob": 0.3,
                    "params": ["hue_shift_limit", "sat_shift_limit"],
                },
                "rgb_shift": {
                    "name": "RGB Kaydır",
                    "description": "RGB kanallarını kaydırır",
                    "default_prob": 0.2,
                    "params": ["r_shift_limit"],
                },
                "clahe": {
                    "name": "CLAHE",
                    "description": "Adaptive histogram equalization",
                    "default_prob": 0.2,
                    "params": ["clip_limit"],
                },
                "to_gray": {
                    "name": "Gri Tonlama",
                    "description": "Resmi gri tonlamaya çevirir",
                    "default_prob": 0.1,
                    "params": [],
                },
            },
        },
        "blur_noise": {
            "name": "Blur / Noise",
            "description": "Bulanıklık ve gürültü ekleyen augmentationlar",
            "icon": "🌫️",
            "augmentations": {
                "gaussian_blur": {
                    "name": "Gaussian Blur",
                    "description": "Gaussian bulanıklık ekler",
                    "default_prob": 0.1,
                    "params": ["blur_limit"],
                },
                "motion_blur": {
                    "name": "Motion Blur",
                    "description": "Hareket bulanıklığı ekler",
                    "default_prob": 0.1,
                    "params": ["blur_limit"],
                },
                "gaussian_noise": {
                    "name": "Gaussian Noise",
                    "description": "Gaussian gürültü ekler",
                    "default_prob": 0.1,
                    "params": ["var_limit"],
                },
            },
        },
        "weather": {
            "name": "Hava / Occlusion",
            "description": "Hava durumu ve kapatma efektleri",
            "icon": "🌧️",
            "augmentations": {
                "random_rain": {
                    "name": "Yağmur",
                    "description": "Yağmur efekti ekler",
                    "default_prob": 0.1,
                    "params": [],
                },
                "random_fog": {
                    "name": "Sis",
                    "description": "Sis efekti ekler",
                    "default_prob": 0.1,
                    "params": [],
                },
                "random_shadow": {
                    "name": "Gölge",
                    "description": "Rastgele gölge ekler",
                    "default_prob": 0.1,
                    "params": [],
                },
                "coarse_dropout": {
                    "name": "Coarse Dropout",
                    "description": "Rastgele dikdörtgenler siler",
                    "default_prob": 0.1,
                    "params": ["max_holes", "max_height", "max_width"],
                },
            },
        },
    }
