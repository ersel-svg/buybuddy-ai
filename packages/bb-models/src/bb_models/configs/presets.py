"""
Model-specific training configuration presets.

Each model family has optimized default hyperparameters based on:
- Model size and architecture
- Memory requirements
- Training stability characteristics
- Empirical best practices

Users can override any setting while starting from a solid baseline.
"""

import copy
from typing import Dict, Any, Optional, List


# =============================================
# DINOv2 Family Presets
# =============================================

DINOV2_SMALL_PRESET = {
    "training": {
        "epochs": 30,
        "batch_size": 32,
        "lr": 4e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 3,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
    },
    "model": {
        "proj_dim": 384,
        "dropout": 0.15,
        "arcface_scale": 30.0,
        "arcface_margin": 0.30,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 4,
        "mixed_precision": True,
    },
}

DINOV2_BASE_PRESET = {
    "training": {
        "epochs": 30,
        "batch_size": 24,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 3,
        "grad_clip": 1.0,
        "label_smoothing": 0.15,
    },
    "model": {
        "proj_dim": 768,
        "dropout": 0.15,
        "arcface_scale": 30.0,
        "arcface_margin": 0.30,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 4,
        "mixed_precision": True,
    },
}

DINOV2_LARGE_PRESET = {
    "training": {
        "epochs": 30,
        "batch_size": 16,       # Smaller batch (memory)
        "lr": 2e-5,             # Lower LR (stability)
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 5,     # Longer warmup
        "grad_clip": 0.5,       # Aggressive clipping
        "label_smoothing": 0.2,
    },
    "model": {
        "proj_dim": 1024,
        "dropout": 0.15,
        "arcface_scale": 30.0,
        "arcface_margin": 0.30,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 2,       # Reduced (memory)
        "mixed_precision": True,
    },
}

# =============================================
# DINOv3 Family Presets
# =============================================

DINOV3_SMALL_PRESET = {
    "training": {
        "epochs": 25,           # DINOv3 converges faster
        "batch_size": 32,
        "lr": 3.5e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 3,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
    },
    "model": {
        "proj_dim": 384,
        "dropout": 0.1,
        "arcface_scale": 30.0,
        "arcface_margin": 0.28,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 4,
        "mixed_precision": True,
    },
}

DINOV3_BASE_PRESET = {
    "training": {
        "epochs": 25,
        "batch_size": 24,
        "lr": 2.5e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 3,
        "grad_clip": 1.0,
        "label_smoothing": 0.15,
    },
    "model": {
        "proj_dim": 768,
        "dropout": 0.1,
        "arcface_scale": 30.0,
        "arcface_margin": 0.28,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 4,
        "mixed_precision": True,
    },
}

DINOV3_LARGE_PRESET = {
    "training": {
        "epochs": 25,
        "batch_size": 16,
        "lr": 1.8e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.9,
        "warmup_epochs": 5,
        "grad_clip": 0.5,
        "label_smoothing": 0.2,
    },
    "model": {
        "proj_dim": 1024,
        "dropout": 0.1,
        "arcface_scale": 30.0,
        "arcface_margin": 0.28,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 384,
        "num_workers": 2,
        "mixed_precision": True,
    },
}

# =============================================
# CLIP Family Presets
# =============================================

CLIP_VIT_L_14_PRESET = {
    "training": {
        "epochs": 30,
        "batch_size": 16,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "llrd_decay": 0.85,
        "warmup_epochs": 5,
        "grad_clip": 0.5,
        "label_smoothing": 0.2,
    },
    "model": {
        "proj_dim": 768,
        "dropout": 0.15,
        "arcface_scale": 30.0,
        "arcface_margin": 0.30,
    },
    "sampling": {
        "domain_aware_ratio": 0.57,
        "hard_negative_pool_size": 5,
        "use_hardest_negatives": True,
    },
    "infrastructure": {
        "image_size": 224,
        "num_workers": 2,
        "mixed_precision": True,
    },
}

# =============================================
# Combined Registry
# =============================================

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # DINOv2 Family
    "dinov2-small": DINOV2_SMALL_PRESET,
    "dinov2-base": DINOV2_BASE_PRESET,
    "dinov2-large": DINOV2_LARGE_PRESET,
    # DINOv3 Family
    "dinov3-small": DINOV3_SMALL_PRESET,
    "dinov3-base": DINOV3_BASE_PRESET,
    "dinov3-large": DINOV3_LARGE_PRESET,
    # CLIP Family
    "clip-vit-l-14": CLIP_VIT_L_14_PRESET,
}


def get_preset(model_id: str) -> Dict[str, Any]:
    """
    Get configuration preset for a model.

    Args:
        model_id: Model identifier (e.g., "dinov3-base", "clip-vit-b-16")

    Returns:
        Deep copy of the preset configuration dict.

    Raises:
        ValueError: If model_id has no preset.

    Example:
        >>> config = get_preset("dinov3-base")
        >>> config["training"]["epochs"] = 50  # Customize
    """
    if model_id not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"No preset for model: {model_id}. Available: {available}")

    return copy.deepcopy(MODEL_PRESETS[model_id])


def merge_config(
    preset: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge preset with user overrides.

    Performs deep merge so nested values can be overridden individually.

    Args:
        preset: Base preset configuration.
        overrides: User-specified overrides.

    Returns:
        Merged configuration.

    Example:
        >>> preset = get_preset("dinov3-base")
        >>> overrides = {"training": {"epochs": 50, "batch_size": 32}}
        >>> config = merge_config(preset, overrides)
    """
    result = copy.deepcopy(preset)

    for section, values in overrides.items():
        if section in result and isinstance(values, dict) and isinstance(result[section], dict):
            # Deep merge for dict sections
            result[section].update(values)
        else:
            # Direct replacement for non-dict values
            result[section] = values

    return result


def list_presets() -> List[str]:
    """
    List all available preset model IDs.

    Returns:
        List of model IDs with presets.
    """
    return list(MODEL_PRESETS.keys())


def get_preset_summary(model_id: str) -> Dict[str, Any]:
    """
    Get a summary of key preset values.

    Args:
        model_id: Model identifier.

    Returns:
        Dict with key configuration values.
    """
    preset = get_preset(model_id)

    return {
        "model_id": model_id,
        "epochs": preset["training"]["epochs"],
        "batch_size": preset["training"]["batch_size"],
        "lr": preset["training"]["lr"],
        "proj_dim": preset["model"]["proj_dim"],
        "image_size": preset["infrastructure"]["image_size"],
        "mixed_precision": preset["infrastructure"]["mixed_precision"],
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a training configuration.

    Args:
        config: Configuration dict to validate.

    Returns:
        List of validation warnings (empty if valid).
    """
    warnings = []

    # Training section
    training = config.get("training", {})
    if training.get("batch_size", 0) < 4:
        warnings.append("batch_size < 4 may cause unstable training")
    if training.get("lr", 0) > 1e-3:
        warnings.append("lr > 1e-3 is very high for fine-tuning")
    if training.get("grad_clip", 1.0) < 0.1:
        warnings.append("grad_clip < 0.1 may over-constrain gradients")

    # Model section
    model = config.get("model", {})
    if model.get("dropout", 0) > 0.5:
        warnings.append("dropout > 0.5 may cause underfitting")
    if model.get("arcface_margin", 0) > 0.5:
        warnings.append("arcface_margin > 0.5 is aggressive, may hurt training")

    # Infrastructure section
    infra = config.get("infrastructure", {})
    if infra.get("num_workers", 0) > 8:
        warnings.append("num_workers > 8 may not improve performance")

    return warnings


# =============================================
# Quick preset creation for custom models
# =============================================

def create_custom_preset(
    base_preset: str = "dinov3-base",
    embedding_dim: int = 768,
    **overrides,
) -> Dict[str, Any]:
    """
    Create a custom preset based on an existing one.

    Useful for custom fine-tuned models.

    Args:
        base_preset: Base model preset to start from.
        embedding_dim: Embedding dimension of the custom model.
        **overrides: Additional configuration overrides.

    Returns:
        Custom preset configuration.

    Example:
        >>> preset = create_custom_preset(
        ...     base_preset="dinov3-base",
        ...     embedding_dim=768,
        ...     training={"epochs": 20, "lr": 1e-5},
        ... )
    """
    preset = get_preset(base_preset)

    # Set embedding dim
    preset["model"]["proj_dim"] = embedding_dim

    # Apply overrides
    return merge_config(preset, overrides)
