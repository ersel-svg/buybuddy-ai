"""
Model registry with configurations for all supported models.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Any, List
import importlib


# Standard normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP uses different normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@dataclass
class ModelConfig:
    """Configuration for a model."""

    model_id: str
    family: str  # "dinov2", "dinov3", "clip", "custom"
    hf_model_id: str
    embedding_dim: int
    image_size: int
    params_millions: float
    description: str
    backbone_class: str  # Module path to backbone class
    image_mean: List[float] = field(default_factory=lambda: IMAGENET_MEAN.copy())
    image_std: List[float] = field(default_factory=lambda: IMAGENET_STD.copy())

    @property
    def input_size(self) -> int:
        """Alias for image_size (backward compatibility)."""
        return self.image_size

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration for this model."""
        return {
            "image_size": self.image_size,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }


# =============================================
# DINOv2 Family (Giant excluded for memory)
# =============================================
DINOV2_MODELS = {
    "dinov2-small": ModelConfig(
        model_id="dinov2-small",
        family="dinov2",
        hf_model_id="facebook/dinov2-small",
        embedding_dim=384,
        image_size=518,
        params_millions=22,
        description="DINOv2 Small - Fast and lightweight",
        backbone_class="bb_models.backbones.dinov2.DINOv2Backbone",
    ),
    "dinov2-base": ModelConfig(
        model_id="dinov2-base",
        family="dinov2",
        hf_model_id="facebook/dinov2-base",
        embedding_dim=768,
        image_size=518,
        params_millions=86,
        description="DINOv2 Base - Balanced performance",
        backbone_class="bb_models.backbones.dinov2.DINOv2Backbone",
    ),
    "dinov2-large": ModelConfig(
        model_id="dinov2-large",
        family="dinov2",
        hf_model_id="facebook/dinov2-large",
        embedding_dim=1024,
        image_size=518,
        params_millions=300,
        description="DINOv2 Large - High accuracy",
        backbone_class="bb_models.backbones.dinov2.DINOv2Backbone",
    ),
}

# =============================================
# DINOv3 Family (Giant excluded for memory)
# =============================================
DINOV3_MODELS = {
    "dinov3-small": ModelConfig(
        model_id="dinov3-small",
        family="dinov3",
        hf_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        embedding_dim=384,
        image_size=518,
        params_millions=21,
        description="DINOv3 Small - Latest architecture, fast",
        backbone_class="bb_models.backbones.dinov3.DINOv3Backbone",
    ),
    "dinov3-base": ModelConfig(
        model_id="dinov3-base",
        family="dinov3",
        hf_model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        embedding_dim=768,
        image_size=518,
        params_millions=86,
        description="DINOv3 Base - Recommended for most use cases",
        backbone_class="bb_models.backbones.dinov3.DINOv3Backbone",
    ),
    "dinov3-large": ModelConfig(
        model_id="dinov3-large",
        family="dinov3",
        hf_model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        embedding_dim=1024,
        image_size=518,
        params_millions=300,
        description="DINOv3 Large - Best accuracy",
        backbone_class="bb_models.backbones.dinov3.DINOv3Backbone",
    ),
}

# =============================================
# CLIP Family
# =============================================
CLIP_MODELS = {
    "clip-vit-l-14": ModelConfig(
        model_id="clip-vit-l-14",
        family="clip",
        hf_model_id="openai/clip-vit-large-patch14",
        embedding_dim=1024,
        image_size=224,
        params_millions=304,
        description="CLIP ViT-L/14 - High capacity",
        backbone_class="bb_models.backbones.clip.CLIPBackbone",
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
    ),
}

# =============================================
# Combined Registry
# =============================================
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    **DINOV2_MODELS,
    **DINOV3_MODELS,
    **CLIP_MODELS,
}


def get_model_config(model_id: str) -> ModelConfig:
    """
    Get configuration for a model by ID.

    Args:
        model_id: Model identifier (e.g., "dinov2-base", "clip-vit-b-16")

    Returns:
        ModelConfig object

    Raises:
        ValueError: If model_id is not supported
    """
    if model_id not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")
    return MODEL_CONFIGS[model_id]


def get_backbone(
    model_id: str,
    checkpoint_url: Optional[str] = None,
    load_pretrained: bool = True,
) -> Any:
    """
    Get a backbone instance by model ID.

    Args:
        model_id: Model identifier (e.g., "dinov2-base")
        checkpoint_url: Optional URL to custom checkpoint
        load_pretrained: Whether to load pretrained weights

    Returns:
        Backbone instance

    Example:
        >>> backbone = get_backbone("dinov2-base")
        >>> features = backbone(images)
    """
    config = get_model_config(model_id)

    # Dynamically import the backbone class
    module_path, class_name = config.backbone_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    backbone_class = getattr(module, class_name)

    # Create instance
    backbone = backbone_class(model_id=model_id)

    # Load weights if requested
    if load_pretrained:
        backbone.load_pretrained(checkpoint_url=checkpoint_url)

    return backbone


def list_available_models(family: Optional[str] = None) -> list:
    """
    List all available models.

    Args:
        family: Optional filter by family ("dinov2", "dinov3", "clip")

    Returns:
        List of model IDs
    """
    if family is None:
        return list(MODEL_CONFIGS.keys())

    return [
        model_id
        for model_id, config in MODEL_CONFIGS.items()
        if config.family == family
    ]


def is_model_supported(model_id: str) -> bool:
    """Check if a model ID is supported."""
    return model_id in MODEL_CONFIGS


def get_models_by_family() -> Dict[str, list]:
    """
    Get models grouped by family.

    Returns:
        Dict mapping family name to list of model IDs
    """
    result: Dict[str, list] = {}
    for model_id, config in MODEL_CONFIGS.items():
        if config.family not in result:
            result[config.family] = []
        result[config.family].append(model_id)
    return result


def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a model.

    Args:
        model_id: Model identifier

    Returns:
        Dict with model information including preprocessing config
    """
    config = get_model_config(model_id)
    return {
        "model_id": config.model_id,
        "family": config.family,
        "hf_model_id": config.hf_model_id,
        "embedding_dim": config.embedding_dim,
        "image_size": config.image_size,
        "image_mean": config.image_mean,
        "image_std": config.image_std,
        "params_millions": config.params_millions,
        "description": config.description,
        "preprocessing": config.get_preprocessing_config(),
    }
