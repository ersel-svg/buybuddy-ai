"""
Classification Model Zoo - SOTA Models

Supported Models:
- ViT (Vision Transformer): vit_base_patch16_224, vit_large_patch16_224
- ConvNeXt: convnext_tiny, convnext_small, convnext_base
- EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
- Swin Transformer: swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
- ResNet: resnet50, resnet101
- DINOv2: dinov2_vits14, dinov2_vitb14 (for fine-tuning)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import timm
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    pretrained: bool = True
    num_classes: int = 1000
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    global_pool: str = "avg"
    # For transfer learning
    freeze_backbone: bool = False
    freeze_bn: bool = False
    # For ViT/Swin
    img_size: int = 224


# Model registry with default configs
MODEL_REGISTRY = {
    # ViT models
    "vit_tiny_patch16_224": {
        "timm_name": "vit_tiny_patch16_224",
        "input_size": 224,
        "params_m": 5.7,
        "flops_g": 1.3,
        "features_dim": 192,
    },
    "vit_small_patch16_224": {
        "timm_name": "vit_small_patch16_224",
        "input_size": 224,
        "params_m": 22.1,
        "flops_g": 4.6,
        "features_dim": 384,
    },
    "vit_base_patch16_224": {
        "timm_name": "vit_base_patch16_224",
        "input_size": 224,
        "params_m": 86.6,
        "flops_g": 17.6,
        "features_dim": 768,
    },
    "vit_large_patch16_224": {
        "timm_name": "vit_large_patch16_224",
        "input_size": 224,
        "params_m": 304.4,
        "flops_g": 61.6,
        "features_dim": 1024,
    },
    # ConvNeXt models
    "convnext_tiny": {
        "timm_name": "convnext_tiny",
        "input_size": 224,
        "params_m": 28.6,
        "flops_g": 4.5,
        "features_dim": 768,
    },
    "convnext_small": {
        "timm_name": "convnext_small",
        "input_size": 224,
        "params_m": 50.2,
        "flops_g": 8.7,
        "features_dim": 768,
    },
    "convnext_base": {
        "timm_name": "convnext_base",
        "input_size": 224,
        "params_m": 88.6,
        "flops_g": 15.4,
        "features_dim": 1024,
    },
    # EfficientNet models
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "input_size": 224,
        "params_m": 5.3,
        "flops_g": 0.4,
        "features_dim": 1280,
    },
    "efficientnet_b1": {
        "timm_name": "efficientnet_b1",
        "input_size": 240,
        "params_m": 7.8,
        "flops_g": 0.7,
        "features_dim": 1280,
    },
    "efficientnet_b2": {
        "timm_name": "efficientnet_b2",
        "input_size": 260,
        "params_m": 9.2,
        "flops_g": 1.0,
        "features_dim": 1408,
    },
    "efficientnet_b3": {
        "timm_name": "efficientnet_b3",
        "input_size": 300,
        "params_m": 12.2,
        "flops_g": 1.8,
        "features_dim": 1536,
    },
    # Swin Transformer models
    "swin_tiny_patch4_window7_224": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "input_size": 224,
        "params_m": 28.3,
        "flops_g": 4.5,
        "features_dim": 768,
    },
    "swin_small_patch4_window7_224": {
        "timm_name": "swin_small_patch4_window7_224",
        "input_size": 224,
        "params_m": 49.6,
        "flops_g": 8.7,
        "features_dim": 768,
    },
    "swin_base_patch4_window7_224": {
        "timm_name": "swin_base_patch4_window7_224",
        "input_size": 224,
        "params_m": 87.8,
        "flops_g": 15.4,
        "features_dim": 1024,
    },
    # ResNet models
    "resnet50": {
        "timm_name": "resnet50",
        "input_size": 224,
        "params_m": 25.6,
        "flops_g": 4.1,
        "features_dim": 2048,
    },
    "resnet101": {
        "timm_name": "resnet101",
        "input_size": 224,
        "params_m": 44.5,
        "flops_g": 7.8,
        "features_dim": 2048,
    },
    # EfficientNetV2 (newer, faster)
    "efficientnetv2_s": {
        "timm_name": "efficientnetv2_s",
        "input_size": 384,
        "params_m": 21.5,
        "flops_g": 8.8,
        "features_dim": 1280,
    },
    "efficientnetv2_m": {
        "timm_name": "efficientnetv2_m",
        "input_size": 480,
        "params_m": 54.1,
        "flops_g": 25.0,
        "features_dim": 1280,
    },
    # DINOv2 (for fine-tuning)
    "dinov2_vits14": {
        "timm_name": "vit_small_patch14_dinov2",
        "input_size": 518,
        "params_m": 22.0,
        "flops_g": 5.5,
        "features_dim": 384,
    },
    "dinov2_vitb14": {
        "timm_name": "vit_base_patch14_dinov2",
        "input_size": 518,
        "params_m": 86.6,
        "flops_g": 17.5,
        "features_dim": 768,
    },
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Create a classification model.

    Args:
        model_name: Name of the model (from MODEL_REGISTRY)
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        drop_rate: Dropout rate for classifier
        drop_path_rate: Stochastic depth rate

    Returns:
        model: The created model
        model_info: Dict with model metadata
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_info = MODEL_REGISTRY[model_name].copy()
    timm_name = model_info["timm_name"]

    # Create model with timm
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs
    )

    model_info["num_classes"] = num_classes
    model_info["pretrained"] = pretrained

    return model, model_info


def freeze_backbone(model: nn.Module, freeze_bn: bool = True) -> None:
    """
    Freeze backbone, keeping only classifier trainable.

    Args:
        model: The model to freeze
        freeze_bn: Also freeze batch norm layers
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier (head)
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Optionally freeze BN
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def get_model_input_size(model_name: str) -> int:
    """Get the input size for a model."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]["input_size"]
    return 224


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models with their info."""
    return MODEL_REGISTRY.copy()


class ModelWithEmbedding(nn.Module):
    """
    Wrapper that provides both classification logits and embeddings.

    Useful for:
    - ArcFace / CosFace loss
    - Contrastive learning
    - Feature extraction
    """

    def __init__(
        self,
        base_model: nn.Module,
        embedding_dim: int,
        num_classes: int,
        normalize_embeddings: bool = True,
    ):
        super().__init__()
        self.backbone = base_model

        # Remove the original classifier
        if hasattr(self.backbone, 'head'):
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError("Could not find classifier layer")

        # Embedding projector
        self.embedding_proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.normalize_embeddings = normalize_embeddings

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dict with 'logits' and 'embeddings'
        """
        features = self.backbone(x)

        embeddings = self.embedding_proj(features)

        if self.normalize_embeddings:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        logits = self.classifier(embeddings)

        return {
            "logits": logits,
            "embeddings": embeddings,
            "features": features,
        }


if __name__ == "__main__":
    # Test model creation
    print("Available models:")
    for name, info in MODEL_REGISTRY.items():
        print(f"  {name}: {info['params_m']:.1f}M params, {info['input_size']}px")

    # Test creating a model
    model, info = create_model("vit_base_patch16_224", num_classes=10)
    print(f"\nCreated: {info}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
