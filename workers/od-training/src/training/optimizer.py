"""
Layer-wise Learning Rate Decay (LLRD) Optimizer Builder.

LLRD applies different learning rates to different layers:
- Deep layers (pretrained): Lower LR to preserve learned features
- Shallow layers (new): Higher LR to learn task-specific features

Formula:
    lr(layer_i) = base_lr × decay^(num_layers - i - 1)

For object detection:
- Backbone layers: Decaying LR (0.1x to 1x)
- Encoder/Decoder: Base LR
- Detection Head: 10x Base LR

Usage:
    optimizer = build_llrd_optimizer(
        model,
        base_lr=0.0001,
        weight_decay=0.0001,
        llrd_decay=0.9,
        head_lr_factor=10.0,
    )
"""

import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD


def get_num_layers(model_type: str) -> int:
    """Get number of layers for LLRD based on model type."""
    layer_counts = {
        "rt-detr": 12,      # ResNet stages + encoder + decoder
        "rt-detr-r18": 8,
        "rt-detr-r50": 12,
        "rt-detr-r101": 16,
        "d-fine": 12,
        "d-fine-s": 8,
        "d-fine-m": 10,
        "d-fine-l": 12,
        "d-fine-x": 14,
        "rf-detr": 12,
        "detr": 12,
        "default": 12,
    }
    return layer_counts.get(model_type, layer_counts["default"])


def get_layer_id_for_rtdetr(name: str, num_layers: int) -> int:
    """
    Get layer ID for RT-DETR model parameter.

    Layer assignment:
    - backbone.conv1, bn1: 0
    - backbone.layer1: 1
    - backbone.layer2: 2
    - backbone.layer3: 3
    - backbone.layer4: 4
    - encoder: 5-8
    - decoder: 9-11
    - head (class_embed, bbox_embed): num_layers (highest LR)

    Args:
        name: Parameter name
        num_layers: Total number of layers

    Returns:
        Layer ID (0 = deepest/lowest LR, num_layers = head/highest LR)
    """
    # Detection head - highest LR
    if any(x in name for x in ["class_embed", "bbox_embed", "query_embed", "label_enc"]):
        return num_layers

    # Backbone stem
    if "backbone" in name:
        if any(x in name for x in ["conv1", "bn1", "layer1"]):
            return 0
        elif "layer2" in name:
            return 1
        elif "layer3" in name:
            return 2
        elif "layer4" in name:
            return 3
        else:
            return 0

    # Encoder layers
    if "encoder" in name:
        # Try to extract layer number
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_num = int(match.group(1))
            return min(4 + layer_num, num_layers - 4)
        return 5

    # Decoder layers
    if "decoder" in name:
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_num = int(match.group(1))
            return min(num_layers - 3 + layer_num, num_layers - 1)
        return num_layers - 2

    # Input projection, position embedding, etc.
    if any(x in name for x in ["input_proj", "position", "level_embed"]):
        return 4

    # Default to middle layer
    return num_layers // 2


def get_layer_id_for_dfine(name: str, num_layers: int) -> int:
    """
    Get layer ID for D-FINE model parameter.

    Similar structure to RT-DETR with some differences.
    """
    # Detection head
    if any(x in name for x in ["class_head", "bbox_head", "reg_head", "cls_head"]):
        return num_layers

    # Backbone
    if "backbone" in name:
        if "stem" in name or "layer1" in name or "stage1" in name:
            return 0
        elif "layer2" in name or "stage2" in name:
            return 1
        elif "layer3" in name or "stage3" in name:
            return 2
        elif "layer4" in name or "stage4" in name:
            return 3
        return 0

    # Encoder
    if "encoder" in name:
        match = re.search(r"(\d+)", name)
        if match:
            return min(4 + int(match.group(1)), num_layers - 4)
        return 5

    # Decoder
    if "decoder" in name:
        match = re.search(r"(\d+)", name)
        if match:
            return min(num_layers - 3 + int(match.group(1)), num_layers - 1)
        return num_layers - 2

    # Neck, FPN
    if any(x in name for x in ["neck", "fpn", "pan"]):
        return num_layers - 3

    return num_layers // 2


def get_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.0001,
    llrd_decay: float = 0.9,
    head_lr_factor: float = 10.0,
    model_type: str = "rt-detr",
    no_weight_decay_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with layer-wise learning rate decay.

    Args:
        model: Model to create param groups for
        base_lr: Base learning rate
        weight_decay: Weight decay for regularization
        llrd_decay: Decay factor per layer (0.9 = 10% reduction per layer)
        head_lr_factor: LR multiplier for detection head
        model_type: Model type for layer assignment
        no_weight_decay_names: Parameter names to exclude from weight decay

    Returns:
        List of parameter groups for optimizer
    """
    if no_weight_decay_names is None:
        no_weight_decay_names = ["bias", "LayerNorm", "layer_norm", "bn", "norm"]

    num_layers = get_num_layers(model_type)

    # Choose layer ID function based on model type
    if "dfine" in model_type.lower() or "d-fine" in model_type.lower():
        get_layer_id = lambda name: get_layer_id_for_dfine(name, num_layers)
    else:
        get_layer_id = lambda name: get_layer_id_for_rtdetr(name, num_layers)

    # Group parameters by layer ID and weight decay
    param_groups: Dict[Tuple[int, bool], List[torch.nn.Parameter]] = {}
    param_group_names: Dict[Tuple[int, bool], List[str]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer ID
        layer_id = get_layer_id(name)

        # Determine if weight decay should be applied
        apply_wd = not any(nd in name for nd in no_weight_decay_names)

        key = (layer_id, apply_wd)

        if key not in param_groups:
            param_groups[key] = []
            param_group_names[key] = []

        param_groups[key].append(param)
        param_group_names[key].append(name)

    # Create optimizer param groups
    optimizer_groups = []

    for (layer_id, apply_wd), params in param_groups.items():
        # Calculate learning rate for this layer
        if layer_id == num_layers:
            # Detection head: highest LR
            lr = base_lr * head_lr_factor
        else:
            # LLRD: lr = base_lr × decay^(num_layers - layer_id - 1)
            lr = base_lr * (llrd_decay ** (num_layers - layer_id - 1))

        group = {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay if apply_wd else 0.0,
            "layer_id": layer_id,
            "param_names": param_group_names[(layer_id, apply_wd)],
        }
        optimizer_groups.append(group)

    # Sort by layer ID for logging
    optimizer_groups.sort(key=lambda x: x["layer_id"])

    # Log parameter groups
    print(f"LLRD Parameter Groups (base_lr={base_lr}, decay={llrd_decay}):")
    for group in optimizer_groups:
        print(f"  Layer {group['layer_id']:2d}: lr={group['lr']:.6f}, "
              f"wd={group['weight_decay']:.4f}, params={len(group['params'])}")

    return optimizer_groups


def build_llrd_optimizer(
    model: nn.Module,
    base_lr: float = 0.0001,
    weight_decay: float = 0.0001,
    llrd_decay: float = 0.9,
    head_lr_factor: float = 10.0,
    model_type: str = "rt-detr",
    optimizer_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Build optimizer with layer-wise learning rate decay.

    Args:
        model: Model to optimize
        base_lr: Base learning rate
        weight_decay: Weight decay
        llrd_decay: LLRD decay factor
        head_lr_factor: LR multiplier for head
        model_type: Model type
        optimizer_type: "adamw" or "sgd"
        betas: AdamW betas
        eps: AdamW epsilon
        momentum: SGD momentum

    Returns:
        Configured optimizer
    """
    param_groups = get_param_groups(
        model=model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        llrd_decay=llrd_decay,
        head_lr_factor=head_lr_factor,
        model_type=model_type,
    )

    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(param_groups, betas=betas, eps=eps)
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(param_groups, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def build_simple_optimizer(
    model: nn.Module,
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    optimizer_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Build simple optimizer without LLRD (for comparison/fallback).

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_type: "adamw" or "sgd"
        betas: AdamW betas

    Returns:
        Configured optimizer
    """
    # Separate params with/without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in ["bias", "LayerNorm", "layer_norm", "bn", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(param_groups, lr=lr, betas=betas)
    else:
        optimizer = SGD(param_groups, lr=lr, momentum=0.9, nesterov=True)

    return optimizer
