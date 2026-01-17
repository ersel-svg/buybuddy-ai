"""
Layer-wise Learning Rate Decay (LLRD) utilities.

LLRD applies different learning rates to different layers of the model,
with earlier layers having lower learning rates. This helps with:
- Preserving pretrained features in early layers
- Allowing later layers to adapt more quickly
- Preventing catastrophic forgetting

Reference:
    Universal Language Model Fine-tuning for Text Classification
    https://arxiv.org/abs/1801.06146
"""

from typing import List, Dict, Any, Optional, Iterator
import torch
import torch.nn as nn


def get_llrd_optimizer_params(
    model: nn.Module,
    base_lr: float,
    llrd_decay: float = 0.9,
    weight_decay: float = 0.01,
    no_decay_keywords: Optional[List[str]] = None,
    head_lr_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Get optimizer parameter groups with layer-wise learning rate decay.

    Args:
        model: Model with backbone and optional head layers.
        base_lr: Base learning rate for the last layer.
        llrd_decay: Decay factor for each layer going backwards.
                   0.9 means each earlier layer has 0.9x the LR.
        weight_decay: Weight decay for parameters (except biases/norms).
        no_decay_keywords: Parameter name patterns that should have no weight decay.
        head_lr_multiplier: LR multiplier for head parameters.

    Returns:
        List of parameter group dicts for optimizer.

    Example:
        >>> params = get_llrd_optimizer_params(model, base_lr=3e-5, llrd_decay=0.9)
        >>> optimizer = torch.optim.AdamW(params, lr=3e-5)
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layernorm"]

    # Try to get layer groups from the model
    layer_groups = _get_layer_groups(model)

    if not layer_groups:
        # Fallback: simple parameter grouping without LLRD
        print("Warning: Could not determine layer groups for LLRD, using uniform LR")
        return _get_simple_params(model, base_lr, weight_decay, no_decay_keywords)

    num_layers = len(layer_groups)
    optimizer_params = []

    # Process backbone layers with LLRD
    for layer_idx, params in enumerate(layer_groups):
        # Earlier layers get lower LR
        layer_lr = base_lr * (llrd_decay ** (num_layers - layer_idx - 1))

        # Separate params with/without weight decay
        decay_params = []
        no_decay_params = []

        for param in params:
            if param.requires_grad:
                # Check if this param should have no weight decay
                param_name = _get_param_name(model, param)
                if any(nd in param_name for nd in no_decay_keywords):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        if decay_params:
            optimizer_params.append({
                "params": decay_params,
                "lr": layer_lr,
                "weight_decay": weight_decay,
                "layer_idx": layer_idx,
            })

        if no_decay_params:
            optimizer_params.append({
                "params": no_decay_params,
                "lr": layer_lr,
                "weight_decay": 0.0,
                "layer_idx": layer_idx,
            })

    # Head parameters (projection, classifier, etc.) at base_lr * multiplier
    head_params = _get_head_params(model, layer_groups)
    head_lr = base_lr * head_lr_multiplier

    if head_params:
        decay_params = []
        no_decay_params = []

        for param in head_params:
            if param.requires_grad:
                param_name = _get_param_name(model, param)
                if any(nd in param_name for nd in no_decay_keywords):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        if decay_params:
            optimizer_params.append({
                "params": decay_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "layer_idx": "head",
            })

        if no_decay_params:
            optimizer_params.append({
                "params": no_decay_params,
                "lr": head_lr,
                "weight_decay": 0.0,
                "layer_idx": "head",
            })

    return optimizer_params


def _get_layer_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """
    Extract layer groups from the model.

    Tries different model architectures (DINOv2, CLIP, etc.).
    """
    groups = []

    # Check if model has a get_layer_groups method
    if hasattr(model, "get_layer_groups"):
        return model.get_layer_groups()

    # Check for backbone attribute
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "get_layer_groups"):
        return backbone.get_layer_groups()

    # Try common architectures
    # DINOv2/ViT-style: embeddings + encoder.layers
    embeddings = getattr(model, "embeddings", None)
    if embeddings is None and backbone is not None:
        embeddings = getattr(backbone, "embeddings", None)

    encoder = getattr(model, "encoder", None)
    if encoder is None and backbone is not None:
        encoder = getattr(backbone, "encoder", None)

    if embeddings is not None:
        groups.append(list(embeddings.parameters()))

    if encoder is not None:
        layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
        if layers is not None:
            for layer in layers:
                groups.append(list(layer.parameters()))

    # Final layer norm
    layernorm = getattr(model, "layernorm", None)
    if layernorm is None and backbone is not None:
        layernorm = getattr(backbone, "layernorm", None)
    if layernorm is not None:
        groups.append(list(layernorm.parameters()))

    return groups


def _get_head_params(
    model: nn.Module,
    backbone_groups: List[List[nn.Parameter]],
) -> List[nn.Parameter]:
    """Get parameters that are not in the backbone groups."""
    backbone_params = set()
    for group in backbone_groups:
        for param in group:
            backbone_params.add(id(param))

    head_params = []
    for param in model.parameters():
        if id(param) not in backbone_params:
            head_params.append(param)

    return head_params


def _get_param_name(model: nn.Module, param: nn.Parameter) -> str:
    """Get the name of a parameter in the model."""
    for name, p in model.named_parameters():
        if p is param:
            return name
    return ""


def _get_simple_params(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    no_decay_keywords: List[str],
) -> List[Dict[str, Any]]:
    """Fallback: simple parameter groups without LLRD."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    params = []
    if decay_params:
        params.append({
            "params": decay_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })
    if no_decay_params:
        params.append({
            "params": no_decay_params,
            "lr": lr,
            "weight_decay": 0.0,
        })

    return params


def apply_llrd(
    optimizer: torch.optim.Optimizer,
    decay_factor: float,
) -> None:
    """
    Apply LLRD decay to an existing optimizer.

    Reduces learning rate for each parameter group based on its layer_idx.

    Args:
        optimizer: Optimizer with parameter groups.
        decay_factor: Factor to multiply LR by for each layer going back.
    """
    # Find max layer index
    max_layer_idx = 0
    for group in optimizer.param_groups:
        layer_idx = group.get("layer_idx", 0)
        if isinstance(layer_idx, int) and layer_idx > max_layer_idx:
            max_layer_idx = layer_idx

    # Apply decay
    for group in optimizer.param_groups:
        layer_idx = group.get("layer_idx", max_layer_idx)
        if isinstance(layer_idx, int):
            group["lr"] = group["lr"] * (decay_factor ** (max_layer_idx - layer_idx))


def print_llrd_info(optimizer: torch.optim.Optimizer) -> None:
    """Print learning rates for each parameter group."""
    print("LLRD Parameter Groups:")
    for i, group in enumerate(optimizer.param_groups):
        layer_idx = group.get("layer_idx", "unknown")
        lr = group["lr"]
        wd = group["weight_decay"]
        num_params = sum(p.numel() for p in group["params"])
        print(f"  Group {i}: layer={layer_idx}, lr={lr:.2e}, wd={wd}, params={num_params:,}")
