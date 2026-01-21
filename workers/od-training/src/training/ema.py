"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a shadow copy of model weights that is updated as:
    θ_ema = decay × θ_ema + (1 - decay) × θ_current

This provides a smoother version of the model that typically performs
better at inference time (+0.5-2% mAP improvement).

Usage:
    ema = ModelEMA(model, decay=0.9999)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        ema.update(model)  # Update EMA after each step

    # For validation, use EMA weights:
    ema.apply_shadow(model)
    metrics = validate(model)
    ema.restore(model)

    # For saving, save EMA weights:
    torch.save(ema.state_dict(), "model_ema.pt")
"""

import math
from copy import deepcopy
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    Keeps a shadow copy of model parameters that is smoothly updated
    during training. The EMA model typically generalizes better than
    the trained model.

    Args:
        model: The model to track
        decay: EMA decay factor (default: 0.9999)
        warmup_steps: Number of steps for decay warmup (default: 2000)
        device: Device for EMA weights (default: same as model)

    Example:
        >>> model = MyModel()
        >>> ema = ModelEMA(model, decay=0.9999)
        >>> for step, batch in enumerate(dataloader):
        ...     loss = model(batch)
        ...     optimizer.step()
        ...     ema.update(model, step)
        >>> # Use EMA for inference
        >>> ema.apply_shadow(model)
        >>> predictions = model(test_data)
        >>> ema.restore(model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

        # Create shadow model (deep copy of parameters)
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

        # Move to device if specified
        if device is not None:
            self.shadow.to(device)

        # Store original parameters for restore
        self.backup: Dict[str, torch.Tensor] = {}

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Get current decay value with optional warmup.

        During warmup, decay increases from 0 to target decay:
            decay = min(target_decay, (1 + step) / (warmup_steps + step))

        Args:
            step: Current training step (uses self.updates if None)

        Returns:
            Current decay value
        """
        if step is None:
            step = self.updates

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Warmup: decay increases from ~0.1 to target
            return min(self.decay, (1 + step) / (self.warmup_steps + step))

        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> None:
        """
        Update EMA weights.

        θ_ema = decay × θ_ema + (1 - decay) × θ_current

        Args:
            model: Current model with updated weights
            step: Current step for decay warmup (optional)
        """
        if step is not None:
            self.updates = step
        else:
            self.updates += 1

        decay = self.get_decay(self.updates)

        # Update shadow parameters
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, shadow_param in shadow_params.items():
            if name in model_params:
                model_param = model_params[name]
                # EMA update: shadow = decay * shadow + (1 - decay) * current
                shadow_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

        # Also update buffers (e.g., BatchNorm running stats)
        model_buffers = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())

        for name, shadow_buffer in shadow_buffers.items():
            if name in model_buffers:
                model_buffer = model_buffers[name]
                shadow_buffer.data.copy_(model_buffer.data)

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Apply EMA weights to model (for inference/validation).

        Saves current model weights to backup before applying shadow.
        Call restore() to revert to original weights.

        Args:
            model: Model to apply EMA weights to
        """
        # Backup current parameters
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.data.clone()

        # Apply shadow parameters
        shadow_params = dict(self.shadow.named_parameters())
        for name, param in model.named_parameters():
            if name in shadow_params:
                param.data.copy_(shadow_params[name].data)

    def restore(self, model: nn.Module) -> None:
        """
        Restore original weights to model (after inference/validation).

        Args:
            model: Model to restore original weights to
        """
        if not self.backup:
            return

        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        """
        Get EMA state dict for checkpointing.

        Returns:
            State dict containing shadow model and metadata
        """
        return {
            "shadow": self.shadow.state_dict(),
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load EMA state from checkpoint.

        Args:
            state_dict: State dict from state_dict()
        """
        self.shadow.load_state_dict(state_dict["shadow"])
        self.decay = state_dict.get("decay", self.decay)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.updates = state_dict.get("updates", 0)

    def to(self, device: torch.device) -> "ModelEMA":
        """Move EMA model to device."""
        self.shadow.to(device)
        return self

    def eval(self) -> nn.Module:
        """Get EMA model in eval mode for direct inference."""
        self.shadow.eval()
        return self.shadow

    @property
    def module(self) -> nn.Module:
        """Direct access to shadow model."""
        return self.shadow


def copy_model_weights(src: nn.Module, dst: nn.Module) -> None:
    """
    Copy weights from source model to destination model.

    Args:
        src: Source model
        dst: Destination model
    """
    src_params = dict(src.named_parameters())
    for name, param in dst.named_parameters():
        if name in src_params:
            param.data.copy_(src_params[name].data)
