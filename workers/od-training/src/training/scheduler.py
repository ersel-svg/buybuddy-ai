"""
Learning Rate Schedulers for SOTA OD Training.

Implements:
- Warmup + Cosine Annealing (recommended)
- Warmup + Step Decay
- Warmup + Linear Decay

Formula (Warmup + Cosine):
    Warmup:  lr = base_lr × (step / warmup_steps)
    Cosine:  lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))

Usage:
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=3,
        total_epochs=100,
        steps_per_epoch=500,
        min_lr_ratio=0.01,
    )

    for epoch in range(epochs):
        for batch in dataloader:
            loss = model(batch)
            optimizer.step()
            scheduler.step()  # Per-step update!

        print(f"LR: {scheduler.get_last_lr()}")
"""

import math
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.

    During warmup:
        lr = base_lr × (current_step / warmup_steps)

    After warmup (cosine annealing):
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of base LR (default: 0.01)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Store base LRs (will be set in parent __init__)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch  # Actually current step

        if step < self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            # Clamp progress to [0, 1]
            progress = min(1.0, max(0.0, progress))

            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            return [
                base_lr * self.min_lr_ratio +
                (base_lr - base_lr * self.min_lr_ratio) * cosine_factor
                for base_lr in self.base_lrs
            ]


class WarmupStepScheduler(LRScheduler):
    """
    Learning rate scheduler with warmup and step decay.

    Drops LR by gamma at specified milestones.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        milestones: Steps at which to decay LR
        gamma: Decay factor (default: 0.1)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Step decay
            decay_factor = 1.0
            for milestone in self.milestones:
                if step >= milestone:
                    decay_factor *= self.gamma

            return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupLinearScheduler(LRScheduler):
    """
    Learning rate scheduler with warmup and linear decay.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum LR ratio (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps and self.warmup_steps > 0:
            # Linear warmup
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)

            decay_factor = 1.0 - progress * (1.0 - self.min_lr_ratio)

            return [base_lr * decay_factor for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int = 3,
    total_epochs: int = 100,
    steps_per_epoch: int = 500,
    scheduler_type: str = "cosine",
    min_lr_ratio: float = 0.01,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
) -> LRScheduler:
    """
    Build learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        steps_per_epoch: Number of steps per epoch
        scheduler_type: "cosine", "step", or "linear"
        min_lr_ratio: Minimum LR ratio for cosine/linear
        milestones: Epoch milestones for step scheduler
        gamma: Decay factor for step scheduler

    Returns:
        Configured LR scheduler
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    print(f"Building {scheduler_type} scheduler:")
    print(f"  Warmup: {warmup_epochs} epochs ({warmup_steps} steps)")
    print(f"  Total: {total_epochs} epochs ({total_steps} steps)")

    if scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
        print(f"  Min LR ratio: {min_lr_ratio}")

    elif scheduler_type == "step":
        if milestones is None:
            # Default: decay at 60% and 80% of training
            milestones = [
                int(0.6 * total_steps),
                int(0.8 * total_steps),
            ]
        scheduler = WarmupStepScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            milestones=milestones,
            gamma=gamma,
        )
        print(f"  Milestones: {milestones}")
        print(f"  Gamma: {gamma}")

    elif scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
        print(f"  Min LR ratio: {min_lr_ratio}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def get_lr_at_step(
    scheduler_type: str,
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.01,
) -> float:
    """
    Calculate LR at a specific step (for visualization/debugging).

    Args:
        scheduler_type: "cosine", "linear"
        step: Current step
        warmup_steps: Warmup steps
        total_steps: Total steps
        base_lr: Base learning rate
        min_lr_ratio: Min LR ratio

    Returns:
        Learning rate at step
    """
    if step < warmup_steps and warmup_steps > 0:
        return base_lr * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    if scheduler_type == "cosine":
        min_lr = base_lr * min_lr_ratio
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    elif scheduler_type == "linear":
        return base_lr * (1.0 - progress * (1.0 - min_lr_ratio))
    else:
        return base_lr
