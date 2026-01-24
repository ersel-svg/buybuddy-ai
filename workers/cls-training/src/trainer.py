"""
SOTA Classification Trainer

Features:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Learning Rate Schedulers (Cosine, OneCycle, StepLR)
- EMA (Exponential Moving Average)
- Early Stopping
- Class Weighting
- MixUp / CutMix
- Gradient Clipping
- Progress Tracking
"""

import os
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    ReduceLROnPlateau,
)

from .augmentations import MixUpCutMix


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9  # for SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam

    # Learning rate schedule
    scheduler: str = "cosine"  # cosine, cosine_warmup, onecycle, step, plateau
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    step_size: int = 3  # for StepLR
    step_gamma: float = 0.5  # for StepLR

    # Mixed precision
    use_amp: bool = True

    # Gradient
    gradient_accumulation: int = 1
    gradient_clip: float = 1.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # MixUp / CutMix
    use_mixup: bool = False
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.5

    # Early stopping
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001

    # Class weighting
    use_class_weights: bool = False

    # Misc
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EMA:
    """
    Exponential Moving Average for model weights.

    Maintains a shadow copy of model weights that is updated
    with exponential moving average during training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",  # 'max' for accuracy, 'min' for loss
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ClassificationTrainer:
    """
    SOTA Classification Trainer with all bells and whistles.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler (will be set after dataloader is known)
        self.scheduler = None

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None

        # EMA
        self.ema = EMA(model, config.ema_decay) if config.use_ema else None

        # Early stopping
        self.early_stopping = (
            EarlyStopping(config.patience, config.min_delta)
            if config.early_stopping
            else None
        )

        # MixUp / CutMix
        self.mixup = None
        if config.use_mixup:
            self.mixup = MixUpCutMix(
                mixup_alpha=config.mixup_alpha,
                cutmix_alpha=config.cutmix_alpha,
                mixup_prob=config.mixup_prob,
                cutmix_prob=config.cutmix_prob,
                num_classes=num_classes,
            )

        # Metrics tracking
        self.history = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self, steps_per_epoch: int):
        """Create learning rate scheduler."""
        total_steps = steps_per_epoch * self.config.epochs

        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "cosine_warmup":
            warmup_steps = steps_per_epoch * self.config.warmup_epochs

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(
                    self.config.min_lr / self.config.learning_rate,
                    0.5 * (1 + math.cos(math.pi * progress)),
                )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        elif self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
            )
        elif self.config.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.step_size * steps_per_epoch,
                gamma=self.config.step_gamma,
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )
        else:
            self.scheduler = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        accumulation_steps = 0

        oom_count = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # MixUp / CutMix
                use_soft_targets = False
                if self.mixup is not None and self.model.training:
                    images, targets = self.mixup(images, targets)
                    use_soft_targets = True

                # Forward pass with AMP
                with autocast('cuda', enabled=self.config.use_amp):
                    outputs = self.model(images)

                    # Handle different output types
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs

                    # Compute loss
                    if use_soft_targets:
                        # Soft targets from MixUp/CutMix
                        loss = self._soft_cross_entropy(logits, targets)
                    else:
                        loss = self.loss_fn(logits, targets)

                    loss = loss / self.config.gradient_accumulation

                # NaN/Inf detection - stop early to avoid corrupted model
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(
                        f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. "
                        "Try reducing learning rate or using gradient clipping."
                    )

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    # Clear memory and skip this batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    if oom_count <= 3:
                        print(f"⚠️  OOM on batch {batch_idx}, skipping (count: {oom_count})")
                        continue
                    else:
                        print(f"❌ Too many OOM errors ({oom_count}), stopping training")
                        raise
                else:
                    raise

            accumulation_steps += 1

            # Optimizer step
            if accumulation_steps >= self.config.gradient_accumulation:
                if self.config.gradient_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                # Update scheduler (step-wise)
                if self.scheduler is not None and not isinstance(
                    self.scheduler, ReduceLROnPlateau
                ):
                    self.scheduler.step()

                accumulation_steps = 0

            # Metrics
            total_loss += loss.item() * self.config.gradient_accumulation

            if not use_soft_targets:
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                # For soft targets, use argmax
                _, predicted = logits.max(1)
                _, hard_targets = targets.max(1)
                total += hard_targets.size(0)
                correct += predicted.eq(hard_targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {
            "train_loss": avg_loss,
            "train_acc": accuracy,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _soft_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross entropy with soft targets."""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1)
        return loss.mean()

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """Validate the model."""
        # Use EMA weights if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Per-class metrics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for images, targets in val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = self.loss_fn(logits, targets)
            total_loss += loss.item()

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class accuracy
            for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1

        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # Per-class accuracy
        per_class_acc = {}
        for c in range(self.num_classes):
            if class_total[c] > 0:
                per_class_acc[self.class_names[c]] = (
                    100.0 * class_correct[c] / class_total[c]
                )

        return {
            "val_loss": avg_loss,
            "val_acc": accuracy,
            "per_class_acc": per_class_acc,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[str] = None,
        callback: Optional[Callable] = None,
        shutdown_checker: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            callback: Optional callback function(epoch, metrics)
            shutdown_checker: Optional function that returns True if shutdown requested

        Returns:
            Training history and best metrics
        """
        # Setup scheduler
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation
        self._create_scheduler(steps_per_epoch)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Check for shutdown request at start of each epoch
            if shutdown_checker and shutdown_checker():
                print(f"⚠️  Shutdown requested. Stopping training at epoch {epoch + 1}.")
                break

            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Memory cleanup between epochs to prevent gradual buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update plateau scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_acc"])

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}

            # Update history
            for key, value in metrics.items():
                if key != "per_class_acc":
                    self.history[key].append(value)

            # Track best model
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_epoch = epoch + 1

                if save_dir:
                    # Save FP16 inference-only checkpoint for best model
                    # (smaller file, uses EMA weights for better inference)
                    self._save_checkpoint(
                        os.path.join(save_dir, "best_model.pt"),
                        epoch,
                        val_metrics,
                        inference_only=True,  # FP16 + EMA weights
                    )
                    print(f"  New best model! Val Acc: {val_metrics['val_acc']:.2f}%")

            # Callback
            if callback:
                callback(epoch, metrics)

            # Logging
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} "
                f"| Train Loss: {train_metrics['train_loss']:.4f} "
                f"| Train Acc: {train_metrics['train_acc']:.2f}% "
                f"| Val Loss: {val_metrics['val_loss']:.4f} "
                f"| Val Acc: {val_metrics['val_acc']:.2f}% "
                f"| LR: {train_metrics['lr']:.2e} "
                f"| Time: {epoch_time:.1f}s"
            )

            # Early stopping
            if self.early_stopping and self.early_stopping(val_metrics["val_acc"]):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        total_time = time.time() - start_time

        # Final checkpoint (full for potential resume, not inference_only)
        if save_dir:
            self._save_checkpoint(
                os.path.join(save_dir, "final_model.pt"),
                epoch,
                val_metrics,
                inference_only=False,
            )

        return {
            "history": dict(self.history),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "final_metrics": metrics,
        }

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
        inference_only: bool = False,
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Validation metrics
            inference_only: If True, save FP16 inference-only checkpoint (smaller, faster)
                           Uses EMA weights if available for better inference quality.
        """
        if inference_only:
            # FP16 inference-only checkpoint (~50% smaller file size)
            # Use EMA weights if available (usually better for inference)
            if self.ema is not None:
                model_weights = self.ema.shadow
            else:
                model_weights = self.model.state_dict()

            # Convert to FP16 for smaller file size
            # Note: Some state_dict values might not be tensors (e.g., num_batches_tracked)
            def to_fp16(v):
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    return v.half()
                return v

            inference_checkpoint = {
                "model_state_dict": {
                    k: to_fp16(v)
                    for k, v in model_weights.items()
                },
                "epoch": epoch,
                "metrics": metrics,
                "config": self.config.to_dict(),
                "precision": "fp16",
                "inference_only": True,
                "used_ema": self.ema is not None,
            }
            torch.save(inference_checkpoint, path)

            # Log file size
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Saved FP16 inference checkpoint: {file_size_mb:.1f} MB (EMA: {self.ema is not None})")
        else:
            # Full checkpoint for resuming training
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config.to_dict(),
            }

            if self.ema is not None:
                checkpoint["ema_shadow"] = self.ema.shadow

            torch.save(checkpoint, path)


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = "inverse",
) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Args:
        labels: List of class labels
        num_classes: Number of classes
        method: 'inverse', 'inverse_sqrt', or 'effective_samples'

    Returns:
        Class weight tensor
    """
    import numpy as np
    from collections import Counter

    counter = Counter(labels)
    counts = [counter.get(i, 0) for i in range(num_classes)]
    counts = np.array(counts, dtype=float)
    counts = np.maximum(counts, 1)  # Avoid division by zero

    if method == "inverse":
        weights = 1.0 / counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif method == "effective_samples":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Quick test
    print("Trainer module loaded successfully")
    print(f"Config example: {TrainingConfig()}")
