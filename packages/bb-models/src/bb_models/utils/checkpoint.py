"""
Checkpoint management utilities for model training.

Handles:
- Saving/loading checkpoints with metadata
- Automatic best checkpoint tracking
- Checkpoint cleanup based on retention policy
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    epoch: int
    step: Optional[int]
    train_loss: float
    val_loss: Optional[float]
    val_recall_at_1: Optional[float]
    val_recall_at_5: Optional[float]
    is_best: bool
    is_final: bool
    model_id: str
    timestamp: str
    config: Dict[str, Any]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Union[str, Path],
    train_loss: float,
    val_loss: Optional[float] = None,
    val_recall_at_1: Optional[float] = None,
    val_recall_at_5: Optional[float] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    is_best: bool = False,
    is_final: bool = False,
    model_id: str = "unknown",
    config: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        path: Save path.
        train_loss: Training loss at this checkpoint.
        val_loss: Validation loss (if available).
        val_recall_at_1: Validation recall@1 metric.
        val_recall_at_5: Validation recall@5 metric.
        scheduler: Learning rate scheduler (optional).
        scaler: Gradient scaler for mixed precision (optional).
        is_best: Whether this is the best checkpoint so far.
        is_final: Whether this is the final epoch.
        model_id: Model identifier.
        config: Training configuration dict.
        extra_state: Additional state to save.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_recall_at_1": val_recall_at_1,
        "val_recall_at_5": val_recall_at_5,
        "is_best": is_best,
        "is_final": is_final,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
    }

    # Add scheduler state
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Add scaler state (for mixed precision)
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Add extra state
    if extra_state:
        checkpoint.update(extra_state)

    # Save checkpoint
    torch.save(checkpoint, path)

    # Save metadata as JSON (for easy inspection)
    metadata = CheckpointMetadata(
        epoch=epoch,
        step=extra_state.get("step") if extra_state else None,
        train_loss=train_loss,
        val_loss=val_loss,
        val_recall_at_1=val_recall_at_1,
        val_recall_at_5=val_recall_at_5,
        is_best=is_best,
        is_final=is_final,
        model_id=model_id,
        timestamp=checkpoint["timestamp"],
        config=config or {},
    )
    metadata_path = path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load a checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into (optional).
        optimizer: Optimizer to restore state (optional).
        scheduler: Scheduler to restore state (optional).
        scaler: Gradient scaler to restore state (optional).
        device: Device to load tensors to.
        strict: Whether to strictly enforce state dict keys.

    Returns:
        Checkpoint dict with all saved data.
    """
    path = Path(path)

    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load model weights
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


class CheckpointManager:
    """
    Manages checkpoint saving with automatic cleanup.

    Retention policy:
    - Always keep best checkpoint
    - Always keep final checkpoint
    - Keep last N checkpoints
    - Delete older checkpoints automatically
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        keep_best: bool = True,
        keep_final: bool = True,
        keep_last_n: int = 3,
        model_id: str = "model",
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints.
            keep_best: Always keep the best checkpoint.
            keep_final: Always keep the final checkpoint.
            keep_last_n: Number of recent checkpoints to keep.
            model_id: Model identifier for naming.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keep_best = keep_best
        self.keep_final = keep_final
        self.keep_last_n = keep_last_n
        self.model_id = model_id

        # Track checkpoints
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: Optional[float] = None

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_recall_at_1: Optional[float] = None,
        val_recall_at_5: Optional[float] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        is_final: bool = False,
        config: Optional[Dict[str, Any]] = None,
        metric_for_best: str = "val_loss",
        minimize_metric: bool = True,
    ) -> Path:
        """
        Save checkpoint and manage retention.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            train_loss: Training loss.
            val_loss: Validation loss.
            val_recall_at_1: Recall@1 metric.
            val_recall_at_5: Recall@5 metric.
            scheduler: LR scheduler.
            scaler: Gradient scaler.
            is_final: Whether this is the final epoch.
            config: Training config.
            metric_for_best: Which metric determines "best".
            minimize_metric: Whether lower is better for the metric.

        Returns:
            Path to saved checkpoint.
        """
        # Determine if this is the best checkpoint
        current_metric = val_loss if metric_for_best == "val_loss" else val_recall_at_1
        is_best = False

        if current_metric is not None:
            if self.best_metric is None:
                is_best = True
            elif minimize_metric and current_metric < self.best_metric:
                is_best = True
            elif not minimize_metric and current_metric > self.best_metric:
                is_best = True

        # Save checkpoint
        checkpoint_name = f"{self.model_id}_epoch{epoch:03d}.pth"
        checkpoint_path = self.output_dir / checkpoint_name

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=checkpoint_path,
            train_loss=train_loss,
            val_loss=val_loss,
            val_recall_at_1=val_recall_at_1,
            val_recall_at_5=val_recall_at_5,
            scheduler=scheduler,
            scaler=scaler,
            is_best=is_best,
            is_final=is_final,
            model_id=self.model_id,
            config=config,
        )

        # Track this checkpoint
        self.checkpoints.append({
            "path": checkpoint_path,
            "epoch": epoch,
            "metric": current_metric,
            "is_best": is_best,
            "is_final": is_final,
        })

        # Update best checkpoint
        if is_best:
            self.best_metric = current_metric
            self.best_checkpoint = checkpoint_path

            # Also save as "best"
            best_path = self.output_dir / f"{self.model_id}_best.pth"
            torch.save(torch.load(checkpoint_path, weights_only=False), best_path)

        # Save as "last" for easy resumption
        last_path = self.output_dir / f"{self.model_id}_last.pth"
        torch.save(torch.load(checkpoint_path, weights_only=False), last_path)

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_path

    def _cleanup(self):
        """Remove old checkpoints according to retention policy."""
        if len(self.checkpoints) <= self.keep_last_n:
            return

        # Identify protected checkpoints
        protected = set()

        if self.keep_best and self.best_checkpoint:
            protected.add(str(self.best_checkpoint))

        # Keep final checkpoints
        if self.keep_final:
            for cp in self.checkpoints:
                if cp["is_final"]:
                    protected.add(str(cp["path"]))

        # Keep last N
        recent = self.checkpoints[-self.keep_last_n:]
        for cp in recent:
            protected.add(str(cp["path"]))

        # Delete unprotected checkpoints
        for cp in self.checkpoints[:-self.keep_last_n]:
            if str(cp["path"]) not in protected:
                try:
                    cp["path"].unlink(missing_ok=True)
                    # Also delete metadata JSON
                    cp["path"].with_suffix(".json").unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete checkpoint {cp['path']}: {e}")

        # Update checkpoint list
        self.checkpoints = [
            cp for cp in self.checkpoints
            if cp["path"].exists() or str(cp["path"]) in protected
        ]

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        return self.best_checkpoint

    def get_last_checkpoint(self) -> Optional[Path]:
        """Get path to the most recent checkpoint."""
        if self.checkpoints:
            return self.checkpoints[-1]["path"]
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all managed checkpoints."""
        return self.checkpoints.copy()
