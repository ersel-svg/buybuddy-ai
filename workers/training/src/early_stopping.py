"""
Early Stopping Module for Training.

Provides configurable early stopping to prevent overfitting.
"""

from typing import Optional, Callable
import json


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training if no improvement
    is seen for a specified number of epochs (patience).

    Usage:
        early_stopping = EarlyStopping(patience=5, mode='max')
        for epoch in range(epochs):
            train_metrics = train_epoch()
            val_metrics = validate()

            if early_stopping(val_metrics['recall@1']):
                print(f"Early stopping at epoch {epoch}")
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
        restore_best: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for metrics like recall
            verbose: Print messages when stopping
            restore_best: Track best value for restoration
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best = restore_best

        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.should_stop = False

    def __call__(self, value: float, epoch: Optional[int] = None) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value
            epoch: Current epoch number (for tracking)

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        if self._is_improvement(value):
            if self.verbose:
                improvement = value - self.best_value if self.mode == "max" else self.best_value - value
                print(f"EarlyStopping: Improvement of {improvement:.6f} detected")

            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training. Best value: {self.best_value:.6f} at epoch {self.best_epoch}")

        return self.should_stop

    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over best."""
        if self.mode == "max":
            return value > (self.best_value + self.min_delta)
        else:
            return value < (self.best_value - self.min_delta)

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.best_epoch = None
        self.should_stop = False

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "counter": self.counter,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.counter = state.get("counter", 0)
        self.best_value = state.get("best_value")
        self.best_epoch = state.get("best_epoch")
        self.should_stop = state.get("should_stop", False)


class MetricTracker:
    """
    Track multiple metrics during training.

    Provides:
    - Best value tracking for each metric
    - Moving average computation
    - Improvement detection
    """

    def __init__(self, metrics: list[str], modes: Optional[dict[str, str]] = None):
        """
        Args:
            metrics: List of metric names to track
            modes: Dict of metric_name -> 'min' or 'max'
        """
        self.metrics = metrics
        self.modes = modes or {}

        # Default modes
        for m in metrics:
            if m not in self.modes:
                # Assume 'loss' should be minimized, others maximized
                self.modes[m] = "min" if "loss" in m.lower() else "max"

        self.history = {m: [] for m in metrics}
        self.best_values = {m: None for m in metrics}
        self.best_epochs = {m: None for m in metrics}

    def update(self, values: dict, epoch: int):
        """
        Update metrics with new values.

        Args:
            values: Dict of metric_name -> value
            epoch: Current epoch
        """
        for name, value in values.items():
            if name in self.metrics:
                self.history[name].append(value)

                # Update best
                if self.best_values[name] is None:
                    self.best_values[name] = value
                    self.best_epochs[name] = epoch
                elif self._is_better(name, value, self.best_values[name]):
                    self.best_values[name] = value
                    self.best_epochs[name] = epoch

    def _is_better(self, metric: str, value: float, best: float) -> bool:
        """Check if value is better than best."""
        if self.modes[metric] == "max":
            return value > best
        return value < best

    def get_best(self, metric: str) -> tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best_values[metric], self.best_epochs[metric]

    def get_moving_average(self, metric: str, window: int = 5) -> float:
        """Get moving average of a metric."""
        if metric not in self.history or not self.history[metric]:
            return 0.0

        values = self.history[metric]
        if len(values) < window:
            return sum(values) / len(values)

        return sum(values[-window:]) / window

    def improved(self, metric: str, threshold: float = 0.0) -> bool:
        """Check if metric improved in last update."""
        if metric not in self.history or len(self.history[metric]) < 2:
            return True

        current = self.history[metric][-1]
        previous = self.history[metric][-2]

        if self.modes[metric] == "max":
            return current > (previous + threshold)
        return current < (previous - threshold)

    def summary(self) -> dict:
        """Get summary of all metrics."""
        return {
            m: {
                "best": self.best_values[m],
                "best_epoch": self.best_epochs[m],
                "last": self.history[m][-1] if self.history[m] else None,
                "moving_avg": self.get_moving_average(m),
            }
            for m in self.metrics
        }


class CurriculumScheduler:
    """
    Scheduler for curriculum learning phases.

    Automatically transitions between phases based on epoch.
    """

    def __init__(
        self,
        warmup_epochs: int = 2,
        easy_epochs: int = 5,
        hard_epochs: int = 10,
        finetune_epochs: int = 3,
    ):
        """
        Args:
            warmup_epochs: Epochs for warmup phase
            easy_epochs: Epochs for easy phase
            hard_epochs: Epochs for hard phase
            finetune_epochs: Epochs for finetune phase
        """
        self.warmup_epochs = warmup_epochs
        self.easy_epochs = easy_epochs
        self.hard_epochs = hard_epochs
        self.finetune_epochs = finetune_epochs

        self.total_epochs = warmup_epochs + easy_epochs + hard_epochs + finetune_epochs
        self.current_phase = "warmup"

    def get_phase(self, epoch: int) -> str:
        """Get curriculum phase for given epoch."""
        if epoch < self.warmup_epochs:
            return "warmup"
        elif epoch < self.warmup_epochs + self.easy_epochs:
            return "easy"
        elif epoch < self.warmup_epochs + self.easy_epochs + self.hard_epochs:
            return "hard"
        else:
            return "finetune"

    def update(self, epoch: int) -> str:
        """Update phase based on epoch and return current phase."""
        new_phase = self.get_phase(epoch)
        if new_phase != self.current_phase:
            print(f"Curriculum: Transitioning from {self.current_phase} to {new_phase}")
            self.current_phase = new_phase
        return self.current_phase

    def get_loss_weights(self) -> dict:
        """Get recommended loss weights for current phase."""
        weights = {
            "warmup": {"arcface": 1.0, "triplet": 0.0, "domain": 0.0},
            "easy": {"arcface": 1.0, "triplet": 0.3, "domain": 0.05},
            "hard": {"arcface": 1.0, "triplet": 0.5, "domain": 0.1},
            "finetune": {"arcface": 0.5, "triplet": 0.8, "domain": 0.15},
        }
        return weights.get(self.current_phase, weights["hard"])

    def get_mining_ratio(self) -> dict:
        """Get hard negative mining ratios for current phase."""
        ratios = {
            "warmup": {"hard": 0.0, "semi_hard": 0.2, "random": 0.8},
            "easy": {"hard": 0.2, "semi_hard": 0.3, "random": 0.5},
            "hard": {"hard": 0.5, "semi_hard": 0.3, "random": 0.2},
            "finetune": {"hard": 0.7, "semi_hard": 0.2, "random": 0.1},
        }
        return ratios.get(self.current_phase, ratios["hard"])
