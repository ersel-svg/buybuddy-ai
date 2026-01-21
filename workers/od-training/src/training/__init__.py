"""
Training utilities and SOTA base trainer for Object Detection.

Components:
- ModelEMA: Exponential Moving Average for smoother weights
- LLRD Optimizer: Layer-wise Learning Rate Decay
- Scheduler: Warmup + Cosine Annealing
- SOTABaseTrainer: Base trainer with all SOTA features

Usage:
    from training import SOTABaseTrainer, TrainingConfig

    class MyTrainer(SOTABaseTrainer):
        def setup_model(self):
            self.model = load_my_model()

        def compute_loss(self, outputs, targets):
            return my_loss(outputs, targets)

        def forward_pass(self, images):
            return self.model(images)

        def postprocess(self, outputs, targets):
            return convert_to_predictions(outputs)
"""

# EMA
from .ema import ModelEMA

# Optimizer
from .optimizer import (
    build_llrd_optimizer,
    build_simple_optimizer as build_optimizer,
    get_param_groups,
)

# Scheduler
from .scheduler import (
    build_scheduler,
    WarmupCosineScheduler,
    WarmupStepScheduler,
    WarmupLinearScheduler,
)

# Base trainer
from .base_trainer import (
    SOTABaseTrainer,
    TrainingConfig,
    DatasetConfig,
    OutputConfig,
    TrainingResult,
)

__all__ = [
    # EMA
    "ModelEMA",
    # Optimizer
    "build_llrd_optimizer",
    "build_optimizer",
    "get_param_groups",
    # Scheduler
    "build_scheduler",
    "WarmupCosineScheduler",
    "WarmupStepScheduler",
    "WarmupLinearScheduler",
    # Base trainer
    "SOTABaseTrainer",
    "TrainingConfig",
    "DatasetConfig",
    "OutputConfig",
    "TrainingResult",
]
