"""
SOTA Trainers for Object Detection.

Model-specific trainers that inherit from SOTABaseTrainer:
- RTDETRSOTATrainer: RT-DETR with SOTA features
- DFINESOTATrainer: D-FINE with SOTA features
"""

from .rtdetr_sota_trainer import RTDETRSOTATrainer
from .dfine_sota_trainer import DFINESOTATrainer

__all__ = [
    "RTDETRSOTATrainer",
    "DFINESOTATrainer",
]
