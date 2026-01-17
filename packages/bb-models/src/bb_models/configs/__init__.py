"""
Configuration presets for model training.
"""

from bb_models.configs.presets import (
    MODEL_PRESETS,
    get_preset,
    merge_config,
    list_presets,
)

__all__ = [
    "MODEL_PRESETS",
    "get_preset",
    "merge_config",
    "list_presets",
]
