"""
Configuration for OD Annotation Worker.
Environment variables and model paths.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    weights_path: str
    device: str = "cuda"
    enabled: bool = True


@dataclass
class Config:
    """Main configuration for the worker."""

    # Model weights paths
    grounding_dino_weights: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDING_DINO_WEIGHTS",
            "/app/weights/groundingdino_swint_ogc.pth"
        )
    )
    grounding_dino_config: str = field(
        default_factory=lambda: os.getenv(
            "GROUNDING_DINO_CONFIG",
            "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
    )
    sam2_weights: str = field(
        default_factory=lambda: os.getenv(
            "SAM2_WEIGHTS",
            "/app/weights/sam2.1_hiera_large.pt"
        )
    )
    sam2_config: str = field(
        default_factory=lambda: os.getenv(
            "SAM2_CONFIG",
            "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
    )

    # Florence-2 (loaded from HuggingFace)
    florence2_model_id: str = field(
        default_factory=lambda: os.getenv(
            "FLORENCE2_MODEL_ID",
            "microsoft/Florence-2-large"
        )
    )

    # HuggingFace cache directory
    hf_cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "HF_HOME",
            "/app/huggingface_cache"
        )
    )

    # HuggingFace token (for SAM3 model download)
    hf_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )

    # Device configuration
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cuda")
    )

    # Default thresholds
    default_box_threshold: float = 0.3
    default_text_threshold: float = 0.25
    default_nms_threshold: float = 0.5

    # Image processing
    max_image_size: int = 1024  # Max dimension for processing
    image_download_timeout: int = 30  # seconds

    # Logging
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    # Model loading behavior
    preload_models: bool = field(
        default_factory=lambda: os.getenv("PRELOAD_MODELS", "true").lower() == "true"
    )

    def get_enabled_models(self) -> list[str]:
        """Get list of enabled models based on environment.

        All models are loaded from HuggingFace, so they're always available.
        SAM3 requires HF_TOKEN for gated model access.
        """
        # Base models - always available via HuggingFace
        models = ["grounding_dino", "florence2", "sam2"]

        # SAM3 requires HF token for gated model access
        if self.hf_token:
            models.append("sam3")

        return models


# Global config instance
config = Config()


# Supported models and their display names
SUPPORTED_MODELS = {
    "grounding_dino": {
        "name": "Grounding DINO",
        "description": "Open-vocabulary object detection with text prompts",
        "tasks": ["detect"],
        "requires_prompt": True,
    },
    "florence2": {
        "name": "Florence-2",
        "description": "Versatile vision model for detection and captioning",
        "tasks": ["detect", "caption", "phrase_grounding"],
        "requires_prompt": False,  # Can work without prompt
    },
    "sam2": {
        "name": "SAM 2.1",
        "description": "Segment Anything Model for interactive segmentation",
        "tasks": ["segment"],
        "requires_prompt": False,  # Uses point/box prompts
    },
    "sam3": {
        "name": "SAM 3",
        "description": "Segment Anything Model 3 with text prompt support",
        "tasks": ["detect", "segment"],
        "requires_prompt": True,  # Uses text prompts (like Grounding DINO)
    },
    "custom": {
        "name": "Custom Model",
        "description": "User-trained YOLO/DETR models",
        "tasks": ["detect"],
        "requires_prompt": False,
    },
}


# Task types
TASK_DETECT = "detect"
TASK_SEGMENT = "segment"
TASK_BATCH = "batch"

VALID_TASKS = [TASK_DETECT, TASK_SEGMENT, TASK_BATCH]
