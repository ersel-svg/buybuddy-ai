"""
Model factory for OD Annotation Worker.

Provides lazy loading and caching of models to optimize cold starts.
"""

from typing import Any
from loguru import logger

from config import config


def get_model(
    model_name: str,
    cache: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> Any:
    """
    Get a model instance, using cache if available.

    Args:
        model_name: Name of the model to load
        cache: Dict to store loaded models
        model_config: Additional config for dynamic models (Roboflow)
                     Required keys for "roboflow": checkpoint_url, architecture, classes

    Returns:
        Model instance
    """
    # For Roboflow models, use checkpoint_url as cache key for uniqueness
    if model_name == "roboflow" and model_config:
        cache_key = f"roboflow:{model_config.get('checkpoint_url', '')}"
    else:
        cache_key = model_name

    if cache_key in cache:
        logger.debug(f"Using cached model: {cache_key}")
        return cache[cache_key]

    logger.info(f"Loading model: {model_name}")

    if model_name == "grounding_dino":
        from .grounding_dino import GroundingDINOModel
        model = GroundingDINOModel(
            weights_path=config.grounding_dino_weights,
            config_path=config.grounding_dino_config,
            device=config.device,
        )

    elif model_name == "sam2":
        from .sam2 import SAM2Model
        model = SAM2Model(
            weights_path=config.sam2_weights,
            config_path=config.sam2_config,
            device=config.device,
        )

    elif model_name == "florence2":
        from .florence2 import Florence2Model
        model = Florence2Model(
            model_id=config.florence2_model_id,
            device=config.device,
            cache_dir=config.hf_cache_dir,
        )

    elif model_name == "sam3":
        from .sam3 import SAM3Model
        model = SAM3Model(
            device=config.device,
            hf_token=config.hf_token,
        )

    elif model_name == "custom":
        from .custom_model import CustomModel
        # Custom model needs model_path from job input
        raise ValueError("Custom model requires model_path in job input")

    elif model_name == "roboflow":
        # Roboflow trained models (YOLOv8, RF-DETR, etc.)
        # Requires model_config with checkpoint_url, architecture, classes
        if not model_config:
            raise ValueError("model_config required for Roboflow models")

        required_keys = ["checkpoint_url", "architecture", "classes"]
        missing = [k for k in required_keys if k not in model_config]
        if missing:
            raise ValueError(f"model_config missing required keys: {missing}")

        from .roboflow_model import RoboflowModel
        model = RoboflowModel(
            checkpoint_url=model_config["checkpoint_url"],
            architecture=model_config["architecture"],
            classes=model_config["classes"],
            device=config.device,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    cache[cache_key] = model
    logger.info(f"Model loaded: {model_name}")

    return model


def preload_models(cache: dict[str, Any]) -> None:
    """
    Preload commonly used models into cache.
    Called at startup for faster cold starts.
    """
    enabled_models = config.get_enabled_models()
    logger.info(f"Preloading models: {enabled_models}")

    # Load Grounding DINO first (most commonly used)
    if "grounding_dino" in enabled_models:
        try:
            get_model("grounding_dino", cache)
        except Exception as e:
            logger.warning(f"Failed to preload Grounding DINO: {e}")

    # Load SAM 2 (for interactive segmentation)
    if "sam2" in enabled_models:
        try:
            get_model("sam2", cache)
        except Exception as e:
            logger.warning(f"Failed to preload SAM 2: {e}")

    # Florence-2 is loaded on demand (large model)


# Global model cache (populated by handler.py)
MODEL_CACHE: dict[str, Any] = {}
