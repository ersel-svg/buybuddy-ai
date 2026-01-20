"""
Model factory for OD Annotation Worker.

Provides lazy loading and caching of models to optimize cold starts.
"""

from typing import Any
from loguru import logger

from config import config


def get_model(model_name: str, cache: dict[str, Any]) -> Any:
    """
    Get a model instance, using cache if available.

    Args:
        model_name: Name of the model to load
        cache: Dict to store loaded models

    Returns:
        Model instance
    """
    if model_name in cache:
        logger.debug(f"Using cached model: {model_name}")
        return cache[model_name]

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

    elif model_name == "owlv2":
        from .owlv2 import OWLv2Model
        model = OWLv2Model(
            device=config.device,
            cache_dir=config.hf_cache_dir,
        )

    elif model_name == "custom":
        from .custom_model import CustomModel
        # Custom model needs model_path from job input
        raise ValueError("Custom model requires model_path in job input")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    cache[model_name] = model
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
