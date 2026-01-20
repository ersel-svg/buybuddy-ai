"""
RunPod Serverless Handler for OD Annotation Worker.

Supports:
- detect: Single image detection with Grounding DINO, Florence-2, or OWLv2
- segment: Interactive segmentation with SAM 2.1
- batch: Bulk detection for multiple images

Input format:
{
    "task": "detect" | "segment" | "batch",
    "model": "grounding_dino" | "florence2" | "sam2" | "owlv2" | "custom",
    "image_url": "https://...",
    "images": [...],  # For batch mode
    "text_prompt": "shelf . product . price tag",
    "box_threshold": 0.3,
    "text_threshold": 0.25,
    "prompt_type": "point" | "box",  # For SAM
    "point": [0.5, 0.3],  # Normalized coords
    "box": [x, y, w, h],  # Normalized coords
}
"""

import runpod
from loguru import logger
import sys
import traceback
from typing import Any

from config import config, VALID_TASKS, TASK_DETECT, TASK_SEGMENT, TASK_BATCH
from models import get_model, preload_models, MODEL_CACHE


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=config.log_level,
)


def validate_input(job_input: dict) -> tuple[bool, str | None]:
    """Validate job input and return (is_valid, error_message)."""
    task = job_input.get("task")
    if not task:
        return False, "Missing required field: task"

    if task not in VALID_TASKS:
        return False, f"Invalid task: {task}. Must be one of {VALID_TASKS}"

    model = job_input.get("model")
    if not model and task != TASK_SEGMENT:
        return False, "Missing required field: model"

    # For detect and segment, need image_url
    if task in [TASK_DETECT, TASK_SEGMENT]:
        if not job_input.get("image_url"):
            return False, "Missing required field: image_url"

    # For batch, need images list
    if task == TASK_BATCH:
        if not job_input.get("images"):
            return False, "Missing required field: images"

    # For grounding models, need text_prompt
    if model in ["grounding_dino", "owlv2"] and task == TASK_DETECT:
        if not job_input.get("text_prompt"):
            return False, f"Missing required field: text_prompt for model {model}"

    # For segment, need prompt_type and corresponding prompt
    if task == TASK_SEGMENT:
        prompt_type = job_input.get("prompt_type")
        if not prompt_type:
            return False, "Missing required field: prompt_type for segment task"

        if prompt_type == "point" and not job_input.get("point"):
            return False, "Missing required field: point for point prompt"

        if prompt_type == "box" and not job_input.get("box"):
            return False, "Missing required field: box for box prompt"

    return True, None


def handle_detect(job_input: dict) -> dict[str, Any]:
    """Handle single image detection."""
    model_name = job_input.get("model", "grounding_dino")
    image_url = job_input["image_url"]
    text_prompt = job_input.get("text_prompt", "")
    box_threshold = job_input.get("box_threshold", config.default_box_threshold)
    text_threshold = job_input.get("text_threshold", config.default_text_threshold)

    logger.info(f"Detection request: model={model_name}, prompt='{text_prompt}'")

    # Get model (cached)
    model = get_model(model_name, MODEL_CACHE)

    # Run inference
    predictions = model.predict(
        image_url=image_url,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    logger.info(f"Detection complete: {len(predictions)} objects found")

    return {
        "status": "success",
        "predictions": predictions,
        "model": model_name,
        "prompt": text_prompt,
    }


def handle_segment(job_input: dict) -> dict[str, Any]:
    """Handle interactive segmentation with SAM 2.1."""
    image_url = job_input["image_url"]
    prompt_type = job_input["prompt_type"]
    return_mask = job_input.get("return_mask", True)

    logger.info(f"Segmentation request: prompt_type={prompt_type}")

    # Get SAM model
    model = get_model("sam2", MODEL_CACHE)

    if prompt_type == "point":
        point = job_input["point"]
        label = job_input.get("label", 1)  # 1=foreground, 0=background

        result = model.segment_point(
            image_url=image_url,
            point=point,
            label=label,
            return_mask=return_mask,
        )

    elif prompt_type == "box":
        box = job_input["box"]

        result = model.segment_box(
            image_url=image_url,
            box=box,
            return_mask=return_mask,
        )

    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

    logger.info("Segmentation complete")

    return {
        "status": "success",
        "bbox": result["bbox"],
        "confidence": result.get("confidence", 1.0),
        "mask": result.get("mask") if return_mask else None,
    }


def handle_batch(job_input: dict) -> dict[str, Any]:
    """Handle batch detection for multiple images."""
    model_name = job_input.get("model", "grounding_dino")
    images = job_input["images"]  # List of {"id": str, "url": str}
    text_prompt = job_input.get("text_prompt", "")
    box_threshold = job_input.get("box_threshold", config.default_box_threshold)
    text_threshold = job_input.get("text_threshold", config.default_text_threshold)

    logger.info(f"Batch request: {len(images)} images, model={model_name}")

    # Get model (cached)
    model = get_model(model_name, MODEL_CACHE)

    results = []
    errors = []

    for i, image_info in enumerate(images):
        image_id = image_info.get("id", str(i))
        image_url = image_info.get("url")

        if not image_url:
            errors.append({"id": image_id, "error": "Missing image URL"})
            continue

        try:
            predictions = model.predict(
                image_url=image_url,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            results.append({
                "id": image_id,
                "predictions": predictions,
                "status": "success",
            })

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            errors.append({
                "id": image_id,
                "error": str(e),
            })

        # Log progress for long batches
        if (i + 1) % 10 == 0:
            logger.info(f"Batch progress: {i + 1}/{len(images)}")

    logger.info(f"Batch complete: {len(results)} success, {len(errors)} errors")

    return {
        "status": "success" if not errors else "partial",
        "results": results,
        "errors": errors,
        "model": model_name,
        "prompt": text_prompt,
        "total": len(images),
        "successful": len(results),
        "failed": len(errors),
    }


def handler(job: dict) -> dict[str, Any]:
    """
    Main RunPod handler function.

    Args:
        job: RunPod job dict with "input" key

    Returns:
        Result dict with predictions or error
    """
    try:
        job_input = job.get("input", {})

        # Validate input
        is_valid, error_msg = validate_input(job_input)
        if not is_valid:
            return {"status": "error", "error": error_msg}

        task = job_input["task"]

        # Route to appropriate handler
        if task == TASK_DETECT:
            return handle_detect(job_input)

        elif task == TASK_SEGMENT:
            return handle_segment(job_input)

        elif task == TASK_BATCH:
            return handle_batch(job_input)

        else:
            return {"status": "error", "error": f"Unknown task: {task}"}

    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Preload models on startup if configured
if config.preload_models:
    logger.info("Preloading models for faster cold starts...")
    try:
        preload_models(MODEL_CACHE)
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload some models: {e}")


# Start RunPod serverless
if __name__ == "__main__":
    logger.info("Starting OD Annotation Worker...")
    runpod.serverless.start({"handler": handler})
