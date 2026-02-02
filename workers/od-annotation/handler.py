"""
RunPod Serverless Handler for OD Annotation Worker.

Supports:
- detect: Single image detection with Grounding DINO, Florence-2, or OWLv2
- segment: Interactive segmentation with SAM 2.1
- batch: Bulk detection for multiple images (with chunked webhook support)

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
    "chunk_webhook_url": "https://...",  # Optional: webhook for chunk results
    "chunk_size": 500,  # Optional: images per chunk (default 500)
}
"""

import runpod
import httpx
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
    model_config = job_input.get("model_config")  # For Roboflow models
    image_url = job_input["image_url"]
    text_prompt = job_input.get("text_prompt", "")
    box_threshold = job_input.get("box_threshold", config.default_box_threshold)
    text_threshold = job_input.get("text_threshold", config.default_text_threshold)
    use_nms = job_input.get("use_nms", False)
    nms_threshold = job_input.get("nms_threshold", 0.5)

    # Log request details
    if model_name == "roboflow" and model_config:
        logger.info(f"Detection request: model=roboflow ({model_config.get('architecture')}), nms={use_nms}")
    else:
        logger.info(f"Detection request: model={model_name}, prompt='{text_prompt}', nms={use_nms}")

    # Get model (cached) - pass model_config for Roboflow models
    model = get_model(model_name, MODEL_CACHE, model_config=model_config)

    # Run inference (with or without NMS)
    if use_nms and hasattr(model, "predict_with_nms"):
        predictions = model.predict_with_nms(
            image_url=image_url,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
        )
        logger.info(f"Detection with NMS complete: {len(predictions)} objects (nms_threshold={nms_threshold})")
    else:
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
        "nms_applied": use_nms,
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


def send_chunk_webhook(webhook_url: str, payload: dict, chunk_index: int, total_chunks: int) -> bool:
    """Send chunk results to webhook. Returns True on success."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info(f"Chunk {chunk_index + 1}/{total_chunks} sent to webhook successfully")
                return True
            else:
                logger.error(f"Chunk webhook failed: {response.status_code} - {response.text}")
                return False
    except Exception as e:
        logger.error(f"Chunk webhook error: {e}")
        return False


def handle_batch(job_input: dict) -> dict[str, Any]:
    """Handle batch detection for multiple images with chunked webhook support."""
    model_name = job_input.get("model", "grounding_dino")
    model_config = job_input.get("model_config")  # For Roboflow models
    images = job_input["images"]  # List of {"id": str, "url": str}
    text_prompt = job_input.get("text_prompt", "")
    box_threshold = job_input.get("box_threshold", config.default_box_threshold)
    text_threshold = job_input.get("text_threshold", config.default_text_threshold)
    filter_classes = job_input.get("filter_classes")  # Optional class filter

    # Chunked webhook support for large batches
    chunk_webhook_url = job_input.get("chunk_webhook_url")
    chunk_size = job_input.get("chunk_size", 500)  # Default 500 images per chunk
    job_id = job_input.get("job_id")

    # Log request details
    if model_name == "roboflow" and model_config:
        logger.info(f"Batch request: {len(images)} images, model=roboflow ({model_config.get('architecture')})")
        logger.info(f"[DEBUG] model_config.classes: {model_config.get('classes')}")
        logger.info(f"[DEBUG] filter_classes: {filter_classes}")
    else:
        logger.info(f"Batch request: {len(images)} images, model={model_name}")

    if chunk_webhook_url:
        logger.info(f"Chunked mode enabled: chunk_size={chunk_size}, webhook={chunk_webhook_url}")

    # Get model (cached) - pass model_config for Roboflow models
    model = get_model(model_name, MODEL_CACHE, model_config=model_config)

    # Track overall stats
    total_results = 0
    total_errors = 0
    total_predictions = 0

    # Current chunk results
    chunk_results = []
    chunk_errors = []

    # Calculate total chunks for progress
    total_chunks = (len(images) + chunk_size - 1) // chunk_size if chunk_webhook_url else 1
    current_chunk = 0

    for i, image_info in enumerate(images):
        image_id = image_info.get("id", str(i))
        image_url = image_info.get("url")

        if not image_url:
            chunk_errors.append({"id": image_id, "error": "Missing image URL"})
            continue

        try:
            predictions = model.predict(
                image_url=image_url,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Apply class filter if specified (for Roboflow models)
            if filter_classes:
                original_count = len(predictions)
                # Log what labels we got before filtering
                if original_count > 0:
                    labels_found = set(p.get("label") for p in predictions)
                    logger.info(f"[DEBUG] Labels found in predictions: {labels_found}")
                predictions = [p for p in predictions if p.get("label") in filter_classes]
                logger.info(f"[DEBUG] Class filter applied: {original_count} -> {len(predictions)} (filter={filter_classes})")

            chunk_results.append({
                "id": image_id,
                "predictions": predictions,
                "status": "success",
            })
            total_predictions += len(predictions)

        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            chunk_errors.append({
                "id": image_id,
                "error": str(e),
            })

        # Log progress for long batches
        if (i + 1) % 10 == 0:
            logger.info(f"Batch progress: {i + 1}/{len(images)}")

        # Send chunk to webhook if chunk is full or this is the last image
        is_chunk_full = len(chunk_results) + len(chunk_errors) >= chunk_size
        is_last_image = i == len(images) - 1

        if chunk_webhook_url and (is_chunk_full or is_last_image) and (chunk_results or chunk_errors):
            is_final = is_last_image

            chunk_payload = {
                "job_id": job_id,
                "chunk_index": current_chunk,
                "total_chunks": total_chunks,
                "is_final": is_final,
                "results": chunk_results,
                "errors": chunk_errors,
                "chunk_stats": {
                    "successful": len(chunk_results),
                    "failed": len(chunk_errors),
                    "predictions": sum(len(r.get("predictions", [])) for r in chunk_results),
                },
            }

            send_chunk_webhook(chunk_webhook_url, chunk_payload, current_chunk, total_chunks)

            # Update totals
            total_results += len(chunk_results)
            total_errors += len(chunk_errors)

            # Reset chunk
            chunk_results = []
            chunk_errors = []
            current_chunk += 1

    # If no chunked webhook, results are in chunk_results/chunk_errors
    if not chunk_webhook_url:
        total_results = len(chunk_results)
        total_errors = len(chunk_errors)

    logger.info(f"Batch complete: {total_results} success, {total_errors} errors, {total_predictions} total predictions")

    # Return minimal summary (not full results) to avoid payload limit
    if chunk_webhook_url:
        return {
            "status": "success" if total_errors == 0 else "partial",
            "chunked": True,
            "total_chunks_sent": current_chunk,
            "model": model_name,
            "prompt": text_prompt,
            "total": len(images),
            "successful": total_results,
            "failed": total_errors,
            "total_predictions": total_predictions,
        }
    else:
        # Original behavior for small batches (no chunking)
        return {
            "status": "success" if not chunk_errors else "partial",
            "results": chunk_results,
            "errors": chunk_errors,
            "model": model_name,
            "prompt": text_prompt,
            "total": len(images),
            "successful": len(chunk_results),
            "failed": len(chunk_errors),
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

        # Override config hf_token if provided in job input (for SAM3 access)
        if job_input.get("hf_token"):
            config.hf_token = job_input["hf_token"]
            logger.debug("Using hf_token from job input")

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
