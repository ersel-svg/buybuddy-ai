"""
AI Annotation API Router

Endpoints for AI-powered auto-annotation:
- /predict: Single image prediction (real-time)
- /segment: Interactive SAM segmentation (real-time)
- /batch: Bulk annotation job (async)
- /jobs/{job_id}: Job status
- /webhook: RunPod callback
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from config import settings
from schemas.od import (
    AIPredictRequest,
    AIPredictResponse,
    AIPrediction,
    AISegmentRequest,
    AISegmentResponse,
    AIBatchAnnotateRequest,
    AIBatchJobResponse,
    AIJobStatusResponse,
    AIWebhookPayload,
    BBox,
)
from services.runpod import runpod_service, EndpointType
from services.supabase import supabase_service

router = APIRouter()

# Supported models for each task type
# Open-vocabulary models (require text_prompt)
OPEN_VOCAB_MODELS = ["grounding_dino", "sam3", "florence2"]
SEGMENT_MODELS = ["sam2", "sam3"]


async def get_all_detect_models() -> list[str]:
    """Get all available detection models including Roboflow models."""
    models = OPEN_VOCAB_MODELS.copy()

    # Add active Roboflow models from database
    try:
        result = supabase_service.client.table("od_roboflow_models").select(
            "id"
        ).eq("is_active", True).execute()

        for rf_model in result.data or []:
            models.append(f"rf:{rf_model['id']}")
    except Exception:
        # If table doesn't exist yet, just return open-vocab models
        pass

    return models


async def get_roboflow_model(model_id: str) -> dict | None:
    """Get Roboflow model details by ID."""
    try:
        result = supabase_service.client.table("od_roboflow_models").select(
            "id, name, display_name, architecture, classes, checkpoint_url, map"
        ).eq("id", model_id).eq("is_active", True).single().execute()
        return result.data
    except Exception:
        return None


def is_roboflow_model(model: str) -> bool:
    """Check if model ID refers to a Roboflow model."""
    return model.startswith("rf:")


def _compute_iou(box1: BBox, box2: BBox) -> float:
    """Compute IoU (Intersection over Union) between two bboxes."""
    # Convert to x1, y1, x2, y2 format
    x1_1, y1_1 = box1.x, box1.y
    x2_1, y2_1 = box1.x + box1.width, box1.y + box1.height

    x1_2, y1_2 = box2.x, box2.y
    x2_2, y2_2 = box2.x + box2.width, box2.y + box2.height

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _apply_class_agnostic_nms(
    predictions: list[AIPrediction],
    iou_threshold: float = 0.5,
) -> list[AIPrediction]:
    """
    Apply class-agnostic NMS to filter overlapping predictions.

    Unlike class-aware NMS, this compares ALL boxes regardless of label.
    Keeps the prediction with highest confidence when boxes overlap.

    Args:
        predictions: List of predictions to filter
        iou_threshold: IoU threshold for suppression (lower = more aggressive)

    Returns:
        Filtered list of predictions
    """
    if not predictions:
        return predictions

    # Sort by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    kept = []
    suppressed = set()

    for i, pred in enumerate(sorted_preds):
        if i in suppressed:
            continue

        kept.append(pred)

        # Suppress all lower-confidence predictions that overlap
        for j in range(i + 1, len(sorted_preds)):
            if j in suppressed:
                continue

            iou = _compute_iou(pred.bbox, sorted_preds[j].bbox)
            if iou >= iou_threshold:
                suppressed.add(j)

    return kept


@router.post("/predict", response_model=AIPredictResponse)
async def predict_single(request: AIPredictRequest) -> AIPredictResponse:
    """
    Run AI prediction on a single image (real-time).

    Supported models:
    - grounding_dino: Text prompt → bounding boxes (SOTA open-vocabulary)
    - sam3: Text prompt → mask → bounding box
    - florence2: Versatile vision tasks
    - rf:{model_id}: Roboflow trained models (closed-vocabulary, no text prompt needed)
    - trained:{model_id}: Custom trained models (closed-vocabulary, no text prompt needed)
    """
    # Check model type
    is_rf_model = is_roboflow_model(request.model)
    is_trained_model = request.model.startswith("trained:")
    rf_model_data = None
    trained_model_data = None

    if is_rf_model:
        # Extract Roboflow model ID and fetch from database
        rf_id = request.model.replace("rf:", "")
        rf_model_data = await get_roboflow_model(rf_id)

        if not rf_model_data:
            raise HTTPException(
                status_code=404,
                detail=f"Roboflow model '{rf_id}' not found or not active"
            )
    elif is_trained_model:
        # Extract trained model ID and fetch from database
        trained_id = request.model.replace("trained:", "")

        try:
            result = supabase_service.client.table("od_trained_models").select(
                "id, name, model_type, checkpoint_url, class_mapping, is_active"
            ).eq("id", trained_id).single().execute()

            if not result.data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trained model '{trained_id}' not found"
                )

            trained_model_data = result.data

            if not trained_model_data.get("is_active"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{trained_model_data['name']}' is not active"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch trained model: {str(e)}"
            )
    else:
        # Validate open-vocab model
        if request.model not in OPEN_VOCAB_MODELS:
            all_models = await get_all_detect_models()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{request.model}'. Supported: {all_models}"
            )

        # Open-vocab models require text_prompt
        if not request.text_prompt:
            raise HTTPException(
                status_code=400,
                detail=f"text_prompt is required for model '{request.model}'"
            )

    # Check endpoint configuration
    if not runpod_service.is_configured(EndpointType.OD_ANNOTATION):
        raise HTTPException(
            status_code=503,
            detail="OD Annotation endpoint not configured"
        )

    # Get image URL from database
    image_result = supabase_service.client.table("od_images").select(
        "id, image_url"
    ).eq("id", request.image_id).single().execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_result.data["image_url"]

    # Prepare RunPod input based on model type
    if is_rf_model and rf_model_data:
        # Roboflow model - send model_config for dynamic loading
        runpod_input = {
            "task": "detect",
            "model": "roboflow",  # Generic handler in worker
            "model_config": {
                "checkpoint_url": rf_model_data["checkpoint_url"],
                "architecture": rf_model_data["architecture"],
                "classes": rf_model_data["classes"],
            },
            "image_url": image_url,
            "box_threshold": request.box_threshold,
            # text_prompt not needed for Roboflow models
        }
    elif is_trained_model and trained_model_data:
        # Trained model - send model_config similar to Roboflow
        # class_mapping format: {class_id: index}
        class_mapping = trained_model_data.get("class_mapping", {})

        # Get class names from od_classes table
        classes = []
        if class_mapping:
            # Sort by index
            sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])  # x[1] is the index
            class_ids = [class_id for class_id, _ in sorted_classes]

            if class_ids:
                classes_result = supabase_service.client.table("od_classes").select(
                    "id, name"
                ).in_("id", class_ids).execute()

                class_names_map = {c["id"]: c["name"] for c in classes_result.data or []}
                classes = [class_names_map.get(class_id, f"class_{idx}") for class_id, idx in sorted_classes]

        runpod_input = {
            "task": "detect",
            "model": "trained",  # Generic handler in worker
            "model_config": {
                "checkpoint_url": trained_model_data["checkpoint_url"],
                "architecture": trained_model_data["model_type"],  # rt-detr, d-fine
                "classes": classes,
                "class_mapping": class_mapping,  # Full mapping for reference
            },
            "image_url": image_url,
            "box_threshold": request.box_threshold,
            # text_prompt not needed for trained models
        }
    else:
        # Open-vocab model - existing logic
        runpod_input = {
            "task": "detect",
            "model": request.model,
            "image_url": image_url,
            "text_prompt": request.text_prompt,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "use_nms": request.use_nms,
            "nms_threshold": request.nms_threshold,
            "hf_token": settings.hf_token,  # For SAM3 model access
        }

    start_time = time.time()

    try:
        # Call RunPod synchronously (real-time)
        result = await runpod_service.submit_job_sync(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data=runpod_input,
            timeout=60,  # 60 second timeout for single image
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Check for errors
        if result.get("status") == "FAILED":
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"AI prediction failed: {error}")

        # Extract predictions from output
        output = result.get("output", {})
        raw_predictions = output.get("predictions", [])

        # Convert to response format
        predictions = []
        for pred in raw_predictions:
            bbox_data = pred.get("bbox", {})
            predictions.append(AIPrediction(
                bbox=BBox(
                    x=bbox_data.get("x", 0),
                    y=bbox_data.get("y", 0),
                    width=bbox_data.get("width", 0),
                    height=bbox_data.get("height", 0),
                ),
                label=pred.get("label", ""),
                confidence=pred.get("confidence", 0),
                mask=pred.get("mask"),
            ))

        # Apply class filter if specified
        if request.filter_classes:
            filter_set = set(request.filter_classes)
            original_count = len(predictions)
            predictions = [p for p in predictions if p.label in filter_set]
            filtered_count = original_count - len(predictions)
            if filtered_count > 0:
                print(f"[AI Predict] Class filter removed {filtered_count} predictions (keeping: {request.filter_classes})")

        # Apply class-agnostic NMS to remove duplicate/overlapping boxes
        # This runs regardless of model - ensures consistent deduplication
        original_count = len(predictions)
        if request.use_nms and len(predictions) > 1:
            predictions = _apply_class_agnostic_nms(predictions, request.nms_threshold)
            filtered_count = original_count - len(predictions)
            if filtered_count > 0:
                print(f"[AI Predict] NMS filtered {filtered_count} overlapping boxes (IoU threshold: {request.nms_threshold})")

        return AIPredictResponse(
            predictions=predictions,
            model=request.model,
            processing_time_ms=processing_time_ms,
            nms_applied=request.use_nms,
        )

    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(status_code=504, detail="AI prediction timed out")
    except Exception as e:
        print(f"[AI Predict] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment", response_model=AISegmentResponse)
async def segment_interactive(request: AISegmentRequest) -> AISegmentResponse:
    """
    Interactive SAM segmentation (real-time).

    Point mode: Click on object → get mask + bbox
    Box mode: Draw box → get refined mask + bbox

    Supported models:
    - sam2: Point/box prompt → mask (fast, no text)
    - sam3: Point/box + optional text → mask
    """
    # Validate model
    if request.model not in SEGMENT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Supported: {SEGMENT_MODELS}"
        )

    # Validate prompt type
    if request.prompt_type not in ["point", "box"]:
        raise HTTPException(
            status_code=400,
            detail="prompt_type must be 'point' or 'box'"
        )

    # Validate prompt data
    if request.prompt_type == "point" and not request.point:
        raise HTTPException(status_code=400, detail="point is required for point prompt")
    if request.prompt_type == "box" and not request.box:
        raise HTTPException(status_code=400, detail="box is required for box prompt")

    # Check endpoint configuration
    if not runpod_service.is_configured(EndpointType.OD_ANNOTATION):
        raise HTTPException(
            status_code=503,
            detail="OD Annotation endpoint not configured"
        )

    # Get image URL
    image_result = supabase_service.client.table("od_images").select(
        "id, image_url"
    ).eq("id", request.image_id).single().execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_result.data["image_url"]

    # Prepare RunPod input
    runpod_input = {
        "task": "segment",
        "model": request.model,
        "image_url": image_url,
        "prompt_type": request.prompt_type,
        "label": request.label,
        "hf_token": settings.hf_token,  # For SAM3 model access
    }

    if request.point:
        runpod_input["point"] = list(request.point)
    if request.box:
        runpod_input["box"] = request.box
    if request.text_prompt and request.model == "sam3":
        runpod_input["text_prompt"] = request.text_prompt

    start_time = time.time()

    try:
        result = await runpod_service.submit_job_sync(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data=runpod_input,
            timeout=30,  # Segmentation should be fast
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        if result.get("status") == "FAILED":
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {error}")

        output = result.get("output", {})
        bbox_data = output.get("bbox", {})

        return AISegmentResponse(
            bbox=BBox(
                x=bbox_data.get("x", 0),
                y=bbox_data.get("y", 0),
                width=bbox_data.get("width", 0),
                height=bbox_data.get("height", 0),
            ),
            confidence=output.get("confidence", 1.0),
            mask=output.get("mask"),
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Segmentation timed out")
    except Exception as e:
        print(f"[AI Segment] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=AIBatchJobResponse)
async def batch_annotate(
    request: AIBatchAnnotateRequest,
    background_tasks: BackgroundTasks,
) -> AIBatchJobResponse:
    """
    Start a bulk AI annotation job (async).

    Processes multiple images and optionally saves predictions as annotations.
    Use /jobs/{job_id} to check progress.

    Supports:
    - Open-vocabulary models (require text_prompt)
    - Roboflow closed-vocabulary models (rf:{model_id})
    - Custom trained models (trained:{model_id})
    """
    # Check model type
    is_rf_model = is_roboflow_model(request.model)
    is_trained_model = request.model.startswith("trained:")
    rf_model_data = None
    trained_model_data = None

    if is_rf_model:
        # Extract Roboflow model ID and fetch from database
        rf_id = request.model.replace("rf:", "")
        rf_model_data = await get_roboflow_model(rf_id)

        if not rf_model_data:
            raise HTTPException(
                status_code=404,
                detail=f"Roboflow model '{rf_id}' not found or not active"
            )
    elif is_trained_model:
        # Extract trained model ID and fetch from database
        trained_id = request.model.replace("trained:", "")

        try:
            result = supabase_service.client.table("od_trained_models").select(
                "id, name, model_type, checkpoint_url, class_mapping, is_active"
            ).eq("id", trained_id).single().execute()

            if not result.data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trained model '{trained_id}' not found"
                )

            trained_model_data = result.data

            if not trained_model_data.get("is_active"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{trained_model_data['name']}' is not active"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch trained model: {str(e)}"
            )
    else:
        # Validate open-vocab model
        all_models = await get_all_detect_models()
        if request.model not in all_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{request.model}'. Supported: {all_models}"
            )

        # Open-vocab models require text_prompt
        if not request.text_prompt:
            raise HTTPException(
                status_code=400,
                detail=f"text_prompt is required for open-vocabulary model '{request.model}'"
            )

    # Check endpoint configuration
    if not runpod_service.is_configured(EndpointType.OD_ANNOTATION):
        raise HTTPException(
            status_code=503,
            detail="OD Annotation endpoint not configured"
        )

    # Verify dataset exists
    dataset_result = supabase_service.client.table("od_datasets").select(
        "id, name"
    ).eq("id", request.dataset_id).single().execute()

    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get images to process - use pagination to bypass Supabase 1000 row limit
    images = []
    batch_size = 1000
    offset = 0
    max_images = request.limit if request.limit else None  # Early limit optimization

    while True:
        # Stop early if we've reached the limit
        if max_images and len(images) >= max_images:
            images = images[:max_images]
            break

        if request.image_ids:
            # Specific images - batch the .in_() query to avoid URL length limits
            # Process in chunks of 50 UUIDs at a time
            chunk_size = 50
            ids_to_fetch = request.image_ids[:max_images] if max_images else request.image_ids
            for i in range(0, len(ids_to_fetch), chunk_size):
                chunk = ids_to_fetch[i:i + chunk_size]
                chunk_result = supabase_service.client.table("od_dataset_images").select(
                    "image_id, od_images!inner(id, image_url)"
                ).eq("dataset_id", request.dataset_id).in_("image_id", chunk).execute()
                images.extend(chunk_result.data or [])
            break  # No pagination needed for specific IDs
        else:
            # All unannotated images in dataset - paginate
            images_query = supabase_service.client.table("od_dataset_images").select(
                "image_id, od_images!inner(id, image_url)"
            ).eq("dataset_id", request.dataset_id).eq("status", "pending").range(offset, offset + batch_size - 1)

            images_result = images_query.execute()
            batch = images_result.data or []
            images.extend(batch)

            if len(batch) < batch_size:
                break  # No more records
            offset += batch_size

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images found to process"
        )

    # Apply limit if specified
    if request.limit and request.limit < len(images):
        images = images[:request.limit]

    # Create job record
    job_id = str(uuid.uuid4())

    job_data = {
        "id": job_id,
        "type": "od_annotation",
        "status": "pending",
        "config": {
            "dataset_id": request.dataset_id,
            "model": request.model,
            "text_prompt": request.text_prompt,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "auto_accept": request.auto_accept,
            "class_mapping": request.class_mapping,
            "filter_classes": request.filter_classes,  # For Roboflow models
            "total_images": len(images),
            "image_ids": [img["image_id"] for img in images],
        },
        "created_at": datetime.utcnow().isoformat(),
    }

    supabase_service.client.table("jobs").insert(job_data).execute()

    # Prepare image URLs for batch processing
    # Worker expects: {"id": str, "url": str}
    image_data = []
    for img in images:
        od_image = img.get("od_images", {})
        if od_image:
            image_url = od_image.get("image_url")
            if image_url:
                image_data.append({
                    "id": img["image_id"],  # Worker expects "id"
                    "url": image_url,        # Worker expects "url"
                })

    # Prepare RunPod input based on model type
    if is_rf_model and rf_model_data:
        # Roboflow model - send model_config for dynamic loading
        runpod_input = {
            "task": "batch",
            "model": "roboflow",  # Generic handler in worker
            "model_config": {
                "checkpoint_url": rf_model_data["checkpoint_url"],
                "architecture": rf_model_data["architecture"],
                "classes": rf_model_data["classes"],
            },
            "images": image_data,
            "box_threshold": request.box_threshold,
            "filter_classes": request.filter_classes,  # Filter specific classes
            "job_id": job_id,
        }
    elif is_trained_model and trained_model_data:
        # Trained model - send model_config similar to Roboflow
        # class_mapping format: {class_id: index}
        class_mapping = trained_model_data.get("class_mapping", {})

        # Get class names from od_classes table
        classes = []
        if class_mapping:
            # Sort by index
            sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])  # x[1] is the index
            class_ids = [class_id for class_id, _ in sorted_classes]

            if class_ids:
                classes_result = supabase_service.client.table("od_classes").select(
                    "id, name"
                ).in_("id", class_ids).execute()

                class_names_map = {c["id"]: c["name"] for c in classes_result.data or []}
                classes = [class_names_map.get(class_id, f"class_{idx}") for class_id, idx in sorted_classes]

        runpod_input = {
            "task": "batch",
            "model": "trained",  # Generic handler in worker
            "model_config": {
                "checkpoint_url": trained_model_data["checkpoint_url"],
                "architecture": trained_model_data["model_type"],  # rt-detr, d-fine
                "classes": classes,
                "class_mapping": class_mapping,
            },
            "images": image_data,
            "box_threshold": request.box_threshold,
            "filter_classes": request.filter_classes,  # Filter specific classes
            "job_id": job_id,
        }
    else:
        # Open-vocab model
        runpod_input = {
            "task": "batch",
            "model": request.model,
            "images": image_data,
            "text_prompt": request.text_prompt,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "filter_classes": request.filter_classes,  # Optional filter
            "job_id": job_id,
            "hf_token": settings.hf_token,  # For SAM3 model access
        }

    # Webhook URL for RunPod callback (primary completion mechanism)
    webhook_url = None
    if settings.api_url:
        webhook_url = f"{settings.api_url.rstrip('/')}/api/v1/od/ai/webhook"
        print(f"[AI Batch] Webhook URL configured: {webhook_url}")
    else:
        print("[AI Batch] Warning: api_url not configured, webhook disabled. Job completion relies on polling.")

    try:
        # Submit async job to RunPod
        runpod_response = await runpod_service.submit_job(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data=runpod_input,
            webhook_url=webhook_url,
        )

        runpod_job_id = runpod_response.get("id")

        # Update job with RunPod ID
        supabase_service.client.table("jobs").update({
            "runpod_job_id": runpod_job_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()

        return AIBatchJobResponse(
            job_id=job_id,
            status="running",
            total_images=len(images),
            message=f"Started AI annotation for {len(images)} images",
        )

    except Exception as e:
        # Update job as failed
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": str(e),
        }).eq("id", job_id).execute()

        raise HTTPException(status_code=500, detail=f"Failed to start batch job: {e}")


@router.get("/jobs/{job_id}", response_model=AIJobStatusResponse)
async def get_job_status(job_id: str) -> AIJobStatusResponse:
    """
    Get batch annotation job status.

    LAST RESORT completion mechanism - checks RunPod status on-demand.
    Only processes completion if job is still in 'running' state (idempotent).
    """
    # Get job from database
    job_result = supabase_service.client.table("jobs").select(
        "*"
    ).eq("id", job_id).eq("type", "od_annotation").single().execute()

    if not job_result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_result.data
    config = job.get("config", {})

    # Only check RunPod if job is still running (idempotency - don't reprocess completed jobs)
    if job.get("status") == "running" and job.get("runpod_job_id"):
        try:
            runpod_status = await runpod_service.get_job_status(
                endpoint_type=EndpointType.OD_ANNOTATION,
                job_id=job["runpod_job_id"],
            )

            # Update job if completed
            if runpod_status.get("status") == "COMPLETED":
                output = runpod_status.get("output", {})

                # Count total predictions from results array
                results = output.get("results", [])
                predictions_count = sum(
                    len(r.get("predictions", [])) for r in results
                )

                # Always save predictions as annotations (user can review/delete later)
                annotations_created = 0
                if results:
                    predictions_by_image = {
                        r["id"]: r.get("predictions", [])
                        for r in results if r.get("id")
                    }
                    try:
                        annotations_created = await _save_predictions_as_annotations(
                            dataset_id=config["dataset_id"],
                            predictions_by_image=predictions_by_image,
                            class_mapping=config.get("class_mapping"),
                            model=config.get("model"),
                            filter_classes=config.get("filter_classes"),
                        )
                        print(f"[AI Job Status] Created {annotations_created} annotations for job {job_id[:8]}...")
                    except Exception as e:
                        print(f"[AI Job Status] Failed to save annotations: {e}")

                # Store predictions count in result for status response
                result_data = {
                    **output,
                    "total_predictions": predictions_count,
                    "annotations_created": annotations_created,
                    "completed_by": "status_check",
                }
                supabase_service.client.table("jobs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": result_data,
                }).eq("id", job_id).execute()

                job["status"] = "completed"
                job["result"] = result_data
                print(f"[AI Job Status] Job {job_id[:8]}... completed ({predictions_count} predictions)")

            elif runpod_status.get("status") == "FAILED":
                error = runpod_status.get("error", "Unknown error")
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error": error,
                    "result": {"failed_by": "status_check"},
                }).eq("id", job_id).execute()

                job["status"] = "failed"
                job["error"] = error
                print(f"[AI Job Status] Job {job_id[:8]}... failed: {error}")

        except Exception as e:
            error_str = str(e)
            # 404 means job expired from RunPod
            if "404" in error_str:
                print(f"[AI Job Status] Job {job_id[:8]}... expired from RunPod (404)")
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error": "Job expired from RunPod. Results could not be retrieved. Please re-run the job.",
                    "result": {"failed_by": "status_check", "reason": "runpod_expired"},
                }).eq("id", job_id).execute()
                job["status"] = "failed"
                job["error"] = "Job expired from RunPod"
            else:
                print(f"[AI Job Status] Error checking RunPod: {e}")

    # Get predictions count from result (set when job completes)
    result = job.get("result", {}) or {}
    predictions_count = result.get("total_predictions", 0)

    return AIJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=config.get("progress", 0),
        total_images=config.get("total_images", 0),
        predictions_generated=predictions_count,
        error_message=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.post("/webhook")
async def runpod_webhook(payload: AIWebhookPayload) -> dict:
    """
    Handle RunPod webhook callback for batch jobs.

    PRIMARY completion mechanism - called by RunPod when job finishes.
    Always saves predictions as annotations (user can review/delete later).
    """
    print(f"[AI Webhook] Received: job_id={payload.id}, status={payload.status}")

    # Find job by RunPod ID
    job_result = supabase_service.client.table("jobs").select(
        "*"
    ).eq("runpod_job_id", payload.id).eq("type", "od_annotation").single().execute()

    if not job_result.data:
        print(f"[AI Webhook] Job not found for RunPod ID: {payload.id}")
        return {"status": "ignored", "reason": "job_not_found"}

    job = job_result.data
    job_id = job["id"]
    config = job.get("config", {})

    # Check if job is already completed (avoid duplicate processing)
    if job.get("status") == "completed":
        print(f"[AI Webhook] Job {job_id[:8]}... already completed, skipping")
        return {"status": "ignored", "reason": "already_completed"}

    if payload.status == "COMPLETED" and payload.output:
        output = payload.output

        # Worker returns results array: [{"id": ..., "predictions": [...]}]
        results = output.get("results", [])
        total_predictions = sum(
            len(r.get("predictions", [])) for r in results
        )

        # Convert results array to dict for processing
        predictions_by_image = {
            r["id"]: r.get("predictions", [])
            for r in results if r.get("id")
        }

        # Always save predictions as annotations (user can review/delete later)
        # This ensures no predictions are lost even if user closed the modal
        annotations_created = 0
        if predictions_by_image:
            try:
                annotations_created = await _save_predictions_as_annotations(
                    dataset_id=config["dataset_id"],
                    predictions_by_image=predictions_by_image,
                    class_mapping=config.get("class_mapping"),
                    model=config.get("model"),
                    filter_classes=config.get("filter_classes"),
                )
                print(f"[AI Webhook] Created {annotations_created} annotations for job {job_id[:8]}...")
            except Exception as e:
                print(f"[AI Webhook] Failed to save annotations: {e}")

        # Update job status
        supabase_service.client.table("jobs").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": {
                "total_predictions": total_predictions,
                "images_processed": len(results),
                "annotations_created": annotations_created,
                "completed_by": "webhook",
            },
        }).eq("id", job_id).execute()

        print(f"[AI Webhook] Job {job_id[:8]}... completed ({total_predictions} predictions, {annotations_created} annotations)")
        return {"status": "success", "job_id": job_id, "annotations_created": annotations_created}

    elif payload.status == "FAILED":
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": payload.error or "Unknown error",
            "result": {"failed_by": "webhook"},
        }).eq("id", job_id).execute()

        print(f"[AI Webhook] Job {job_id[:8]}... failed: {payload.error}")
        return {"status": "failed", "job_id": job_id, "error": payload.error}

    return {"status": "ignored", "reason": f"unhandled_status_{payload.status}"}


async def _save_predictions_as_annotations(
    dataset_id: str,
    predictions_by_image: dict,
    class_mapping: Optional[dict],
    model: str,
    filter_classes: Optional[list[str]] = None,
) -> int:
    """
    Save AI predictions as annotations in the database.
    Auto-creates classes for labels that don't exist.
    Uses batch insert for performance.

    Args:
        dataset_id: Target dataset ID
        predictions_by_image: Dict of image_id -> list of predictions
        class_mapping: Dict mapping detected labels to class IDs or '__new__:classname'
        model: Model name used for predictions
        filter_classes: Optional list of classes to include. If set, other classes are skipped.

    Returns number of annotations created.
    """
    import random

    # Cache for class IDs to avoid repeated lookups
    class_cache: dict[str, str] = {}

    # Collect all annotations for batch insert
    annotations_batch: list[dict] = []
    now = datetime.utcnow().isoformat()

    for image_id, predictions in predictions_by_image.items():
        for pred in predictions:
            bbox = pred.get("bbox", {})
            label = pred.get("label", "")
            confidence = pred.get("confidence", 0)

            if not label:
                continue

            # Apply class filter if specified (for Roboflow models)
            if filter_classes and label not in filter_classes:
                continue

            # Check cache first
            if label in class_cache:
                class_id = class_cache[label]
            else:
                class_id = None

                # Check class_mapping
                if class_mapping and label in class_mapping:
                    mapped_value = class_mapping[label]
                    # Handle "__new__:classname" format from frontend
                    if mapped_value.startswith("__new__:"):
                        class_id = None  # Will create new below
                    else:
                        class_id = mapped_value

                if not class_id:
                    # Try to find existing class by name
                    try:
                        class_result = supabase_service.client.table("od_classes").select(
                            "id"
                        ).eq("dataset_id", dataset_id).eq("name", label).single().execute()
                        if class_result.data:
                            class_id = class_result.data["id"]
                    except Exception:
                        pass  # Class doesn't exist

                if not class_id:
                    # Auto-create class with random color
                    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
                              "#3b82f6", "#8b5cf6", "#ec4899", "#6b7280"]
                    new_class_id = str(uuid.uuid4())
                    try:
                        supabase_service.client.table("od_classes").insert({
                            "id": new_class_id,
                            "dataset_id": dataset_id,
                            "name": label,
                            "display_name": label.replace("_", " ").title(),
                            "color": random.choice(colors),
                            "created_at": now,
                        }).execute()
                        class_id = new_class_id
                        print(f"[AI Annotation] Created new class: {label} ({new_class_id})")
                    except Exception as e:
                        print(f"[AI Annotation] Failed to create class {label}: {e}")
                        continue

                # Cache the class_id
                class_cache[label] = class_id

            # Add to batch instead of inserting one by one
            annotations_batch.append({
                "id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "image_id": image_id,
                "class_id": class_id,
                "bbox_x": bbox.get("x", 0),
                "bbox_y": bbox.get("y", 0),
                "bbox_width": bbox.get("width", 0),
                "bbox_height": bbox.get("height", 0),
                "is_ai_generated": True,
                "confidence": confidence,
                "ai_model": model,
                "is_reviewed": False,
                "created_at": now,
                "updated_at": now,
            })

    # Batch insert all annotations (in chunks of 500 to avoid payload limits)
    annotations_created = 0
    BATCH_SIZE = 500

    for i in range(0, len(annotations_batch), BATCH_SIZE):
        chunk = annotations_batch[i:i + BATCH_SIZE]
        try:
            supabase_service.client.table("od_annotations").insert(chunk).execute()
            annotations_created += len(chunk)
            print(f"[AI Annotation] Inserted batch {i // BATCH_SIZE + 1}: {len(chunk)} annotations")
        except Exception as e:
            print(f"[AI Annotation] Failed to insert batch: {e}")
            # Fallback to one-by-one for this chunk
            for annotation in chunk:
                try:
                    supabase_service.client.table("od_annotations").insert(annotation).execute()
                    annotations_created += 1
                except Exception as inner_e:
                    print(f"[AI Annotation] Failed single insert: {inner_e}")

    # Update annotation counts using batch RPC (prevents system lockup with large batches)
    if annotations_created > 0:
        image_ids = list(predictions_by_image.keys())
        class_ids = list(set(class_cache.values()))

        # Try RPC first (most efficient - single query for all counts)
        rpc_success = False
        try:
            result = supabase_service.client.rpc(
                "update_annotation_counts_batch",
                {
                    "p_dataset_id": dataset_id,
                    "p_image_ids": image_ids,
                    "p_class_ids": class_ids,
                }
            ).execute()

            if result.data:
                print(f"[AI Annotation] Batch count update: {result.data}")
                rpc_success = True
            else:
                print(f"[AI Annotation] Batch count update completed")
                rpc_success = True

        except Exception as e:
            print(f"[AI Annotation] RPC not available, using legacy method: {e}")

        # Fallback to legacy method if RPC failed
        if not rpc_success:
            try:
                # 1. Dataset total count (single query)
                count_result = supabase_service.client.table("od_annotations").select(
                    "id", count="exact"
                ).eq("dataset_id", dataset_id).execute()
                total_count = count_result.count or 0
                supabase_service.client.table("od_datasets").update({
                    "annotation_count": total_count
                }).eq("id", dataset_id).execute()

                # 2. Class counts (one query per class - usually just 1-2 classes)
                for class_id in class_ids:
                    class_count_result = supabase_service.client.table("od_annotations").select(
                        "id", count="exact"
                    ).eq("dataset_id", dataset_id).eq("class_id", class_id).execute()
                    class_count = class_count_result.count or 0
                    supabase_service.client.table("od_classes").update({
                        "annotation_count": class_count
                    }).eq("id", class_id).execute()

                # 3. Image counts - SKIP for large batches to prevent lockup
                # User can trigger manual recount if needed
                if len(image_ids) <= 1000:
                    for image_id in image_ids:
                        img_count_result = supabase_service.client.table("od_annotations").select(
                            "id", count="exact"
                        ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
                        img_count = img_count_result.count or 0
                        supabase_service.client.table("od_dataset_images").update({
                            "annotation_count": img_count,
                            "status": "annotated" if img_count > 0 else "pending",
                            "last_annotated_at": now,
                        }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

                    # Update annotated_image_count
                    annotated_images_result = supabase_service.client.table("od_dataset_images").select(
                        "id", count="exact"
                    ).eq("dataset_id", dataset_id).gt("annotation_count", 0).execute()
                    supabase_service.client.table("od_datasets").update({
                        "annotated_image_count": annotated_images_result.count or 0
                    }).eq("id", dataset_id).execute()
                else:
                    print(f"[AI Annotation] Skipping image count updates for {len(image_ids)} images (too large, use RPC)")
                    print(f"[AI Annotation] Run migration 071 to enable batch count updates")

            except Exception as fallback_e:
                print(f"[AI Annotation] Legacy fallback failed: {fallback_e}")

    return annotations_created


@router.get("/models")
async def list_available_models() -> dict:
    """List available AI models and their capabilities."""

    # Base open-vocabulary detection models
    detection_models = [
        {
            "id": "grounding_dino",
            "name": "Grounding DINO",
            "description": "SOTA open-vocabulary object detection. Best for text→bbox.",
            "tasks": ["detect"],
            "requires_prompt": True,
            "model_type": "open_vocab",
        },
        {
            "id": "sam3",
            "name": "SAM 3",
            "description": "Segment Anything Model 3 with text prompt support.",
            "tasks": ["detect", "segment"],
            "requires_prompt": True,
            "model_type": "open_vocab",
        },
        {
            "id": "florence2",
            "name": "Florence-2",
            "description": "Microsoft's versatile vision model.",
            "tasks": ["detect", "caption"],
            "requires_prompt": False,
            "model_type": "open_vocab",
        },
    ]

    # Fetch active Roboflow models from database
    try:
        rf_result = supabase_service.client.table("od_roboflow_models").select(
            "id, name, display_name, description, architecture, classes, map"
        ).eq("is_active", True).execute()

        for rf in rf_result.data or []:
            map_str = f"{rf['map']*100:.1f}% mAP" if rf.get('map') else ""
            description = rf.get('description') or f"{rf['architecture']} {map_str}".strip()

            detection_models.append({
                "id": f"rf:{rf['id']}",
                "name": rf["display_name"],
                "description": description,
                "tasks": ["detect"],
                "requires_prompt": False,  # Closed-vocabulary - no text prompt needed
                "model_type": "closed_vocab",
                "classes": rf["classes"],  # Fixed classes
                "architecture": rf["architecture"],
            })
    except Exception as e:
        # Table might not exist yet, just continue with base models
        print(f"[AI Models] Could not fetch Roboflow models: {e}")

    # Fetch active trained models from database
    try:
        trained_result = supabase_service.client.table("od_trained_models").select(
            "id, name, model_type, class_mapping, map, map_50, is_active"
        ).eq("is_active", True).execute()

        for trained in trained_result.data or []:
            # Extract class list from class_mapping JSON
            # Format: {class_id: index}
            class_mapping = trained.get("class_mapping", {})

            # Get class names from od_classes table
            classes = []
            if class_mapping:
                # Create a list of (index, class_id) tuples and sort by index
                sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])  # x[1] is the index

                # Fetch class names in batch
                class_ids = [class_id for class_id, _ in sorted_classes]
                if class_ids:
                    classes_result = supabase_service.client.table("od_classes").select(
                        "id, name"
                    ).in_("id", class_ids).execute()

                    # Create a mapping of class_id -> name
                    class_names_map = {c["id"]: c["name"] for c in classes_result.data or []}

                    # Build ordered class list
                    classes = [class_names_map.get(class_id, f"class_{idx}") for class_id, idx in sorted_classes]

            # Format metrics
            map_val = trained.get("map", 0)
            map_50_val = trained.get("map_50", 0)
            metrics_str = f"mAP: {map_val:.2%}" if map_val else ""
            if map_50_val:
                metrics_str += f" | mAP@50: {map_50_val:.2%}" if metrics_str else f"mAP@50: {map_50_val:.2%}"

            description = f"{trained['model_type'].upper()}"
            if metrics_str:
                description += f" - {metrics_str}"

            detection_models.append({
                "id": f"trained:{trained['id']}",
                "name": trained["name"],
                "description": description,
                "tasks": ["detect"],
                "requires_prompt": False,  # Closed-vocabulary - trained on specific classes
                "model_type": "closed_vocab",
                "classes": classes,  # Fixed classes from training
                "architecture": trained["model_type"],  # rt-detr, d-fine
                "metrics": {
                    "map": trained.get("map"),
                    "map_50": trained.get("map_50"),
                },
            })
    except Exception as e:
        # Table might not exist yet, just continue
        print(f"[AI Models] Could not fetch trained models: {e}")

    return {
        "detection_models": detection_models,
        "segmentation_models": [
            {
                "id": "sam2",
                "name": "SAM 2.1",
                "description": "Fast interactive segmentation with point/box prompts.",
                "tasks": ["segment"],
                "requires_prompt": False,
            },
            {
                "id": "sam3",
                "name": "SAM 3",
                "description": "SAM 3 also supports segmentation with optional text.",
                "tasks": ["segment"],
                "requires_prompt": False,
            },
        ],
    }


@router.get("/models/{model_id}/debug")
async def get_model_debug_info(model_id: str) -> dict:
    """
    Debug endpoint to get full model configuration from database.
    Use this to verify model classes, architecture, and checkpoint URL.

    model_id format:
    - "rf:uuid" for Roboflow models
    - "trained:uuid" for custom trained models
    """
    if model_id.startswith("rf:"):
        rf_id = model_id.replace("rf:", "")
        try:
            result = supabase_service.client.table("od_roboflow_models").select(
                "id, name, display_name, architecture, classes, checkpoint_url, map, class_count, is_active"
            ).eq("id", rf_id).single().execute()

            if not result.data:
                raise HTTPException(status_code=404, detail=f"Roboflow model not found: {rf_id}")

            model_data = result.data
            return {
                "model_type": "roboflow",
                "id": model_data["id"],
                "name": model_data["name"],
                "display_name": model_data["display_name"],
                "architecture": model_data["architecture"],
                "classes": model_data["classes"],
                "class_count": model_data["class_count"],
                "num_classes_in_array": len(model_data["classes"]) if model_data["classes"] else 0,
                "checkpoint_url": model_data["checkpoint_url"],
                "map": model_data["map"],
                "is_active": model_data["is_active"],
                "_debug_info": {
                    "what_worker_receives": {
                        "model": "roboflow",
                        "model_config": {
                            "checkpoint_url": model_data["checkpoint_url"],
                            "architecture": model_data["architecture"],
                            "classes": model_data["classes"],
                        }
                    }
                }
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    elif model_id.startswith("trained:"):
        trained_id = model_id.replace("trained:", "")
        try:
            result = supabase_service.client.table("od_trained_models").select(
                "id, name, model_type, checkpoint_url, class_mapping, map, map_50, is_active"
            ).eq("id", trained_id).single().execute()

            if not result.data:
                raise HTTPException(status_code=404, detail=f"Trained model not found: {trained_id}")

            model_data = result.data

            # Resolve class names from class_mapping
            class_mapping = model_data.get("class_mapping", {})
            classes = []
            if class_mapping:
                sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
                class_ids = [class_id for class_id, _ in sorted_classes]

                if class_ids:
                    classes_result = supabase_service.client.table("od_classes").select(
                        "id, name"
                    ).in_("id", class_ids).execute()

                    class_names_map = {c["id"]: c["name"] for c in classes_result.data or []}
                    classes = [class_names_map.get(class_id, f"class_{idx}") for class_id, idx in sorted_classes]

            return {
                "model_type": "trained",
                "id": model_data["id"],
                "name": model_data["name"],
                "architecture": model_data["model_type"],
                "class_mapping": class_mapping,
                "resolved_classes": classes,
                "num_classes": len(classes),
                "checkpoint_url": model_data["checkpoint_url"],
                "map": model_data["map"],
                "map_50": model_data["map_50"],
                "is_active": model_data["is_active"],
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model_id format. Use 'rf:uuid' or 'trained:uuid'"
        )
