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
DETECT_MODELS = ["grounding_dino", "sam3", "florence2"]
SEGMENT_MODELS = ["sam2", "sam3"]


@router.post("/predict", response_model=AIPredictResponse)
async def predict_single(request: AIPredictRequest) -> AIPredictResponse:
    """
    Run AI prediction on a single image (real-time).

    Supported models:
    - grounding_dino: Text prompt → bounding boxes (SOTA open-vocabulary)
    - sam3: Text prompt → mask → bounding box
    - florence2: Versatile vision tasks
    """
    # Validate model
    if request.model not in DETECT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Supported: {DETECT_MODELS}"
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

    # Prepare RunPod input
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

        return AIPredictResponse(
            predictions=predictions,
            model=request.model,
            processing_time_ms=processing_time_ms,
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
    """
    # Validate model
    if request.model not in DETECT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Supported: {DETECT_MODELS}"
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

    # Get images to process
    if request.image_ids:
        # Specific images
        images_query = supabase_service.client.table("od_dataset_images").select(
            "image_id, od_images!inner(id, image_url)"
        ).eq("dataset_id", request.dataset_id).in_("image_id", request.image_ids)
    else:
        # All unannotated images in dataset
        images_query = supabase_service.client.table("od_dataset_images").select(
            "image_id, od_images!inner(id, image_url)"
        ).eq("dataset_id", request.dataset_id).eq("status", "pending")

    images_result = images_query.execute()
    images = images_result.data or []

    if not images:
        raise HTTPException(
            status_code=400,
            detail="No images found to process"
        )

    # Create job record
    job_id = str(uuid.uuid4())

    job_data = {
        "id": job_id,
        "type": "od_annotation",
        "status": "pending",
        "metadata": {
            "dataset_id": request.dataset_id,
            "model": request.model,
            "text_prompt": request.text_prompt,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "auto_accept": request.auto_accept,
            "class_mapping": request.class_mapping,
            "total_images": len(images),
            "image_ids": [img["image_id"] for img in images],
        },
        "created_at": datetime.utcnow().isoformat(),
    }

    supabase_service.client.table("jobs").insert(job_data).execute()

    # Prepare image URLs for batch processing
    image_data = []
    for img in images:
        od_image = img.get("od_images", {})
        if od_image:
            image_data.append({
                "image_id": img["image_id"],
                "image_url": od_image.get("image_url"),
            })

    # Prepare RunPod input
    runpod_input = {
        "task": "batch",
        "model": request.model,
        "images": image_data,
        "text_prompt": request.text_prompt,
        "box_threshold": request.box_threshold,
        "text_threshold": request.text_threshold,
        "job_id": job_id,
        "hf_token": settings.hf_token,  # For SAM3 model access
    }

    # TODO: Add webhook URL for production
    # webhook_url = f"{settings.api_base_url}/api/v1/od/ai/webhook"

    try:
        # Submit async job to RunPod
        runpod_response = await runpod_service.submit_job(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data=runpod_input,
            # webhook_url=webhook_url,
        )

        runpod_job_id = runpod_response.get("id")

        # Update job with RunPod ID
        supabase_service.client.table("jobs").update({
            "runpod_job_id": runpod_job_id,
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()

        return AIBatchJobResponse(
            job_id=job_id,
            status="processing",
            total_images=len(images),
            message=f"Started AI annotation for {len(images)} images",
        )

    except Exception as e:
        # Update job as failed
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", job_id).execute()

        raise HTTPException(status_code=500, detail=f"Failed to start batch job: {e}")


@router.get("/jobs/{job_id}", response_model=AIJobStatusResponse)
async def get_job_status(job_id: str) -> AIJobStatusResponse:
    """Get batch annotation job status."""
    # Get job from database
    job_result = supabase_service.client.table("jobs").select(
        "*"
    ).eq("id", job_id).eq("type", "od_annotation").single().execute()

    if not job_result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_result.data
    metadata = job.get("metadata", {})

    # If job is still processing, check RunPod status
    if job.get("status") == "processing" and job.get("runpod_job_id"):
        try:
            runpod_status = await runpod_service.get_job_status(
                endpoint_type=EndpointType.OD_ANNOTATION,
                job_id=job["runpod_job_id"],
            )

            # Update job if completed
            if runpod_status.get("status") == "COMPLETED":
                output = runpod_status.get("output", {})
                predictions_count = output.get("total_predictions", 0)

                supabase_service.client.table("jobs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": output,
                }).eq("id", job_id).execute()

                job["status"] = "completed"
                metadata["predictions_generated"] = predictions_count

            elif runpod_status.get("status") == "FAILED":
                error = runpod_status.get("error", "Unknown error")
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error_message": error,
                }).eq("id", job_id).execute()

                job["status"] = "failed"
                job["error_message"] = error

        except Exception as e:
            print(f"[AI Job Status] Error checking RunPod: {e}")

    return AIJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=metadata.get("progress", 0),
        total_images=metadata.get("total_images", 0),
        predictions_generated=metadata.get("predictions_generated", 0),
        error_message=job.get("error_message"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


@router.post("/webhook")
async def runpod_webhook(payload: AIWebhookPayload) -> dict:
    """
    Handle RunPod webhook callback for batch jobs.

    Updates job status and optionally saves predictions as annotations.
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
    metadata = job.get("metadata", {})

    if payload.status == "COMPLETED" and payload.output:
        output = payload.output
        predictions_by_image = output.get("predictions", {})
        total_predictions = output.get("total_predictions", 0)

        # Optionally save predictions as annotations
        if metadata.get("auto_accept") and predictions_by_image:
            await _save_predictions_as_annotations(
                dataset_id=metadata["dataset_id"],
                predictions_by_image=predictions_by_image,
                class_mapping=metadata.get("class_mapping"),
                model=metadata.get("model"),
            )

        # Update job status
        supabase_service.client.table("jobs").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": {
                "total_predictions": total_predictions,
                "images_processed": len(predictions_by_image),
            },
        }).eq("id", job_id).execute()

        return {"status": "success", "job_id": job_id}

    elif payload.status == "FAILED":
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error_message": payload.error or "Unknown error",
        }).eq("id", job_id).execute()

        return {"status": "failed", "job_id": job_id, "error": payload.error}

    return {"status": "ignored", "reason": f"unhandled_status_{payload.status}"}


async def _save_predictions_as_annotations(
    dataset_id: str,
    predictions_by_image: dict,
    class_mapping: Optional[dict],
    model: str,
) -> int:
    """
    Save AI predictions as annotations in the database.

    Returns number of annotations created.
    """
    annotations_created = 0

    for image_id, predictions in predictions_by_image.items():
        for pred in predictions:
            bbox = pred.get("bbox", {})
            label = pred.get("label", "")
            confidence = pred.get("confidence", 0)

            # Map label to class ID
            class_id = None
            if class_mapping and label in class_mapping:
                class_id = class_mapping[label]
            else:
                # Try to find existing class by name
                class_result = supabase_service.client.table("od_classes").select(
                    "id"
                ).eq("dataset_id", dataset_id).eq("name", label).single().execute()

                if class_result.data:
                    class_id = class_result.data["id"]

            if not class_id:
                # Skip predictions without class mapping
                continue

            # Create annotation
            annotation_data = {
                "id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "image_id": image_id,
                "class_id": class_id,
                "bbox": bbox,
                "is_ai_generated": True,
                "confidence": confidence,
                "ai_model": model,
                "is_reviewed": False,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

            try:
                supabase_service.client.table("od_annotations").insert(
                    annotation_data
                ).execute()
                annotations_created += 1
            except Exception as e:
                print(f"[AI Annotation] Failed to save annotation: {e}")

    return annotations_created


@router.get("/models")
async def list_available_models() -> dict:
    """List available AI models and their capabilities."""
    return {
        "detection_models": [
            {
                "id": "grounding_dino",
                "name": "Grounding DINO",
                "description": "SOTA open-vocabulary object detection. Best for text→bbox.",
                "tasks": ["detect"],
                "requires_prompt": True,
            },
            {
                "id": "sam3",
                "name": "SAM 3",
                "description": "Segment Anything Model 3 with text prompt support.",
                "tasks": ["detect", "segment"],
                "requires_prompt": True,
            },
            {
                "id": "florence2",
                "name": "Florence-2",
                "description": "Microsoft's versatile vision model.",
                "tasks": ["detect", "caption"],
                "requires_prompt": False,
            },
        ],
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
