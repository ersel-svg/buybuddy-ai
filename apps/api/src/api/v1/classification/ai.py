"""
Classification AI Labeling Router

AI-powered auto-labeling for image classification:
- /predict: Single image classification (real-time)
- /batch: Bulk classification job (async)
- /jobs/{job_id}: Job status
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import settings
from services.runpod import runpod_service, EndpointType
from services.supabase import supabase_service

router = APIRouter()


# Request/Response Models
class CLSAIPredictRequest(BaseModel):
    """Request for single image classification."""
    image_id: str
    dataset_id: str
    model: str = "clip"  # clip, dinov2, or trained model ID
    top_k: int = 5
    threshold: float = 0.1


class CLSAIPrediction(BaseModel):
    """Single classification prediction."""
    class_id: str
    class_name: str
    confidence: float


class CLSAIPredictResponse(BaseModel):
    """Response for single image classification."""
    predictions: list[CLSAIPrediction]
    model: str
    processing_time_ms: int


class CLSAIBatchRequest(BaseModel):
    """Request for batch classification."""
    dataset_id: str
    image_ids: Optional[list[str]] = None  # If None, process all pending
    model: str = "clip"
    top_k: int = 1  # For single-label, use top_k=1
    threshold: float = 0.3
    auto_accept: bool = False  # Automatically accept predictions above threshold
    overwrite_existing: bool = False


class CLSAIBatchResponse(BaseModel):
    """Response for batch classification job."""
    job_id: str
    status: str
    total_images: int
    message: str


class CLSAIJobStatusResponse(BaseModel):
    """Response for job status."""
    job_id: str
    status: str
    progress: int
    total_images: int
    predictions_generated: int
    labels_created: int
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# Available models
CLIP_MODELS = ["clip", "clip-vit-b-32", "clip-vit-l-14"]
DINOV2_MODELS = ["dinov2", "dinov2-vit-s-14", "dinov2-vit-b-14"]


@router.get("/models")
async def list_available_models() -> dict:
    """List available AI models for classification."""

    base_models = [
        {
            "id": "clip",
            "name": "CLIP ViT-B/32",
            "description": "OpenAI CLIP for zero-shot classification. Fast and accurate.",
            "model_type": "zero_shot",
            "requires_classes": True,
        },
        {
            "id": "clip-vit-l-14",
            "name": "CLIP ViT-L/14",
            "description": "Larger CLIP model. More accurate but slower.",
            "model_type": "zero_shot",
            "requires_classes": True,
        },
        {
            "id": "dinov2",
            "name": "DINOv2 ViT-B/14",
            "description": "Meta DINOv2 for zero-shot classification.",
            "model_type": "zero_shot",
            "requires_classes": True,
        },
    ]

    # Add trained models from database
    trained_models = []
    try:
        result = supabase_service.client.table("cls_trained_models").select(
            "id, name, model_type, model_size, num_classes, accuracy"
        ).eq("is_active", True).execute()

        for model in result.data or []:
            acc_str = f"{model['accuracy']*100:.1f}%" if model.get('accuracy') else ""
            trained_models.append({
                "id": f"trained:{model['id']}",
                "name": model["name"],
                "description": f"{model['model_type']} {model.get('model_size', '')} - {model['num_classes']} classes {acc_str}".strip(),
                "model_type": "trained",
                "requires_classes": False,  # Trained models have their own classes
                "num_classes": model["num_classes"],
            })
    except Exception as e:
        print(f"[CLS AI] Failed to fetch trained models: {e}")

    return {
        "zero_shot_models": base_models,
        "trained_models": trained_models,
    }


@router.post("/predict", response_model=CLSAIPredictResponse)
async def predict_single(request: CLSAIPredictRequest) -> CLSAIPredictResponse:
    """
    Run AI classification on a single image (real-time).

    Uses CLIP or DINOv2 for zero-shot classification based on dataset classes.
    Can also use trained classification models.
    """
    # Check endpoint configuration
    if not runpod_service.is_configured(EndpointType.CLS_ANNOTATION):
        # Fallback to OD annotation endpoint which can also run CLIP
        if not runpod_service.is_configured(EndpointType.OD_ANNOTATION):
            raise HTTPException(
                status_code=503,
                detail="Classification AI endpoint not configured"
            )

    # Get image URL from database
    image_result = supabase_service.client.table("cls_images").select(
        "id, image_url"
    ).eq("id", request.image_id).single().execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_result.data["image_url"]

    # Get dataset classes
    classes_result = supabase_service.client.table("cls_classes").select(
        "id, name, display_name"
    ).eq("dataset_id", request.dataset_id).eq("is_active", True).execute()

    classes = classes_result.data or []
    if not classes:
        raise HTTPException(
            status_code=400,
            detail="Dataset has no classes defined"
        )

    # Prepare class names for CLIP
    class_names = [c.get("display_name") or c["name"] for c in classes]
    class_map = {(c.get("display_name") or c["name"]): c["id"] for c in classes}

    # Check if using trained model
    is_trained_model = request.model.startswith("trained:")

    if is_trained_model:
        # Get trained model details
        model_id = request.model.replace("trained:", "")
        model_result = supabase_service.client.table("cls_trained_models").select(
            "id, model_type, model_path, class_names"
        ).eq("id", model_id).eq("is_active", True).single().execute()

        if not model_result.data:
            raise HTTPException(status_code=404, detail="Trained model not found")

        runpod_input = {
            "task": "classify",
            "model": "trained",
            "model_path": model_result.data["model_path"],
            "model_type": model_result.data["model_type"],
            "image_url": image_url,
            "top_k": request.top_k,
        }
        # Map class indices back to dataset classes
        model_classes = model_result.data.get("class_names", [])
        class_map = {}
        for i, name in enumerate(model_classes):
            # Try to find matching class in dataset
            matching = next((c for c in classes if c["name"] == name or c.get("display_name") == name), None)
            if matching:
                class_map[str(i)] = matching["id"]
    else:
        # Zero-shot model (CLIP, DINOv2)
        runpod_input = {
            "task": "classify",
            "model": request.model,
            "image_url": image_url,
            "class_names": class_names,
            "top_k": request.top_k,
        }

    start_time = time.time()

    try:
        # Try CLS endpoint first, fallback to OD endpoint
        endpoint = EndpointType.CLS_ANNOTATION if runpod_service.is_configured(EndpointType.CLS_ANNOTATION) else EndpointType.OD_ANNOTATION

        result = await runpod_service.submit_job_sync(
            endpoint_type=endpoint,
            input_data=runpod_input,
            timeout=30,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        if result.get("status") == "FAILED":
            error = result.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Classification failed: {error}")

        output = result.get("output", {})
        raw_predictions = output.get("predictions", [])

        # Convert to response format
        predictions = []
        for pred in raw_predictions:
            class_name = pred.get("label", pred.get("class_name", ""))
            confidence = pred.get("confidence", 0)

            if confidence < request.threshold:
                continue

            # Map class name to ID
            class_id = class_map.get(class_name)
            if not class_id and is_trained_model:
                class_id = class_map.get(str(pred.get("class_index", "")))

            if class_id:
                predictions.append(CLSAIPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                ))

        return CLSAIPredictResponse(
            predictions=predictions[:request.top_k],
            model=request.model,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Classification timed out")
    except Exception as e:
        print(f"[CLS AI Predict] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=CLSAIBatchResponse)
async def batch_classify(
    request: CLSAIBatchRequest,
    background_tasks: BackgroundTasks,
) -> CLSAIBatchResponse:
    """
    Start a bulk AI classification job (async).

    Processes multiple images and optionally saves predictions as labels.
    """
    # Check endpoint configuration
    endpoint = None
    if runpod_service.is_configured(EndpointType.CLS_ANNOTATION):
        endpoint = EndpointType.CLS_ANNOTATION
    elif runpod_service.is_configured(EndpointType.OD_ANNOTATION):
        endpoint = EndpointType.OD_ANNOTATION
    else:
        raise HTTPException(
            status_code=503,
            detail="Classification AI endpoint not configured"
        )

    # Verify dataset exists
    dataset_result = supabase_service.client.table("cls_datasets").select(
        "id, name, task_type"
    ).eq("id", request.dataset_id).single().execute()

    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task_type = dataset_result.data.get("task_type", "single_label")

    # Get dataset classes
    classes_result = supabase_service.client.table("cls_classes").select(
        "id, name, display_name"
    ).eq("dataset_id", request.dataset_id).eq("is_active", True).execute()

    classes = classes_result.data or []
    if not classes:
        raise HTTPException(
            status_code=400,
            detail="Dataset has no classes defined"
        )

    class_names = [c.get("display_name") or c["name"] for c in classes]
    class_map = {(c.get("display_name") or c["name"]): c["id"] for c in classes}

    # Get images to process
    if request.image_ids:
        images_query = supabase_service.client.table("cls_dataset_images").select(
            "image_id, cls_images!inner(id, image_url)"
        ).eq("dataset_id", request.dataset_id).in_("image_id", request.image_ids)
    else:
        # All pending images in dataset
        images_query = supabase_service.client.table("cls_dataset_images").select(
            "image_id, cls_images!inner(id, image_url)"
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
        "type": "cls_annotation",
        "status": "pending",
        "config": {
            "dataset_id": request.dataset_id,
            "model": request.model,
            "class_names": class_names,
            "class_map": class_map,
            "top_k": request.top_k,
            "threshold": request.threshold,
            "auto_accept": request.auto_accept,
            "overwrite_existing": request.overwrite_existing,
            "task_type": task_type,
            "total_images": len(images),
            "image_ids": [img["image_id"] for img in images],
        },
        "created_at": datetime.utcnow().isoformat(),
    }

    supabase_service.client.table("jobs").insert(job_data).execute()

    # Prepare image data for batch processing
    image_data = []
    for img in images:
        cls_image = img.get("cls_images", {})
        if cls_image:
            image_url = cls_image.get("image_url")
            if image_url:
                image_data.append({
                    "id": img["image_id"],
                    "url": image_url,
                })

    # Prepare RunPod input
    runpod_input = {
        "task": "batch_classify",
        "model": request.model,
        "images": image_data,
        "class_names": class_names,
        "top_k": request.top_k,
        "job_id": job_id,
    }

    try:
        # Submit async job to RunPod
        runpod_response = await runpod_service.submit_job(
            endpoint_type=endpoint,
            input_data=runpod_input,
        )

        runpod_job_id = runpod_response.get("id")

        # Update job with RunPod ID
        supabase_service.client.table("jobs").update({
            "runpod_job_id": runpod_job_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()

        return CLSAIBatchResponse(
            job_id=job_id,
            status="running",
            total_images=len(images),
            message=f"Started AI classification for {len(images)} images",
        )

    except Exception as e:
        # Update job as failed
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": str(e),
        }).eq("id", job_id).execute()

        raise HTTPException(status_code=500, detail=f"Failed to start batch job: {e}")


@router.get("/jobs/{job_id}", response_model=CLSAIJobStatusResponse)
async def get_job_status(job_id: str) -> CLSAIJobStatusResponse:
    """Get batch classification job status."""
    # Get job from database
    job_result = supabase_service.client.table("jobs").select(
        "*"
    ).eq("id", job_id).eq("type", "cls_annotation").single().execute()

    if not job_result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_result.data
    config = job.get("config", {})

    # If job is still running, check RunPod status
    if job.get("status") == "running" and job.get("runpod_job_id"):
        try:
            endpoint = EndpointType.CLS_ANNOTATION if runpod_service.is_configured(EndpointType.CLS_ANNOTATION) else EndpointType.OD_ANNOTATION

            runpod_status = await runpod_service.get_job_status(
                endpoint_type=endpoint,
                job_id=job["runpod_job_id"],
            )

            if runpod_status.get("status") == "COMPLETED":
                output = runpod_status.get("output", {})
                results = output.get("results", [])

                # Save predictions as labels
                labels_created = 0
                if results:
                    predictions_by_image = {
                        r["id"]: r.get("predictions", [])
                        for r in results if r.get("id")
                    }
                    labels_created = await _save_predictions_as_labels(
                        dataset_id=config["dataset_id"],
                        predictions_by_image=predictions_by_image,
                        class_map=config.get("class_map", {}),
                        threshold=config.get("threshold", 0.3),
                        auto_accept=config.get("auto_accept", False),
                        overwrite_existing=config.get("overwrite_existing", False),
                        task_type=config.get("task_type", "single_label"),
                    )

                result_data = {
                    **output,
                    "total_predictions": sum(len(r.get("predictions", [])) for r in results),
                    "labels_created": labels_created,
                }
                supabase_service.client.table("jobs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "result": result_data,
                }).eq("id", job_id).execute()

                job["status"] = "completed"
                job["result"] = result_data

            elif runpod_status.get("status") == "FAILED":
                error = runpod_status.get("error", "Unknown error")
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error": error,
                }).eq("id", job_id).execute()

                job["status"] = "failed"
                job["error"] = error

        except Exception as e:
            print(f"[CLS AI Job Status] Error checking RunPod: {e}")

    result = job.get("result", {}) or {}

    return CLSAIJobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=config.get("progress", 0),
        total_images=config.get("total_images", 0),
        predictions_generated=result.get("total_predictions", 0),
        labels_created=result.get("labels_created", 0),
        error_message=job.get("error"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
    )


async def _save_predictions_as_labels(
    dataset_id: str,
    predictions_by_image: dict,
    class_map: dict,
    threshold: float,
    auto_accept: bool,
    overwrite_existing: bool,
    task_type: str,
) -> int:
    """
    Save AI predictions as labels in the database.

    Returns number of labels created.
    """
    labels_created = 0

    for image_id, predictions in predictions_by_image.items():
        # Filter by threshold
        valid_predictions = [p for p in predictions if p.get("confidence", 0) >= threshold]

        if not valid_predictions:
            continue

        # For single-label, only use top prediction
        if task_type == "single_label":
            valid_predictions = valid_predictions[:1]

        for pred in valid_predictions:
            class_name = pred.get("label", pred.get("class_name", ""))
            confidence = pred.get("confidence", 0)

            class_id = class_map.get(class_name)
            if not class_id:
                continue

            # Check for existing label
            existing = supabase_service.client.table("cls_labels").select(
                "id"
            ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

            if existing.data and not overwrite_existing:
                continue

            # Delete existing if overwriting
            if existing.data and overwrite_existing:
                supabase_service.client.table("cls_labels").delete().eq(
                    "dataset_id", dataset_id
                ).eq("image_id", image_id).execute()

            # Create new label
            label_data = {
                "id": str(uuid.uuid4()),
                "dataset_id": dataset_id,
                "image_id": image_id,
                "class_id": class_id,
                "confidence": confidence,
                "is_ai_generated": True,
                "is_reviewed": auto_accept,
                "created_at": datetime.utcnow().isoformat(),
            }

            try:
                supabase_service.client.table("cls_labels").insert(label_data).execute()
                labels_created += 1

                # Update image status
                new_status = "labeled" if auto_accept else "review"
                supabase_service.client.table("cls_dataset_images").update({
                    "status": new_status,
                }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

            except Exception as e:
                print(f"[CLS AI] Failed to save label for {image_id}: {e}")

    # Update dataset stats
    if labels_created > 0:
        try:
            supabase_service.client.rpc("update_cls_dataset_stats", {
                "p_dataset_id": dataset_id
            }).execute()
        except Exception as e:
            print(f"[CLS AI] Failed to update stats: {e}")

    return labels_created
