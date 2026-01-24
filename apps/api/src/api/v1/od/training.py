"""
Object Detection - Training Router

Endpoints for managing OD training runs.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Header

from config import settings
from services.supabase import supabase_service
from services.runpod import runpod_service, EndpointType
from schemas.od import (
    ODTrainingRunCreate,
    ODTrainingRunResponse,
)

# Thread pool for CPU-bound/blocking operations
_executor = ThreadPoolExecutor(max_workers=2)

router = APIRouter()


@router.get("", response_model=list[ODTrainingRunResponse])
async def list_training_runs(
    dataset_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    """List training runs with optional filters."""
    query = supabase_service.client.table("od_training_runs").select("*")

    if dataset_id:
        query = query.eq("dataset_id", dataset_id)
    if status:
        query = query.eq("status", status)

    query = query.order("created_at", desc=True).limit(limit)
    result = query.execute()

    return result.data or []


@router.get("/{training_id}", response_model=ODTrainingRunResponse)
async def get_training_run(training_id: str):
    """Get a single training run by ID."""
    result = supabase_service.client.table("od_training_runs").select("*").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    return result.data


@router.post("", response_model=ODTrainingRunResponse)
async def create_training_run(data: ODTrainingRunCreate, background_tasks: BackgroundTasks):
    """
    Create and start a new training run.

    This will:
    1. Create training run record
    2. Trigger RunPod training job (URL-based mode)
    3. Worker fetches data directly from Supabase (no ZIP export!)
    4. Track progress via direct Supabase writes
    """
    import json
    print(f"[DEBUG] Received training request: name={data.name}, dataset_id={data.dataset_id}")
    print(f"[DEBUG] Model: {data.model_type}/{data.model_size}")
    if data.config:
        try:
            config_dict = data.config.model_dump()
            print(f"[DEBUG] Config keys: {list(config_dict.keys())}")
        except Exception as e:
            print(f"[DEBUG] Error dumping config: {e}")

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", data.dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if dataset has annotations
    if dataset.data.get("annotation_count", 0) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no annotations")

    # Get dataset version if specified
    version = None
    if data.dataset_version_id:
        version = supabase_service.client.table("od_dataset_versions").select("*").eq("id", data.dataset_version_id).single().execute()
        if not version.data:
            raise HTTPException(status_code=404, detail="Dataset version not found")

    # Get classes used in this dataset's annotations
    annotations = supabase_service.client.table("od_annotations").select("class_id").eq("dataset_id", data.dataset_id).execute()
    class_ids = list(set(a["class_id"] for a in annotations.data or []))

    if class_ids:
        classes = supabase_service.client.table("od_classes").select("id, name").in_("id", class_ids).eq("is_active", True).order("name").execute()
        class_names = [c["name"] for c in classes.data or []]
    else:
        class_names = []

    # Create training config
    config = data.config.model_dump() if data.config else {}
    training_id = str(uuid4())

    # Create training run record
    training_data = {
        "id": training_id,
        "name": data.name,
        "description": data.description,
        "dataset_id": data.dataset_id,
        "dataset_version_id": data.dataset_version_id,
        "model_type": data.model_type,
        "model_size": data.model_size,
        "config": config,
        "status": "pending",
        "current_epoch": 0,
        "total_epochs": config.get("epochs", 100),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("od_training_runs").insert(training_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create training run")

    # Start training in background
    background_tasks.add_task(
        start_training_job,
        training_id=training_id,
        dataset_id=data.dataset_id,
        version_id=data.dataset_version_id,
        model_type=data.model_type,
        model_size=data.model_size,
        config=config,
        class_names=class_names,
    )

    return result.data[0]


async def start_training_job(
    training_id: str,
    dataset_id: str,
    version_id: Optional[str],
    model_type: str,
    model_size: str,
    config: dict,
    class_names: list,
):
    """
    Background task to start training job.

    NEW: URL-based approach - no ZIP export!
    Worker fetches data directly from Supabase with pagination.
    This is much faster and doesn't block the API for large datasets.
    """
    try:
        print(f"[TRAINING] Starting job {training_id}")

        # Update status to queued (no more preparing/export phase!)
        print(f"[TRAINING] Updating status to queued...")
        supabase_service.client.table("od_training_runs").update({
            "status": "queued",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", training_id).execute()
        print(f"[TRAINING] Status updated to queued")

        # Determine format based on model type (worker may use this for annotation format)
        export_format = "yolo" if model_type == "yolo-nas" else "coco"

        # NEW: Build lightweight job payload - no ZIP export needed!
        # Worker will fetch data directly from Supabase
        job_input = {
            "training_run_id": training_id,
            # NEW: URL-based loading - worker fetches from Supabase directly
            "dataset_id": dataset_id,
            "dataset_version_id": version_id,
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            # Legacy field (kept for backward compatibility, worker checks this first)
            "dataset_url": None,
            "dataset_format": export_format,
            # Model config
            "model_type": model_type,
            "model_size": model_size,
            "config": config,
            "class_names": class_names,
            "num_classes": len(class_names),
        }

        print(f"[TRAINING] Submitting job to RunPod (URL-based mode)...")

        # Submit job to RunPod
        # Note: We don't use RunPod's built-in webhook mechanism because
        # the worker implements its own Supabase writes for progress updates.
        # This is more reliable than webhooks.
        job_result = await runpod_service.submit_job(
            endpoint_type=EndpointType.OD_TRAINING,
            input_data=job_input,
            webhook_url=None,  # Worker uses direct Supabase writes
        )

        # Update with job ID
        supabase_service.client.table("od_training_runs").update({
            "runpod_job_id": job_result.get("id"),
            "status": "training",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", training_id).execute()

        print(f"[TRAINING] Job submitted successfully: {job_result.get('id')}")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[TRAINING] Error in job {training_id}: {e}")
        print(f"[TRAINING] Traceback:\n{error_traceback}")

        # Update status to failed
        supabase_service.client.table("od_training_runs").update({
            "status": "failed",
            "error_message": str(e),
            "error_traceback": error_traceback,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", training_id).execute()
        raise


@router.post("/{training_id}/cancel")
async def cancel_training_run(training_id: str):
    """Cancel a training run."""
    # Get training run
    result = supabase_service.client.table("od_training_runs").select("*").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    training = result.data

    # Check if can be cancelled
    if training["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel training in {training['status']} state")

    # Cancel RunPod job if running
    if training.get("runpod_job_id"):
        try:
            await runpod_service.cancel_job(
                endpoint_type=EndpointType.OD_TRAINING,
                job_id=training["runpod_job_id"]
            )
        except Exception:
            pass  # Ignore errors when cancelling

    # Update status
    supabase_service.client.table("od_training_runs").update({
        "status": "cancelled",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", training_id).execute()

    return {"status": "cancelled", "id": training_id}


@router.delete("/{training_id}")
async def delete_training_run(training_id: str):
    """Delete a training run."""
    # Get training run
    result = supabase_service.client.table("od_training_runs").select("*").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    training = result.data

    # Cancel if still running
    if training["status"] in ["pending", "preparing", "queued", "training", "running"]:
        if training.get("runpod_job_id"):
            try:
                await runpod_service.cancel_job(
                    endpoint_type=EndpointType.OD_TRAINING,
                    job_id=training["runpod_job_id"]
                )
            except Exception:
                pass

    # Delete from database
    supabase_service.client.table("od_training_runs").delete().eq("id", training_id).execute()

    return {"status": "deleted", "id": training_id}


@router.get("/{training_id}/metrics")
async def get_training_metrics(training_id: str):
    """Get training metrics history."""
    result = supabase_service.client.table("od_training_runs").select("metrics_history").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    return result.data.get("metrics_history", [])


@router.get("/{training_id}/logs")
async def get_training_logs(training_id: str, limit: int = 100):
    """Get training logs."""
    result = supabase_service.client.table("od_training_runs").select("logs").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    logs = result.data.get("logs", [])
    return logs[-limit:] if logs else []


@router.post("/webhook")
async def training_webhook(
    payload: dict,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
    authorization: Optional[str] = Header(None),
    apikey: Optional[str] = Header(None),
):
    """
    Webhook endpoint for training progress updates from RunPod.

    Authentication: Accepts X-Webhook-Secret, Authorization Bearer, or apikey headers.
    The webhook is called by the RunPod training worker.

    Expected payload:
    {
        "training_run_id": "uuid",
        "status": "training" | "completed" | "failed",
        "progress": 50,
        "current_epoch": 25,
        "metrics": {"loss": 0.5, "map": 0.6},
        "model_url": "https://...",
        "error": "error message"
    }
    """
    # Verify authentication - accept multiple methods
    is_authenticated = False

    # Method 1: X-Webhook-Secret header
    webhook_secret = getattr(settings, 'webhook_secret', None) or getattr(settings, 'WEBHOOK_SECRET', None)
    if x_webhook_secret and webhook_secret and x_webhook_secret == webhook_secret:
        is_authenticated = True

    # Method 2: Supabase service key (apikey or Authorization Bearer)
    service_key = settings.supabase_service_role_key
    if service_key:
        if apikey == service_key:
            is_authenticated = True
        if authorization and authorization.replace("Bearer ", "") == service_key:
            is_authenticated = True

    # Method 3: If no secrets configured, allow (development mode)
    if not webhook_secret and not service_key:
        is_authenticated = True

    if not is_authenticated:
        raise HTTPException(status_code=401, detail="Webhook authentication failed")

    training_run_id = payload.get("training_run_id")
    if not training_run_id:
        raise HTTPException(status_code=400, detail="training_run_id required")

    status = payload.get("status")
    progress = payload.get("progress", 0)
    current_epoch = payload.get("current_epoch", 0)
    metrics = payload.get("metrics", {})
    model_url = payload.get("model_url")
    error = payload.get("error")

    # Get current training run
    result = supabase_service.client.table("od_training_runs").select("*").eq("id", training_run_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    training = result.data

    # Build update
    update_data = {
        "status": status,
        "current_epoch": current_epoch,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Update metrics
    if metrics:
        current_map = metrics.get("map", 0)
        if current_map > (training.get("best_map") or 0):
            update_data["best_map"] = current_map
            update_data["best_epoch"] = current_epoch

        # Append to metrics history
        metrics_history = training.get("metrics_history", []) or []
        metrics_history.append({
            "epoch": current_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        })
        update_data["metrics_history"] = metrics_history

    # Handle completion
    if status == "completed":
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Create trained model record
        if model_url:
            model_data = {
                "id": str(uuid4()),
                "training_run_id": training_run_id,
                "name": f"{training['name']} Model",
                "model_type": training["model_type"],
                "checkpoint_url": model_url,
                "map": update_data.get("best_map"),
                "map_50": metrics.get("map_50"),
                "class_count": len(training.get("config", {}).get("class_names", [])),
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            supabase_service.client.table("od_trained_models").insert(model_data).execute()

    # Handle failure
    if status == "failed" and error:
        update_data["error_message"] = error

    # Update training run
    supabase_service.client.table("od_training_runs").update(update_data).eq("id", training_run_id).execute()

    return {"status": "ok"}


@router.get("/{training_run_id}/metrics-history")
async def get_metrics_history(training_run_id: str):
    """
    Get epoch-by-epoch metrics history for a training run.
    Fetches from unified training_metrics_history table.
    """
    result = supabase_service.client.table("training_metrics_history").select(
        "epoch, train_loss, val_loss, map, map50, map75, learning_rate, created_at"
    ).eq(
        "training_run_id", training_run_id
    ).eq(
        "training_type", "od"
    ).order(
        "epoch", desc=False
    ).execute()

    return result.data or []
