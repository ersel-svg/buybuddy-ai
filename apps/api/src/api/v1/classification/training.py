"""
Classification - Training Router

Endpoints for managing classification model training runs.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSTrainingRunCreate,
    CLSTrainingRunUpdate,
    CLSTrainingRunResponse,
)

router = APIRouter()

# Augmentation presets
AUGMENTATION_PRESETS = {
    "sota": {
        "name": "SOTA (Recommended)",
        "description": "RandAugment + MixUp + CutMix + Label Smoothing",
        "training_time_factor": 1.3,
        "accuracy_boost": "+2-4%",
        "config": {
            "randaugment": {"n": 2, "m": 9},
            "mixup": {"alpha": 0.8, "prob": 0.5},
            "cutmix": {"alpha": 1.0, "prob": 0.5},
            "random_erasing": {"prob": 0.25},
            "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
            "horizontal_flip": {"prob": 0.5},
        }
    },
    "heavy": {
        "name": "Heavy (Small Datasets)",
        "description": "TrivialAugmentWide + Strong regularization",
        "training_time_factor": 1.8,
        "accuracy_boost": "+4-6%",
        "config": {
            "trivialaugment": True,
            "mixup": {"alpha": 1.0, "prob": 0.8},
            "cutmix": {"alpha": 1.0, "prob": 0.8},
            "random_erasing": {"prob": 0.4},
            "color_jitter": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.2},
            "horizontal_flip": {"prob": 0.5},
            "vertical_flip": {"prob": 0.2},
            "rotation": {"degrees": 15},
        }
    },
    "medium": {
        "name": "Medium (Balanced)",
        "description": "Standard augmentations",
        "training_time_factor": 1.2,
        "accuracy_boost": "+1-2%",
        "config": {
            "mixup": {"alpha": 0.4, "prob": 0.3},
            "cutmix": {"alpha": 0.5, "prob": 0.3},
            "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.05},
            "horizontal_flip": {"prob": 0.5},
        }
    },
    "light": {
        "name": "Light (Large Datasets)",
        "description": "Basic augmentations only",
        "training_time_factor": 1.05,
        "accuracy_boost": "+0.5-1%",
        "config": {
            "horizontal_flip": {"prob": 0.5},
            "color_jitter": {"brightness": 0.2, "contrast": 0.2},
        }
    },
    "none": {
        "name": "None (Baseline)",
        "description": "No augmentation",
        "training_time_factor": 1.0,
        "accuracy_boost": "Baseline",
        "config": {}
    }
}

# Model configurations
MODEL_CONFIGS = {
    "vit": {
        "name": "Vision Transformer",
        "sizes": {
            "tiny": {"hf_id": "google/vit-base-patch16-224", "embed_dim": 192, "params": "5.7M"},
            "small": {"hf_id": "google/vit-small-patch16-224", "embed_dim": 384, "params": "22M"},
            "base": {"hf_id": "google/vit-base-patch16-224", "embed_dim": 768, "params": "86M"},
            "large": {"hf_id": "google/vit-large-patch16-224", "embed_dim": 1024, "params": "304M"},
        }
    },
    "convnext": {
        "name": "ConvNeXt v2",
        "sizes": {
            "tiny": {"hf_id": "facebook/convnext-tiny-224", "embed_dim": 768, "params": "28M"},
            "small": {"hf_id": "facebook/convnext-small-224", "embed_dim": 768, "params": "50M"},
            "base": {"hf_id": "facebook/convnext-base-224", "embed_dim": 1024, "params": "89M"},
            "large": {"hf_id": "facebook/convnext-large-224", "embed_dim": 1536, "params": "198M"},
        }
    },
    "efficientnet": {
        "name": "EfficientNet v2",
        "sizes": {
            "s": {"timm_id": "efficientnetv2_rw_s", "embed_dim": 1792, "params": "21M"},
            "m": {"timm_id": "efficientnetv2_rw_m", "embed_dim": 1792, "params": "54M"},
            "l": {"timm_id": "efficientnetv2_rw_l", "embed_dim": 1792, "params": "120M"},
        }
    },
    "swin": {
        "name": "Swin Transformer v2",
        "sizes": {
            "tiny": {"hf_id": "microsoft/swin-tiny-patch4-window7-224", "embed_dim": 768, "params": "28M"},
            "small": {"hf_id": "microsoft/swin-small-patch4-window7-224", "embed_dim": 768, "params": "50M"},
            "base": {"hf_id": "microsoft/swin-base-patch4-window7-224", "embed_dim": 1024, "params": "88M"},
        }
    },
    "dinov2": {
        "name": "DINOv2 (Transfer Learning)",
        "sizes": {
            "small": {"hf_id": "facebook/dinov2-small", "embed_dim": 384, "params": "22M"},
            "base": {"hf_id": "facebook/dinov2-base", "embed_dim": 768, "params": "86M"},
            "large": {"hf_id": "facebook/dinov2-large", "embed_dim": 1024, "params": "300M"},
        }
    },
    "clip": {
        "name": "CLIP (Zero-shot capable)",
        "sizes": {
            "vit-b-16": {"hf_id": "openai/clip-vit-base-patch16", "embed_dim": 512, "params": "86M"},
            "vit-l-14": {"hf_id": "openai/clip-vit-large-patch14", "embed_dim": 768, "params": "304M"},
        }
    },
}


@router.get("/presets")
async def get_augmentation_presets():
    """Get available augmentation presets."""
    return AUGMENTATION_PRESETS


@router.get("/model-configs")
async def get_model_configs():
    """Get supported model configurations."""
    return MODEL_CONFIGS


@router.get("", response_model=list[CLSTrainingRunResponse])
async def list_training_runs(
    status: Optional[str] = None,
    dataset_id: Optional[str] = None,
    model_type: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
):
    """List classification training runs."""
    offset = (page - 1) * limit

    query = supabase_service.client.table("cls_training_runs").select("*", count="exact")

    if status:
        query = query.eq("status", status)
    if dataset_id:
        query = query.eq("dataset_id", dataset_id)
    if model_type:
        query = query.eq("model_type", model_type)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return result.data or []


@router.get("/{training_id}", response_model=CLSTrainingRunResponse)
async def get_training_run(training_id: str):
    """Get a single training run by ID."""
    result = supabase_service.client.table("cls_training_runs").select("*").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    return result.data


@router.post("", response_model=CLSTrainingRunResponse)
async def create_training_run(data: CLSTrainingRunCreate, background_tasks: BackgroundTasks):
    """Create and start a new training run."""
    # Get dataset info
    dataset = supabase_service.client.table("cls_datasets").select("*").eq("id", data.dataset_id).single().execute()

    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get class count from dataset
    labels = supabase_service.client.table("cls_labels").select("class_id").eq("dataset_id", data.dataset_id).execute()
    class_ids = list(set(l["class_id"] for l in labels.data or []))
    num_classes = len(class_ids)

    if num_classes < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 2 classes for training")

    # Create training run record
    training_data = {
        "name": data.name,
        "description": data.description,
        "dataset_id": data.dataset_id,
        "dataset_version_id": data.dataset_version_id,
        "task_type": dataset.data.get("task_type", "single_label"),
        "num_classes": num_classes,
        "model_type": data.config.model_type,
        "model_size": data.config.model_size,
        "config": data.config.model_dump(),
        "total_epochs": data.config.epochs,
        "status": "pending",
    }

    result = supabase_service.client.table("cls_training_runs").insert(training_data).execute()

    training_run = result.data[0]

    # TODO: Submit to RunPod in background
    # background_tasks.add_task(submit_training_job, training_run["id"])

    return training_run


@router.patch("/{training_id}", response_model=CLSTrainingRunResponse)
async def update_training_run(training_id: str, data: CLSTrainingRunUpdate):
    """Update a training run (name, description only)."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("cls_training_runs").update(update_data).eq("id", training_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    return result.data[0]


@router.post("/{training_id}/cancel")
async def cancel_training_run(training_id: str):
    """Cancel a running training job."""
    # Get training run
    training = supabase_service.client.table("cls_training_runs").select("status, runpod_job_id").eq("id", training_id).single().execute()

    if not training.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    if training.data["status"] not in ["pending", "preparing", "queued", "training"]:
        raise HTTPException(status_code=400, detail="Training run cannot be cancelled in current state")

    # TODO: Cancel RunPod job if running
    # if training.data.get("runpod_job_id"):
    #     cancel_runpod_job(training.data["runpod_job_id"])

    # Update status
    supabase_service.client.table("cls_training_runs").update({
        "status": "cancelled",
    }).eq("id", training_id).execute()

    return {"success": True, "message": "Training run cancelled"}


@router.delete("/{training_id}")
async def delete_training_run(training_id: str):
    """Delete a training run."""
    # Get training run
    training = supabase_service.client.table("cls_training_runs").select("status").eq("id", training_id).single().execute()

    if not training.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    if training.data["status"] in ["training", "preparing"]:
        raise HTTPException(status_code=400, detail="Cannot delete a running training job")

    supabase_service.client.table("cls_training_runs").delete().eq("id", training_id).execute()

    return {"success": True, "message": "Training run deleted"}


@router.get("/{training_id}/metrics")
async def get_training_metrics(training_id: str):
    """Get metrics history for a training run."""
    result = supabase_service.client.table("cls_training_runs").select("metrics_history, current_epoch, total_epochs, best_accuracy, best_f1, best_top5_accuracy, best_epoch").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    return {
        "metrics_history": result.data.get("metrics_history") or [],
        "current_epoch": result.data.get("current_epoch", 0),
        "total_epochs": result.data.get("total_epochs", 0),
        "best_metrics": {
            "accuracy": result.data.get("best_accuracy"),
            "f1": result.data.get("best_f1"),
            "top5_accuracy": result.data.get("best_top5_accuracy"),
            "epoch": result.data.get("best_epoch"),
        }
    }


@router.get("/{training_id}/checkpoints")
async def get_training_checkpoints(training_id: str):
    """Get checkpoints for a training run."""
    # Checkpoints would be stored in a separate table or in the training run's metadata
    # For now, return from metrics history
    result = supabase_service.client.table("cls_training_runs").select("metrics_history").eq("id", training_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Training run not found")

    history = result.data.get("metrics_history") or []

    # Filter to checkpoints (those with checkpoint_url)
    checkpoints = [m for m in history if m.get("checkpoint_url")]

    return checkpoints


# ===========================================
# Webhook Endpoint (for RunPod callbacks)
# ===========================================

@router.post("/webhook")
async def training_webhook(payload: dict):
    """Webhook endpoint for training progress updates from RunPod."""
    training_id = payload.get("training_run_id")

    if not training_id:
        raise HTTPException(status_code=400, detail="training_run_id required")

    update_data = {}

    # Update status
    if "status" in payload:
        update_data["status"] = payload["status"]

    # Update progress
    if "current_epoch" in payload:
        update_data["current_epoch"] = payload["current_epoch"]

    # Update metrics
    if "metrics" in payload:
        metrics = payload["metrics"]

        # Update best metrics if improved
        if metrics.get("accuracy"):
            current = supabase_service.client.table("cls_training_runs").select("best_accuracy").eq("id", training_id).single().execute()
            if not current.data.get("best_accuracy") or metrics["accuracy"] > current.data["best_accuracy"]:
                update_data["best_accuracy"] = metrics["accuracy"]
                update_data["best_epoch"] = payload.get("current_epoch")

        if metrics.get("f1"):
            current = supabase_service.client.table("cls_training_runs").select("best_f1").eq("id", training_id).single().execute()
            if not current.data.get("best_f1") or metrics["f1"] > current.data["best_f1"]:
                update_data["best_f1"] = metrics["f1"]

        if metrics.get("top5_accuracy"):
            update_data["best_top5_accuracy"] = metrics["top5_accuracy"]

        # Append to metrics history
        result = supabase_service.client.table("cls_training_runs").select("metrics_history").eq("id", training_id).single().execute()
        history = result.data.get("metrics_history") or [] if result.data else []
        history.append({
            "epoch": payload.get("current_epoch"),
            **metrics,
        })
        update_data["metrics_history"] = history

    # Update timestamps
    if payload.get("status") == "training" and "started_at" not in update_data:
        # Check if already started
        current = supabase_service.client.table("cls_training_runs").select("started_at").eq("id", training_id).single().execute()
        if not current.data.get("started_at"):
            update_data["started_at"] = "now()"

    if payload.get("status") in ["completed", "failed"]:
        update_data["completed_at"] = "now()"

    # Update error message if failed
    if payload.get("error"):
        update_data["error_message"] = payload["error"]

    # Apply updates
    if update_data:
        supabase_service.client.table("cls_training_runs").update(update_data).eq("id", training_id).execute()

    # If completed, create trained model record
    if payload.get("status") == "completed" and payload.get("model_url"):
        await _create_trained_model(training_id, payload)

    return {"status": "ok"}


async def _create_trained_model(training_id: str, payload: dict):
    """Create a trained model record from completed training."""
    # Get training run info
    training = supabase_service.client.table("cls_training_runs").select("*").eq("id", training_id).single().execute()

    if not training.data:
        return

    # Get class names from dataset version or labels
    dataset_id = training.data.get("dataset_id")
    labels = supabase_service.client.table("cls_labels").select("class_id").eq("dataset_id", dataset_id).execute()
    class_ids = list(set(l["class_id"] for l in labels.data or []))

    classes = supabase_service.client.table("cls_classes").select("id, name, color").in_("id", class_ids).execute()
    class_names = [c["name"] for c in classes.data or []]
    class_mapping = {i: {"id": c["id"], "name": c["name"], "color": c["color"]} for i, c in enumerate(classes.data or [])}

    model_data = {
        "training_run_id": training_id,
        "name": f"{training.data['name']} - Model",
        "model_type": training.data["model_type"],
        "model_size": training.data["model_size"],
        "task_type": training.data["task_type"],
        "checkpoint_url": payload.get("model_url"),
        "onnx_url": payload.get("onnx_url"),
        "num_classes": training.data["num_classes"],
        "class_names": class_names,
        "class_mapping": class_mapping,
        "accuracy": training.data.get("best_accuracy"),
        "f1_score": training.data.get("best_f1"),
        "top5_accuracy": training.data.get("best_top5_accuracy"),
        "confusion_matrix": payload.get("confusion_matrix"),
        "per_class_metrics": payload.get("per_class_metrics"),
    }

    supabase_service.client.table("cls_trained_models").insert(model_data).execute()
