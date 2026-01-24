"""
Classification - Training Router

Endpoints for managing classification model training runs.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks

logger = logging.getLogger(__name__)

from config import settings
from services.supabase import supabase_service
from services.runpod import runpod_service, EndpointType
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

    # Get active class count from dataset
    active_classes = supabase_service.client.table("cls_classes").select("id").eq("dataset_id", data.dataset_id).eq("is_active", True).execute()
    active_class_ids = set(c["id"] for c in active_classes.data or [])
    num_classes = len(active_class_ids)

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

    # Submit to RunPod in background
    background_tasks.add_task(submit_training_job, training_run["id"])

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

    # Cancel RunPod job if running
    if training.data.get("runpod_job_id"):
        try:
            if runpod_service.is_configured(EndpointType.CLS_TRAINING):
                await runpod_service.cancel_job(
                    endpoint_type=EndpointType.CLS_TRAINING,
                    job_id=training.data["runpod_job_id"],
                )
                logger.info(f"[CLS Training] Cancelled RunPod job: {training.data['runpod_job_id']}")
        except Exception as e:
            logger.warning(f"[CLS Training] Failed to cancel RunPod job: {e}")

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
# Background Tasks
# ===========================================

async def submit_training_job(training_run_id: str):
    """
    Submit training job to RunPod in background.

    Fetches dataset, prepares training data in worker's expected format, and submits to CLS_TRAINING endpoint.
    """
    import random

    try:
        logger.info(f"[CLS Training] Submitting job for training run: {training_run_id}")

        # Update status to preparing
        supabase_service.client.table("cls_training_runs").update({
            "status": "preparing",
        }).eq("id", training_run_id).execute()

        # Fetch training run
        training_result = supabase_service.client.table("cls_training_runs").select(
            "*"
        ).eq("id", training_run_id).single().execute()

        if not training_result.data:
            raise Exception("Training run not found")

        training_run = training_result.data
        dataset_id = training_run["dataset_id"]
        config = training_run.get("config", {})

        # Fetch dataset classes
        classes_result = supabase_service.client.table("cls_classes").select(
            "id, name, display_name"
        ).eq("dataset_id", dataset_id).eq("is_active", True).order("created_at").execute()

        classes = classes_result.data or []
        if len(classes) < 2:
            raise Exception("Dataset must have at least 2 classes")

        # Create class name list and ID to index mapping
        class_names = [c.get("display_name") or c["name"] for c in classes]
        class_id_to_index = {c["id"]: i for i, c in enumerate(classes)}

        # Fetch labeled images with their labels (with pagination to handle >1000 records)
        labels_data = []
        page_size = 1000
        offset = 0

        while True:
            labels_result = supabase_service.client.table("cls_labels").select(
                "image_id, class_id, cls_images!inner(id, image_url)"
            ).eq("dataset_id", dataset_id).range(offset, offset + page_size - 1).execute()

            batch = labels_result.data or []
            labels_data.extend(batch)

            if len(batch) < page_size:
                break
            offset += page_size

        logger.info(f"[CLS Training] Fetched {len(labels_data)} labels with pagination")

        if not labels_data:
            raise Exception("No labeled images found in dataset")

        # Build image data list with URL and label
        all_images = []
        seen_images = set()

        for label in labels_data:
            image_id = label["image_id"]
            class_id = label["class_id"]
            cls_image = label.get("cls_images", {})

            if not cls_image or image_id in seen_images:
                continue

            image_url = cls_image.get("image_url")
            if not image_url:
                continue

            class_index = class_id_to_index.get(class_id)
            if class_index is None:
                continue

            all_images.append({
                "url": image_url,
                "label": class_index,
            })
            seen_images.add(image_id)

        if len(all_images) < 10:
            raise Exception(f"Not enough labeled images ({len(all_images)}). Minimum 10 required.")

        # Split into train/val sets
        train_split = config.get("train_split", 0.8)
        random.seed(42)  # Reproducible split
        random.shuffle(all_images)

        split_idx = int(len(all_images) * train_split)
        train_urls = all_images[:split_idx]
        val_urls = all_images[split_idx:]

        # Ensure at least some validation samples
        if len(val_urls) < 2:
            # Move some from train to val
            val_urls = all_images[-2:]
            train_urls = all_images[:-2]

        logger.info(f"[CLS Training] Prepared {len(train_urls)} train, {len(val_urls)} val images with {len(classes)} classes")

        # Map model config to worker's expected format
        model_type = training_run.get("model_type", "efficientnet")
        model_size = training_run.get("model_size", "b0")

        # Map model type + size to timm model name
        model_name_map = {
            ("efficientnet", "b0"): "efficientnet_b0",
            ("efficientnet", "b1"): "efficientnet_b1",
            ("efficientnet", "b2"): "efficientnet_b2",
            ("efficientnet", "s"): "efficientnetv2_rw_s",
            ("efficientnet", "m"): "efficientnetv2_rw_m",
            ("vit", "tiny"): "vit_tiny_patch16_224",
            ("vit", "small"): "vit_small_patch16_224",
            ("vit", "base"): "vit_base_patch16_224",
            ("convnext", "tiny"): "convnext_tiny",
            ("convnext", "small"): "convnext_small",
            ("convnext", "base"): "convnext_base",
            ("swin", "tiny"): "swin_tiny_patch4_window7_224",
            ("swin", "small"): "swin_small_patch4_window7_224",
            ("resnet", "50"): "resnet50",
            ("resnet", "101"): "resnet101",
        }
        model_name = model_name_map.get((model_type.lower(), model_size.lower()), "efficientnet_b0")

        # Map loss function
        loss_map = {
            "cross_entropy": "cross_entropy",
            "label_smoothing": "label_smoothing",
            "focal": "focal",
            "arcface": "arcface",
            "cosface": "cosface",
            "circle": "circle",
        }
        loss = loss_map.get(config.get("loss_function", "cross_entropy"), "label_smoothing")

        # ===========================================
        # Smart Defaults Based on Dataset Size
        # ===========================================
        num_train_samples = len(train_urls)
        is_small_dataset = num_train_samples < 5000

        # Augmentation preset - auto-select heavy for small datasets unless overridden
        augmentation_preset = config.get("augmentation_preset")
        if augmentation_preset is None:
            augmentation_preset = "heavy" if is_small_dataset else "sota"
            logger.info(f"[CLS Training] Auto-selected '{augmentation_preset}' augmentation for {num_train_samples} samples")

        # MixUp/CutMix - enable for sota and heavy presets
        use_mixup = config.get("use_mixup", augmentation_preset in ["sota", "heavy"])

        # Dropout - higher for small datasets to prevent overfitting
        drop_rate = config.get("drop_rate", 0.3 if is_small_dataset else 0.1)
        drop_path_rate = config.get("drop_path_rate", 0.2 if is_small_dataset else 0.1)

        # Weight decay - stronger regularization for small datasets
        weight_decay = config.get("weight_decay", 0.05 if is_small_dataset else 0.01)

        # Learning rate - lower default for stability
        learning_rate = config.get("learning_rate", 0.0001)

        # Class weights - auto-enable for better class balance
        use_class_weights = config.get("use_class_weights", True)

        logger.info(f"[CLS Training] Anti-overfitting config: drop_rate={drop_rate}, weight_decay={weight_decay}, mixup={use_mixup}")

        # Prepare RunPod input in worker's expected format
        runpod_input = {
            "training_run_id": training_run_id,
            "dataset": {
                "train_urls": train_urls,
                "val_urls": val_urls,
                "class_names": class_names,
            },
            "config": {
                "model_name": model_name,
                "epochs": config.get("epochs", 30),
                "batch_size": config.get("batch_size", 32),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": config.get("optimizer", "adamw"),
                "scheduler": "cosine_warmup",
                "warmup_epochs": config.get("warmup_epochs", 5),
                "loss": loss,
                "loss_smoothing": config.get("label_smoothing", 0.1),
                "augmentation": augmentation_preset,
                "use_mixup": use_mixup,
                "mixup_alpha": config.get("mixup_alpha", 1.0 if augmentation_preset == "heavy" else 0.8),
                "cutmix_alpha": config.get("cutmix_alpha", 1.0),
                "use_amp": config.get("mixed_precision", True),
                "use_ema": True,
                "use_class_weights": use_class_weights,
                "early_stopping": True,
                "patience": config.get("early_stopping_patience", 10),
                # Anti-overfitting parameters
                "drop_rate": drop_rate,
                "drop_path_rate": drop_path_rate,
                "image_size": config.get("image_size", 224),
                # Data loading configuration
                "data_loading": config.get("data_loading"),
            },
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_service_role_key,
            "api_url": settings.api_url,  # For webhook calls on completion
        }

        # Check if CLS_TRAINING endpoint is configured
        if not runpod_service.is_configured(EndpointType.CLS_TRAINING):
            raise Exception("CLS_TRAINING endpoint not configured")

        # Submit job to RunPod
        runpod_response = await runpod_service.submit_job(
            endpoint_type=EndpointType.CLS_TRAINING,
            input_data=runpod_input,
        )

        runpod_job_id = runpod_response.get("id")
        if not runpod_job_id:
            raise Exception(f"Failed to get RunPod job ID: {runpod_response}")

        logger.info(f"[CLS Training] RunPod job submitted: {runpod_job_id}")

        # Update training run with RunPod job ID and status
        supabase_service.client.table("cls_training_runs").update({
            "status": "queued",
            "runpod_job_id": runpod_job_id,
        }).eq("id", training_run_id).execute()

    except Exception as e:
        logger.error(f"[CLS Training] Failed to submit job: {e}")

        # Update training run as failed
        supabase_service.client.table("cls_training_runs").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", training_run_id).execute()


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


@router.get("/{training_run_id}/metrics-history")
async def get_metrics_history(training_run_id: str):
    """
    Get epoch-by-epoch metrics history for a training run.
    Fetches from unified training_metrics_history table.
    """
    result = supabase_service.client.table("training_metrics_history").select(
        "epoch, train_loss, val_loss, val_accuracy, val_f1, learning_rate, created_at"
    ).eq(
        "training_run_id", training_run_id
    ).eq(
        "training_type", "cls"
    ).order(
        "epoch", desc=False
    ).execute()

    return result.data or []
