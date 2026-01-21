"""
OD Training Worker - RunPod Handler

Handles training jobs for object detection models with SOTA features:
- RT-DETR (Apache 2.0)
- D-FINE (Apache 2.0)

SOTA Features:
- EMA (Exponential Moving Average)
- LLRD (Layer-wise Learning Rate Decay)
- Warmup + Cosine LR Scheduler
- Mixed Precision (FP16)
- Mosaic, MixUp, CopyPaste augmentations
- COCO mAP evaluation
"""

import os
import json
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Dict
import httpx

import runpod

from config import (
    TrainingConfig,
    DatasetConfig,
    OutputConfig,
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    WEBHOOK_URL,
    MODEL_CONFIGS,
    AUGMENTATION_PRESETS,
)


def convert_frontend_augmentation_config(aug_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert frontend augmentation_config format to backend augmentation_overrides format.

    Frontend format:
    {
        "mosaic": { "enabled": true, "probability": 0.5, ... },
        "mixup": { "enabled": true, "probability": 0.3, "alpha": 8.0 },
        "copypaste": { "enabled": true, "probability": 0.3, "blend_ratio": 0.5 },
        ...
    }

    Backend format:
    {
        "mosaic": {"enabled": True, "prob": 0.5, "params": {...}},
        "mixup": {"enabled": True, "prob": 0.3, "params": {"alpha": 8.0}},
        "copypaste": {"enabled": True, "prob": 0.3, "params": {"blend_ratio": 0.5}},
        ...
    }

    Supports all 40+ augmentations from the Training Wizard:
    - Multi-image: mosaic, mosaic9, mixup, cutmix, copypaste
    - Geometric: horizontal_flip, vertical_flip, rotate90, random_rotate,
                 shift_scale_rotate, affine, perspective, safe_rotate,
                 random_crop, random_scale, grid_distortion, elastic_transform,
                 optical_distortion, piecewise_affine
    - Color: brightness_contrast, color_jitter, hue_saturation, random_gamma,
             rgb_shift, channel_shuffle, clahe, equalize, random_tone_curve,
             posterize, solarize, sharpen, unsharp_mask, fancy_pca, invert_img, to_gray
    - Blur: gaussian_blur, motion_blur, median_blur, defocus, zoom_blur,
            glass_blur, advanced_blur
    - Noise: gaussian_noise, iso_noise, multiplicative_noise
    - Quality: image_compression, downscale
    - Dropout: coarse_dropout, grid_dropout, pixel_dropout, mask_dropout
    - Weather: random_rain, random_fog, random_shadow, random_sun_flare,
               random_snow, spatter, plasma_brightness
    """
    if not aug_config:
        return None

    # Mapping from frontend keys to backend keys (most are 1:1)
    key_mapping = {
        # Multi-image
        "mosaic": "mosaic",
        "mosaic9": "mosaic9",
        "mixup": "mixup",
        "cutmix": "cutmix",
        "copypaste": "copypaste",
        "copy_paste": "copypaste",  # Legacy alias

        # Geometric
        "horizontal_flip": "horizontal_flip",
        "vertical_flip": "vertical_flip",
        "rotate90": "rotate90",
        "random_rotate": "random_rotate",
        "shift_scale_rotate": "shift_scale_rotate",
        "affine": "affine",
        "perspective": "perspective",
        "safe_rotate": "safe_rotate",
        "random_crop": "random_crop",
        "random_scale": "random_scale",
        "grid_distortion": "grid_distortion",
        "elastic_transform": "elastic_transform",
        "optical_distortion": "optical_distortion",
        "piecewise_affine": "piecewise_affine",

        # Color
        "brightness_contrast": "brightness_contrast",
        "color_jitter": "color_jitter",
        "hue_saturation": "hue_saturation",
        "random_gamma": "random_gamma",
        "rgb_shift": "rgb_shift",
        "channel_shuffle": "channel_shuffle",
        "clahe": "clahe",
        "equalize": "equalize",
        "random_tone_curve": "random_tone_curve",
        "posterize": "posterize",
        "solarize": "solarize",
        "sharpen": "sharpen",
        "unsharp_mask": "unsharp_mask",
        "fancy_pca": "fancy_pca",
        "invert_img": "invert_img",
        "to_gray": "to_gray",

        # Blur
        "gaussian_blur": "gaussian_blur",
        "motion_blur": "motion_blur",
        "median_blur": "median_blur",
        "defocus": "defocus",
        "zoom_blur": "zoom_blur",
        "glass_blur": "glass_blur",
        "advanced_blur": "advanced_blur",

        # Noise
        "gaussian_noise": "gaussian_noise",
        "iso_noise": "iso_noise",
        "multiplicative_noise": "multiplicative_noise",

        # Quality
        "image_compression": "image_compression",
        "downscale": "downscale",

        # Dropout
        "coarse_dropout": "coarse_dropout",
        "grid_dropout": "grid_dropout",
        "pixel_dropout": "pixel_dropout",
        "mask_dropout": "mask_dropout",

        # Weather
        "random_rain": "random_rain",
        "random_fog": "random_fog",
        "random_shadow": "random_shadow",
        "random_sun_flare": "random_sun_flare",
        "random_snow": "random_snow",
        "spatter": "spatter",
        "plasma_brightness": "plasma_brightness",
    }

    # Keys that should not be treated as params
    reserved_keys = {"enabled", "probability", "prob"}

    overrides = {}

    for frontend_key, frontend_val in aug_config.items():
        if not isinstance(frontend_val, dict):
            continue

        # Get backend key (use frontend key if no mapping exists)
        backend_key = key_mapping.get(frontend_key, frontend_key)

        # Build backend config
        backend_val = {"enabled": frontend_val.get("enabled", False)}

        # Convert probability to prob
        if "probability" in frontend_val:
            backend_val["prob"] = frontend_val["probability"]
        elif "prob" in frontend_val:
            backend_val["prob"] = frontend_val["prob"]

        # Extract params (all other keys except reserved ones)
        params = {}
        for param_key, param_val in frontend_val.items():
            if param_key not in reserved_keys:
                params[param_key] = param_val

        if params:
            backend_val["params"] = params

        overrides[backend_key] = backend_val

    return overrides if overrides else None


def download_dataset(dataset_url: str, output_path: str) -> str:
    """Download and extract dataset from URL."""
    import zipfile

    print(f"Downloading dataset from {dataset_url}")

    response = httpx.get(dataset_url, timeout=300, follow_redirects=True)
    response.raise_for_status()

    zip_path = os.path.join(output_path, "dataset.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract
    extract_path = os.path.join(output_path, "dataset")
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_path)

    os.remove(zip_path)

    # Handle nested folder case: if there's a single subfolder, use it as root
    contents = os.listdir(extract_path)
    if len(contents) == 1:
        potential_root = os.path.join(extract_path, contents[0])
        if os.path.isdir(potential_root):
            # Check if this subfolder contains the expected structure
            if os.path.exists(os.path.join(potential_root, "annotations")) or \
               os.path.exists(os.path.join(potential_root, "images")):
                print(f"Found nested dataset folder: {contents[0]}")
                extract_path = potential_root

    print(f"Dataset extracted to {extract_path}")

    return extract_path


def send_webhook(
    job_id: str,
    training_run_id: str,
    status: str,
    progress: int = 0,
    current_epoch: int = 0,
    metrics: Optional[dict] = None,
    error: Optional[str] = None,
    model_url: Optional[str] = None,
):
    """Send progress update to webhook."""
    if not WEBHOOK_URL:
        return

    payload = {
        "job_id": job_id,
        "training_run_id": training_run_id,
        "status": status,
        "progress": progress,
        "current_epoch": current_epoch,
        "metrics": metrics or {},
        "error": error,
        "model_url": model_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = httpx.post(
            WEBHOOK_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        print(f"Webhook sent: {status}, response: {response.status_code}")
    except Exception as e:
        print(f"Failed to send webhook: {e}")


def upload_model_to_supabase(
    model_path: str,
    training_run_id: str,
    model_type: str,
) -> str:
    """Upload trained model to Supabase storage."""
    from supabase import create_client

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase credentials not configured")

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Generate unique filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{training_run_id}/{model_type}_{timestamp}.pt"

    # Upload
    with open(model_path, "rb") as f:
        client.storage.from_("od-models").upload(
            filename,
            f.read(),
            {"content-type": "application/octet-stream"},
        )

    # Get public URL
    url = client.storage.from_("od-models").get_public_url(filename)
    print(f"Model uploaded to {url}")

    return url


def train_sota(
    job_id: str,
    training_run_id: str,
    model_type: str,
    training_config: TrainingConfig,
    dataset_config: DatasetConfig,
    output_config: OutputConfig,
) -> dict:
    """
    Train model using SOTA trainer.

    Supports RT-DETR and D-FINE with all SOTA features.
    """
    # Import SOTA trainers
    from src.trainers import RTDETRSOTATrainer, DFINESOTATrainer

    # Create progress callback
    def progress_callback(epoch: int, metrics: Dict[str, float]):
        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="training",
            progress=int((epoch / training_config.epochs) * 90) + 10,  # 10-100%
            current_epoch=epoch,
            metrics=metrics,
        )

    # Convert configs to SOTA format
    sota_training_config = training_config.to_sota_config()
    sota_dataset_config = dataset_config.to_sota_config()
    sota_output_config = output_config.to_sota_config()

    # Select trainer based on model type
    if model_type == "rt-detr":
        trainer = RTDETRSOTATrainer(
            training_config=sota_training_config,
            dataset_config=sota_dataset_config,
            output_config=sota_output_config,
            progress_callback=progress_callback,
        )
    elif model_type == "d-fine":
        trainer = DFINESOTATrainer(
            training_config=sota_training_config,
            dataset_config=sota_dataset_config,
            output_config=sota_output_config,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    result = trainer.train()

    return {
        "best_checkpoint": result.best_checkpoint,
        "best_metrics": result.best_metrics,
        "best_epoch": result.best_epoch,
        "total_epochs": result.total_epochs,
        "training_time": result.training_time,
    }


def handler(job: dict) -> dict:
    """
    RunPod handler for training jobs.

    Expected input:
    {
        "training_run_id": "uuid",
        "dataset_url": "https://...",
        "dataset_format": "coco",
        "model_type": "rt-detr" | "d-fine",
        "model_size": "s" | "m" | "l" | "x",
        "config": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.0001,
            "image_size": 640,
            "augmentation_preset": "sota",
            "use_ema": true,
            "llrd_decay": 0.9,
            ...
        },
        "class_names": ["class1", "class2", ...],
        "num_classes": 10
    }
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    print(f"=== Training Job Started ===")
    print(f"Job ID: {job_id}")
    print(f"Input: {json.dumps(job_input, indent=2)}")

    # Extract parameters
    training_run_id = job_input.get("training_run_id")
    dataset_url = job_input.get("dataset_url")
    dataset_format = job_input.get("dataset_format", "coco")
    model_type = job_input.get("model_type", "rt-detr")
    model_size = job_input.get("model_size", "l")
    config = job_input.get("config", {})
    class_names = job_input.get("class_names", [])
    num_classes = job_input.get("num_classes", len(class_names))

    # Validate
    if not training_run_id:
        return {"error": "training_run_id is required"}
    if not dataset_url:
        return {"error": "dataset_url is required"}
    if model_type not in MODEL_CONFIGS:
        return {"error": f"Invalid model_type: {model_type}. Supported: {list(MODEL_CONFIGS.keys())}"}

    try:
        # Send started webhook
        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="started",
            progress=0,
        )

        # Create temp directories
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        logs_dir = os.path.join(temp_dir, "logs")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Download dataset
        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="downloading",
            progress=5,
        )
        dataset_path = download_dataset(dataset_url, temp_dir)

        # Handle custom augmentation config from frontend
        aug_preset = config.get("augmentation_preset", "sota")
        aug_overrides = config.get("augmentation_overrides")

        # If frontend sends augmentation_config (custom mode), convert it
        if config.get("augmentation_config"):
            aug_overrides = convert_frontend_augmentation_config(
                config["augmentation_config"]
            )
            print(f"Custom augmentation config converted: {aug_overrides}")

        # Setup training config with SOTA features
        training_config = TrainingConfig(
            model_type=model_type,
            model_size=model_size,
            epochs=config.get("epochs", 100),
            batch_size=config.get("batch_size", 16),
            learning_rate=config.get("learning_rate", 0.0001),
            weight_decay=config.get("weight_decay", 0.0001),
            llrd_decay=config.get("llrd_decay", 0.9),
            head_lr_factor=config.get("head_lr_factor", 10.0),
            warmup_epochs=config.get("warmup_epochs", 3),
            use_ema=config.get("use_ema", True),
            ema_decay=config.get("ema_decay", 0.9999),
            mixed_precision=config.get("mixed_precision", True),
            gradient_clip=config.get("gradient_clip", 1.0),
            image_size=config.get("image_size", 640),
            multi_scale=config.get("multi_scale", False),
            augmentation_preset=aug_preset,
            augmentation_overrides=aug_overrides,
            patience=config.get("patience", 20),
            save_freq=config.get("save_freq", 5),
        )

        # Setup dataset config
        dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            format=dataset_format,
            num_classes=num_classes,
            class_names=class_names,
        )

        # Set paths based on format
        if dataset_format == "coco":
            dataset_config.train_path = os.path.join(dataset_path, "images", "train")
            dataset_config.val_path = os.path.join(dataset_path, "images", "val")
            dataset_config.train_ann_file = os.path.join(dataset_path, "annotations", "train.json")
            dataset_config.val_ann_file = os.path.join(dataset_path, "annotations", "val.json")

            # Try alternative annotation paths
            if not os.path.exists(dataset_config.train_ann_file):
                dataset_config.train_ann_file = os.path.join(dataset_path, "annotations", "instances_train.json")
            if not os.path.exists(dataset_config.val_ann_file):
                dataset_config.val_ann_file = os.path.join(dataset_path, "annotations", "instances_val.json")
        else:
            # YOLO format - convert to COCO
            from src.data import convert_yolo_to_coco, get_yolo_class_names

            print("Converting YOLO format to COCO...")
            coco_path = os.path.join(temp_dir, "coco_converted")

            # Get class names from YOLO data.yaml if not provided
            if not class_names:
                try:
                    class_names = get_yolo_class_names(dataset_path)
                    num_classes = len(class_names)
                except Exception as e:
                    print(f"Warning: Could not read class names from data.yaml: {e}")

            convert_yolo_to_coco(
                yolo_dataset_path=dataset_path,
                output_path=coco_path,
                class_names=class_names if class_names else None,
            )

            # Update paths to converted COCO format
            dataset_config.train_path = os.path.join(coco_path, "images", "train")
            dataset_config.val_path = os.path.join(coco_path, "images", "val")
            dataset_config.train_ann_file = os.path.join(coco_path, "annotations", "train.json")
            dataset_config.val_ann_file = os.path.join(coco_path, "annotations", "val.json")
            dataset_config.num_classes = num_classes
            dataset_config.class_names = class_names

        output_config = OutputConfig(
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
        )

        # Send training started
        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="training",
            progress=10,
        )

        # Train using SOTA trainer
        result = train_sota(
            job_id=job_id,
            training_run_id=training_run_id,
            model_type=model_type,
            training_config=training_config,
            dataset_config=dataset_config,
            output_config=output_config,
        )

        # Upload best model
        model_url = None
        if result.get("best_checkpoint"):
            model_url = upload_model_to_supabase(
                result["best_checkpoint"],
                training_run_id,
                model_type,
            )

        # Send completed webhook
        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="completed",
            progress=100,
            current_epoch=result.get("total_epochs", 0),
            metrics=result.get("best_metrics", {}),
            model_url=model_url,
        )

        return {
            "status": "completed",
            "training_run_id": training_run_id,
            "model_type": model_type,
            "model_url": model_url,
            "best_metrics": result.get("best_metrics", {}),
            "best_epoch": result.get("best_epoch", 0),
            "total_epochs": result.get("total_epochs", 0),
            "training_time_seconds": result.get("training_time", 0),
        }

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()

        send_webhook(
            job_id=job_id,
            training_run_id=training_run_id,
            status="failed",
            error=error_msg,
        )

        return {
            "status": "failed",
            "error": error_msg,
            "training_run_id": training_run_id,
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
