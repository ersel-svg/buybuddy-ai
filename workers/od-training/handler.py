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

Progress Updates:
- Direct Supabase writes (no webhook dependency)
- More reliable than webhook-based updates
"""

import os
import sys
import json
import signal
import tempfile
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Dict
import httpx

import runpod


# ===========================================
# Graceful Shutdown Handling
# ===========================================
_shutdown_requested = False
_current_training_run_id = None


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[SHUTDOWN] Signal {signum} received, initiating graceful shutdown...")

    # Update training run status if we have one
    if _current_training_run_id:
        try:
            update_training_run(
                training_run_id=_current_training_run_id,
                status="cancelled",
                error="Training cancelled due to server shutdown",
            )
            print(f"[SHUTDOWN] Training run {_current_training_run_id} marked as cancelled")
        except Exception as e:
            print(f"[SHUTDOWN] Failed to update training run: {e}")


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


# Note: Signal handlers registered at end of file (after update_training_run is defined)

from config import (
    TrainingConfig,
    DatasetConfig,
    OutputConfig,
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    MODEL_CONFIGS,
    AUGMENTATION_PRESETS,
)


# ===========================================
# Supabase Client Singleton
# ===========================================
_supabase_client = None


def get_supabase_client():
    """Get or create Supabase client singleton."""
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[WARNING] Supabase credentials not configured, progress updates disabled")
        return None

    from supabase import create_client
    _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


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
    # Fix v2: Properly detect and use nested dataset folders
    contents = os.listdir(extract_path)
    print(f"[Dataset Fix v2] Extract contents: {contents}")
    if len(contents) == 1:
        potential_root = os.path.join(extract_path, contents[0])
        if os.path.isdir(potential_root):
            # Check if this subfolder contains the expected structure
            has_annotations = os.path.exists(os.path.join(potential_root, "annotations"))
            has_images = os.path.exists(os.path.join(potential_root, "images"))
            print(f"[Dataset Fix v2] Checking {contents[0]}: annotations={has_annotations}, images={has_images}")
            if has_annotations or has_images:
                print(f"[Dataset Fix v2] Using nested folder: {contents[0]}")
                extract_path = potential_root

    print(f"Dataset extracted to {extract_path}")

    return extract_path


def update_training_run(
    training_run_id: str,
    status: str,
    progress: int = 0,
    current_epoch: int = 0,
    total_epochs: int = 0,
    metrics: Optional[dict] = None,
    error: Optional[str] = None,
    model_url: Optional[str] = None,
    class_mapping: Optional[Dict] = None,
    max_retries: int = 3,
):
    """
    Update training run directly in Supabase with retry logic.

    This is more reliable than webhook-based updates as it doesn't
    require a publicly accessible API endpoint.
    """
    import time

    client = get_supabase_client()
    if not client:
        print(f"[WARNING] Cannot update training run: Supabase client not available")
        return

    last_error = None
    for attempt in range(max_retries):
        try:
            # Build update data
            update_data = {
                "status": status,
                "current_epoch": current_epoch,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add total_epochs if provided
            if total_epochs > 0:
                update_data["total_epochs"] = total_epochs

            # Handle metrics and metrics_history
            if metrics:
                # Get current training run to append to metrics_history
                result = client.table("od_training_runs").select("metrics_history, best_map, best_epoch").eq("id", training_run_id).single().execute()

                if result.data:
                    current_data = result.data
                    metrics_history = current_data.get("metrics_history") or []

                    # Append new metrics to history
                    metrics_history.append({
                        "epoch": current_epoch,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **metrics,
                    })
                    update_data["metrics_history"] = metrics_history

                    # Update best_map if improved
                    current_map = metrics.get("map", 0)
                    best_map = current_data.get("best_map") or 0
                    if current_map > best_map:
                        update_data["best_map"] = current_map
                        update_data["best_epoch"] = current_epoch

            # Handle completion
            if status == "completed":
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

                # Create trained model record if model_url provided
                if model_url:
                    try:
                        # Get training run details for model creation
                        run_result = client.table("od_training_runs").select("name, model_type, config, best_map").eq("id", training_run_id).single().execute()

                        if run_result.data:
                            run_data = run_result.data
                            from uuid import uuid4

                            # Extract class_count from config if available
                            config = run_data.get("config", {})
                            class_count = config.get("num_classes", 0) if isinstance(config, dict) else 0

                            model_data = {
                                "id": str(uuid4()),
                                "training_run_id": training_run_id,
                                "name": f"{run_data.get('name', 'OD Model')} Model",
                                "model_type": run_data.get("model_type", "rt-detr"),
                                "checkpoint_url": model_url,
                                "map": update_data.get("best_map") or run_data.get("best_map"),
                                "map_50": metrics.get("map_50") if metrics else None,
                                "class_mapping": class_mapping or {},
                                "class_count": class_count,
                                "is_active": True,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            }
                            client.table("od_trained_models").insert(model_data).execute()
                            print(f"[INFO] Created trained model record: {model_data['id']} (class_count={class_count}, class_mapping_keys={len(class_mapping or {})})")
                    except Exception as model_err:
                        print(f"[WARNING] Failed to create trained model record: {model_err}")

            # Handle failure
            if status == "failed" and error:
                update_data["error_message"] = error

            # Handle start
            if status in ["started", "training"] and "started_at" not in update_data:
                # Check if started_at already set
                check_result = client.table("od_training_runs").select("started_at").eq("id", training_run_id).single().execute()
                if check_result.data and not check_result.data.get("started_at"):
                    update_data["started_at"] = datetime.now(timezone.utc).isoformat()

            # Update training run
            client.table("od_training_runs").update(update_data).eq("id", training_run_id).execute()
            print(f"[Supabase] Training run updated: status={status}, epoch={current_epoch}")
            return  # Success, exit retry loop

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[WARNING] Supabase update failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Failed to update training run after {max_retries} attempts: {e}")
                traceback.print_exc()


def upload_model_to_supabase(
    model_path: str,
    training_run_id: str,
    model_type: str,
    max_retries: int = 3,
    timeout_seconds: int = 300,
) -> str:
    """
    Upload trained model to Supabase storage with retry logic.

    Args:
        model_path: Path to the model file
        training_run_id: Training run ID for organizing uploads
        model_type: Type of model (e.g., 'rt-detr', 'd-fine')
        max_retries: Maximum number of upload attempts
        timeout_seconds: Timeout for upload in seconds (default 5 min)

    Returns:
        Public URL of uploaded model
    """
    import time
    from supabase import create_client

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Supabase credentials not configured")

    # Get file size for logging
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[INFO] Uploading model: {model_path} ({file_size_mb:.2f} MB)")

    # Create client
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Extend storage client timeout for large files (default is 20s)
    try:
        client.storage._client.timeout = httpx.Timeout(float(timeout_seconds))
        print(f"[INFO] Storage timeout set to {timeout_seconds}s")
    except Exception as e:
        print(f"[WARNING] Could not set storage timeout: {e}")

    # Generate unique filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{training_run_id}/{model_type}_{timestamp}.pt"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Upload attempt {attempt}/{max_retries}...")

            # Stream upload - don't load entire file into RAM
            with open(model_path, "rb") as f:
                # Upload with streaming (more memory efficient for large files)
                client.storage.from_("od-models").upload(
                    filename,
                    f,
                    {"content-type": "application/octet-stream"},
                )

            # Get public URL
            url = client.storage.from_("od-models").get_public_url(filename)
            print(f"[SUCCESS] Model uploaded to {url}")
            return url

        except Exception as e:
            last_error = e
            print(f"[WARNING] Upload attempt {attempt} failed: {type(e).__name__}: {e}")

            if attempt < max_retries:
                # Exponential backoff: 5s, 10s, 20s
                wait_time = 5 * (2 ** (attempt - 1))
                print(f"[INFO] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # All retries failed
    raise RuntimeError(f"Failed to upload model after {max_retries} attempts: {last_error}")


def train_sota(
    job_id: str,
    training_run_id: str,
    model_type: str,
    training_config: TrainingConfig,
    dataset_config: Optional[DatasetConfig],
    output_config: OutputConfig,
    url_dataset_data: Optional[Dict] = None,
) -> dict:
    """
    Train model using SOTA trainer.

    Supports RT-DETR and D-FINE with all SOTA features.

    Args:
        job_id: RunPod job ID
        training_run_id: Training run UUID
        model_type: "rt-detr" or "d-fine"
        training_config: Training hyperparameters
        dataset_config: Dataset paths (for file-based mode, can be None for URL mode)
        output_config: Output directories
        url_dataset_data: If provided, use URL-based data loading (new mode)
    """
    # Import SOTA trainers
    from src.trainers import RTDETRSOTATrainer, DFINESOTATrainer

    # Create progress callback
    def progress_callback(epoch: int, metrics: Dict[str, float]):
        update_training_run(
            training_run_id=training_run_id,
            status="training",
            progress=int((epoch / training_config.epochs) * 90) + 10,  # 10-100%
            current_epoch=epoch,
            total_epochs=training_config.epochs,
            metrics=metrics,
        )

    # Convert configs to SOTA format
    sota_training_config = training_config.to_sota_config()
    sota_output_config = output_config.to_sota_config()

    # Dataset config: use provided or create minimal for URL mode
    if url_dataset_data is not None:
        # URL-based mode: create minimal dataset config
        from src.training.base_trainer import DatasetConfig as SOTADatasetConfig
        sota_dataset_config = SOTADatasetConfig(
            num_classes=url_dataset_data.get("num_classes", 0),
            class_names=url_dataset_data.get("class_names", []),
        )
    else:
        # File-based mode: use provided config
        sota_dataset_config = dataset_config.to_sota_config()

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

    # Train (pass url_dataset_data if available)
    result = trainer.train(url_dataset_data=url_dataset_data)

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
    dataset_url = job_input.get("dataset_url")  # Legacy: ZIP URL
    dataset_id = job_input.get("dataset_id")  # NEW: Direct Supabase fetch
    dataset_format = job_input.get("dataset_format", "coco")
    model_type = job_input.get("model_type", "rt-detr")
    model_size = job_input.get("model_size", "l")
    config = job_input.get("config", {})
    class_names = job_input.get("class_names", [])
    num_classes = job_input.get("num_classes", len(class_names))

    # NEW: Supabase credentials for URL-based loading
    supabase_url_param = job_input.get("supabase_url")
    supabase_service_key = job_input.get("supabase_service_key")

    # Validate
    if not training_run_id:
        return {"error": "training_run_id is required"}
    # NEW: Either dataset_id (URL-based) OR dataset_url (ZIP-based) is required
    if not dataset_id and not dataset_url:
        return {"error": "Either dataset_id or dataset_url is required"}
    if model_type not in MODEL_CONFIGS:
        return {"error": f"Invalid model_type: {model_type}. Supported: {list(MODEL_CONFIGS.keys())}"}

    # Set current training run for graceful shutdown
    global _current_training_run_id
    _current_training_run_id = training_run_id

    try:
        # Check for early shutdown
        if is_shutdown_requested():
            return {"error": "Server shutdown in progress", "status": "cancelled"}

        # Update status to started
        update_training_run(
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

        # =====================================================
        # DATA LOADING: URL-based (new) or ZIP-based (legacy)
        # =====================================================
        use_url_based = bool(dataset_id and supabase_url_param and supabase_service_key)
        url_dataset_data = None

        if use_url_based:
            # NEW: URL-based approach - fetch from Supabase directly
            print(f"[INFO] Using URL-based data loading (dataset_id: {dataset_id})")
            update_training_run(
                training_run_id=training_run_id,
                status="preparing",
                progress=5,
            )

            try:
                from src.data.supabase_fetcher import build_url_dataset_data
                url_dataset_data = build_url_dataset_data(
                    supabase_url=supabase_url_param,
                    supabase_key=supabase_service_key,
                    dataset_id=dataset_id,
                    train_split=config.get("train_split", 0.8),
                    seed=config.get("seed", 42),
                )

                # Update class info from fetched data
                if url_dataset_data["class_names"]:
                    class_names = url_dataset_data["class_names"]
                    num_classes = url_dataset_data["num_classes"]
                    print(f"[INFO] Loaded {num_classes} classes: {class_names}")

                dataset_path = None  # Not used in URL mode

            except Exception as e:
                print(f"[ERROR] Failed to fetch dataset from Supabase: {e}")
                # Fall back to ZIP if available
                if dataset_url:
                    print("[INFO] Falling back to ZIP-based loading...")
                    use_url_based = False
                    url_dataset_data = None
                else:
                    raise

        if not use_url_based:
            # LEGACY: ZIP-based approach - download and extract
            print(f"[INFO] Using ZIP-based data loading (dataset_url: {dataset_url})")
            update_training_run(
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

        # Setup dataset config (only needed for ZIP-based mode)
        dataset_config = None
        if not use_url_based:
            dataset_config = DatasetConfig(
                dataset_path=dataset_path,
                format=dataset_format,
                num_classes=num_classes,
                class_names=class_names,
            )

        # Set paths based on format (ZIP-based only)
        if not use_url_based and dataset_format == "coco":
            dataset_config.train_path = os.path.join(dataset_path, "images", "train")
            dataset_config.val_path = os.path.join(dataset_path, "images", "val")
            dataset_config.train_ann_file = os.path.join(dataset_path, "annotations", "train.json")
            dataset_config.val_ann_file = os.path.join(dataset_path, "annotations", "val.json")

            # Try alternative annotation paths
            if not os.path.exists(dataset_config.train_ann_file):
                dataset_config.train_ann_file = os.path.join(dataset_path, "annotations", "instances_train.json")
            if not os.path.exists(dataset_config.val_ann_file):
                dataset_config.val_ann_file = os.path.join(dataset_path, "annotations", "instances_val.json")

            # Fallback: if val doesn't exist, use train for validation (with a warning)
            if not os.path.exists(dataset_config.val_ann_file):
                print(f"[WARNING] No validation annotations found, using train data for validation")
                dataset_config.val_ann_file = dataset_config.train_ann_file
                dataset_config.val_path = dataset_config.train_path

            # Read num_classes from COCO annotation if not provided
            if num_classes == 0 or not class_names:
                try:
                    with open(dataset_config.train_ann_file) as f:
                        coco_data = json.load(f)
                    categories = coco_data.get("categories", [])
                    if categories:
                        num_classes = len(categories)
                        class_names = [c.get("name", f"class_{c['id']}") for c in categories]
                        dataset_config.num_classes = num_classes
                        dataset_config.class_names = class_names
                        print(f"[INFO] Read {num_classes} classes from COCO annotations: {class_names}")
                except Exception as e:
                    print(f"[WARNING] Could not read classes from COCO file: {e}")
        elif not use_url_based:
            # YOLO format - convert to COCO (ZIP-based only)
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

        # Update status to training
        update_training_run(
            training_run_id=training_run_id,
            status="training",
            progress=10,
            total_epochs=training_config.epochs,
        )

        # Train using SOTA trainer
        result = train_sota(
            job_id=job_id,
            training_run_id=training_run_id,
            model_type=model_type,
            training_config=training_config,
            dataset_config=dataset_config,
            output_config=output_config,
            url_dataset_data=url_dataset_data,  # NEW: URL-based data if available
        )

        # Upload best model
        model_url = None
        if result.get("best_checkpoint"):
            model_url = upload_model_to_supabase(
                result["best_checkpoint"],
                training_run_id,
                model_type,
            )

        # Update status to completed
        update_training_run(
            training_run_id=training_run_id,
            status="completed",
            progress=100,
            current_epoch=result.get("total_epochs", 0),
            total_epochs=result.get("total_epochs", 0),
            metrics=result.get("best_metrics", {}),
            model_url=model_url,
            class_mapping=url_dataset_data.get("class_mapping") if url_dataset_data else None,
        )

        # Clear current training run
        _current_training_run_id = None

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

        # Check if this was a shutdown
        if is_shutdown_requested():
            error_msg = "Training cancelled due to server shutdown"
            status = "cancelled"
        else:
            status = "failed"

        # Update status
        update_training_run(
            training_run_id=training_run_id,
            status=status,
            error=error_msg,
        )

        return {
            "status": status,
            "error": error_msg,
            "training_run_id": training_run_id,
        }

    finally:
        # Always clear current training run
        _current_training_run_id = None


# ===========================================
# Register Signal Handlers (after all functions defined)
# ===========================================
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# RunPod serverless entry point
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
