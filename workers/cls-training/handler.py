"""
CLS Training Worker - RunPod Handler

SOTA Classification Training with:
- Models: ViT, ConvNeXt, EfficientNet, Swin, ResNet, DINOv2
- Losses: CE, LabelSmoothing, Focal, Circle, ArcFace, CosFace, Poly
- Augmentations: SOTA presets with MixUp, CutMix, RandAugment
- Training: AMP, EMA, Gradient Accumulation, Early Stopping, etc.

Input:
{
    "training_run_id": "uuid",
    "config": {
        "model_name": "vit_base_patch16_224",
        "num_classes": 4,
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "loss": "label_smoothing",
        "augmentation": "sota",
        "use_mixup": true,
        ...
    },
    "dataset": {
        "train_urls": [{"url": "...", "label": 0}, ...],
        "val_urls": [{"url": "...", "label": 0}, ...],
        "class_names": ["class1", "class2", ...]
    },
    "supabase_url": "...",
    "supabase_key": "..."
}

Output:
{
    "status": "completed",
    "metrics": {
        "best_val_acc": 0.92,
        "best_epoch": 8,
        "train_loss": 0.15,
        ...
    },
    "model_url": "https://..."
}
"""

import os
import sys
import json
import time
import signal
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import requests

# Load environment variables from .env file (if available)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import create_model, MODEL_REGISTRY
from src.losses import get_loss, AVAILABLE_LOSSES
from src.augmentations import get_train_transforms, get_val_transforms, AUGMENTATION_PRESETS
from src.trainer import ClassificationTrainer, TrainingConfig, compute_class_weights
from src.production import (
    ProductionConfig,
    validate_dataset,
    load_image_with_retry,
    check_memory_available,
    export_model_onnx,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    get_shutdown_handler,
    TrainingStats,
)

# Check ONNX availability
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# ============================================
# Graceful Shutdown Tracking
# ============================================

_shutdown_requested = False
_current_training_run_id = None
_current_supabase_url = None
_current_supabase_key = None


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


def _update_training_run_status(status: str, error_message: str = None):
    """Update training run status in Supabase (called during shutdown)."""
    global _current_training_run_id, _current_supabase_url, _current_supabase_key

    if not _current_training_run_id or not _current_supabase_url or not _current_supabase_key:
        return

    try:
        from datetime import datetime
        import httpx

        data = {
            "status": status,
            "completed_at": datetime.utcnow().isoformat(),
        }
        if error_message:
            data["error_message"] = error_message[:1000]

        # Direct HTTP call to Supabase (avoid circular dependencies)
        headers = {
            "apikey": _current_supabase_key,
            "Authorization": f"Bearer {_current_supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        url = f"{_current_supabase_url}/rest/v1/cls_training_runs?id=eq.{_current_training_run_id}"

        with httpx.Client(timeout=15) as client:
            response = client.patch(url, json=data, headers=headers)
            if response.status_code in [200, 204]:
                print(f"✓ Training run status updated to: {status}")
            else:
                print(f"⚠️  Failed to update status: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Failed to update training run status: {e}")


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n⚠️  Received {signal_name}. Marking training as cancelled...")
    _update_training_run_status("cancelled", f"Training interrupted by {signal_name}")


# ============================================
# Retry Helper for Supabase/HTTP calls
# ============================================

def retry_request(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: tuple = (ConnectionError, TimeoutError),
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to call (should be a lambda or callable)
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        retry_on: Exception types to retry on

    Returns:
        Result of the function call
    """
    import httpx

    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as e:
            last_exception = e
            if attempt < max_retries:
                print(f"⚠️  Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise
        except Exception as e:
            # Don't retry on other exceptions
            raise

    raise last_exception


def supabase_update(
    supabase_url: str,
    supabase_key: str,
    table: str,
    id_field: str,
    id_value: str,
    data: dict,
    timeout: int = 30,
):
    """
    Update a Supabase record with retry logic.
    """
    import httpx

    def do_request():
        return httpx.patch(
            f"{supabase_url}/rest/v1/{table}?{id_field}=eq.{id_value}",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=data,
            timeout=timeout,
        )

    return retry_request(do_request)


def supabase_upload_streaming(
    supabase_url: str,
    supabase_key: str,
    bucket: str,
    path: str,
    file_path: str,
    content_type: str = "application/octet-stream",
    timeout: int = 300,  # 5 minutes for large files
):
    """
    Upload a file to Supabase Storage with streaming (memory efficient).

    Unlike loading the entire file into RAM, this streams the file content
    directly to the HTTP request, preventing RAM exhaustion for large models.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Service key
        bucket: Storage bucket name
        path: Path within bucket
        file_path: Local file path to upload
        content_type: MIME type
        timeout: Request timeout in seconds
    """
    import httpx

    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)

    # For very large files (> 100MB), increase timeout proportionally
    if file_size_mb > 100:
        timeout = max(timeout, int(file_size_mb * 3))  # ~3 seconds per MB
        print(f"Large file ({file_size_mb:.1f}MB), timeout set to {timeout}s")

    def do_request():
        # Stream the file instead of loading into memory
        with open(file_path, "rb") as f:
            return httpx.post(
                f"{supabase_url}/storage/v1/object/{bucket}/{path}",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                },
                content=f,  # Stream file handle directly
                timeout=timeout,
            )

    return retry_request(do_request, max_retries=2)  # Fewer retries for large uploads


def supabase_upload(
    supabase_url: str,
    supabase_key: str,
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    timeout: int = 300,  # 5 minutes for large files
):
    """
    Upload bytes data to Supabase Storage with retry logic.
    For streaming large files, use supabase_upload_streaming() instead.
    """
    import httpx

    file_size_mb = len(data) / (1024 * 1024)

    # For very large files (> 100MB), increase timeout proportionally
    if file_size_mb > 100:
        timeout = max(timeout, int(file_size_mb * 3))  # ~3 seconds per MB
        print(f"Large file ({file_size_mb:.1f}MB), timeout set to {timeout}s")

    def do_request():
        return httpx.post(
            f"{supabase_url}/storage/v1/object/{bucket}/{path}",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": content_type,
            },
            content=data,
            timeout=timeout,
        )

    return retry_request(do_request, max_retries=2)  # Fewer retries for large uploads


# Default config
DEFAULT_CONFIG = {
    "model_name": "vit_base_patch16_224",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "scheduler": "cosine_warmup",
    "warmup_epochs": 2,
    "loss": "label_smoothing",
    "loss_smoothing": 0.1,
    "augmentation": "medium",
    "use_mixup": False,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.9999,
    "gradient_accumulation": 1,
    "gradient_clip": 1.0,
    "early_stopping": True,
    "patience": 5,
    "use_class_weights": False,
    "num_workers": 4,
    "image_size": 224,
    "seed": 42,
    # Anti-overfitting parameters (API can override based on dataset size)
    "drop_rate": 0.1,
    "drop_path_rate": 0.1,
}


class URLImageDataset(Dataset):
    """
    Dataset that loads images from URLs.

    Supports caching, parallel loading, and production error handling.
    """

    def __init__(
        self,
        image_data: List[Dict[str, Any]],
        transform=None,
        cache_dir: Optional[str] = None,
        max_workers: int = 8,
        production_config: Optional[ProductionConfig] = None,
        stats: Optional[TrainingStats] = None,
    ):
        """
        Args:
            image_data: List of {"url": str, "label": int}
            transform: Transform pipeline
            cache_dir: Directory to cache downloaded images
            max_workers: Number of download workers
            production_config: Production settings for retry/timeout
            stats: TrainingStats for tracking failures
        """
        self.image_data = image_data
        self.transform = transform
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.production_config = production_config or ProductionConfig()
        self.stats = stats

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Pre-download images (optional)
        self._cache = {}
        self._failed_urls = set()

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        url = item["url"]
        label = item["label"]

        # Load image
        if url in self._cache:
            img = self._cache[url]
        else:
            img = self._load_image(url)

        # Apply transforms
        if self.transform is not None:
            # Check if albumentations by module name only
            # (torchvision.transforms also has Compose with 'transforms' attr)
            transform_module = type(self.transform).__module__
            is_albumentations = 'albumentations' in transform_module

            if is_albumentations:
                # Albumentations style: pass image as keyword argument
                transformed = self.transform(image=np.array(img))
                img = transformed["image"]
            else:
                # Torchvision style: pass image directly
                img = self.transform(img)

        return img, label

    def _load_image(self, url: str) -> Image.Image:
        """Load image from URL with retry logic and caching."""
        if self.cache_dir:
            # Check file cache
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
            cache_path = os.path.join(self.cache_dir, f"{url_hash}.jpg")

            if os.path.exists(cache_path):
                try:
                    return Image.open(cache_path).convert("RGB")
                except Exception:
                    pass  # Corrupted cache, re-download

        # Use production utility for retry logic
        img, error = load_image_with_retry(url, self.production_config)

        if error:
            self._failed_urls.add(url)
            if self.stats:
                self.stats.record_failed_url(url, error)
            print(f"⚠️  Failed to load {url[:60]}...: {error}")
        else:
            # Cache to file
            if self.cache_dir and img:
                try:
                    img.save(cache_path, "JPEG", quality=95)
                except Exception:
                    pass

        return img

    def preload(self, progress_callback=None):
        """Pre-download all images in parallel."""
        print(f"Pre-loading {len(self.image_data)} images...")

        def download(item):
            url = item["url"]
            try:
                img = self._load_image(url)
                self._cache[url] = img
                return url not in self._failed_urls
            except:
                return False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(download, self.image_data))

        loaded = sum(results)
        failed = len(self.image_data) - loaded
        if failed > 0:
            print(f"⚠️  {failed} images failed to load")
        print(f"Pre-loaded {loaded}/{len(self.image_data)} images")
        return loaded


def train_model(
    config: Dict[str, Any],
    dataset: Dict[str, Any],
    device: str = "cuda",
    save_dir: str = "/tmp/cls_training",
    progress_callback=None,
    validation: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        config: Training configuration
        dataset: Dataset with train_urls, val_urls, class_names
        device: Device to train on
        save_dir: Directory to save checkpoints
        progress_callback: Optional callback for progress updates
        validation: Optional validation results with imbalance detection

    Returns:
        Training results
    """
    validation = validation or {}
    os.makedirs(save_dir, exist_ok=True)

    # Merge with defaults
    full_config = {**DEFAULT_CONFIG, **config}

    # Extract settings
    model_name = full_config["model_name"]
    num_classes = len(dataset.get("class_names", []))
    epochs = full_config["epochs"]
    batch_size = full_config["batch_size"]
    learning_rate = full_config["learning_rate"]
    loss_name = full_config["loss"]
    augmentation = full_config["augmentation"]
    image_size = full_config.get("image_size") or MODEL_REGISTRY.get(model_name, {}).get("input_size", 224)

    print(f"=" * 60)
    print("CLS TRAINING - SOTA")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"Classes: {num_classes} ({dataset.get('class_names', [])})")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"LR: {learning_rate}")
    print(f"Loss: {loss_name}")
    print(f"Augmentation: {augmentation}")
    print(f"Image Size: {image_size}")
    print(f"Device: {device}")
    print(f"=" * 60)

    # Set seed
    torch.manual_seed(full_config["seed"])
    np.random.seed(full_config["seed"])

    # Create transforms
    train_transform = get_train_transforms(
        img_size=image_size,
        preset=augmentation,
    )
    val_transform = get_val_transforms(img_size=image_size)

    # Create datasets
    cache_dir = os.path.join(save_dir, "image_cache")
    train_dataset = URLImageDataset(
        dataset["train_urls"],
        transform=train_transform,
        cache_dir=cache_dir,
    )
    val_dataset = URLImageDataset(
        dataset["val_urls"],
        transform=val_transform,
        cache_dir=cache_dir,
    )

    # Pre-load images
    train_dataset.preload()
    val_dataset.preload()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=full_config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=full_config["num_workers"],
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Compute class weights if needed (auto-enable if imbalance detected)
    class_weights = None
    use_class_weights = full_config["use_class_weights"]

    # Auto-enable class weights if validation recommends it
    if not use_class_weights and validation.get("recommend_class_weights"):
        imbalance_ratio = validation.get("imbalance_ratio", 1.0)
        print(f"⚠️  Auto-enabling class weights due to imbalance ratio: {imbalance_ratio:.1f}:1")
        use_class_weights = True

    if use_class_weights:
        train_labels = [item["label"] for item in dataset["train_urls"]]
        class_weights = compute_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(device)
        print(f"✓ Class weights enabled: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # Create model with configurable dropout rates
    drop_rate = full_config.get("drop_rate", 0.1)
    drop_path_rate = full_config.get("drop_path_rate", 0.1)
    print(f"Dropout config: drop_rate={drop_rate}, drop_path_rate={drop_path_rate}")

    model, model_info = create_model(
        model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    print(f"Model params: {model_info['params_m']:.1f}M")

    # Create loss
    loss_kwargs = {}
    if loss_name == "label_smoothing":
        loss_kwargs["smoothing"] = full_config.get("loss_smoothing", 0.1)
    elif loss_name == "focal":
        loss_kwargs["gamma"] = full_config.get("focal_gamma", 2.0)

    loss_fn = get_loss(
        loss_name,
        num_classes=num_classes,
        class_weights=class_weights,
        **loss_kwargs,
    )

    # Create training config
    training_config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=full_config["weight_decay"],
        optimizer=full_config["optimizer"],
        scheduler=full_config["scheduler"],
        warmup_epochs=full_config["warmup_epochs"],
        use_amp=full_config["use_amp"],
        use_ema=full_config["use_ema"],
        ema_decay=full_config["ema_decay"],
        gradient_accumulation=full_config["gradient_accumulation"],
        gradient_clip=full_config["gradient_clip"],
        early_stopping=full_config["early_stopping"],
        patience=full_config["patience"],
        use_mixup=full_config["use_mixup"],
        mixup_alpha=full_config["mixup_alpha"],
        cutmix_alpha=full_config["cutmix_alpha"],
        num_workers=full_config["num_workers"],
        seed=full_config["seed"],
    )

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        config=training_config,
        device=device,
        num_classes=num_classes,
        class_names=dataset.get("class_names"),
    )

    # Train (with shutdown checker for graceful termination)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        callback=progress_callback,
        shutdown_checker=is_shutdown_requested,
    )

    # Save final model info
    results["model_info"] = model_info
    results["config"] = full_config
    results["num_classes"] = num_classes
    results["class_names"] = dataset.get("class_names", [])

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED")
    print(f"Best Val Accuracy: {results['best_val_acc']:.2f}% (epoch {results['best_epoch']})")
    print(f"Total Time: {results['total_time']:.1f}s")
    print(f"{'=' * 60}")

    return results


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.

    Args:
        job: RunPod job dict with "input" key

    Returns:
        Result dict
    """
    stats = TrainingStats()

    # Initialize variables for exception handler access
    training_run_id = None
    supabase_url = None
    supabase_key = None

    try:
        job_input = job.get("input", job)

        # Extract inputs
        training_run_id = job_input.get("training_run_id", "unknown")
        config = job_input.get("config", {})
        dataset = job_input.get("dataset", {})
        supabase_url = job_input.get("supabase_url")
        supabase_key = job_input.get("supabase_key")
        resume_from = job_input.get("resume_from")  # Checkpoint path for resume

        # Track current training run for graceful shutdown
        global _current_training_run_id, _current_supabase_url, _current_supabase_key
        _current_training_run_id = training_run_id
        _current_supabase_url = supabase_url
        _current_supabase_key = supabase_key

        # Check for shutdown before starting
        if is_shutdown_requested():
            return {
                "error": "Shutdown requested before training started",
                "status": "CANCELLED",
            }

        print(f"\n{'=' * 60}")
        print(f"CLS TRAINING JOB: {training_run_id}")
        print(f"{'=' * 60}")

        # === PRODUCTION VALIDATION ===
        validation = validate_dataset(
            train_urls=dataset.get("train_urls", []),
            val_urls=dataset.get("val_urls", []),
            class_names=dataset.get("class_names", []),
        )

        if not validation["valid"]:
            return {
                "error": f"Dataset validation failed: {validation['issues']}",
                "status": "FAILED",
                "validation": validation,
            }

        if validation["warnings"]:
            print(f"⚠️  Validation warnings: {validation['warnings']}")

        print(f"✓ Dataset validated: {validation['train_samples']} train, {validation['val_samples']} val")

        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

            # === MEMORY CHECK ===
            memory_info = check_memory_available(
                model=None,  # Will check after model creation
                batch_size=config.get("batch_size", 32),
                img_size=config.get("image_size", 224),
                device=device,
            )
            if memory_info.get("memory_warning"):
                original_batch = config.get("batch_size", 32)
                new_batch = memory_info["recommended_batch_size"]

                # Calculate gradient accumulation to maintain effective batch size
                if new_batch > 0 and original_batch > new_batch:
                    grad_accum = max(1, original_batch // new_batch)
                    config["gradient_accumulation"] = grad_accum
                    print(f"⚠️  Memory warning: Reducing batch size {original_batch} → {new_batch}")
                    print(f"   Gradient accumulation set to {grad_accum} (effective batch: {new_batch * grad_accum})")
                else:
                    print(f"⚠️  Memory warning: Reducing batch size from {original_batch} to {new_batch}")

                config["batch_size"] = new_batch

        # Training directory - prefer /workspace over /tmp for more space
        workspace_dir = "/workspace" if os.path.exists("/workspace") else "/tmp"
        save_dir = f"{workspace_dir}/cls_training_{training_run_id}"

        # === UPDATE STATUS TO TRAINING ===
        if supabase_url and supabase_key:
            try:
                from datetime import datetime
                supabase_update(
                    supabase_url, supabase_key,
                    table="cls_training_runs",
                    id_field="id",
                    id_value=training_run_id,
                    data={
                        "status": "training",
                        "started_at": datetime.utcnow().isoformat(),
                    },
                )
                print(f"✓ Status updated: training")
            except Exception as e:
                print(f"⚠️  Failed to update training status: {e}")

        # Progress callback for Supabase updates (with retry)
        best_val_acc_so_far = 0.0

        def progress_callback(epoch, metrics):
            nonlocal best_val_acc_so_far
            if supabase_url and supabase_key:
                try:
                    update_data = {
                        "current_epoch": epoch + 1,
                    }

                    # Update best metrics if improved
                    val_acc = metrics.get("val_acc", metrics.get("accuracy", 0))
                    if val_acc > best_val_acc_so_far:
                        best_val_acc_so_far = val_acc
                        update_data["best_accuracy"] = val_acc
                        update_data["best_epoch"] = epoch + 1

                    if metrics.get("val_f1"):
                        update_data["best_f1"] = metrics["val_f1"]

                    # === METRICS HISTORY TRACKING ===
                    # Append current epoch metrics to metrics_history JSONB array
                    try:
                        from datetime import datetime
                        import httpx

                        # First, fetch current metrics_history
                        headers = {
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                        }
                        fetch_url = f"{supabase_url}/rest/v1/cls_training_runs?id=eq.{training_run_id}&select=metrics_history"

                        with httpx.Client(timeout=10) as client:
                            response = client.get(fetch_url, headers=headers)
                            if response.status_code == 200:
                                data = response.json()
                                if data and len(data) > 0:
                                    metrics_history = data[0].get("metrics_history") or []
                                else:
                                    metrics_history = []
                            else:
                                metrics_history = []

                        # Append new epoch metrics
                        epoch_metrics = {
                            "epoch": epoch + 1,
                            "timestamp": datetime.utcnow().isoformat(),
                            "train_loss": metrics.get("train_loss"),
                            "val_loss": metrics.get("val_loss"),
                            "val_acc": val_acc,
                            "val_f1": metrics.get("val_f1"),
                            "learning_rate": metrics.get("learning_rate"),
                        }
                        metrics_history.append(epoch_metrics)
                        update_data["metrics_history"] = metrics_history

                    except Exception as e:
                        print(f"Metrics history update failed: {e}")

                    supabase_update(
                        supabase_url, supabase_key,
                        table="cls_training_runs",
                        id_field="id",
                        id_value=training_run_id,
                        data=update_data,
                        timeout=15,
                    )
                except Exception as e:
                    print(f"Progress update failed: {e}")

        # Train
        results = train_model(
            config=config,
            dataset=dataset,
            device=device,
            save_dir=save_dir,
            progress_callback=progress_callback,
            validation=validation,
        )

        # Upload model to Supabase Storage (if configured) - using streaming for memory efficiency
        model_url = None
        model_storage_path = None
        if supabase_url and supabase_key:
            try:
                model_path = os.path.join(save_dir, "best_model.pt")
                if os.path.exists(model_path):
                    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"Uploading model ({file_size_mb:.1f}MB) using streaming...")

                    # Storage path within cls-models bucket (private bucket)
                    model_storage_path = f"{training_run_id}/best_model.pt"

                    # Use streaming upload to avoid loading entire file into RAM
                    response = supabase_upload_streaming(
                        supabase_url, supabase_key,
                        bucket="cls-models",
                        path=model_storage_path,
                        file_path=model_path,
                        timeout=300,  # 5 minutes for large models
                    )

                    if response.status_code in [200, 201]:
                        # Store the storage path (not public URL - bucket is private)
                        model_url = f"cls-models/{model_storage_path}"
                        print(f"✓ Model uploaded to storage: {model_url}")
                    else:
                        print(f"⚠️  Model upload failed: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"⚠️  Model upload failed: {e}")

        # === ONNX EXPORT (if requested) ===
        onnx_url = None
        if config.get("export_onnx", False):
            if not ONNX_AVAILABLE:
                print("⚠️  ONNX export requested but onnx package not installed")
            else:
                try:
                    onnx_path = os.path.join(save_dir, "model.onnx")
                    # Need to reload best model for export
                    from src.models import create_model
                    export_model, _ = create_model(
                        config.get("model_name", "efficientnet_b0"),
                        num_classes=len(dataset.get("class_names", [])),
                        pretrained=False,
                    )
                    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location="cpu", weights_only=False)
                    export_model.load_state_dict(best_ckpt["model_state_dict"])

                    export_result = export_model_onnx(
                        export_model,
                        onnx_path,
                        img_size=config.get("image_size", 224),
                    )
                    if export_result["success"]:
                        print(f"✓ ONNX exported: {export_result['size_mb']:.1f}MB")
                        # Upload ONNX to Supabase storage (with retry)
                        if supabase_url and supabase_key:
                            with open(onnx_path, "rb") as f:
                                onnx_data = f.read()
                            onnx_storage_path = f"{training_run_id}/model.onnx"
                            response = supabase_upload(
                                supabase_url, supabase_key,
                                bucket="cls-models",
                                path=onnx_storage_path,
                                data=onnx_data,
                                timeout=300,
                            )
                            if response.status_code in [200, 201]:
                                onnx_url = f"cls-models/{onnx_storage_path}"
                                print(f"✓ ONNX uploaded to storage: {onnx_url}")
                    else:
                        print(f"⚠️  ONNX export failed: {export_result.get('error')}")
                except Exception as e:
                    print(f"⚠️  ONNX export error: {e}")

        # Get training stats summary
        training_stats = stats.get_summary()

        # === UPDATE FINAL STATUS IN SUPABASE ===
        final_metrics = {
            "best_val_acc": results["best_val_acc"],
            "best_epoch": results["best_epoch"],
            "total_time": results["total_time"],
            "final_train_loss": results["history"]["train_loss"][-1] if results["history"]["train_loss"] else None,
            "final_val_loss": results["history"]["val_loss"][-1] if results["history"]["val_loss"] else None,
        }

        if supabase_url and supabase_key:
            try:
                from datetime import datetime
                supabase_update(
                    supabase_url, supabase_key,
                    table="cls_training_runs",
                    id_field="id",
                    id_value=training_run_id,
                    data={
                        "status": "completed",
                        "completed_at": datetime.utcnow().isoformat(),
                        "best_accuracy": results["best_val_acc"],
                        "best_epoch": results["best_epoch"],
                        "model_url": model_url,
                        "onnx_url": onnx_url,
                    },
                )
                print(f"✓ Final status updated in Supabase: completed")
            except Exception as e:
                print(f"⚠️  Failed to update final status: {e}")

        return {
            "status": "COMPLETED",
            "training_run_id": training_run_id,
            "metrics": final_metrics,
            "model_url": model_url,
            "onnx_url": onnx_url,
            "model_info": results["model_info"],
            "class_names": results["class_names"],
            "training_stats": training_stats,
        }

    except Exception as e:
        traceback.print_exc()
        error_msg = str(e)

        # === UPDATE FAILED STATUS IN SUPABASE (with retry) ===
        if supabase_url and supabase_key and training_run_id:
            try:
                from datetime import datetime
                supabase_update(
                    supabase_url, supabase_key,
                    table="cls_training_runs",
                    id_field="id",
                    id_value=training_run_id,
                    data={
                        "status": "failed",
                        "completed_at": datetime.utcnow().isoformat(),
                        "error_message": error_msg[:1000],  # Limit error message length
                    },
                )
                print(f"✓ Failed status updated in Supabase")
            except Exception as update_error:
                print(f"⚠️  Failed to update failed status: {update_error}")

        return {
            "error": error_msg,
            "status": "FAILED",
            "traceback": traceback.format_exc(),
            "training_stats": stats.get_summary() if stats else None,
        }


# ============================================
# Signal Handler Registration (MUST be at end of file)
# ============================================
# Register signal handlers AFTER all functions are defined
# This ensures _signal_handler can access all necessary functions
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ============================================
# RunPod Serverless Entry Point
# ============================================
if __name__ == "__main__":
    import runpod

    # Check if we're running as a RunPod serverless worker
    if os.environ.get("RUNPOD_POD_ID"):
        print("Starting CLS Training Worker (RunPod Serverless)...")
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing with synthetic data
        print("Testing CLS Training Handler (Local Mode)...")

        test_job = {
            "input": {
                "training_run_id": "test_run",
                "config": {
                    "model_name": "efficientnet_b0",
                    "epochs": 2,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "loss": "label_smoothing",
                    "augmentation": "light",
                    "use_amp": True,
                    "use_ema": False,
                    "num_workers": 0,
                },
                "dataset": {
                    "train_urls": [
                        {"url": "https://picsum.photos/id/237/224/224", "label": 0},
                        {"url": "https://picsum.photos/id/238/224/224", "label": 0},
                        {"url": "https://picsum.photos/id/239/224/224", "label": 1},
                        {"url": "https://picsum.photos/id/240/224/224", "label": 1},
                    ],
                    "val_urls": [
                        {"url": "https://picsum.photos/id/241/224/224", "label": 0},
                        {"url": "https://picsum.photos/id/242/224/224", "label": 1},
                    ],
                    "class_names": ["class_a", "class_b"],
                },
            }
        }

        result = handler(test_job)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
