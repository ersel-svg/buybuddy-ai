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
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import requests

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
            # Check if albumentations by checking the module name
            is_albumentations = (
                hasattr(self.transform, '__module__') and
                'albumentations' in self.transform.__module__
            )
            if is_albumentations:
                transformed = self.transform(image=np.array(img))
                img = transformed["image"]
            else:
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
) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        config: Training configuration
        dataset: Dataset with train_urls, val_urls, class_names
        device: Device to train on
        save_dir: Directory to save checkpoints
        progress_callback: Optional callback for progress updates

    Returns:
        Training results
    """
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

    # Compute class weights if needed
    class_weights = None
    if full_config["use_class_weights"]:
        train_labels = [item["label"] for item in dataset["train_urls"]]
        class_weights = compute_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights.tolist()}")

    # Create model
    model, model_info = create_model(
        model_name,
        num_classes=num_classes,
        pretrained=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
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

    # Train
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        callback=progress_callback,
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

    try:
        job_input = job.get("input", job)

        # Extract inputs
        training_run_id = job_input.get("training_run_id", "unknown")
        config = job_input.get("config", {})
        dataset = job_input.get("dataset", {})
        supabase_url = job_input.get("supabase_url")
        supabase_key = job_input.get("supabase_key")
        resume_from = job_input.get("resume_from")  # Checkpoint path for resume

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
                print(f"⚠️  Memory warning: Reducing batch size from {config.get('batch_size', 32)} to {memory_info['recommended_batch_size']}")
                config["batch_size"] = memory_info["recommended_batch_size"]

        # Training directory - prefer /workspace over /tmp for more space
        workspace_dir = "/workspace" if os.path.exists("/workspace") else "/tmp"
        save_dir = f"{workspace_dir}/cls_training_{training_run_id}"

        # Progress callback for Supabase updates
        def progress_callback(epoch, metrics):
            if supabase_url and supabase_key:
                try:
                    # Update training run in Supabase
                    import httpx
                    httpx.patch(
                        f"{supabase_url}/rest/v1/cls_training_runs?id=eq.{training_run_id}",
                        headers={
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "current_epoch": epoch + 1,
                            "metrics": metrics,
                        },
                        timeout=10,
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
        )

        # Upload model to Supabase Storage (if configured)
        model_url = None
        if supabase_url and supabase_key:
            try:
                model_path = os.path.join(save_dir, "best_model.pt")
                if os.path.exists(model_path):
                    # Upload to Supabase storage
                    import httpx
                    with open(model_path, "rb") as f:
                        model_data = f.read()

                    storage_path = f"cls-models/{training_run_id}/best_model.pt"
                    response = httpx.post(
                        f"{supabase_url}/storage/v1/object/cls-models/{training_run_id}/best_model.pt",
                        headers={
                            "apikey": supabase_key,
                            "Authorization": f"Bearer {supabase_key}",
                            "Content-Type": "application/octet-stream",
                        },
                        content=model_data,
                        timeout=120,
                    )

                    if response.status_code in [200, 201]:
                        model_url = f"{supabase_url}/storage/v1/object/public/{storage_path}"
                        print(f"Model uploaded: {model_url}")
            except Exception as e:
                print(f"Model upload failed: {e}")

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
                        # Upload ONNX to Supabase too
                        if supabase_url and supabase_key:
                            import httpx
                            with open(onnx_path, "rb") as f:
                                onnx_data = f.read()
                            response = httpx.post(
                                f"{supabase_url}/storage/v1/object/cls-models/{training_run_id}/model.onnx",
                                headers={
                                    "apikey": supabase_key,
                                    "Authorization": f"Bearer {supabase_key}",
                                    "Content-Type": "application/octet-stream",
                                },
                                content=onnx_data,
                                timeout=120,
                            )
                            if response.status_code in [200, 201]:
                                onnx_url = f"{supabase_url}/storage/v1/object/public/cls-models/{training_run_id}/model.onnx"
                    else:
                        print(f"⚠️  ONNX export failed: {export_result.get('error')}")
                except Exception as e:
                    print(f"⚠️  ONNX export error: {e}")

        # Get training stats summary
        training_stats = stats.get_summary()

        return {
            "status": "COMPLETED",
            "training_run_id": training_run_id,
            "metrics": {
                "best_val_acc": results["best_val_acc"],
                "best_epoch": results["best_epoch"],
                "total_time": results["total_time"],
                "final_train_loss": results["history"]["train_loss"][-1] if results["history"]["train_loss"] else None,
                "final_val_loss": results["history"]["val_loss"][-1] if results["history"]["val_loss"] else None,
            },
            "model_url": model_url,
            "onnx_url": onnx_url,
            "model_info": results["model_info"],
            "class_names": results["class_names"],
            "training_stats": training_stats,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "FAILED",
            "traceback": traceback.format_exc(),
            "training_stats": stats.get_summary() if stats else None,
        }


# For local testing
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing CLS Training Handler...")

    # Create test job
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
