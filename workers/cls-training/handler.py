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

    Supports caching and parallel loading.
    """

    def __init__(
        self,
        image_data: List[Dict[str, Any]],
        transform=None,
        cache_dir: Optional[str] = None,
        max_workers: int = 8,
    ):
        """
        Args:
            image_data: List of {"url": str, "label": int}
            transform: Transform pipeline
            cache_dir: Directory to cache downloaded images
            max_workers: Number of download workers
        """
        self.image_data = image_data
        self.transform = transform
        self.cache_dir = cache_dir
        self.max_workers = max_workers

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Pre-download images (optional)
        self._cache = {}

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
            if hasattr(self.transform, '__call__'):
                # Check if albumentations
                if hasattr(self.transform, 'transforms'):
                    transformed = self.transform(image=np.array(img))
                    img = transformed["image"]
                else:
                    img = self.transform(img)

        return img, label

    def _load_image(self, url: str) -> Image.Image:
        """Load image from URL or cache."""
        if self.cache_dir:
            # Check file cache
            cache_path = os.path.join(
                self.cache_dir,
                url.split("/")[-1].split("?")[0]
            )
            if os.path.exists(cache_path):
                return Image.open(cache_path).convert("RGB")

        # Download
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # Cache to file
            if self.cache_dir:
                img.save(cache_path)

            return img
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            # Return blank image as fallback
            return Image.new("RGB", (224, 224), (128, 128, 128))

    def preload(self, progress_callback=None):
        """Pre-download all images in parallel."""
        print(f"Pre-loading {len(self.image_data)} images...")

        def download(item):
            url = item["url"]
            try:
                img = self._load_image(url)
                self._cache[url] = img
                return True
            except:
                return False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(download, self.image_data))

        loaded = sum(results)
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
    try:
        job_input = job.get("input", job)

        # Extract inputs
        training_run_id = job_input.get("training_run_id", "unknown")
        config = job_input.get("config", {})
        dataset = job_input.get("dataset", {})
        supabase_url = job_input.get("supabase_url")
        supabase_key = job_input.get("supabase_key")

        print(f"\n{'=' * 60}")
        print(f"CLS TRAINING JOB: {training_run_id}")
        print(f"{'=' * 60}")

        # Validate inputs
        if not dataset.get("train_urls"):
            return {"error": "No training data provided", "status": "FAILED"}

        if not dataset.get("class_names"):
            return {"error": "No class names provided", "status": "FAILED"}

        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Training directory
        save_dir = f"/tmp/cls_training_{training_run_id}"

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
            "model_info": results["model_info"],
            "class_names": results["class_names"],
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "FAILED",
            "traceback": traceback.format_exc(),
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
