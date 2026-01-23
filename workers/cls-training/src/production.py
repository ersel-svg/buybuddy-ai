"""
Production Hardening Utilities

Handles:
- Failed URL loads with retry logic
- Corrupted images
- Label validation
- Memory management
- Checkpoint resume
- ONNX export
- Graceful shutdown
"""

import os
import sys
import time
import signal
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from io import BytesIO
import traceback

import torch
import torch.nn as nn
import requests
from PIL import Image
import numpy as np


@dataclass
class ProductionConfig:
    """Production configuration."""
    # URL loading
    url_timeout: int = 30
    url_retries: int = 3
    url_retry_delay: float = 1.0

    # Validation
    max_label_value: Optional[int] = None
    min_images_per_class: int = 1

    # Memory
    max_batch_size: int = 64
    auto_reduce_batch: bool = True
    gradient_checkpointing: bool = False

    # Checkpoint
    checkpoint_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3

    # Export
    export_onnx: bool = False
    export_torchscript: bool = False

    # Graceful shutdown
    enable_graceful_shutdown: bool = True


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""

    def __init__(self):
        self.shutdown_requested = False
        self._lock = threading.Lock()

    def request_shutdown(self, signum=None, frame=None):
        with self._lock:
            self.shutdown_requested = True
            print(f"\n⚠️  Shutdown requested (signal {signum}). Finishing current epoch...")

    def should_stop(self) -> bool:
        with self._lock:
            return self.shutdown_requested

    def register(self):
        """Register signal handlers."""
        signal.signal(signal.SIGTERM, self.request_shutdown)
        signal.signal(signal.SIGINT, self.request_shutdown)


# Global shutdown handler
_shutdown_handler = GracefulShutdown()


def get_shutdown_handler() -> GracefulShutdown:
    return _shutdown_handler


def validate_dataset(
    train_urls: List[Dict[str, Any]],
    val_urls: List[Dict[str, Any]],
    class_names: List[str],
    config: ProductionConfig = None,
) -> Dict[str, Any]:
    """
    Validate dataset before training.

    Returns:
        Dict with validation results and any issues found.
    """
    config = config or ProductionConfig()
    issues = []
    warnings = []

    num_classes = len(class_names)

    # Check for empty data
    if not train_urls:
        issues.append("No training data provided")
    if not val_urls:
        warnings.append("No validation data provided")
    if not class_names:
        issues.append("No class names provided")

    # Check label validity
    train_labels = [item.get("label", -1) for item in train_urls]
    val_labels = [item.get("label", -1) for item in val_urls]
    all_labels = train_labels + val_labels

    invalid_labels = [l for l in all_labels if l < 0 or l >= num_classes]
    if invalid_labels:
        issues.append(f"Invalid labels found: {set(invalid_labels)} (valid: 0-{num_classes-1})")

    # Check class distribution
    from collections import Counter
    train_dist = Counter(train_labels)
    for class_idx in range(num_classes):
        count = train_dist.get(class_idx, 0)
        if count == 0:
            warnings.append(f"Class '{class_names[class_idx]}' has no training samples")
        elif count < config.min_images_per_class:
            warnings.append(f"Class '{class_names[class_idx]}' has only {count} samples")

    # Calculate class imbalance ratio
    imbalance_ratio = 1.0
    recommend_class_weights = False
    if train_dist and len(train_dist) > 0:
        counts = [train_dist.get(i, 0) for i in range(num_classes)]
        non_zero_counts = [c for c in counts if c > 0]
        if non_zero_counts:
            max_count = max(non_zero_counts)
            min_count = min(non_zero_counts)
            if min_count > 0:
                imbalance_ratio = max_count / min_count
                # Recommend class weights if imbalance > 3:1
                if imbalance_ratio > 3.0:
                    recommend_class_weights = True
                    warnings.append(
                        f"Class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
                        "Consider enabling class_weights or using focal loss."
                    )

    # Check URLs
    empty_urls = sum(1 for item in train_urls + val_urls if not item.get("url"))
    if empty_urls > 0:
        issues.append(f"{empty_urls} items have empty URLs")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "num_classes": num_classes,
        "train_samples": len(train_urls),
        "val_samples": len(val_urls),
        "class_distribution": dict(train_dist),
        "imbalance_ratio": imbalance_ratio,
        "recommend_class_weights": recommend_class_weights,
    }


def load_image_with_retry(
    url: str,
    config: ProductionConfig = None,
    fallback_size: Tuple[int, int] = (224, 224),
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Load image from URL with retry logic.

    Returns:
        (image, error_message) - error_message is None on success
    """
    config = config or ProductionConfig()

    for attempt in range(config.url_retries):
        try:
            response = requests.get(url, timeout=config.url_timeout)
            response.raise_for_status()

            # Try to open the image
            img = Image.open(BytesIO(response.content))

            # Validate it's a real image by accessing pixels
            img.load()

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            return img, None

        except requests.exceptions.Timeout:
            error = f"Timeout after {config.url_timeout}s"
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}"
            if e.response.status_code == 404:
                break  # Don't retry 404s
        except requests.exceptions.RequestException as e:
            error = f"Request failed: {str(e)[:50]}"
        except Exception as e:
            error = f"Image error: {str(e)[:50]}"

        if attempt < config.url_retries - 1:
            time.sleep(config.url_retry_delay)

    # Return fallback gray image
    fallback = Image.new("RGB", fallback_size, (128, 128, 128))
    return fallback, error


def check_memory_available(
    model: nn.Module,
    batch_size: int,
    img_size: int = 224,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Check if there's enough GPU memory for training.

    Returns:
        Dict with memory info and recommended batch size.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return {
            "device": device,
            "check_performed": False,
            "recommended_batch_size": batch_size,
        }

    # Get current memory status
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = total_memory - allocated_memory

    # Estimate memory per sample (rough estimate)
    # Input: batch_size * 3 * img_size * img_size * 4 bytes
    # Gradients and activations: ~10x input size for transformers
    input_size = 3 * img_size * img_size * 4  # bytes
    estimated_per_sample = input_size * 20  # Conservative estimate

    # Estimate max batch size
    max_batch = int(free_memory * 0.7 / estimated_per_sample)  # Use 70% of free memory
    recommended_batch = min(batch_size, max(1, max_batch))

    return {
        "device": device,
        "check_performed": True,
        "total_memory_gb": total_memory / 1e9,
        "free_memory_gb": free_memory / 1e9,
        "requested_batch_size": batch_size,
        "recommended_batch_size": recommended_batch,
        "memory_warning": recommended_batch < batch_size,
    }


def export_model_onnx(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
    num_classes: int = 10,
    opset_version: int = 14,
) -> Dict[str, Any]:
    """
    Export model to ONNX format.

    Returns:
        Dict with export status and path.
    """
    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)

        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        # Verify
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        return {
            "success": True,
            "path": save_path,
            "size_mb": os.path.getsize(save_path) / 1e6,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def export_model_torchscript(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
) -> Dict[str, Any]:
    """
    Export model to TorchScript format.

    Returns:
        Dict with export status and path.
    """
    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)

        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()

        # Script the model
        scripted = torch.jit.trace(model, dummy_input)
        scripted.save(save_path)

        return {
            "success": True,
            "path": save_path,
            "size_mb": os.path.getsize(save_path) / 1e6,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    save_dir: str,
    config: Dict[str, Any],
    ema_shadow: Optional[Dict] = None,
    is_best: bool = False,
) -> str:
    """
    Save training checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    if ema_shadow is not None:
        checkpoint["ema_shadow"] = ema_shadow

    # Save epoch checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Save as best if applicable
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(checkpoint, best_path)

    # Save latest
    latest_path = os.path.join(save_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Returns:
        Dict with checkpoint data.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
        "ema_shadow": checkpoint.get("ema_shadow"),
    }


def cleanup_old_checkpoints(
    save_dir: str,
    keep_last_n: int = 3,
    pattern: str = "checkpoint_epoch_*.pt",
):
    """Remove old checkpoints, keeping only the last N."""
    import glob

    checkpoints = sorted(glob.glob(os.path.join(save_dir, pattern)))

    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            try:
                os.remove(old_ckpt)
            except:
                pass


class TrainingStats:
    """Track training statistics for monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.epoch_times = []
        self.failed_urls = []
        self.memory_usage = []

    def record_epoch(self, epoch: int, duration: float, metrics: Dict[str, Any]):
        self.epoch_times.append({
            "epoch": epoch,
            "duration": duration,
            "metrics": metrics,
        })

    def record_failed_url(self, url: str, error: str):
        self.failed_urls.append({"url": url, "error": error})

    def record_memory(self):
        if torch.cuda.is_available():
            self.memory_usage.append({
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            })

    def get_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        avg_epoch_time = sum(e["duration"] for e in self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

        return {
            "total_time_seconds": total_time,
            "total_epochs": len(self.epoch_times),
            "avg_epoch_time": avg_epoch_time,
            "failed_url_count": len(self.failed_urls),
            "failed_urls": self.failed_urls[:10],  # First 10 only
            "peak_memory_gb": max((m["allocated_gb"] for m in self.memory_usage), default=0),
        }
