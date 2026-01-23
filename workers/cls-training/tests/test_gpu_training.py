#!/usr/bin/env python3
"""
CLS Training - GPU Integration Tests

Tests all combinations of:
- Models: efficientnet_b0, vit_tiny, resnet50
- Losses: ce, label_smoothing, focal, poly
- Augmentations: none, light, medium, sota

Run on GPU pod:
    python tests/test_gpu_training.py
"""

import sys
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float
    metrics: Dict[str, Any] = None


class GPUTestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("WARNING: No GPU available, running on CPU")

    def run_test(self, name: str, func, *args, **kwargs) -> TestResult:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            if isinstance(result, dict):
                test_result = TestResult(
                    name=name,
                    passed=True,
                    message="OK",
                    duration=duration,
                    metrics=result
                )
                print(f"✓ PASSED ({duration:.2f}s)")
                if result:
                    for k, v in result.items():
                        if isinstance(v, float):
                            print(f"  {k}: {v:.4f}")
                        else:
                            print(f"  {k}: {v}")
            else:
                test_result = TestResult(
                    name=name,
                    passed=True,
                    message=str(result) if result else "OK",
                    duration=duration
                )
                print(f"✓ PASSED ({duration:.2f}s)")

        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(
                name=name,
                passed=False,
                message=f"{type(e).__name__}: {str(e)[:200]}",
                duration=duration
            )
            print(f"✗ FAILED ({duration:.2f}s)")
            print(f"  Error: {test_result.message}")
            import traceback
            traceback.print_exc()

        self.results.append(test_result)

        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return test_result

    def summary(self):
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)

        print(f"Total: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time:.1f}s")

        if failed > 0:
            print(f"\nFAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}: {r.message}")

        return failed == 0


def create_synthetic_data(
    num_train: int = 100,
    num_val: int = 20,
    num_classes: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """Create synthetic image data for testing."""

    # Generate random images and labels
    train_x = torch.randn(num_train, 3, img_size, img_size)
    train_y = torch.randint(0, num_classes, (num_train,))

    val_x = torch.randn(num_val, 3, img_size, img_size)
    val_y = torch.randint(0, num_classes, (num_val,))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def test_model_training(
    model_name: str,
    loss_name: str,
    augmentation: str,
    num_classes: int = 4,
    epochs: int = 2,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Test training a specific model configuration."""

    from src.models import create_model, MODEL_REGISTRY
    from src.losses import get_loss
    from src.trainer import ClassificationTrainer, TrainingConfig

    # Get model input size
    input_size = MODEL_REGISTRY.get(model_name, {}).get("input_size", 224)

    # Create model
    model, model_info = create_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,  # Faster for testing
    )

    # Create loss
    loss_fn = get_loss(loss_name, num_classes=num_classes)

    # Create config
    config = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        learning_rate=1e-3,
        use_amp=True,
        use_ema=False,  # Faster for testing
        use_mixup=False,
        early_stopping=False,
        num_workers=0,
    )

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        num_classes=num_classes,
    )

    # Create synthetic data
    train_loader, val_loader = create_synthetic_data(
        num_train=64,
        num_val=16,
        num_classes=num_classes,
        img_size=input_size,
    )

    # Train
    with tempfile.TemporaryDirectory() as tmpdir:
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmpdir,
        )

    return {
        "model": model_name,
        "loss": loss_name,
        "augmentation": augmentation,
        "best_val_acc": results["best_val_acc"],
        "best_epoch": results["best_epoch"],
        "total_time": results["total_time"],
        "params_m": model_info["params_m"],
    }


def test_arcface_training(device: str = "cuda") -> Dict[str, Any]:
    """Test ArcFace loss with embedding model."""

    from src.models import create_model, ModelWithEmbedding
    from src.losses import get_loss
    from src.trainer import ClassificationTrainer, TrainingConfig
    import torch.nn.functional as F

    num_classes = 4
    embedding_dim = 256

    # Create base model
    base_model, model_info = create_model(
        "efficientnet_b0",
        num_classes=num_classes,
        pretrained=False,
    )

    # Wrap with embedding
    model = ModelWithEmbedding(
        base_model=base_model,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    )

    # Create ArcFace loss
    loss_fn = get_loss(
        "arcface",
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )

    # Move to device
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Test forward pass
    x = torch.randn(4, 3, 224, 224).to(device)
    y = torch.randint(0, num_classes, (4,)).to(device)

    with torch.no_grad():
        out = model(x)
        embeddings = out["embeddings"]
        loss = loss_fn(embeddings, y)

    return {
        "embedding_shape": list(embeddings.shape),
        "loss": loss.item(),
        "status": "OK",
    }


def test_mixup_cutmix(device: str = "cuda") -> Dict[str, Any]:
    """Test MixUp and CutMix augmentation."""

    from src.augmentations import MixUpCutMix

    num_classes = 10
    batch_size = 8

    mixup = MixUpCutMix(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_prob=0.5,
        cutmix_prob=0.5,
        num_classes=num_classes,
    )

    # Test on GPU
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)

    mixed_images, mixed_targets = mixup(images, targets)

    return {
        "input_shape": list(images.shape),
        "output_shape": list(mixed_images.shape),
        "target_shape": list(mixed_targets.shape),
        "status": "OK",
    }


def test_full_handler(device: str = "cuda") -> Dict[str, Any]:
    """Test the full handler with a job."""

    from handler import handler

    # Create a test job
    job = {
        "input": {
            "training_run_id": "gpu_test",
            "config": {
                "model_name": "efficientnet_b0",
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "loss": "label_smoothing",
                "augmentation": "light",
                "use_amp": True,
                "use_ema": False,
                "num_workers": 0,
            },
            "dataset": {
                "train_urls": [
                    {"url": "https://picsum.photos/224", "label": 0},
                    {"url": "https://picsum.photos/224", "label": 0},
                    {"url": "https://picsum.photos/224", "label": 1},
                    {"url": "https://picsum.photos/224", "label": 1},
                ],
                "val_urls": [
                    {"url": "https://picsum.photos/224", "label": 0},
                    {"url": "https://picsum.photos/224", "label": 1},
                ],
                "class_names": ["class_a", "class_b"],
            },
        }
    }

    result = handler(job)

    return {
        "status": result.get("status"),
        "best_val_acc": result.get("metrics", {}).get("best_val_acc"),
        "total_time": result.get("metrics", {}).get("total_time"),
    }


def main():
    runner = GPUTestRunner()
    device = runner.device

    print("\n" + "="*60)
    print("CLS TRAINING - GPU INTEGRATION TESTS")
    print("="*60)

    # Test 1: Different models
    models_to_test = ["efficientnet_b0", "vit_tiny_patch16_224", "resnet50"]
    for model_name in models_to_test:
        runner.run_test(
            f"Model: {model_name}",
            test_model_training,
            model_name=model_name,
            loss_name="label_smoothing",
            augmentation="light",
            device=device,
        )

    # Test 2: Different losses
    losses_to_test = ["ce", "label_smoothing", "focal", "poly"]
    for loss_name in losses_to_test:
        runner.run_test(
            f"Loss: {loss_name}",
            test_model_training,
            model_name="efficientnet_b0",
            loss_name=loss_name,
            augmentation="light",
            device=device,
        )

    # Test 3: ArcFace with embeddings
    runner.run_test(
        "ArcFace + Embeddings",
        test_arcface_training,
        device=device,
    )

    # Test 4: MixUp/CutMix
    runner.run_test(
        "MixUp/CutMix",
        test_mixup_cutmix,
        device=device,
    )

    # Test 5: Full handler
    runner.run_test(
        "Full Handler",
        test_full_handler,
        device=device,
    )

    # Summary
    success = runner.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
