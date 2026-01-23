#!/usr/bin/env python3
"""
CLS Training Pipeline - Comprehensive Test Suite

Phase 1: Local CPU Tests
- Tests all components without GPU
- Catches import errors, logic bugs, API mismatches

Usage:
    python tests/test_pipeline.py

    # Or with pytest
    pytest tests/test_pipeline.py -v
"""

import sys
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestStatus(Enum):
    PASS = "✓"
    FAIL = "✗"
    SKIP = "○"
    WARN = "⚠"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str = ""
    duration: float = 0.0


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.current_section = ""

    def section(self, name: str):
        self.current_section = name
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

    def test(self, name: str, func, *args, **kwargs) -> TestResult:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            if result is True or result is None:
                status = TestStatus.PASS
                msg = ""
            elif isinstance(result, str):
                status = TestStatus.WARN
                msg = result
            else:
                status = TestStatus.PASS
                msg = str(result) if result else ""

            test_result = TestResult(name, status, msg, duration)
        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(
                name,
                TestStatus.FAIL,
                f"{type(e).__name__}: {str(e)[:100]}",
                duration
            )

        self.results.append(test_result)

        icon = test_result.status.value
        time_str = f"({test_result.duration*1000:.0f}ms)" if test_result.duration > 0.01 else ""
        msg_str = f" - {test_result.message}" if test_result.message else ""
        print(f"  {icon} {name} {time_str}{msg_str}")

        return test_result

    def summary(self):
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        warned = sum(1 for r in self.results if r.status == TestStatus.WARN)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIP)

        total = len(self.results)

        print(f"  Total:   {total}")
        print(f"  Passed:  {passed} {TestStatus.PASS.value}")
        print(f"  Failed:  {failed} {TestStatus.FAIL.value}")
        print(f"  Warned:  {warned} {TestStatus.WARN.value}")
        print(f"  Skipped: {skipped} {TestStatus.SKIP.value}")

        if failed > 0:
            print(f"\n  FAILED TESTS:")
            for r in self.results:
                if r.status == TestStatus.FAIL:
                    print(f"    {TestStatus.FAIL.value} {r.name}: {r.message}")

        print(f"\n{'='*60}")

        return failed == 0


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_import_torch():
    """Test PyTorch import."""
    import torch
    return f"torch {torch.__version__}"


def test_import_timm():
    """Test timm import."""
    import timm
    return f"timm {timm.__version__}"


def test_import_albumentations():
    """Test albumentations import."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        return f"albumentations {A.__version__}"
    except ImportError:
        return "albumentations not available (will use torchvision)"


def test_import_src_modules():
    """Test src module imports."""
    from src.models import create_model, MODEL_REGISTRY
    from src.losses import get_loss, AVAILABLE_LOSSES
    from src.augmentations import get_train_transforms, get_val_transforms
    from src.trainer import ClassificationTrainer, TrainingConfig
    return True


def test_model_registry():
    """Test MODEL_REGISTRY has models."""
    from src.models import MODEL_REGISTRY
    count = len(MODEL_REGISTRY)
    assert count > 0, "No models in registry"
    return f"{count} models available"


def test_create_each_model():
    """Test creating each model type."""
    from src.models import create_model, MODEL_REGISTRY
    import torch

    # Test subset of models (faster)
    test_models = ["efficientnet_b0", "vit_tiny_patch16_224", "resnet50"]

    for model_name in test_models:
        if model_name not in MODEL_REGISTRY:
            continue
        model, info = create_model(model_name, num_classes=10, pretrained=False)
        # Quick forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10), f"Wrong output shape for {model_name}"

    return f"Tested {len(test_models)} models"


def test_loss_functions():
    """Test all loss functions."""
    from src.losses import get_loss, AVAILABLE_LOSSES
    import torch

    batch_size = 4
    num_classes = 10

    # Dummy inputs
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    targets_soft = torch.softmax(torch.randn(batch_size, num_classes), dim=1)

    tested = []
    for loss_name in AVAILABLE_LOSSES:
        loss_fn = get_loss(loss_name, num_classes=num_classes)

        # Test with hard targets
        try:
            loss = loss_fn(logits, targets)
            assert loss.dim() == 0, f"Loss should be scalar for {loss_name}"
            tested.append(loss_name)
        except Exception as e:
            # Some losses need soft targets
            try:
                loss = loss_fn(logits, targets_soft)
                tested.append(loss_name)
            except:
                raise

    return f"Tested {len(tested)} losses: {', '.join(tested)}"


def test_augmentation_presets():
    """Test all augmentation presets."""
    from src.augmentations import get_train_transforms, get_val_transforms, AUGMENTATION_PRESETS
    import numpy as np
    from PIL import Image

    # Create dummy image
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img_np = np.array(img)

    for preset in AUGMENTATION_PRESETS.keys():
        transform = get_train_transforms(img_size=224, preset=preset)

        # Check if albumentations or torchvision
        is_albu = hasattr(transform, '__module__') and 'albumentations' in transform.__module__

        if is_albu:
            result = transform(image=img_np)
            tensor = result["image"]
        else:
            tensor = transform(img)

        assert tensor.shape == (3, 224, 224), f"Wrong shape for preset {preset}"

    return f"Tested {len(AUGMENTATION_PRESETS)} presets"


def test_val_transforms():
    """Test validation transforms."""
    from src.augmentations import get_val_transforms
    import numpy as np
    from PIL import Image

    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img_np = np.array(img)

    transform = get_val_transforms(img_size=224)

    is_albu = hasattr(transform, '__module__') and 'albumentations' in transform.__module__

    if is_albu:
        result = transform(image=img_np)
        tensor = result["image"]
    else:
        tensor = transform(img)

    assert tensor.shape == (3, 224, 224), "Wrong val transform output shape"
    return True


def test_url_dataset_mock():
    """Test URLImageDataset with mock data."""
    from handler import URLImageDataset
    from src.augmentations import get_train_transforms
    from PIL import Image
    import tempfile
    import os

    # Create temp images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create local test images
        test_images = []
        for i in range(4):
            img_path = os.path.join(tmpdir, f"test_{i}.jpg")
            img = Image.new("RGB", (224, 224), color=(i*50, i*50, i*50))
            img.save(img_path)
            test_images.append({"url": f"file://{img_path}", "label": i % 2})

        transform = get_train_transforms(img_size=224, preset="light")

        # This will fail for file:// URLs, but tests the structure
        dataset = URLImageDataset(test_images, transform=transform)

        assert len(dataset) == 4, "Dataset length mismatch"

    return True


def test_training_config():
    """Test TrainingConfig creation."""
    from src.trainer import TrainingConfig

    config = TrainingConfig(
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
    )

    assert config.epochs == 10
    assert config.batch_size == 32
    return True


def test_class_weights():
    """Test class weight computation."""
    from src.trainer import compute_class_weights
    import torch

    # Imbalanced labels
    labels = [0, 0, 0, 0, 1, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)

    assert weights.shape == (3,)
    # Class 1 should have highest weight (least samples)
    assert weights[1] > weights[0]
    return True


def test_mixup_cutmix():
    """Test MixUp/CutMix augmentation."""
    from src.augmentations import MixUpCutMix
    import torch

    mixup = MixUpCutMix(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        mixup_prob=0.5,
        cutmix_prob=0.5,
        num_classes=10,
    )

    images = torch.randn(4, 3, 224, 224)
    targets = torch.randint(0, 10, (4,))

    mixed_images, mixed_targets = mixup(images, targets)

    assert mixed_images.shape == images.shape
    assert mixed_targets.shape == (4, 10)  # One-hot
    return True


def test_trainer_creation():
    """Test ClassificationTrainer creation."""
    from src.trainer import ClassificationTrainer, TrainingConfig
    from src.models import create_model
    from src.losses import get_loss
    import torch

    model, _ = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    loss_fn = get_loss("cross_entropy", num_classes=5)
    config = TrainingConfig(epochs=1, batch_size=2, learning_rate=1e-4)

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device="cpu",
        num_classes=5,
    )

    return True


def test_single_training_step():
    """Test a single training step."""
    from src.trainer import ClassificationTrainer, TrainingConfig
    from src.models import create_model
    from src.losses import get_loss
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Small model
    model, _ = create_model("efficientnet_b0", num_classes=3, pretrained=False)
    loss_fn = get_loss("cross_entropy", num_classes=3)
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        use_amp=False,  # CPU doesn't support AMP
        use_ema=False,
        num_workers=0,
    )

    trainer = ClassificationTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device="cpu",
        num_classes=3,
    )

    # Dummy data
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 3, (4,))

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=2)

    # Single epoch
    train_loss, train_acc = trainer._train_epoch(train_loader, 0)

    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    return f"loss={train_loss:.4f}, acc={train_acc:.1f}%"


def test_handler_import():
    """Test handler.py imports without error."""
    import handler
    return True


def test_handler_validation():
    """Test handler input validation."""
    from handler import handler

    # Empty input
    result = handler({"input": {}})
    assert result["status"] == "FAILED"
    assert "error" in result

    # Missing class names
    result = handler({"input": {
        "dataset": {"train_urls": [{"url": "http://x", "label": 0}]}
    }})
    assert result["status"] == "FAILED"

    return True


def test_model_with_embedding():
    """Test ModelWithEmbedding wrapper."""
    from src.models import ModelWithEmbedding, create_model
    import torch

    base_model, _ = create_model("efficientnet_b0", num_classes=10, pretrained=False)

    model = ModelWithEmbedding(
        base_model=base_model,
        embedding_dim=256,
        num_classes=10,
    )

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert "logits" in out
    assert "embeddings" in out
    assert out["logits"].shape == (2, 10)
    assert out["embeddings"].shape == (2, 256)
    return True


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    from src.models import create_model
    import torch
    import tempfile
    import os

    model, _ = create_model("efficientnet_b0", num_classes=5, pretrained=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "best_acc": 0.85,
        }, checkpoint_path)

        # Load
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        assert checkpoint["epoch"] == 5
        assert checkpoint["best_acc"] == 0.85

    return True


def test_transform_api_detection():
    """Test that transform API detection works correctly."""
    from src.augmentations import get_train_transforms, get_val_transforms
    import numpy as np
    from PIL import Image

    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    img_np = np.array(img)

    # Test with albumentations
    transform = get_train_transforms(img_size=224, preset="medium", use_albumentations=True)
    is_albu = hasattr(transform, '__module__') and 'albumentations' in transform.__module__

    if is_albu:
        result = transform(image=img_np)
        tensor = result["image"]
        assert tensor.shape == (3, 224, 224), "Albumentations transform failed"

    # Test with torchvision
    transform_tv = get_train_transforms(img_size=224, preset="medium", use_albumentations=False)
    is_albu_tv = hasattr(transform_tv, '__module__') and 'albumentations' in transform_tv.__module__

    assert not is_albu_tv, "Torchvision transform incorrectly detected as albumentations"
    tensor_tv = transform_tv(img)
    assert tensor_tv.shape == (3, 224, 224), "Torchvision transform failed"

    return "Both backends work correctly"


def test_dataloader_with_transforms():
    """Test DataLoader with transforms works end-to-end."""
    from src.augmentations import get_train_transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torch
    import numpy as np

    class DummyDataset(Dataset):
        def __init__(self, transform, size=10):
            self.transform = transform
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Create random image
            img = Image.new("RGB", (256, 256), color=(idx*20, idx*10, idx*5))

            # Apply transform (same logic as URLImageDataset)
            is_albu = (
                hasattr(self.transform, '__module__') and
                'albumentations' in self.transform.__module__
            )

            if is_albu:
                result = self.transform(image=np.array(img))
                tensor = result["image"]
            else:
                tensor = self.transform(img)

            return tensor, idx % 3

    transform = get_train_transforms(img_size=224, preset="light")
    dataset = DummyDataset(transform, size=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch_x, batch_y in loader:
        assert batch_x.shape == (4, 3, 224, 224), f"Wrong batch shape: {batch_x.shape}"
        assert batch_y.shape == (4,)
        break

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    runner = TestRunner()

    print("\n" + "="*60)
    print("  CLS TRAINING PIPELINE - TEST SUITE")
    print("  Phase 1: Local CPU Tests")
    print("="*60)

    # Section 1: Imports
    runner.section("1. IMPORTS")
    runner.test("PyTorch import", test_import_torch)
    runner.test("timm import", test_import_timm)
    runner.test("albumentations import", test_import_albumentations)
    runner.test("src modules import", test_import_src_modules)
    runner.test("handler import", test_handler_import)

    # Section 2: Models
    runner.section("2. MODELS")
    runner.test("Model registry", test_model_registry)
    runner.test("Create models (efficientnet, vit, resnet)", test_create_each_model)
    runner.test("ModelWithEmbedding wrapper", test_model_with_embedding)

    # Section 3: Losses
    runner.section("3. LOSSES")
    runner.test("All loss functions", test_loss_functions)

    # Section 4: Augmentations
    runner.section("4. AUGMENTATIONS")
    runner.test("All augmentation presets", test_augmentation_presets)
    runner.test("Validation transforms", test_val_transforms)
    runner.test("Transform API detection", test_transform_api_detection)
    runner.test("MixUp/CutMix", test_mixup_cutmix)

    # Section 5: Data
    runner.section("5. DATA PIPELINE")
    runner.test("DataLoader with transforms", test_dataloader_with_transforms)

    # Section 6: Training
    runner.section("6. TRAINING")
    runner.test("TrainingConfig creation", test_training_config)
    runner.test("Class weight computation", test_class_weights)
    runner.test("Trainer creation", test_trainer_creation)
    runner.test("Single training step (CPU)", test_single_training_step)
    runner.test("Checkpoint save/load", test_checkpoint_save_load)

    # Section 7: Handler
    runner.section("7. HANDLER")
    runner.test("Handler input validation", test_handler_validation)

    # Summary
    success = runner.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
