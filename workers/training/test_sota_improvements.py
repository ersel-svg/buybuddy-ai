#!/usr/bin/env python3
"""
Test script for SOTA improvements:
1. Vectorized OnlineHardTripletLoss
2. CircleLoss (CVPR 2020)
3. CombinedProductLoss with Circle Loss support
4. TTA (Test-Time Augmentation) in evaluator
"""

import sys
sys.path.insert(0, '/workspace/training/src')

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

def test_online_hard_triplet_loss():
    """Test vectorized OnlineHardTripletLoss."""
    print("\n" + "=" * 60)
    print("TEST: OnlineHardTripletLoss (Vectorized)")
    print("=" * 60)

    from losses import OnlineHardTripletLoss

    # Create test data
    batch_size = 32
    embedding_dim = 512
    num_classes = 8

    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test different mining types
    for mining_type in ["hard", "semi_hard", "all"]:
        loss_fn = OnlineHardTripletLoss(margin=0.3, mining_type=mining_type)
        loss = loss_fn(embeddings, labels)

        print(f"  Mining type '{mining_type}': loss = {loss.item():.4f}")
        assert not torch.isnan(loss), f"NaN loss for mining_type={mining_type}"
        assert not torch.isinf(loss), f"Inf loss for mining_type={mining_type}"

    print("  ✓ All mining types work correctly")
    return True


def test_circle_loss():
    """Test CircleLoss implementation."""
    print("\n" + "=" * 60)
    print("TEST: CircleLoss (CVPR 2020)")
    print("=" * 60)

    from losses import CircleLoss

    # Create test data
    batch_size = 32
    embedding_dim = 512
    num_classes = 8

    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test Circle Loss
    loss_fn = CircleLoss(margin=0.25, gamma=256)
    loss = loss_fn(embeddings, labels)

    print(f"  Circle Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "NaN loss from CircleLoss"
    assert not torch.isinf(loss), "Inf loss from CircleLoss"
    assert loss.item() >= 0, "CircleLoss should be non-negative"

    # Test gradient flow
    embeddings.requires_grad = True
    loss = loss_fn(embeddings, labels)
    loss.backward()

    assert embeddings.grad is not None, "No gradient computed"
    print(f"  Gradient norm: {embeddings.grad.norm().item():.4f}")
    print("  ✓ CircleLoss works correctly with gradient flow")
    return True


def test_combined_product_loss_with_circle():
    """Test CombinedProductLoss with Circle Loss enabled."""
    print("\n" + "=" * 60)
    print("TEST: CombinedProductLoss with Circle Loss")
    print("=" * 60)

    from losses import CombinedProductLoss

    # Create test data
    batch_size = 32
    embedding_dim = 512
    num_classes = 100

    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test with Circle Loss enabled
    loss_fn = CombinedProductLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_weight=1.0,
        triplet_weight=0.5,
        circle_weight=0.3,  # Enable Circle Loss
        circle_margin=0.25,
        circle_gamma=256.0,
        domain_weight=0.0,
    )

    losses = loss_fn(embeddings, labels)

    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  ArcFace loss: {losses['arcface'].item():.4f}")
    print(f"  Triplet loss: {losses['triplet'].item():.4f}")
    print(f"  Circle loss: {losses['circle'].item():.4f}")
    print(f"  Domain loss: {losses['domain'].item():.4f}")

    assert 'circle' in losses, "Circle loss not in output"
    assert losses['circle'].item() >= 0, "Circle loss should be non-negative"

    # Verify total loss includes circle
    expected_total = (
        1.0 * losses['arcface'] +
        0.5 * losses['triplet'] +
        0.3 * losses['circle'] +
        0.0 * losses['domain']
    )
    assert torch.isclose(losses['total'], expected_total, atol=1e-5), \
        f"Total loss mismatch: {losses['total'].item()} vs {expected_total.item()}"

    print("  ✓ CombinedProductLoss correctly includes Circle Loss")
    return True


def test_combined_product_loss_circle_disabled():
    """Test CombinedProductLoss with Circle Loss disabled (default)."""
    print("\n" + "=" * 60)
    print("TEST: CombinedProductLoss with Circle Loss disabled")
    print("=" * 60)

    from losses import CombinedProductLoss

    batch_size = 32
    embedding_dim = 512
    num_classes = 100

    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test with Circle Loss disabled (default)
    loss_fn = CombinedProductLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_weight=1.0,
        triplet_weight=0.5,
        # circle_weight defaults to 0.0
        domain_weight=0.0,
    )

    losses = loss_fn(embeddings, labels)

    print(f"  Circle loss value: {losses['circle'].item():.4f}")
    assert losses['circle'].item() == 0.0, "Circle loss should be 0 when disabled"

    print("  ✓ Circle Loss correctly disabled by default")
    return True


def test_tta_transforms():
    """Test TTA transforms."""
    print("\n" + "=" * 60)
    print("TEST: TTA Transforms")
    print("=" * 60)

    from evaluator import get_tta_transforms, get_tta_transforms_light

    # Create test image batch
    batch_size = 4
    image_size = 224
    images = torch.randn(batch_size, 3, image_size, image_size)

    # Test full TTA transforms
    full_transforms = get_tta_transforms(image_size)
    print(f"  Full TTA: {len(full_transforms)} transforms")

    for i, transform in enumerate(full_transforms):
        aug_images = transform(images)
        assert aug_images.shape == images.shape, f"Transform {i} changed shape"

    # Test light TTA transforms
    light_transforms = get_tta_transforms_light()
    print(f"  Light TTA: {len(light_transforms)} transforms")

    for i, transform in enumerate(light_transforms):
        aug_images = transform(images)
        assert aug_images.shape == images.shape, f"Light transform {i} changed shape"

    print("  ✓ All TTA transforms preserve tensor shape")
    return True


def test_evaluator_tta():
    """Test ModelEvaluator TTA functionality."""
    print("\n" + "=" * 60)
    print("TEST: ModelEvaluator TTA")
    print("=" * 60)

    from evaluator import ModelEvaluator

    # Create a simple mock model
    class MockModel(nn.Module):
        def __init__(self, embedding_dim=512):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, embedding_dim)

        def forward(self, x):
            batch_size = x.shape[0]
            flat = x.view(batch_size, -1)
            embeddings = self.fc(flat)
            return {"embeddings": embeddings}

        def get_embedding(self, x):
            return self.forward(x)["embeddings"]

    model = MockModel()

    # Test evaluator with TTA enabled
    evaluator = ModelEvaluator(
        model=model,
        model_type="mock",
        use_tta=True,
        tta_mode="light",
    )

    assert evaluator.use_tta is True
    assert evaluator.tta_transforms is not None
    assert len(evaluator.tta_transforms) == 2  # light mode has 2 transforms

    # Test set_tta method
    evaluator.set_tta(enabled=False)
    assert evaluator.use_tta is False
    assert evaluator.tta_transforms is None

    evaluator.set_tta(enabled=True, mode="full")
    assert evaluator.use_tta is True
    assert len(evaluator.tta_transforms) == 5  # full mode has 5 transforms

    print("  ✓ ModelEvaluator TTA configuration works correctly")
    return True


def run_all_tests():
    """Run all SOTA improvement tests."""
    print("\n" + "=" * 70)
    print("SOTA IMPROVEMENTS TEST SUITE")
    print("=" * 70)

    results = []

    tests = [
        ("OnlineHardTripletLoss (Vectorized)", test_online_hard_triplet_loss),
        ("CircleLoss (CVPR 2020)", test_circle_loss),
        ("CombinedProductLoss + Circle", test_combined_product_loss_with_circle),
        ("CombinedProductLoss (Circle disabled)", test_combined_product_loss_circle_disabled),
        ("TTA Transforms", test_tta_transforms),
        ("ModelEvaluator TTA", test_evaluator_tta),
    ]

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
