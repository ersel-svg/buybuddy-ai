#!/usr/bin/env python3
"""
Local Worker Simulator - Run the training worker locally on pod.

This script simulates the RunPod worker by directly using the trainer components,
allowing you to see full output and test new features like Circle Loss and TTA.

Usage on pod:
    cd /workspace/training
    python run_worker_local.py

Environment variables (from .env or export):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import sys
import os
import json
import time

# Add src to path
sys.path.insert(0, '/workspace/training/src')

# Load environment before other imports
from pathlib import Path

def load_env():
    """Load environment variables from .env file."""
    env_paths = [
        Path("/workspace/.env"),
        Path("/workspace/training/.env"),
        Path.cwd() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            for line in env_path.read_text().strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key:
                        os.environ[key] = value
            return True
    return False

# Load env first
if not load_env():
    print("WARNING: No .env file found")

# Now import training components directly (not handler.py to avoid runpod.start)
import torch
from supabase import create_client

from trainer import UnifiedTrainer
from dataset import ProductDataset
from evaluator import ModelEvaluator, DomainAwareEvaluator
from bb_models import is_model_supported, list_available_models


def fetch_sample_products(client, limit=30):
    """Fetch sample products with frames for testing (legacy format)."""
    print(f"\nFetching {limit} products with frames (legacy format)...")

    # Get products that have frames_path and frame_count (legacy format)
    response = client.table("products").select(
        "id, barcode, brand_name, category, product_name, frames_path, frame_count"
    ).gt("frame_count", 0).limit(limit).execute()

    products = response.data

    if not products:
        print("ERROR: No products found in database")
        return None

    # Filter products with valid frames_path
    products_with_frames = [p for p in products if p.get("frames_path")]

    print(f"Found {len(products_with_frames)} products with frames")

    total_frames = sum(p.get("frame_count", 1) for p in products_with_frames)
    print(f"Total frames: {total_frames}")

    return products_with_frames


def run_local_training(num_products=30, epochs=3, circle_weight=0.3, model_type="dinov2-base"):
    """Run training locally with full output visibility."""

    print("=" * 70)
    print("LOCAL WORKER SIMULATOR")
    print("=" * 70)

    # Check environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        print("Please ensure .env file exists with these variables")
        return None

    print(f"Supabase URL: {supabase_url[:50]}...")
    print(f"Model Type: {model_type}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if not is_model_supported(model_type):
        print(f"ERROR: Unsupported model type: {model_type}")
        print(f"Available: {list_available_models()}")
        return None

    # Create Supabase client
    client = create_client(supabase_url, supabase_key)

    # Fetch sample products (legacy format with frames_path)
    products = fetch_sample_products(client, limit=num_products)

    if not products:
        return None

    # Split products into train/val/test
    n = len(products)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_products = products[:train_end]
    val_products = products[train_end:val_end]
    test_products = products[val_end:]

    train_frames = sum(p.get("frame_count", 1) for p in train_products)
    val_frames = sum(p.get("frame_count", 1) for p in val_products)
    test_frames = sum(p.get("frame_count", 1) for p in test_products)

    print(f"\nData Split (legacy format with frames_path):")
    print(f"  Train: {len(train_products)} products, ~{train_frames} frames")
    print(f"  Val: {len(val_products)} products, ~{val_frames} frames")
    print(f"  Test: {len(test_products)} products, ~{test_frames} frames")

    if len(train_products) < 5:
        print("ERROR: Not enough training products")
        return None

    # Training config with SOTA features
    config = {
        "epochs": epochs,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "use_arcface": True,
        "use_gem_pooling": True,
        "use_llrd": True,
        "mixed_precision": True,
        "augmentation_strength": "moderate",
        "warmup_epochs": 1,
        "save_every_n_epochs": 1,
        "eval_every_n_epochs": 1,
        "disable_checkpointing": True,  # Disable for local testing (avoids disk issues)

        # SOTA config with Circle Loss
        "sota_config": {
            "enabled": True,
            "use_combined_loss": True,
            "use_pk_sampling": True,
            "use_early_stopping": True,
            "early_stopping_patience": 5,

            # Loss configuration with Circle Loss (CVPR 2020)
            "loss": {
                "arcface_weight": 1.0,
                "triplet_weight": 0.5,
                "circle_weight": circle_weight,
                "circle_margin": 0.25,
                "circle_gamma": 256.0,
                "domain_weight": 0.1,
            },

            # P-K Sampling config
            "pk_sampling": {
                "p": 4,  # products per batch
                "k": 4,  # images per product
            },
        },
    }

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  SOTA Enabled: True")
    print(f"  Combined Loss: True")
    print(f"  Circle Loss Weight: {circle_weight}")
    print(f"  Triplet Loss Weight: {config['sota_config']['loss']['triplet_weight']}")
    print(f"  ArcFace Weight: {config['sota_config']['loss']['arcface_weight']}")

    # Create datasets
    print("\n" + "=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    # Create datasets using legacy format (no training_images, uses frames_path)
    train_dataset = ProductDataset(
        products=train_products,
        model_type=model_type,
        augmentation_strength=config.get("augmentation_strength", "moderate"),
        is_training=True,
    )

    val_dataset = ProductDataset(
        products=val_products,
        model_type=model_type,
        augmentation_strength="none",
        is_training=False,
    )

    # For test set, use is_training=True to get ALL frames (not just first)
    # This gives meaningful recall metrics (finding other images of same product)
    test_dataset = ProductDataset(
        products=test_products,
        model_type=model_type,
        augmentation_strength="none",  # No augmentation for eval
        is_training=True,  # But include all frames for proper recall testing
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Initialize trainer
    print("\n" + "=" * 70)
    print("INITIALIZING TRAINER")
    print("=" * 70)

    training_run_id = f"local-test-{int(time.time())}"

    trainer = UnifiedTrainer(
        model_type=model_type,
        config=config,
        checkpoint_url=None,
        job_id=training_run_id,
    )

    # Progress callback
    def progress_callback(epoch, batch, total_batches, metrics):
        if batch % 10 == 0 or batch == total_batches:
            loss_str = f"loss={metrics.get('loss', 0):.4f}"
            if 'circle_loss' in metrics:
                loss_str += f", circle={metrics['circle_loss']:.4f}"
            if 'triplet_loss' in metrics:
                loss_str += f", triplet={metrics['triplet_loss']:.4f}"
            print(f"  Batch {batch}/{total_batches}: {loss_str}")

    # Epoch callback
    def epoch_callback(epoch, train_metrics, val_metrics, epoch_time, is_best, curriculum_phase, learning_rate):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1} Summary (took {epoch_time:.1f}s)")
        print(f"{'=' * 50}")

        print(f"\nTrain Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        print(f"\nVal Metrics:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        print(f"\nLearning Rate: {learning_rate:.2e}")
        print(f"Is Best: {is_best}")
        if curriculum_phase:
            print(f"Curriculum Phase: {curriculum_phase}")

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    start_time = time.time()

    training_result = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        progress_callback=progress_callback,
        epoch_callback=epoch_callback,
    )

    training_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nTotal Training Time: {training_time:.1f}s")
    print(f"Epochs Trained: {training_result['epochs_trained']}")
    print(f"Best Epoch: {training_result['best_epoch']}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)

    evaluator = ModelEvaluator(
        model=trainer.model,
        model_type=model_type,
        device=trainer.device,
        use_tta=False,  # Disable TTA for faster evaluation
    )

    test_metrics = evaluator.evaluate(test_dataset)

    print(f"\nTest Metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # TTA Comparison
    print("\n" + "=" * 70)
    print("TTA COMPARISON (Test-Time Augmentation)")
    print("=" * 70)

    # Test with TTA enabled
    evaluator_tta = ModelEvaluator(
        model=trainer.model,
        model_type=model_type,
        device=trainer.device,
        use_tta=True,
        tta_mode="light",
    )

    print("\nEvaluating WITH TTA (light mode - 2 views)...")
    test_metrics_tta = evaluator_tta.evaluate(test_dataset)

    print(f"\nTest Metrics WITH TTA:")
    for key, value in test_metrics_tta.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Compare TTA vs no-TTA
    print(f"\nTTA Improvement:")
    for key in ["recall@1", "recall@5", "recall@10", "mAP"]:
        no_tta_val = test_metrics.get(key, 0)
        tta_val = test_metrics_tta.get(key, 0)
        improvement = (tta_val - no_tta_val) * 100
        sign = "+" if improvement >= 0 else ""
        print(f"  {key}: {no_tta_val:.4f} → {tta_val:.4f} ({sign}{improvement:.2f}%)")

    # Cross-domain evaluation
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN EVALUATION")
    print("=" * 70)

    domain_evaluator = DomainAwareEvaluator(
        model=trainer.model,
        device=trainer.device,
    )

    cross_domain_metrics = domain_evaluator.evaluate_cross_domain(test_dataset)

    print(f"\nCross-Domain Metrics:")
    for key, value in cross_domain_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key in ("per_category", "hard_examples", "confused_pairs"):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTraining Run ID: {training_run_id}")
    print(f"Model: {model_type}")
    print(f"Products: {len(products)}")
    print(f"Epochs: {training_result['epochs_trained']}")
    print(f"Best Epoch: {training_result['best_epoch']}")
    print(f"Training Time: {training_time:.1f}s")

    print(f"\nFinal Metrics:")
    final_metrics = training_result.get('final_metrics', {})
    print(f"  Train Loss: {final_metrics.get('train_loss', 'N/A')}")
    print(f"  Val Loss: {final_metrics.get('val_loss', 'N/A')}")
    if 'val_recall@1' in final_metrics:
        print(f"  Val Recall@1: {final_metrics.get('val_recall@1', 'N/A')}")

    print(f"\nTest Results (without TTA):")
    print(f"  Recall@1: {test_metrics.get('recall@1', 'N/A')}")
    print(f"  Recall@5: {test_metrics.get('recall@5', 'N/A')}")

    print(f"\nTest Results (with TTA):")
    print(f"  Recall@1: {test_metrics_tta.get('recall@1', 'N/A')}")
    print(f"  Recall@5: {test_metrics_tta.get('recall@5', 'N/A')}")

    print(f"\nSOTA Features Used:")
    print(f"  Circle Loss: weight={circle_weight}")
    print(f"  Combined Loss: ArcFace + Triplet + Circle + Domain")
    print(f"  P-K Sampling: P=4, K=4")
    print(f"  TTA: light mode (2 views)")

    result = {
        "status": "completed",
        "training_run_id": training_run_id,
        "model_type": model_type,
        "epochs_trained": training_result["epochs_trained"],
        "best_epoch": training_result["best_epoch"],
        "training_time": training_time,
        "training_metrics": training_result.get("final_metrics", {}),
        "test_metrics": test_metrics,
        "test_metrics_tta": test_metrics_tta,
        "cross_domain_metrics": {
            k: v for k, v in cross_domain_metrics.items()
            if k not in ("per_category", "hard_examples", "confused_pairs")
        },
    }

    return result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run training worker locally")
    parser.add_argument("--products", type=int, default=30, help="Number of products to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--circle-weight", type=float, default=0.3, help="Circle loss weight (0 to disable)")
    parser.add_argument("--no-circle", action="store_true", help="Disable Circle Loss")
    parser.add_argument("--model", type=str, default="dinov2-base", help="Model type")

    args = parser.parse_args()

    circle_weight = 0.0 if args.no_circle else args.circle_weight

    result = run_local_training(
        num_products=args.products,
        epochs=args.epochs,
        circle_weight=circle_weight,
        model_type=args.model,
    )

    if result and result.get("status") == "completed":
        print("\n" + "=" * 70)
        print("SUCCESS! Training completed.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("FAILED! Check error messages above.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
