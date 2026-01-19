#!/usr/bin/env python3
"""
Comprehensive QA Test for Training System - P0 Items
Tests all critical functionality with real data.
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime

# Add src to path
sys.path.insert(0, "/workspace/training/src")

# Results tracking
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {},
    "summary": {"passed": 0, "failed": 0, "skipped": 0}
}

def log_test(name, status, details=""):
    """Log a test result."""
    emoji = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⏭️")
    print(f"\n{emoji} {name}: {status}")
    if details:
        print(f"   {details}")
    results["tests"][name] = {"status": status, "details": details}
    if status == "PASS":
        results["summary"]["passed"] += 1
    elif status == "FAIL":
        results["summary"]["failed"] += 1
    else:
        results["summary"]["skipped"] += 1


def test_imports():
    """Test all required imports."""
    print("\n" + "="*60)
    print("TEST: Module Imports")
    print("="*60)

    try:
        import torch
        log_test("torch import", "PASS", f"Version: {torch.__version__}")
    except Exception as e:
        log_test("torch import", "FAIL", str(e))
        return False

    try:
        from dataset import ProductDataset
        log_test("ProductDataset import", "PASS")
    except Exception as e:
        log_test("ProductDataset import", "FAIL", str(e))
        return False

    try:
        from trainer import UnifiedTrainer, EmbeddingModel
        log_test("UnifiedTrainer import", "PASS")
    except Exception as e:
        log_test("UnifiedTrainer import", "FAIL", str(e))
        return False

    try:
        from evaluator import ModelEvaluator
        log_test("ModelEvaluator import", "PASS")
    except Exception as e:
        log_test("ModelEvaluator import", "FAIL", str(e))
        return False

    try:
        from checkpoint_upload import (
            upload_checkpoint_to_storage,
            save_metrics_history,
            update_training_progress,
            MAX_UPLOAD_SIZE
        )
        log_test("checkpoint_upload import", "PASS", f"MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB")
    except Exception as e:
        log_test("checkpoint_upload import", "FAIL", str(e))
        return False

    try:
        from supabase import create_client
        log_test("supabase import", "PASS")
    except Exception as e:
        log_test("supabase import", "FAIL", str(e))
        return False

    return True


def test_supabase_connection():
    """Test Supabase connection and data fetch."""
    print("\n" + "="*60)
    print("TEST: Supabase Connection")
    print("="*60)

    try:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not url or not key:
            log_test("Supabase env vars", "FAIL", "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")
            return None

        log_test("Supabase env vars", "PASS")

        client = create_client(url, key)

        # Test query
        response = client.table("products").select("id").limit(1).execute()
        log_test("Supabase connection", "PASS", f"Connected successfully")

        return client

    except Exception as e:
        log_test("Supabase connection", "FAIL", str(e))
        return None


def test_fetch_products(client):
    """Fetch products with frames_path (legacy format)."""
    print("\n" + "="*60)
    print("TEST: Fetch Products")
    print("="*60)

    try:
        # Fetch products with frames_path (legacy format)
        response = client.table("products").select(
            "id, barcode, brand_name, frames_path, frame_count, category"
        ).not_.is_("frames_path", "null").gt("frame_count", 0).limit(20).execute()

        products = response.data

        if not products:
            log_test("Fetch products", "FAIL", "No products with frames_path found")
            return None, None

        log_test("Fetch products", "PASS", f"Found {len(products)} products with frames")

        # Show stats
        total_frames = sum(p.get("frame_count", 0) for p in products)
        log_test("Product stats", "PASS", f"{len(products)} products, {total_frames} total frames")

        # We'll use legacy format (training_images=None), so return None for training_images
        return products, None

    except Exception as e:
        log_test("Fetch products", "FAIL", str(e))
        return None, None


def test_dataset_creation(products, training_images):
    """Test ProductDataset creation."""
    print("\n" + "="*60)
    print("TEST: Dataset Creation")
    print("="*60)

    try:
        from dataset import ProductDataset

        # Create training dataset
        train_dataset = ProductDataset(
            products=products,
            model_type="dinov2-base",
            augmentation_strength="moderate",
            is_training=True,
            training_images=training_images,
        )

        log_test("Create training dataset", "PASS", f"{len(train_dataset)} samples")

        # Test __getitem__ - returns dict
        sample = train_dataset[0]

        if not isinstance(sample, dict):
            log_test("Dataset __getitem__", "FAIL", f"Expected dict, got {type(sample)}")
            return None

        required_keys = ["image", "label", "product_id", "domain"]
        missing_keys = [k for k in required_keys if k not in sample]

        if missing_keys:
            log_test("Dataset __getitem__", "FAIL", f"Missing keys: {missing_keys}")
            return None

        log_test("Dataset __getitem__", "PASS", f"Keys: {list(sample.keys())}")

        # Test image tensor
        image_tensor = sample["image"]
        log_test("Image tensor shape", "PASS", f"{image_tensor.shape}")

        # Create val dataset
        val_dataset = ProductDataset(
            products=products,
            model_type="dinov2-base",
            is_training=False,
            training_images=training_images,
        )

        log_test("Create val dataset", "PASS", f"{len(val_dataset)} samples")

        return train_dataset

    except Exception as e:
        import traceback
        log_test("Dataset creation", "FAIL", str(e))
        traceback.print_exc()
        return None


def test_dataloader(dataset):
    """Test DataLoader creation and iteration."""
    print("\n" + "="*60)
    print("TEST: DataLoader")
    print("="*60)

    try:
        import torch
        from torch.utils.data import DataLoader

        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # No multiprocessing for test
        )

        log_test("Create DataLoader", "PASS", f"Batch size: 4, Total batches: {len(loader)}")

        # Get one batch
        batch = next(iter(loader))

        # Check batch structure (dict with batched tensors)
        if not isinstance(batch, dict):
            log_test("Batch structure", "FAIL", f"Expected dict, got {type(batch)}")
            return False

        log_test("Batch structure", "PASS", f"Keys: {list(batch.keys())}")

        # Check shapes
        images = batch["image"]
        labels = batch["label"]
        domains = batch["domain"]

        log_test("Batch shapes", "PASS",
                 f"images: {images.shape}, labels: {labels.shape}, domains: {domains.shape}")

        return True

    except Exception as e:
        import traceback
        log_test("DataLoader", "FAIL", str(e))
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation."""
    print("\n" + "="*60)
    print("TEST: Model Creation")
    print("="*60)

    try:
        import torch
        from trainer import EmbeddingModel

        # Create model
        model = EmbeddingModel(
            model_type="dinov2-base",
            embedding_dim=512,
            num_classes=20,
            use_arcface=True,
        )

        log_test("Create EmbeddingModel", "PASS")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        log_test("Model parameters", "PASS",
                 f"Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        dummy_labels = torch.tensor([0, 1]).to(device)

        with torch.no_grad():
            output = model(dummy_input, dummy_labels)

        if isinstance(output, dict):
            log_test("Model forward pass", "PASS", f"Output keys: {list(output.keys())}")
            if "embeddings" in output:
                log_test("Embedding shape", "PASS", f"{output['embeddings'].shape}")
        else:
            log_test("Model forward pass", "PASS", f"Output shape: {output.shape}")

        return model

    except Exception as e:
        import traceback
        log_test("Model creation", "FAIL", str(e))
        traceback.print_exc()
        return None


def test_trainer_creation(products, training_images, train_dataset, val_dataset):
    """Test UnifiedTrainer creation."""
    print("\n" + "="*60)
    print("TEST: Trainer Creation")
    print("="*60)

    try:
        from trainer import UnifiedTrainer

        config = {
            "model_type": "dinov2-base",
            "embedding_dim": 512,
            "batch_size": 4,
            "epochs": 2,  # Just 2 epochs for testing
            "learning_rate": 1e-4,
            "loss_type": "arcface",
            "warmup_epochs": 0,
        }

        trainer = UnifiedTrainer(
            model_type="dinov2-base",
            config=config,
        )

        log_test("Create UnifiedTrainer", "PASS")
        log_test("Trainer device", "PASS", f"{trainer.device}")

        # Store datasets for later use
        trainer.train_dataset = train_dataset
        trainer.val_dataset = val_dataset

        return trainer

    except Exception as e:
        import traceback
        log_test("Trainer creation", "FAIL", str(e))
        traceback.print_exc()
        return None


def test_training_epoch(trainer, train_dataset, val_dataset):
    """Test running one training epoch using train method."""
    print("\n" + "="*60)
    print("TEST: Training Epoch")
    print("="*60)

    try:
        # Run training with 1 epoch
        print("Running training for 1 epoch...")

        # Temporarily set epochs to 1
        original_epochs = trainer.config.get("epochs", 2)
        trainer.config["epochs"] = 1

        # Run train method - returns a single dict
        results = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        # Restore epochs
        trainer.config["epochs"] = original_epochs

        if results:
            epochs_trained = results.get("epochs_trained", 0)
            best_loss = results.get("best_val_loss", "N/A")
            best_r1 = results.get("best_recall_at_1", 0)
            final_metrics = results.get("final_metrics", {})

            log_test("Training completed", "PASS",
                     f"Epochs: {epochs_trained}, Best Loss: {best_loss:.4f}, Best R@1: {best_r1:.2%}")

            train_loss = final_metrics.get("train_loss", "N/A")
            val_loss = final_metrics.get("val_loss", "N/A")
            val_r1 = final_metrics.get("val_recall_at_1", 0)

            log_test("Final metrics", "PASS",
                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R@1: {val_r1:.2%}")

            return results, final_metrics
        else:
            log_test("Training", "FAIL", "No results returned")
            return None, None

    except Exception as e:
        import traceback
        log_test("Training epoch", "FAIL", str(e))
        traceback.print_exc()
        return None, None


def test_checkpoint_save_load(trainer):
    """Test checkpoint save and load."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Save/Load")
    print("="*60)

    try:
        import torch

        # Check if checkpoint_manager was initialized during training
        if trainer.checkpoint_manager is None:
            log_test("Checkpoint manager", "SKIP", "Not initialized (training may have failed)")
            return None

        # Save checkpoint directly using torch
        checkpoint_path = "/workspace/test_checkpoint.pth"

        checkpoint_data = {
            "model_state_dict": trainer.model.state_dict(),
            "epoch": 0,
            "val_loss": 0.5,
            "val_recall_at_1": 0.8,
        }

        torch.save(checkpoint_data, checkpoint_path)

        # Check file size
        file_size = os.path.getsize(checkpoint_path)
        log_test("Save checkpoint", "PASS", f"Size: {file_size / 1024 / 1024:.1f}MB")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        required_keys = ["model_state_dict", "epoch"]
        missing = [k for k in required_keys if k not in checkpoint]

        if missing:
            log_test("Checkpoint format", "FAIL", f"Missing: {missing}")
        else:
            log_test("Checkpoint format", "PASS", f"Keys: {list(checkpoint.keys())}")

        return checkpoint_path

    except Exception as e:
        import traceback
        log_test("Checkpoint save/load", "FAIL", str(e))
        traceback.print_exc()
        return None


def test_checkpoint_upload(client, checkpoint_path):
    """Test checkpoint upload with FP16 conversion."""
    print("\n" + "="*60)
    print("TEST: Checkpoint Upload (FP16)")
    print("="*60)

    try:
        from checkpoint_upload import upload_checkpoint_to_storage, MAX_UPLOAD_SIZE

        # Use a test training run ID
        test_run_id = "qa-test-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        print(f"Uploading checkpoint (MAX_UPLOAD_SIZE: {MAX_UPLOAD_SIZE / 1024 / 1024:.0f}MB)...")

        url = upload_checkpoint_to_storage(
            client=client,
            checkpoint_path=checkpoint_path,
            training_run_id=test_run_id,
            epoch=0,
            is_best=True,
        )

        if url:
            log_test("Checkpoint upload", "PASS", f"URL: {url[:80]}...")
            return url
        else:
            log_test("Checkpoint upload", "FAIL", "Upload returned None")
            return None

    except Exception as e:
        import traceback
        log_test("Checkpoint upload", "FAIL", str(e))
        traceback.print_exc()
        return None


def test_evaluator(model, val_dataset, model_type="dinov2-base"):
    """Test ModelEvaluator."""
    print("\n" + "="*60)
    print("TEST: Evaluator")
    print("="*60)

    try:
        import torch
        from evaluator import ModelEvaluator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        evaluator = ModelEvaluator(
            model=model,
            model_type=model_type,
            device=device,
            batch_size=4,
        )

        log_test("Create evaluator", "PASS")

        # Create dataloader
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        # Run evaluation
        print("Running evaluation...")
        metrics = evaluator.evaluate(val_loader)

        log_test("Evaluation", "PASS",
                 f"R@1: {metrics.get('recall@1', 0):.2%}, R@5: {metrics.get('recall@5', 0):.2%}")

        return metrics, evaluator

    except Exception as e:
        import traceback
        log_test("Evaluator", "FAIL", str(e))
        traceback.print_exc()
        return None, None


def test_cross_domain_eval(evaluator, val_dataset):
    """Test cross-domain evaluation."""
    print("\n" + "="*60)
    print("TEST: Cross-Domain Evaluation")
    print("="*60)

    try:
        from torch.utils.data import DataLoader

        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        # Check if we have multiple domains
        domains = set()
        for sample in val_dataset.samples[:100]:  # Check first 100
            domains.add(sample.get("domain", "synthetic"))

        if len(domains) < 2:
            log_test("Cross-domain eval", "SKIP", f"Only {len(domains)} domain(s) present: {domains}")
            return None

        print(f"Domains found: {domains}")

        # Run cross-domain evaluation
        cross_metrics = evaluator.evaluate_cross_domain(val_loader)

        if cross_metrics:
            log_test("Cross-domain eval", "PASS", f"Domains evaluated: {list(cross_metrics.keys())}")
            for domain, m in cross_metrics.items():
                print(f"   {domain}: R@1={m.get('recall@1', 0):.2%}")
        else:
            log_test("Cross-domain eval", "FAIL", "No cross-domain metrics returned")

        return cross_metrics

    except Exception as e:
        import traceback
        log_test("Cross-domain eval", "FAIL", str(e))
        traceback.print_exc()
        return None


def main():
    """Run all QA tests."""
    print("\n" + "="*80)
    print("TRAINING SYSTEM QA TEST - P0 ITEMS")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Critical import failures. Cannot continue.")
        return

    # Test 2: Supabase connection
    client = test_supabase_connection()
    if not client:
        print("\n❌ Cannot connect to Supabase. Cannot continue.")
        return

    # Test 3: Fetch products
    products, training_images = test_fetch_products(client)
    if not products:
        print("\n❌ Cannot fetch products. Cannot continue.")
        return
    # training_images can be None for legacy format

    # Test 4: Dataset creation
    train_dataset = test_dataset_creation(products, training_images)
    if not train_dataset:
        print("\n❌ Dataset creation failed. Cannot continue.")
        return

    # Create val dataset separately
    from dataset import ProductDataset
    val_dataset = ProductDataset(
        products=products,
        model_type="dinov2-base",
        is_training=False,
        training_images=training_images,
    )

    # Test 5: DataLoader
    if not test_dataloader(train_dataset):
        print("\n⚠️ DataLoader test failed, but continuing...")

    # Test 6: Model creation
    model = test_model_creation()
    if not model:
        print("\n❌ Model creation failed. Cannot continue.")
        return

    # Test 7: Trainer creation
    trainer = test_trainer_creation(products, training_images, train_dataset, val_dataset)
    if not trainer:
        print("\n❌ Trainer creation failed. Cannot continue.")
        return

    # Test 8: Training epoch
    training_results, final_metrics = test_training_epoch(trainer, train_dataset, val_dataset)
    if not training_results:
        print("\n⚠️ Training epoch failed, but continuing...")

    # Test 9: Checkpoint save/load
    checkpoint_path = test_checkpoint_save_load(trainer)
    if not checkpoint_path:
        print("\n⚠️ Checkpoint save/load failed, but continuing...")

    # Test 10: Checkpoint upload
    if checkpoint_path and client:
        checkpoint_url = test_checkpoint_upload(client, checkpoint_path)
        # Clean up
        try:
            os.unlink(checkpoint_path)
        except:
            pass

    # Test 11: Evaluator
    if trainer and val_dataset:
        metrics, evaluator = test_evaluator(trainer.model, val_dataset)

        # Test 12: Cross-domain evaluation
        if metrics and evaluator:
            test_cross_domain_eval(evaluator, val_dataset)

    # Print summary
    print("\n" + "="*80)
    print("QA TEST SUMMARY")
    print("="*80)
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Skipped: {results['summary']['skipped']}")
    print(f"Total: {results['summary']['passed'] + results['summary']['failed'] + results['summary']['skipped']}")

    # List failures
    failures = [name for name, r in results["tests"].items() if r["status"] == "FAIL"]
    if failures:
        print("\n❌ FAILED TESTS:")
        for name in failures:
            print(f"   - {name}: {results['tests'][name]['details']}")

    print("\n" + "="*80)

    # Save results to file
    results_path = "/workspace/qa_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
