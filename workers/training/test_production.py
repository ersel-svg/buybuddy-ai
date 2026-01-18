#!/usr/bin/env python3
"""
Production Readiness Test for Training Worker.

Tests all production features:
1. Database integration (training_runs table)
2. Progress reporting
3. Full training cycle with real data
4. Checkpoint saving
"""

import os
import sys
import json
import time
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment
def load_env():
    """Load environment from .env file."""
    env_paths = ["/workspace/.env", ".env", "../.env"]
    for path in env_paths:
        if os.path.exists(path):
            print(f"Loading environment from: {path}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            break

load_env()

from supabase import create_client


def get_client():
    """Get Supabase client."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def test_create_training_run():
    """TEST 1: Create a training_run in database."""
    print("\n[TEST 1] Creating training_run...")

    client = get_client()
    run_id = str(uuid.uuid4())[:8] + "-test-" + str(uuid.uuid4())[:4]

    # Minimal required fields based on schema
    payload = {
        "id": run_id,
        "status": "pending",
        "model_type": "dinov2-small",
        "data_source": "all_products",
        "split_config": {"train": 0.7, "val": 0.15, "test": 0.15},
        "train_product_ids": [],
        "val_product_ids": [],
        "test_product_ids": [],
        "training_config": {"epochs": 1, "batch_size": 8},
        "total_epochs": 1,
        "progress": 0.0,
        "message": "Test run created",
    }

    try:
        response = client.table("training_runs").insert(payload).execute()
        print(f"  PASSED - ID: {run_id[:8]}")
        return run_id
    except Exception as e:
        print(f"  FAILED - {e}")
        return None


def test_progress_updates(run_id: str):
    """TEST 2: Update progress in database."""
    print("\n[TEST 2] Progress updates...")

    if not run_id:
        print("  SKIPPED - No run_id")
        return False

    client = get_client()

    try:
        # Update to running
        client.table("training_runs").update({
            "status": "running",
            "progress": 0.5,
            "message": "Training in progress",
            "metrics": {"loss": 0.5, "accuracy": 0.8},
        }).eq("id", run_id).execute()

        # Verify
        result = client.table("training_runs").select("*").eq("id", run_id).execute()
        data = result.data[0]

        if data["status"] == "running" and data["progress"] == 0.5:
            print("  PASSED")
            return True
        else:
            print(f"  FAILED - Status: {data['status']}, Progress: {data['progress']}")
            return False
    except Exception as e:
        print(f"  FAILED - {e}")
        return False


def test_full_training_cycle(run_id: str):
    """TEST 3: Full training cycle with real data."""
    print("\n[TEST 3] Full training cycle...")

    if not run_id:
        print("  SKIPPED - No run_id")
        return False

    try:
        from splitter import ProductSplitter
        from dataset import ProductDataset
        from trainer import UnifiedTrainer

        client = get_client()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        # Fetch products
        print("  Fetching products...")
        response = client.table("products").select(
            "id,barcode,brand_name,frames_path,frame_count"
        ).gte("frame_count", 1).limit(30).execute()

        products = response.data
        print(f"  Found {len(products)} products")

        if len(products) < 10:
            print(f"  FAILED - Not enough products: {len(products)}")
            return False

        # Split data - IMPORTANT: Use keyword arguments!
        print("  Splitting data...")
        splitter = ProductSplitter(url, key)
        train, val, test = splitter.split(
            products=products,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        # Create datasets
        print("  Creating datasets...")
        train_dataset = ProductDataset(
            products=train,
            model_type="dinov2-small",
            augmentation_strength="light",
            is_training=True,
        )

        val_dataset = ProductDataset(
            products=val,
            model_type="dinov2-small",
            augmentation_strength="none",
            is_training=False,
        )

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Update training run with product IDs
        client.table("training_runs").update({
            "train_product_ids": [p["id"] for p in train],
            "val_product_ids": [p["id"] for p in val],
            "test_product_ids": [p["id"] for p in test],
            "message": "Starting training...",
        }).eq("id", run_id).execute()

        # Train for 1 epoch
        print("  Training model (1 epoch)...")
        trainer = UnifiedTrainer(
            model_type="dinov2-small",
            config={
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "use_arcface": False,
                "use_gem_pooling": True,
                "use_llrd": False,
                "mixed_precision": True,
                "prefetch_workers": 16,
            },
            job_id=run_id,
        )

        def progress_callback(epoch, batch, total_batches, metrics):
            progress = 0.2 + (batch / total_batches) * 0.7
            client.table("training_runs").update({
                "progress": progress,
                "message": f"Epoch {epoch+1}, Batch {batch}/{total_batches}",
                "metrics": metrics,
            }).eq("id", run_id).execute()

        result = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            progress_callback=progress_callback,
        )

        print(f"  Training completed: {result['epochs_trained']} epochs")

        # Update final status
        client.table("training_runs").update({
            "status": "completed",
            "progress": 1.0,
            "message": "Training completed successfully",
            "metrics": result["final_metrics"],
        }).eq("id", run_id).execute()

        print("  PASSED")
        return True

    except Exception as e:
        import traceback
        print(f"  FAILED - {e}")
        traceback.print_exc()

        # Update error status
        try:
            client = get_client()
            client.table("training_runs").update({
                "status": "failed",
                "message": str(e),
            }).eq("id", run_id).execute()
        except:
            pass

        return False


def test_verify_database(run_id: str):
    """TEST 4: Verify final database state."""
    print("\n[TEST 4] Verify database...")

    if not run_id:
        print("  SKIPPED - No run_id")
        return False

    client = get_client()

    try:
        result = client.table("training_runs").select("*").eq("id", run_id).execute()

        if not result.data:
            print("  FAILED - Run not found")
            return False

        data = result.data[0]

        checks = [
            ("status", data.get("status") in ["completed", "failed"]),
            ("progress", data.get("progress") is not None),
            ("message", data.get("message") is not None),
            ("metrics", data.get("metrics") is not None),
        ]

        all_passed = True
        for name, passed in checks:
            status = "OK" if passed else "FAIL"
            print(f"    {name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("  PASSED")
        else:
            print("  FAILED - Some checks failed")

        return all_passed

    except Exception as e:
        print(f"  FAILED - {e}")
        return False


def cleanup(run_id: str):
    """Clean up test data."""
    if not run_id or "test" not in run_id:
        return

    try:
        client = get_client()
        client.table("training_runs").delete().eq("id", run_id).execute()
        print(f"\nCleaned up test run: {run_id[:8]}")
    except Exception as e:
        print(f"\nFailed to cleanup: {e}")


def main():
    print("=" * 70)
    print("PRODUCTION READINESS TEST - FULL")
    print("=" * 70)

    run_id = None

    try:
        # Test 1: Create training_run
        run_id = test_create_training_run()

        # Test 2: Progress updates
        test_progress_updates(run_id)

        # Test 3: Full training cycle
        test_full_training_cycle(run_id)

        # Test 4: Verify database
        test_verify_database(run_id)

    finally:
        print("\n" + "=" * 70)
        print(f"Test Run ID: {run_id[:8] if run_id else 'N/A'}")
        print("=" * 70)

        # Optional: cleanup
        # cleanup(run_id)


if __name__ == "__main__":
    main()
