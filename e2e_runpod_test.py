#!/usr/bin/env python3
"""
E2E Test Script for RunPod Workers
Comprehensive tests for embedding extraction and training with real data.
"""

import os
import json
import time
import uuid
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configuration - Load from environment variables
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
EMBEDDING_ENDPOINT = os.environ.get("RUNPOD_EMBEDDING_ENDPOINT", "51yvrsxi4xf1ky")
TRAINING_ENDPOINT = os.environ.get("RUNPOD_TRAINING_ENDPOINT", "nttbt8es86xwvp")
HF_TOKEN = os.environ.get("HF_TOKEN")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qvyxpfcwfktxnaeavkxx.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Test images (public URLs)
TEST_IMAGES = [
    {"id": "test-1", "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400", "type": "product"},
    {"id": "test-2", "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400", "type": "product"},
    {"id": "test-3", "url": "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400", "type": "product"},
    {"id": "test-4", "url": "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=400", "type": "product"},
    {"id": "test-5", "url": "https://images.unsplash.com/photo-1560343090-f0409e92791a?w=400", "type": "product"},
    {"id": "test-6", "url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400", "type": "product"},
    {"id": "test-7", "url": "https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400", "type": "product"},
    {"id": "test-8", "url": "https://images.unsplash.com/photo-1585386959984-a4155224a1ad?w=400", "type": "product"},
    {"id": "test-9", "url": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=400", "type": "product"},
    {"id": "test-10", "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400", "type": "product"},
]

# Models to test
EMBEDDING_MODELS = ["dinov2-small", "dinov2-base", "dinov3-small", "dinov3-base"]
TRAINING_MODELS = ["dinov2-small", "dinov3-small"]  # Faster models for training test

# SOTA Configuration presets for testing
SOTA_CONFIGS = {
    "disabled": {
        "enabled": False,
    },
    "basic_combined_loss": {
        "enabled": True,
        "use_combined_loss": True,
        "use_pk_sampling": False,
        "use_curriculum": False,
        "use_domain_adaptation": False,
        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "loss": {
            "arcface_weight": 1.0,
            "triplet_weight": 0.5,
            "domain_weight": 0.0,
            "triplet_margin": 0.3,
        },
    },
    "combined_loss_pk_sampling": {
        "enabled": True,
        "use_combined_loss": True,
        "use_pk_sampling": True,
        "use_curriculum": False,
        "use_domain_adaptation": False,
        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "loss": {
            "arcface_weight": 1.0,
            "triplet_weight": 0.5,
            "domain_weight": 0.0,
            "triplet_margin": 0.3,
        },
        "sampling": {
            "products_per_batch": 8,
            "samples_per_product": 4,
        },
    },
    "full_sota": {
        "enabled": True,
        "use_combined_loss": True,
        "use_pk_sampling": True,
        "use_curriculum": True,
        "use_domain_adaptation": True,
        "use_early_stopping": True,
        "early_stopping_patience": 3,
        "loss": {
            "arcface_weight": 1.0,
            "triplet_weight": 0.5,
            "domain_weight": 0.3,
            "triplet_margin": 0.3,
        },
        "sampling": {
            "products_per_batch": 8,
            "samples_per_product": 4,
        },
        "curriculum": {
            "warmup_epochs": 1,
            "easy_epochs": 1,
            "hard_epochs": 1,
            "finetune_epochs": 1,
        },
    },
}


def log(msg: str, level: str = "INFO"):
    """Print with timestamp."""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️", "STEP": "▶️"}
    icon = icons.get(level, "")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon} {msg}")


def run_runpod_job_sync(endpoint_id: str, input_data: Dict[str, Any], timeout: int = 300) -> Dict:
    """Submit a job to RunPod and wait for completion (sync)."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json={"input": input_data}, headers=headers)

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}

        result = response.json()
        if result.get("status") == "FAILED":
            return {"error": result.get("error", "Unknown error")}

        return result


def run_runpod_job_async(endpoint_id: str, input_data: Dict[str, Any]) -> Optional[str]:
    """Submit an async job to RunPod and return job ID."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(url, json={"input": input_data}, headers=headers)
        if response.status_code != 200:
            log(f"Failed to submit async job: {response.text}", "ERROR")
            return None
        result = response.json()
        return result.get("id")


def check_job_status(endpoint_id: str, job_id: str) -> Dict:
    """Check status of an async job."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=headers)
        return response.json()


def wait_for_job(endpoint_id: str, job_id: str, timeout: int = 600, poll_interval: int = 10) -> Dict:
    """Wait for an async job to complete."""
    start = time.time()
    last_status = None

    while time.time() - start < timeout:
        result = check_job_status(endpoint_id, job_id)
        status = result.get("status")

        if status != last_status:
            log(f"  Job {job_id[:8]}... status: {status}")
            last_status = status

        if status == "COMPLETED":
            return result
        elif status == "FAILED":
            return {"error": result.get("error", "Job failed"), "result": result}

        time.sleep(poll_interval)

    return {"error": "Timeout waiting for job"}


def get_supabase_headers() -> Dict[str, str]:
    """Get headers for Supabase API calls."""
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


# ============================================================
# EMBEDDING EXTRACTION TESTS
# ============================================================

def test_embedding_single_model(model_type: str, images: List[Dict]) -> Dict:
    """Test embedding extraction for a single model."""
    log(f"Testing model: {model_type}", "STEP")

    input_data = {
        "images": images,
        "model_type": model_type,
        "batch_size": 8,
        "hf_token": HF_TOKEN,
    }

    start = time.time()
    result = run_runpod_job_sync(EMBEDDING_ENDPOINT, input_data, timeout=180)
    elapsed = time.time() - start

    if "error" in result:
        log(f"  FAILED: {result['error'][:100]}", "ERROR")
        return {"model": model_type, "status": "FAILED", "error": result["error"]}

    output = result.get("output", {})
    embeddings = output.get("embeddings", [])
    failed_ids = output.get("failed_ids", [])
    embedding_dim = output.get("embedding_dim", 0)

    log(f"  SUCCESS: {len(embeddings)} embeddings, dim={embedding_dim}, time={elapsed:.1f}s", "SUCCESS")

    return {
        "model": model_type,
        "status": "SUCCESS",
        "extracted": len(embeddings),
        "failed": len(failed_ids),
        "time": elapsed,
        "embedding_dim": embedding_dim,
    }


def test_embedding_all_models() -> bool:
    """Test embedding extraction with all models."""
    log("\n" + "="*70)
    log("TEST: EMBEDDING EXTRACTION - ALL MODELS")
    log("="*70)

    results = []
    for model in EMBEDDING_MODELS:
        result = test_embedding_single_model(model, TEST_IMAGES[:5])
        results.append(result)
        time.sleep(1)

    # Summary
    passed = sum(1 for r in results if r["status"] == "SUCCESS")
    log(f"\nSummary: {passed}/{len(results)} models passed")

    return passed == len(results)


def test_embedding_multiframe() -> bool:
    """Test multi-frame embedding extraction with metadata."""
    log("\n" + "="*70)
    log("TEST: MULTI-FRAME EMBEDDING (product_id, frame_index, is_primary)")
    log("="*70)

    # Simulate multi-frame product images
    product_id = str(uuid.uuid4())
    multiframe_images = [
        {
            "id": f"{product_id}-frame-0",
            "url": TEST_IMAGES[0]["url"],
            "type": "product",
            "domain": "synthetic",
            "product_id": product_id,
            "frame_index": 0,
            "is_primary": True,
            "category": "electronics",
        },
        {
            "id": f"{product_id}-frame-1",
            "url": TEST_IMAGES[1]["url"],
            "type": "product",
            "domain": "synthetic",
            "product_id": product_id,
            "frame_index": 1,
            "is_primary": False,
            "category": "electronics",
        },
        {
            "id": f"{product_id}-frame-2",
            "url": TEST_IMAGES[2]["url"],
            "type": "product",
            "domain": "synthetic",
            "product_id": product_id,
            "frame_index": 2,
            "is_primary": False,
            "category": "electronics",
        },
    ]

    # Also add cutout images
    cutout_images = [
        {
            "id": f"cutout-{i}",
            "url": TEST_IMAGES[i+3]["url"],
            "type": "cutout",
            "domain": "real",
        }
        for i in range(3)
    ]

    all_images = multiframe_images + cutout_images

    input_data = {
        "images": all_images,
        "model_type": "dinov3-base",
        "batch_size": 8,
        "hf_token": HF_TOKEN,
    }

    log(f"Submitting {len(all_images)} images (3 product frames + 3 cutouts)...")

    start = time.time()
    result = run_runpod_job_sync(EMBEDDING_ENDPOINT, input_data, timeout=180)
    elapsed = time.time() - start

    if "error" in result:
        log(f"FAILED: {result['error']}", "ERROR")
        return False

    output = result.get("output", {})
    embeddings = output.get("embeddings", [])

    # Verify metadata is preserved
    product_embeddings = [e for e in embeddings if e.get("product_id") == product_id]
    cutout_embeddings = [e for e in embeddings if e.get("type") == "cutout"]
    primary_embedding = [e for e in embeddings if e.get("is_primary") == True]

    log(f"Results:", "SUCCESS")
    log(f"  Total embeddings: {len(embeddings)}")
    log(f"  Product frames: {len(product_embeddings)}")
    log(f"  Cutouts: {len(cutout_embeddings)}")
    log(f"  Primary frame found: {len(primary_embedding) > 0}")
    log(f"  Time: {elapsed:.1f}s")

    # Verify frame indices
    if product_embeddings:
        frame_indices = sorted([e.get("frame_index", -1) for e in product_embeddings])
        log(f"  Frame indices: {frame_indices}")

    return len(embeddings) == len(all_images)


def test_embedding_large_batch() -> bool:
    """Test large batch embedding."""
    log("\n" + "="*70)
    log("TEST: LARGE BATCH EMBEDDING")
    log("="*70)

    # Create 50 images by repeating test images
    large_batch = []
    for i in range(50):
        img = TEST_IMAGES[i % len(TEST_IMAGES)].copy()
        img["id"] = f"batch-{i}"
        large_batch.append(img)

    input_data = {
        "images": large_batch,
        "model_type": "dinov2-base",
        "batch_size": 16,
        "hf_token": HF_TOKEN,
    }

    log(f"Submitting {len(large_batch)} images with batch_size=16...")

    start = time.time()
    result = run_runpod_job_sync(EMBEDDING_ENDPOINT, input_data, timeout=300)
    elapsed = time.time() - start

    if "error" in result:
        log(f"FAILED: {result['error']}", "ERROR")
        return False

    output = result.get("output", {})
    embeddings = output.get("embeddings", [])
    failed_count = output.get("failed_count", 0)

    throughput = len(embeddings) / elapsed if elapsed > 0 else 0

    log(f"Results:", "SUCCESS")
    log(f"  Processed: {len(embeddings)}/{len(large_batch)}")
    log(f"  Failed: {failed_count}")
    log(f"  Time: {elapsed:.1f}s")
    log(f"  Throughput: {throughput:.1f} images/sec")

    return len(embeddings) >= len(large_batch) * 0.9  # 90% success rate


def get_real_products(limit: int = 50) -> List[Dict]:
    """Get real products from Supabase."""
    log(f"Fetching {limit} products from Supabase...")

    url = f"{SUPABASE_URL}/rest/v1/products"
    params = {
        "select": "id,barcode,brand_name,primary_image_url,category",
        "primary_image_url": "not.is.null",
        "limit": limit,
    }

    with httpx.Client(timeout=30) as client:
        response = client.get(url, headers=get_supabase_headers(), params=params)

        if response.status_code != 200:
            log(f"Error fetching products: {response.status_code}", "ERROR")
            return []

        products = response.json()
        log(f"Found {len(products)} products")
        return products


def test_embedding_real_products() -> bool:
    """Test embedding extraction with real product images."""
    log("\n" + "="*70)
    log("TEST: REAL PRODUCT IMAGES FROM SUPABASE")
    log("="*70)

    products = get_real_products(limit=30)
    if not products:
        log("No products found, skipping test", "WARN")
        return True  # Don't fail if no data

    images = [
        {
            "id": p["id"],
            "url": p["primary_image_url"],
            "type": "product",
            "domain": "real",
            "product_id": p["id"],
            "category": p.get("category", "unknown"),
        }
        for p in products
        if p.get("primary_image_url")
    ]

    input_data = {
        "images": images,
        "model_type": "dinov3-base",
        "batch_size": 8,
        "hf_token": HF_TOKEN,
    }

    log(f"Processing {len(images)} real product images...")

    start = time.time()
    result = run_runpod_job_sync(EMBEDDING_ENDPOINT, input_data, timeout=300)
    elapsed = time.time() - start

    if "error" in result:
        log(f"FAILED: {result['error']}", "ERROR")
        return False

    output = result.get("output", {})
    embeddings = output.get("embeddings", [])
    failed_ids = output.get("failed_ids", [])

    success_rate = len(embeddings) / len(images) * 100 if images else 0

    log(f"Results:", "SUCCESS")
    log(f"  Processed: {len(embeddings)}/{len(images)}")
    log(f"  Success rate: {success_rate:.1f}%")
    log(f"  Failed: {len(failed_ids)}")
    log(f"  Time: {elapsed:.1f}s")

    if failed_ids:
        log(f"  Failed IDs: {failed_ids[:3]}...", "WARN")

    return success_rate >= 80  # 80% success rate for real images


# ============================================================
# TRAINING TESTS
# ============================================================

def create_training_run(model_type: str, product_ids: List[str]) -> Optional[str]:
    """Create a training run record in Supabase."""
    training_run_id = str(uuid.uuid4())

    url = f"{SUPABASE_URL}/rest/v1/training_runs"
    payload = {
        "id": training_run_id,
        "name": f"E2E Test - {model_type} - {datetime.now().strftime('%H:%M:%S')}",
        "description": "Automated E2E test run",
        "status": "pending",
        "base_model_type": model_type,
        "data_source": "selected_products",
        "train_product_ids": product_ids,
        "split_config": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratify_by": "brand_name",
            "seed": 42,
        },
        "training_config": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "use_arcface": True,
            "use_gem_pooling": True,
            "augmentation_strength": "medium",
        },
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(
            url,
            headers={**get_supabase_headers(), "Prefer": "return=representation"},
            json=payload,
        )

        if response.status_code not in (200, 201):
            log(f"Failed to create training run: {response.text}", "ERROR")
            return None

        log(f"Created training run: {training_run_id}")
        return training_run_id


def update_training_run_status(training_run_id: str, status: str, metrics: Optional[Dict] = None):
    """Update training run status in Supabase."""
    url = f"{SUPABASE_URL}/rest/v1/training_runs?id=eq.{training_run_id}"
    payload = {"status": status, "updated_at": datetime.utcnow().isoformat() + "Z"}
    if metrics:
        payload["metrics"] = metrics

    with httpx.Client(timeout=30) as client:
        client.patch(url, headers={**get_supabase_headers(), "Prefer": "return=minimal"}, json=payload)


def test_training_basic() -> bool:
    """Test basic training pipeline."""
    log("\n" + "="*70)
    log("TEST: TRAINING PIPELINE (BASIC)")
    log("="*70)

    model_type = "dinov2-small"  # Fastest model for testing

    # Get product IDs for training
    products = get_real_products(limit=100)
    if len(products) < 30:
        log(f"Not enough products ({len(products)}), need at least 30", "WARN")
        return True  # Don't fail if no data

    product_ids = [p["id"] for p in products]
    log(f"Using {len(product_ids)} products for training")

    # Generate a unique training run ID (worker will use this)
    training_run_id = str(uuid.uuid4())
    log(f"Training run ID: {training_run_id}")

    # Submit training job directly to worker (worker handles data split)
    input_data = {
        "training_run_id": training_run_id,
        "model_type": model_type,
        "product_ids": product_ids,
        "hf_token": HF_TOKEN,
        "supabase_url": SUPABASE_URL,
        "supabase_key": SUPABASE_KEY,
        "config": {
            "epochs": 2,  # Quick test
            "batch_size": 8,
            "learning_rate": 1e-4,
            "use_arcface": True,
            "use_gem_pooling": True,
            "warmup_epochs": 0,
            "save_every_n_epochs": 1,
            "eval_every_n_epochs": 1,
        },
    }

    log(f"Submitting training job (async)...")
    job_id = run_runpod_job_async(TRAINING_ENDPOINT, input_data)

    if not job_id:
        return False

    log(f"Job ID: {job_id}")
    log(f"Waiting for training to complete (this may take several minutes)...")

    # Wait for completion (training can take a while)
    result = wait_for_job(TRAINING_ENDPOINT, job_id, timeout=900, poll_interval=15)

    if "error" in result:
        log(f"Training FAILED: {result['error']}", "ERROR")
        return False

    output = result.get("output", {})

    # Check results
    status = output.get("status")
    epochs_trained = output.get("epochs_trained", 0)
    test_metrics = output.get("test_metrics", {})
    data_stats = output.get("data_stats", {})

    log(f"Training Results:", "SUCCESS")
    log(f"  Status: {status}")
    log(f"  Epochs trained: {epochs_trained}")
    log(f"  Train products: {data_stats.get('train_products', 'N/A')}")
    log(f"  Val products: {data_stats.get('val_products', 'N/A')}")
    log(f"  Test products: {data_stats.get('test_products', 'N/A')}")

    if test_metrics:
        log(f"  Test Recall@1: {test_metrics.get('recall@1', 'N/A')}")
        log(f"  Test Recall@5: {test_metrics.get('recall@5', 'N/A')}")

    return status == "completed"


def test_training_sota_combined_loss_pk() -> bool:
    """Test SOTA training with Combined Loss + P-K Sampling."""
    log("\n" + "="*70)
    log("TEST: SOTA TRAINING (COMBINED LOSS + P-K SAMPLING)")
    log("="*70)

    model_type = "dinov2-small"

    # Get products
    products = get_real_products(limit=100)
    if len(products) < 50:
        log(f"Not enough products ({len(products)}), need at least 50", "WARN")
        return True

    product_ids = [p["id"] for p in products]
    log(f"Using {len(product_ids)} products for SOTA training")

    # Generate a unique training run ID
    training_run_id = str(uuid.uuid4())
    log(f"Training run ID: {training_run_id}")

    # Use combined_loss_pk_sampling config
    sota_config = SOTA_CONFIGS["combined_loss_pk_sampling"]

    input_data = {
        "training_run_id": training_run_id,
        "model_type": model_type,
        "product_ids": product_ids,
        "hf_token": HF_TOKEN,
        "supabase_url": SUPABASE_URL,
        "supabase_key": SUPABASE_KEY,
        "config": {
            "epochs": 2,  # Quick test
            "batch_size": 32,  # P*K = 8*4 = 32
            "learning_rate": 5e-5,
            "use_arcface": True,
            "use_gem_pooling": True,
            "use_llrd": True,
            "warmup_epochs": 0,
            "augmentation_strength": "moderate",
            "mixed_precision": True,
            "sota_config": sota_config,
        },
    }

    log(f"SOTA Config: Combined Loss={sota_config.get('use_combined_loss')}, P-K Sampling={sota_config.get('use_pk_sampling')}")
    log(f"Submitting SOTA training job...")
    job_id = run_runpod_job_async(TRAINING_ENDPOINT, input_data)

    if not job_id:
        return False

    log(f"Job ID: {job_id}")
    log(f"Waiting for SOTA training (this may take 10-15 minutes)...")

    result = wait_for_job(TRAINING_ENDPOINT, job_id, timeout=1200, poll_interval=20)

    if "error" in result:
        log(f"SOTA Training FAILED: {result['error']}", "ERROR")
        if "result" in result:
            log(f"  Details: {json.dumps(result['result'], indent=2)[:500]}")
        return False

    output = result.get("output", {})

    status = output.get("status")
    epochs_trained = output.get("epochs_trained", 0)
    sota_enabled = output.get("sota_enabled", False)
    test_metrics = output.get("test_metrics", {})
    cross_domain = output.get("cross_domain_metrics", {})
    hard_examples = output.get("hard_examples", [])
    confused_pairs = output.get("confused_pairs", [])

    log(f"SOTA Training Results:", "SUCCESS")
    log(f"  Status: {status}")
    log(f"  SOTA enabled: {sota_enabled}")
    log(f"  Epochs trained: {epochs_trained}")

    if test_metrics:
        log(f"  Test Recall@1: {test_metrics.get('recall@1', 'N/A')}")
        log(f"  Test Recall@5: {test_metrics.get('recall@5', 'N/A')}")
        log(f"  Test mAP: {test_metrics.get('mAP', 'N/A')}")

    if cross_domain:
        log(f"  Cross-domain accuracy: {cross_domain.get('cross_domain_accuracy', 'N/A')}")

    log(f"  Hard examples found: {len(hard_examples)}")
    log(f"  Confused pairs found: {len(confused_pairs)}")

    # Log some hard examples
    if hard_examples:
        log(f"  Sample hard examples:")
        for he in hard_examples[:3]:
            log(f"    - Cutout: {he.get('cutout_image_url', 'N/A')[:50]}...")

    return status == "completed"


def test_training_sota_full() -> bool:
    """Test FULL SOTA training with all features enabled."""
    log("\n" + "="*70)
    log("TEST: FULL SOTA TRAINING (ALL FEATURES)")
    log("="*70)

    model_type = "dinov2-small"

    # Get products
    products = get_real_products(limit=100)
    if len(products) < 50:
        log(f"Not enough products ({len(products)}), need at least 50", "WARN")
        return True

    product_ids = [p["id"] for p in products]
    log(f"Using {len(product_ids)} products for FULL SOTA training")

    # Generate a unique training run ID
    training_run_id = str(uuid.uuid4())
    log(f"Training run ID: {training_run_id}")

    # Use full_sota config
    sota_config = SOTA_CONFIGS["full_sota"]

    input_data = {
        "training_run_id": training_run_id,
        "model_type": model_type,
        "product_ids": product_ids,
        "hf_token": HF_TOKEN,
        "supabase_url": SUPABASE_URL,
        "supabase_key": SUPABASE_KEY,
        "config": {
            "epochs": 4,  # Minimum for curriculum (1+1+1+1)
            "batch_size": 32,  # P*K = 8*4 = 32
            "learning_rate": 5e-5,
            "use_arcface": True,
            "use_gem_pooling": True,
            "use_llrd": True,
            "warmup_epochs": 0,
            "augmentation_strength": "strong",
            "mixed_precision": True,
            "sota_config": sota_config,
        },
    }

    log(f"SOTA Config Features:")
    log(f"  - Combined Loss: {sota_config.get('use_combined_loss')}")
    log(f"  - P-K Sampling: {sota_config.get('use_pk_sampling')}")
    log(f"  - Curriculum Learning: {sota_config.get('use_curriculum')}")
    log(f"  - Domain Adaptation: {sota_config.get('use_domain_adaptation')}")
    log(f"  - Early Stopping: {sota_config.get('use_early_stopping')}")
    log(f"Submitting FULL SOTA training job...")

    job_id = run_runpod_job_async(TRAINING_ENDPOINT, input_data)

    if not job_id:
        return False

    log(f"Job ID: {job_id}")
    log(f"Waiting for FULL SOTA training (this may take 15-20 minutes)...")

    result = wait_for_job(TRAINING_ENDPOINT, job_id, timeout=1800, poll_interval=20)

    if "error" in result:
        log(f"FULL SOTA Training FAILED: {result['error']}", "ERROR")
        if "result" in result:
            log(f"  Details: {json.dumps(result['result'], indent=2)[:500]}")
        return False

    output = result.get("output", {})

    status = output.get("status")
    epochs_trained = output.get("epochs_trained", 0)
    sota_enabled = output.get("sota_enabled", False)
    test_metrics = output.get("test_metrics", {})
    best_recall = output.get("best_recall_at_1", 0)

    log(f"FULL SOTA Training Results:", "SUCCESS")
    log(f"  Status: {status}")
    log(f"  SOTA enabled: {sota_enabled}")
    log(f"  Epochs trained: {epochs_trained}")
    log(f"  Best Recall@1: {best_recall}")

    if test_metrics:
        log(f"  Test Recall@1: {test_metrics.get('recall@1', 'N/A')}")
        log(f"  Test Recall@5: {test_metrics.get('recall@5', 'N/A')}")
        log(f"  Test mAP: {test_metrics.get('mAP', 'N/A')}")

    return status == "completed"


def test_embedding_key_frames() -> bool:
    """Test key_frames extraction (0°, 90°, 180°, 270°) simulation."""
    log("\n" + "="*70)
    log("TEST: KEY FRAMES EXTRACTION (4 CARDINAL ANGLES)")
    log("="*70)

    # Simulate key_frames for multiple products
    key_frame_images = []

    for prod_idx in range(5):  # 5 products
        product_id = str(uuid.uuid4())
        # 4 key frames per product (0°, 90°, 180°, 270°)
        for frame_idx, angle in enumerate([0, 90, 180, 270]):
            key_frame_images.append({
                "id": f"{product_id}-frame-{frame_idx}",
                "url": TEST_IMAGES[(prod_idx * 4 + frame_idx) % len(TEST_IMAGES)]["url"],
                "type": "product",
                "domain": "synthetic",
                "product_id": product_id,
                "frame_index": frame_idx,
                "angle": angle,
                "is_primary": frame_idx == 0,
                "category": "test",
            })

    input_data = {
        "images": key_frame_images,
        "model_type": "dinov2-base",
        "batch_size": 8,
        "hf_token": HF_TOKEN,
    }

    log(f"Submitting {len(key_frame_images)} key frame images (5 products x 4 frames)...")

    start = time.time()
    result = run_runpod_job_sync(EMBEDDING_ENDPOINT, input_data, timeout=300)
    elapsed = time.time() - start

    if "error" in result:
        log(f"FAILED: {result['error']}", "ERROR")
        return False

    output = result.get("output", {})
    embeddings = output.get("embeddings", [])

    # Verify structure
    products_seen = set()
    frame_counts = {}
    primary_count = 0

    for emb in embeddings:
        pid = emb.get("product_id")
        if pid:
            products_seen.add(pid)
            frame_counts[pid] = frame_counts.get(pid, 0) + 1
        if emb.get("is_primary"):
            primary_count += 1

    log(f"Key Frames Results:", "SUCCESS")
    log(f"  Total embeddings: {len(embeddings)}")
    log(f"  Products: {len(products_seen)}")
    log(f"  Primary frames: {primary_count}")
    log(f"  Avg frames per product: {len(embeddings) / len(products_seen) if products_seen else 0:.1f}")
    log(f"  Time: {elapsed:.1f}s")

    # Verify 4 frames per product
    all_have_4_frames = all(count == 4 for count in frame_counts.values())
    log(f"  All products have 4 frames: {all_have_4_frames}")

    return len(embeddings) == len(key_frame_images) and all_have_4_frames


# ============================================================
# MAIN
# ============================================================

def validate_env():
    """Validate required environment variables."""
    missing = []
    if not RUNPOD_API_KEY:
        missing.append("RUNPOD_API_KEY")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")

    if missing:
        log(f"Missing required environment variables: {', '.join(missing)}", "ERROR")
        log("Please set them before running:")
        log("  export RUNPOD_API_KEY='rpa_...'")
        log("  export HF_TOKEN='hf_...'")
        log("  export SUPABASE_SERVICE_ROLE_KEY='eyJ...'")
        return False
    return True


def main():
    """Run all E2E tests."""
    if not validate_env():
        return False

    log("="*70)
    log("BUYBUDDY AI - COMPREHENSIVE E2E TESTS")
    log("="*70)
    log(f"Embedding Endpoint: {EMBEDDING_ENDPOINT}")
    log(f"Training Endpoint: {TRAINING_ENDPOINT}")
    log(f"Supabase URL: {SUPABASE_URL}")
    log("="*70)

    results = {}

    # ============================================
    # EMBEDDING TESTS
    # ============================================
    log("\n\n" + "█"*70)
    log("EMBEDDING EXTRACTION TESTS")
    log("█"*70)

    try:
        results["embedding_all_models"] = test_embedding_all_models()
    except Exception as e:
        log(f"Exception in embedding_all_models: {e}", "ERROR")
        results["embedding_all_models"] = False

    try:
        results["embedding_multiframe"] = test_embedding_multiframe()
    except Exception as e:
        log(f"Exception in embedding_multiframe: {e}", "ERROR")
        results["embedding_multiframe"] = False

    try:
        results["embedding_key_frames"] = test_embedding_key_frames()
    except Exception as e:
        log(f"Exception in embedding_key_frames: {e}", "ERROR")
        results["embedding_key_frames"] = False

    try:
        results["embedding_large_batch"] = test_embedding_large_batch()
    except Exception as e:
        log(f"Exception in embedding_large_batch: {e}", "ERROR")
        results["embedding_large_batch"] = False

    try:
        results["embedding_real_products"] = test_embedding_real_products()
    except Exception as e:
        log(f"Exception in embedding_real_products: {e}", "ERROR")
        results["embedding_real_products"] = False

    # ============================================
    # TRAINING TESTS
    # ============================================
    log("\n\n" + "█"*70)
    log("TRAINING TESTS")
    log("█"*70)

    try:
        results["training_basic"] = test_training_basic()
    except Exception as e:
        log(f"Exception in training_basic: {e}", "ERROR")
        results["training_basic"] = False

    try:
        results["training_sota_combined_loss_pk"] = test_training_sota_combined_loss_pk()
    except Exception as e:
        log(f"Exception in training_sota_combined_loss_pk: {e}", "ERROR")
        results["training_sota_combined_loss_pk"] = False

    try:
        results["training_sota_full"] = test_training_sota_full()
    except Exception as e:
        log(f"Exception in training_sota_full: {e}", "ERROR")
        results["training_sota_full"] = False

    # ============================================
    # FINAL SUMMARY
    # ============================================
    log("\n\n" + "="*70)
    log("FINAL E2E TEST RESULTS")
    log("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log(f"{status}: {test_name}")

    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)

    log("="*70)
    log(f"TOTAL: {total_passed}/{total_tests} tests passed")
    log("="*70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
