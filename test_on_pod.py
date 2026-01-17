#!/usr/bin/env python3
"""
GPU Pod Test Script - Run before deploying workers.

Tests that can be run directly on GPU pod via SSH:
1. Model loading (all 9 models including DINOv3 with HF_TOKEN)
2. Embedding extraction handler (simulated)
3. Training handler components
4. bb-models package functionality

Usage:
    export HF_TOKEN="your_hf_token"
    python test_on_pod.py
"""

import os
import sys
import time
import traceback
from typing import Dict, Any, List, Tuple

# Add paths
sys.path.insert(0, "/app/bb-models/src")
sys.path.insert(0, "/app/src")

# Test results
RESULTS: List[Tuple[str, bool, str]] = []


def log_test(name: str, passed: bool, message: str = ""):
    """Log test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append((name, passed, message))
    print(f"{status}: {name}")
    if message:
        print(f"       {message}")


def test_hf_token():
    """Test HF_TOKEN is available."""
    print("\n" + "=" * 60)
    print("TEST: HuggingFace Token")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        log_test("HF_TOKEN set", True, f"Token length: {len(hf_token)}")
    else:
        log_test("HF_TOKEN set", False, "HF_TOKEN not found - DINOv3 will fail")


def test_bb_models_import():
    """Test bb-models package import."""
    print("\n" + "=" * 60)
    print("TEST: bb-models Package Import")
    print("=" * 60)

    try:
        from bb_models import get_backbone, get_model_config, list_available_models
        from bb_models.registry import MODEL_CONFIGS

        models = list_available_models()
        log_test("bb-models import", True, f"Found {len(models)} models")
        print(f"       Models: {models}")
    except Exception as e:
        log_test("bb-models import", False, str(e))


def test_model_configs():
    """Test all model configurations."""
    print("\n" + "=" * 60)
    print("TEST: Model Configurations")
    print("=" * 60)

    from bb_models import get_model_config, list_available_models

    for model_id in list_available_models():
        try:
            config = get_model_config(model_id)
            log_test(
                f"Config: {model_id}",
                True,
                f"dim={config.embedding_dim}, hf={config.hf_model_id}"
            )
        except Exception as e:
            log_test(f"Config: {model_id}", False, str(e))


def test_model_loading():
    """Test loading all models."""
    print("\n" + "=" * 60)
    print("TEST: Model Loading (All 9 Models)")
    print("=" * 60)

    import torch
    from bb_models import get_backbone, list_available_models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    models_to_test = list_available_models()

    for model_id in models_to_test:
        try:
            print(f"\nLoading {model_id}...")
            start = time.time()

            backbone = get_backbone(model_id, load_pretrained=True)
            backbone.to(device)
            backbone.eval()

            elapsed = time.time() - start
            log_test(
                f"Load: {model_id}",
                True,
                f"dim={backbone.embedding_dim}, time={elapsed:.1f}s"
            )

            # Clean up GPU memory
            del backbone
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            log_test(f"Load: {model_id}", False, str(e))
            traceback.print_exc()


def test_embedding_extraction():
    """Test embedding extraction from URL."""
    print("\n" + "=" * 60)
    print("TEST: Embedding Extraction (Real URL)")
    print("=" * 60)

    import torch
    import httpx
    from PIL import Image
    from io import BytesIO
    from bb_models import get_backbone
    from transformers import AutoImageProcessor

    # Test URL (public image)
    test_url = "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"

    try:
        # Download image
        print("Downloading test image...")
        response = httpx.get(test_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image size: {image.size}")

        # Load model
        model_id = "dinov2-base"
        print(f"Loading {model_id}...")
        backbone = get_backbone(model_id, load_pretrained=True)
        backbone.to("cuda" if torch.cuda.is_available() else "cpu")
        backbone.eval()

        # Process image
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(backbone.device) for k, v in inputs.items()}

        # Extract embedding
        with torch.no_grad():
            embedding = backbone(inputs["pixel_values"])

        embedding_np = embedding.cpu().numpy().flatten()

        log_test(
            "Embedding extraction",
            True,
            f"shape={embedding_np.shape}, norm={sum(embedding_np**2)**0.5:.4f}"
        )

        # Cleanup
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        log_test("Embedding extraction", False, str(e))
        traceback.print_exc()


def test_batch_extraction():
    """Test batch embedding extraction."""
    print("\n" + "=" * 60)
    print("TEST: Batch Embedding Extraction")
    print("=" * 60)

    import torch
    import httpx
    from PIL import Image
    from io import BytesIO
    from bb_models import get_backbone
    from transformers import AutoImageProcessor

    # Multiple test URLs
    test_urls = [
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=200",
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=200",
        "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=200",
    ]

    try:
        # Download images
        images = []
        for url in test_urls:
            response = httpx.get(url, timeout=30)
            response.raise_for_status()
            images.append(Image.open(BytesIO(response.content)).convert("RGB"))

        print(f"Downloaded {len(images)} images")

        # Load model
        model_id = "dinov2-base"
        backbone = get_backbone(model_id, load_pretrained=True)
        backbone.to("cuda" if torch.cuda.is_available() else "cpu")
        backbone.eval()

        # Process batch
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(backbone.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            embeddings = backbone(inputs["pixel_values"])

        log_test(
            "Batch extraction",
            True,
            f"batch_size={len(images)}, output_shape={embeddings.shape}"
        )

        # Cleanup
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        log_test("Batch extraction", False, str(e))
        traceback.print_exc()


def test_dinov3_with_token():
    """Test DINOv3 model loading with HF_TOKEN."""
    print("\n" + "=" * 60)
    print("TEST: DINOv3 with HF_TOKEN")
    print("=" * 60)

    import torch

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log_test("DINOv3 loading", False, "HF_TOKEN not set - skipping")
        return

    try:
        from bb_models import get_backbone

        model_id = "dinov3-base"
        print(f"Loading {model_id} with HF_TOKEN...")

        start = time.time()
        backbone = get_backbone(model_id, load_pretrained=True)
        backbone.to("cuda" if torch.cuda.is_available() else "cpu")
        backbone.eval()
        elapsed = time.time() - start

        log_test(
            "DINOv3 loading",
            True,
            f"dim={backbone.embedding_dim}, time={elapsed:.1f}s"
        )

        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224).to(backbone.device)
        with torch.no_grad():
            output = backbone(dummy_input)

        log_test(
            "DINOv3 inference",
            True,
            f"output_shape={output.shape}"
        )

        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        log_test("DINOv3 loading", False, str(e))
        traceback.print_exc()


def test_embedding_handler_simulation():
    """Simulate embedding handler without RunPod."""
    print("\n" + "=" * 60)
    print("TEST: Embedding Handler Simulation")
    print("=" * 60)

    # Add worker path
    sys.path.insert(0, "/app")

    try:
        # Import handler components
        from extractor import get_extractor, EmbeddingExtractor

        # Test get_extractor
        extractor = get_extractor(model_type="dinov2-base")
        log_test(
            "get_extractor()",
            True,
            f"model={extractor.model_type}, dim={extractor.embedding_dim}"
        )

        # Simulate job input
        job_input = {
            "images": [
                {
                    "id": "test-1",
                    "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=200",
                    "type": "product",
                    "domain": "real"
                }
            ],
            "model_type": "dinov2-base",
            "batch_size": 1
        }

        # Extract
        urls = [img["url"] for img in job_input["images"]]
        results = extractor.extract_batch_from_urls(urls, batch_size=1)

        if results:
            url, embedding = results[0]
            log_test(
                "Handler simulation",
                True,
                f"extracted {len(results)} embeddings, dim={len(embedding)}"
            )
        else:
            log_test("Handler simulation", False, "No results returned")

    except Exception as e:
        log_test("Handler simulation", False, str(e))
        traceback.print_exc()


def test_training_components():
    """Test training-related components."""
    print("\n" + "=" * 60)
    print("TEST: Training Components")
    print("=" * 60)

    try:
        # Test imports
        from bb_models.losses import CombinedLoss, TripletMarginLoss
        from bb_models.training import Trainer, TrainingConfig

        log_test("Training imports", True, "CombinedLoss, Trainer, TrainingConfig")

        # Test loss creation
        loss_fn = CombinedLoss(
            use_triplet=True,
            use_arcface=True,
            triplet_weight=0.5,
            arcface_weight=0.5,
            margin=0.3,
            num_classes=100,
            embedding_dim=768
        )
        log_test("CombinedLoss creation", True, "triplet + arcface")

        # Test config
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            epochs=10,
            use_llrd=True,
            llrd_decay=0.9
        )
        log_test("TrainingConfig creation", True, f"lr={config.learning_rate}")

    except Exception as e:
        log_test("Training components", False, str(e))
        traceback.print_exc()


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)

    print(f"\nTotal: {passed}/{total} tests passed")
    print()

    # Print failures
    failures = [(n, m) for n, p, m in RESULTS if not p]
    if failures:
        print("FAILURES:")
        for name, msg in failures:
            print(f"  ❌ {name}: {msg}")
    else:
        print("All tests passed! ✅")

    return passed == total


def main():
    """Run all tests."""
    print("=" * 60)
    print("GPU POD TEST SCRIPT")
    print("Testing before worker deployment")
    print("=" * 60)

    # Run tests
    test_hf_token()
    test_bb_models_import()
    test_model_configs()
    test_model_loading()
    test_embedding_extraction()
    test_batch_extraction()
    test_dinov3_with_token()
    test_embedding_handler_simulation()
    test_training_components()

    # Summary
    success = print_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
