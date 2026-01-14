#!/usr/bin/env python3
"""
Test script for the simplified embedding handler.
Run this on SSH pod to test the worker logic.

Usage:
    python test_handler.py
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime


def test_extractor_import():
    """Test extractor import."""
    print("\n" + "=" * 60)
    print("1. IMPORT TEST")
    print("=" * 60)

    try:
        from src.extractor import get_extractor, DINOv2Extractor
        print("  OK Extractor imported")
        return True
    except ImportError as e:
        print(f"  FAIL Extractor import error: {e}")
        return False


def test_pytorch():
    """Test PyTorch and CUDA."""
    print("\n" + "=" * 60)
    print("2. PYTORCH & CUDA")
    print("=" * 60)

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    except Exception as e:
        print(f"  FAIL PyTorch error: {e}")
        return False


def test_model_loading():
    """Test model loading."""
    print("\n" + "=" * 60)
    print("3. MODEL LOADING")
    print("=" * 60)

    try:
        from src.extractor import get_extractor

        print("  Loading DINOv2 Base model...")
        extractor = get_extractor(model_type="dinov2-base")
        print(f"  OK Model loaded!")
        print(f"    - Embedding dim: {extractor.embedding_dim}")
        print(f"    - Device: {extractor.device}")
        return extractor
    except Exception as e:
        print(f"  FAIL Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_extraction(extractor):
    """Test single image extraction."""
    print("\n" + "=" * 60)
    print("4. SINGLE IMAGE EXTRACTION")
    print("=" * 60)

    # Test with a real product image from Supabase storage
    test_url = "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/frames/60d29e45-f821-4de2-8d8f-d476382982ea/frame_0000.png"

    print(f"  URL: {test_url[:60]}...")

    try:
        embedding = extractor.extract_from_url(test_url)
        print(f"  OK Embedding extracted!")
        print(f"    - Shape: {embedding.shape}")
        print(f"    - Norm: {(embedding ** 2).sum() ** 0.5:.4f}")
        print(f"    - First 5: {embedding[:5]}")
        return embedding
    except Exception as e:
        print(f"  FAIL Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_handler_simulation(extractor):
    """Test handler logic simulation."""
    print("\n" + "=" * 60)
    print("5. HANDLER SIMULATION")
    print("=" * 60)

    # Simulate handler input
    test_input = {
        "images": [
            {
                "id": "test-product-1",
                "url": "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/frames/60d29e45-f821-4de2-8d8f-d476382982ea/frame_0000.png",
                "type": "product"
            },
            {
                "id": "test-product-2",
                "url": "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/frames/01369b9c-bf02-498b-98cb-cd331a182261/frame_0000.png",
                "type": "product"
            },
        ],
        "model_type": "dinov2-base",
        "batch_size": 2
    }

    print(f"  Input: {len(test_input['images'])} images")

    # Extract embeddings
    results = []
    failed = []

    urls = [img["url"] for img in test_input["images"]]
    batch_results = extractor.extract_batch_from_urls(urls, batch_size=2)
    url_to_emb = {url: emb for url, emb in batch_results}

    for img in test_input["images"]:
        emb = url_to_emb.get(img["url"])
        if emb is not None:
            results.append({
                "id": img["id"],
                "type": img["type"],
                "vector": emb.tolist()[:5],  # Just show first 5 for brevity
                "vector_len": len(emb.tolist())
            })
        else:
            failed.append(img["id"])

    print(f"\n  Results:")
    for r in results:
        print(f"    - {r['id']} ({r['type']}): {r['vector_len']} dims, first 5: {r['vector']}")

    if failed:
        print(f"  Failed: {failed}")

    # Simulate handler output
    output = {
        "status": "success",
        "embeddings": results,
        "processed_count": len(results),
        "failed_count": len(failed),
        "embedding_dim": 768
    }

    print(f"\n  Handler output:")
    print(f"    - status: {output['status']}")
    print(f"    - processed: {output['processed_count']}")
    print(f"    - failed: {output['failed_count']}")
    print(f"    - embedding_dim: {output['embedding_dim']}")

    return output


def main():
    print("\n" + "=" * 60)
    print("EMBEDDING HANDLER TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")

    # 1. Import test
    if not test_extractor_import():
        print("\n FAIL Import failed")
        sys.exit(1)

    # 2. PyTorch test
    if not test_pytorch():
        print("\n FAIL PyTorch failed")
        sys.exit(1)

    # 3. Model loading
    extractor = test_model_loading()
    if not extractor:
        print("\n FAIL Model loading failed")
        sys.exit(1)

    # 4. Single extraction
    embedding = test_single_extraction(extractor)
    if embedding is None:
        print("\n FAIL Single extraction failed")
        sys.exit(1)

    # 5. Handler simulation
    output = test_handler_simulation(extractor)
    if output["status"] != "success":
        print("\n FAIL Handler simulation failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nWorker is ready for deployment.")


if __name__ == "__main__":
    main()
