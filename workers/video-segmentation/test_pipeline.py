#!/usr/bin/env python3
"""Real-world pipeline test with 2 products."""

import sys
sys.path.insert(0, "/app/src")
import torch
import time
import uuid

print("=" * 70)
print("REAL-WORLD PIPELINE TEST - 2 PRODUCTS")
print("=" * 70)

# Test products
test_products = [
    {
        "video_url": "https://bb-item-images.s3.amazonaws.com/cfea09a8-a48c-41c7-a9de-99c7043ae155.mp4",
        "barcode": "078000030914",
        "video_id": 41,
    },
    {
        "video_url": "https://bb-item-images.s3.amazonaws.com/6dd1159d-0de9-46ba-80f0-7937529fc9e1.mp4",
        "barcode": "040400008626",
        "video_id": 40,
    }
]

# Initialize pipeline once
print("\n[INIT] Loading pipeline (SAM3 + Gemini)...")
from pipeline import ProductPipeline
pipe = ProductPipeline()

print(f"[INIT] GPU Memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"[INIT] Gemini configured: {pipe.gemini_model is not None}")
print(f"[INIT] SAM3 configured: {pipe.video_predictor is not None}")

# Process each product
for idx, product in enumerate(test_products, 1):
    print("\n" + "=" * 70)
    product_barcode = product["barcode"]
    print(f"PRODUCT {idx}/2: Barcode {product_barcode}")
    print("=" * 70)

    start_time = time.time()

    # Generate a test product_id (in real usage this comes from the job)
    test_product_id = str(uuid.uuid4())

    try:
        result = pipe.process(
            video_url=product["video_url"],
            barcode=product["barcode"],
            video_id=product["video_id"],
            product_id=test_product_id,
            target_frames=100,  # Test with 100 frames
        )

        elapsed = time.time() - start_time

        print(f"\n[RESULT] Product {idx} completed in {elapsed:.1f}s")
        print(f"[RESULT] Status: {result.get('status', 'unknown')}")
        print(f"[RESULT] Frame count: {result.get('frame_count', 0)}")
        frames_url = result.get('frames_url', 'N/A')
        if frames_url and frames_url != 'N/A':
            print(f"[RESULT] Frames URL: {frames_url[:60]}...")

        # Show metadata if available
        metadata = result.get("metadata", {})
        brand = metadata.get("brand_info", {}).get("brand_name", "N/A")
        product_name = metadata.get("product_identity", {}).get("product_name", "N/A")
        print(f"[RESULT] Brand: {brand}")
        print(f"[RESULT] Product: {product_name}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Product {idx} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()

    # Memory status after each product
    print(f"\n[MEMORY] GPU after product {idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.empty_cache()
    print(f"[MEMORY] GPU after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
