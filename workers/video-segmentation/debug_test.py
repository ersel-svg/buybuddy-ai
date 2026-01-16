#!/usr/bin/env python3
"""Debug test script for SAM3 OOM investigation."""

import sys
sys.path.insert(0, "/app/src")

import torch
import time
import uuid

print("=" * 70)
print("SAM3 OOM DEBUG TEST")
print("=" * 70)

# Test with the problematic 375 frame video
test_video = {
    "video_url": "https://bb-item-images.s3.amazonaws.com/6dd1159d-0de9-46ba-80f0-7937529fc9e1.mp4",
    "barcode": "040400008626",
    "video_id": 40,
}

print(f"\n[INIT] GPU Info:")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    print("  CUDA not available!")
    sys.exit(1)

print("\n[INIT] Loading pipeline...")
from pipeline import ProductPipeline
pipe = ProductPipeline()

print(f"\n[INIT] GPU after model load:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Generate test product_id
test_product_id = str(uuid.uuid4())

print("\n" + "=" * 70)
print(f"PROCESSING: {test_video['barcode']}")
print("=" * 70)

start_time = time.time()

try:
    result = pipe.process(
        video_url=test_video["video_url"],
        barcode=test_video["barcode"],
        video_id=test_video["video_id"],
        product_id=test_product_id,
        target_frames=100,  # Test with 100 target frames
    )

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Completed in {elapsed:.1f}s")
    print(f"  Frame count: {result.get('frame_count', 0)}")
    print(f"  Status: {result.get('status', 'unknown')}")

except torch.cuda.OutOfMemoryError as e:
    elapsed = time.time() - start_time
    print(f"\n[OOM ERROR] After {elapsed:.1f}s")
    print(f"  Error: {e}")
    print(f"\n[GPU STATE AT OOM]:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] After {elapsed:.1f}s")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
