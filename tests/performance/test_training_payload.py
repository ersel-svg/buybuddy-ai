"""
Performance Tests for Training Payload Size.

These tests verify that SOTA pattern keeps payload sizes under control.
"""

import pytest
import json
import time


class TestEmbeddingTrainingPayloadSize:
    """Embedding Training payload size tests."""

    def test_sota_1k_products_under_100kb(self):
        """1000 products with SOTA pattern < 100 KB."""
        product_ids = [f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(1000)]
        payload = {
            "training_run_id": "uuid",
            "model_type": "dinov2-base",
            "source_config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:800],
                "val_product_ids": product_ids[800:],
                "image_config": {"image_types": ["synthetic"], "frame_selection": "first"},
            },
            "config": {"epochs": 100},
            "supabase_url": "https://x.supabase.co",
            "supabase_key": "key",
        }
        size = len(json.dumps(payload))
        print(f"1K products SOTA payload: {size / 1000:.1f} KB")
        assert size < 100_000

    def test_sota_5k_products_under_500kb(self):
        """5000 products with SOTA pattern < 500 KB."""
        product_ids = [f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(5000)]
        payload = {
            "training_run_id": "uuid",
            "source_config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:4000],
                "val_product_ids": product_ids[4000:],
                "image_config": {},
            },
            "config": {},
            "supabase_url": "https://x.supabase.co",
            "supabase_key": "key",
        }
        size = len(json.dumps(payload))
        print(f"5K products SOTA payload: {size / 1000:.1f} KB")
        assert size < 500_000

    def test_old_pattern_size_comparison(self):
        """Compare old vs SOTA pattern sizes."""
        # SOTA pattern (1000 products)
        product_ids = [f"uuid-{i}" for i in range(1000)]
        sota_payload = {
            "source_config": {
                "product_ids": product_ids,
                "image_config": {},
            }
        }
        sota_size = len(json.dumps(sota_payload))

        # Old pattern (1000 products x 10 images)
        training_images = {
            f"product_{i}": [
                {"url": f"https://storage.example.com/p/{i}/frame_{j}.png", "type": "synthetic"}
                for j in range(10)
            ]
            for i in range(1000)
        }
        old_payload = {"training_images": training_images}
        old_size = len(json.dumps(old_payload))

        print(f"SOTA size: {sota_size / 1000:.1f} KB")
        print(f"Old size: {old_size / 1000:.1f} KB")
        print(f"Reduction: {(1 - sota_size / old_size) * 100:.1f}%")

        assert sota_size < old_size / 10, "SOTA should be at least 10x smaller"


class TestCLSTrainingPayloadSize:
    """CLS Training payload size tests."""

    def test_sota_constant_size(self):
        """CLS SOTA payload is constant regardless of dataset size."""
        payload = {
            "training_run_id": "uuid",
            "dataset_id": "uuid",  # Only dataset_id, no URLs
            "config": {"epochs": 30},
            "supabase_url": "https://x.supabase.co",
            "supabase_key": "key",
        }
        size = len(json.dumps(payload))
        print(f"CLS SOTA payload: {size} bytes")
        assert size < 1_000, f"SOTA payload should be < 1KB, got {size}"

    def test_old_pattern_10k_images(self):
        """Old pattern with 10K images - still much larger than SOTA."""
        urls = [{"url": f"https://storage.example.com/img_{i}.jpg", "label": i % 10} for i in range(10_000)]
        old_payload = {"dataset": {"train_urls": urls[:8000], "val_urls": urls[8000:]}}
        size = len(json.dumps(old_payload))
        print(f"Old CLS payload (10K images): {size / 1_000_000:.2f} MB")
        # Old pattern is ~0.65 MB for 10K images vs ~1 KB for SOTA
        assert size > 500_000, "Old pattern should be > 500KB for 10K images"

    def test_old_pattern_50k_images(self):
        """Old pattern with 50K images - typical large dataset."""
        urls = [{"url": f"https://storage.example.com/images/cls/img_{i}.jpg", "label": i % 100} for i in range(50_000)]
        old_payload = {"dataset": {"train_urls": urls[:40000], "val_urls": urls[40000:]}}
        size = len(json.dumps(old_payload))
        print(f"Old CLS payload (50K images): {size / 1_000_000:.2f} MB")
        # Old pattern is ~3.9 MB for 50K images vs ~1 KB for SOTA
        assert size > 3_000_000, "Old pattern should be > 3MB for 50K images"

    def test_runpod_payload_limit_safety(self):
        """SOTA payloads should be well under RunPod's ~10MB limit."""
        # Embedding Training with 10K products
        product_ids = [f"uuid-{i}" for i in range(10_000)]
        embedding_payload = {
            "source_config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:8000],
                "val_product_ids": product_ids[8000:],
                "image_config": {},
            },
            "config": {},
            "supabase_url": "https://x.supabase.co",
            "supabase_key": "x" * 500,  # Simulated JWT
        }

        # CLS Training (constant size)
        cls_payload = {
            "dataset_id": "uuid",
            "config": {},
            "supabase_url": "https://x.supabase.co",
            "supabase_key": "x" * 500,
        }

        embedding_size = len(json.dumps(embedding_payload))
        cls_size = len(json.dumps(cls_payload))

        print(f"Embedding Training (10K products): {embedding_size / 1000:.1f} KB")
        print(f"CLS Training: {cls_size / 1000:.1f} KB")

        # RunPod limit is ~10MB, we want to be well under
        assert embedding_size < 1_000_000, "Embedding payload should be < 1MB"
        assert cls_size < 10_000, "CLS payload should be < 10KB"


class TestPayloadSerializationPerformance:
    """Test serialization/deserialization performance."""

    def test_sota_serialization_fast(self):
        """SOTA payload serialization should be fast."""
        product_ids = [f"uuid-{i}" for i in range(5000)]
        payload = {
            "source_config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:4000],
                "val_product_ids": product_ids[4000:],
                "image_config": {},
            }
        }

        start = time.time()
        for _ in range(100):
            json.dumps(payload)
        elapsed = time.time() - start

        print(f"100x serialization of 5K product SOTA payload: {elapsed * 1000:.1f} ms")
        assert elapsed < 1.0, "Serialization should be < 1 second for 100 iterations"

    def test_old_pattern_serialization_slow(self):
        """Old pattern serialization would be slower."""
        training_images = {
            f"product_{i}": [
                {"url": f"https://storage.example.com/p/{i}/frame_{j}.png", "type": "synthetic"}
                for j in range(10)
            ]
            for i in range(1000)
        }
        payload = {"training_images": training_images}

        start = time.time()
        for _ in range(10):
            json.dumps(payload)
        elapsed = time.time() - start

        print(f"10x serialization of old 1K product payload: {elapsed * 1000:.1f} ms")
        # Just documenting the difference, not asserting
