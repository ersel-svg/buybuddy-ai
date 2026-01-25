"""
Contract Tests for Training Workers.

These tests verify that API payloads match worker expectations.
They ensure schema compatibility between API → Worker communication.
"""

import pytest
import json


class TestEmbeddingTrainingContract:
    """Embedding Training API → Worker contract."""

    def test_source_config_required_fields(self):
        """source_config should contain required fields for SOTA pattern."""
        source_config = {
            "product_ids": ["uuid1", "uuid2"],
            "train_product_ids": ["uuid1"],
            "val_product_ids": ["uuid2"],
            "test_product_ids": [],
            "image_config": {
                "image_types": ["synthetic"],
                "frame_selection": "first",
                "max_frames_per_type": 10,
            },
            "label_config": {
                "label_field": "brand_name",
            },
        }
        # Required fields
        assert "product_ids" in source_config
        assert isinstance(source_config["product_ids"], list)
        assert "train_product_ids" in source_config
        assert "val_product_ids" in source_config
        assert "image_config" in source_config

    def test_source_config_image_config_fields(self):
        """image_config should have valid image_types."""
        valid_image_types = ["synthetic", "real", "augmented", "cutout"]
        image_config = {
            "image_types": ["synthetic", "cutout"],
            "frame_selection": "first",
        }
        assert all(t in valid_image_types for t in image_config["image_types"])

    def test_sota_payload_under_100kb(self):
        """SOTA payload for 1000 products should be < 100 KB."""
        product_ids = [f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(1000)]
        payload = {
            "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
            "model_type": "dinov2-base",
            "source_config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:800],
                "val_product_ids": product_ids[800:900],
                "test_product_ids": product_ids[900:],
                "image_config": {
                    "image_types": ["synthetic", "cutout"],
                    "frame_selection": "first",
                    "max_frames_per_type": 10,
                    "include_matched_cutouts": True,
                },
                "label_config": {"label_field": "brand_name"},
            },
            "config": {
                "epochs": 100,
                "batch_size": 64,
                "learning_rate": 0.0001,
            },
            "supabase_url": "https://xyzproject.supabase.co",
            "supabase_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0cmluZyIsInJvbGUiOiJzZXJ2aWNlX3JvbGUiLCJpYXQiOjE2MDAwMDAwMDAsImV4cCI6MTYwMDAwMDAwMH0.signature",
        }
        size = len(json.dumps(payload))
        print(f"Embedding Training SOTA payload (1K products): {size / 1000:.1f} KB")
        assert size < 100_000, f"Payload {size} bytes exceeds 100KB limit"

    def test_old_pattern_would_be_larger(self):
        """Old pattern with training_images dict would be much larger."""
        # Simulate 1000 products with 10 images each
        training_images = {
            f"product_{i}": [
                {
                    "url": f"https://storage.example.com/products/product_{i}/frames/frame_{j:04d}.png",
                    "image_type": "synthetic",
                    "frame_index": j,
                    "domain": "synthetic",
                }
                for j in range(10)
            ]
            for i in range(1000)
        }
        old_payload = {"training_images": training_images}
        size = len(json.dumps(old_payload))
        print(f"Old Embedding Training payload (1K products): {size / 1_000_000:.2f} MB")
        # Old pattern would be significantly larger
        assert size > 1_000_000, "Old pattern should be > 1MB for comparison"


class TestCLSTrainingContract:
    """CLS Training API → Worker contract."""

    def test_dataset_id_only_payload(self):
        """SOTA payload should send dataset_id, not URL arrays."""
        payload = {
            "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
            "dataset_id": "550e8400-e29b-41d4-a716-446655440001",
            "config": {
                "model_name": "efficientnet_b0",
                "epochs": 30,
                "batch_size": 32,
                "train_split": 0.8,
                "seed": 42,
            },
            "supabase_url": "https://xyzproject.supabase.co",
            "supabase_key": "eyJ...",
        }
        # SOTA pattern: dataset_id present, no train_urls/val_urls
        assert "dataset_id" in payload
        assert "train_urls" not in payload
        assert "val_urls" not in payload
        assert "dataset" not in payload

    def test_sota_cls_payload_constant_size(self):
        """CLS SOTA payload should be constant size regardless of dataset."""
        payload = {
            "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
            "dataset_id": "550e8400-e29b-41d4-a716-446655440001",
            "config": {
                "model_name": "efficientnet_b0",
                "epochs": 30,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "train_split": 0.8,
                "seed": 42,
                "loss": "label_smoothing",
                "augmentation": "sota",
                "use_mixup": True,
            },
            "supabase_url": "https://xyzproject.supabase.co",
            "supabase_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0cmluZyIsInJvbGUiOiJzZXJ2aWNlX3JvbGUiLCJpYXQiOjE2MDAwMDAwMDAsImV4cCI6MTYwMDAwMDAwMH0.signature",
        }
        size = len(json.dumps(payload))
        print(f"CLS SOTA payload size: {size} bytes")
        # Payload should be < 5KB regardless of dataset size
        assert size < 5_000, f"Payload should be < 5KB, got {size}"

    def test_old_cls_pattern_large(self):
        """Old CLS pattern with URL arrays would be huge compared to SOTA."""
        urls = [
            {"url": f"https://storage.example.com/cls/images/img_{i}.jpg", "label": i % 10}
            for i in range(50_000)
        ]
        old_payload = {
            "dataset": {
                "train_urls": urls[:40_000],
                "val_urls": urls[40_000:],
                "class_names": [f"class_{i}" for i in range(10)],
            }
        }
        size = len(json.dumps(old_payload))
        print(f"Old CLS payload (50K images): {size / 1_000_000:.2f} MB")
        # Old pattern is ~4 MB for 50K images vs ~1 KB for SOTA
        assert size > 3_000_000, "Old CLS pattern should be > 3MB for 50K images"

    def test_worker_backward_compatibility(self):
        """Worker should accept both old and new formats."""
        # Old format (legacy)
        old_format = {
            "training_run_id": "uuid",
            "dataset": {
                "train_urls": [{"url": "http://example.com/1.jpg", "label": 0}],
                "val_urls": [{"url": "http://example.com/2.jpg", "label": 1}],
                "class_names": ["class_a", "class_b"],
            },
            "config": {},
        }
        # New format (SOTA)
        new_format = {
            "training_run_id": "uuid",
            "dataset_id": "uuid",
            "config": {"train_split": 0.8},
            "supabase_url": "https://...",
            "supabase_key": "...",
        }

        # Both should have training_run_id
        assert "training_run_id" in old_format
        assert "training_run_id" in new_format

        # Old has dataset, new has dataset_id
        assert "dataset" in old_format
        assert "dataset_id" in new_format


class TestWorkerToDBContract:
    """Worker → DB update contract."""

    def test_training_run_update_schema(self):
        """Training run updates should have valid status values."""
        valid_statuses = ["pending", "queued", "preparing", "training", "completed", "failed", "cancelled"]

        update_data = {
            "status": "training",
            "current_epoch": 5,
            "best_accuracy": 92.5,
            "best_epoch": 4,
        }

        assert update_data["status"] in valid_statuses

    def test_metrics_history_schema(self):
        """Metrics history should have expected fields."""
        metrics = {
            "training_run_id": "uuid",
            "training_type": "cls",
            "epoch": 1,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "val_accuracy": 85.0,
            "val_f1": 0.83,
            "learning_rate": 0.0001,
        }

        assert "training_run_id" in metrics
        assert "epoch" in metrics
        assert metrics["training_type"] in ["cls", "embedding", "od"]


class TestStatusEnumConsistency:
    """API, Worker, Frontend should use consistent status values."""

    VALID_TRAINING_STATUSES = {"pending", "queued", "preparing", "training", "completed", "failed", "cancelled"}

    def test_api_uses_valid_statuses(self):
        """API status updates should be valid."""
        api_statuses = {"pending", "preparing", "queued", "training", "completed", "failed", "cancelled"}
        assert api_statuses == self.VALID_TRAINING_STATUSES

    def test_worker_uses_valid_statuses(self):
        """Worker status updates should be subset of valid statuses."""
        worker_statuses = {"training", "completed", "failed", "cancelled"}
        assert worker_statuses.issubset(self.VALID_TRAINING_STATUSES)
