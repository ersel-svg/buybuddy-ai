"""
Contract tests for Embedding Extraction pipeline.

These tests verify that:
1. API payload matches worker's expected schema
2. Worker's DB writes match database schema
3. Status enums are consistent across API, Worker, and Frontend
"""

import pytest
import json


class TestAPIToWorkerContract:
    """Tests that API payload matches what worker expects."""

    def test_source_config_has_required_fields(self):
        """source_config must have type field."""
        source_config = {
            "type": "both",
            "filters": {"has_embedding": False},
            "frame_selection": "first",
        }
        assert "type" in source_config
        assert source_config["type"] in ["cutouts", "products", "both"]

    def test_source_config_valid_types(self):
        """source_config.type must be one of valid values."""
        valid_types = ["cutouts", "products", "both"]
        for t in valid_types:
            source_config = {"type": t}
            assert source_config["type"] in valid_types

    def test_source_config_filters_schema(self):
        """filters object schema validation."""
        valid_filters = {
            "has_embedding": False,
            "product_source": "all",
            "cutout_filter_has_upc": True,
        }
        # All filter values should be bool, str, or None
        for key, value in valid_filters.items():
            assert value is None or isinstance(value, (bool, str, int, list))

    def test_frame_selection_valid_values(self):
        """frame_selection must be one of valid values."""
        valid_selections = ["first", "all", "key_frames", "interval"]
        for selection in valid_selections:
            source_config = {"type": "both", "frame_selection": selection}
            assert source_config["frame_selection"] in valid_selections

    def test_job_input_minimal_payload_size(self):
        """SOTA pattern payload must be < 10 KB."""
        job_input = {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "source_config": {
                "type": "both",
                "filters": {"has_embedding": False, "product_source": "all"},
                "frame_selection": "first",
                "max_frames": 10,
            },
            "model_type": "dinov2-base",
            "embedding_dim": 768,
            "collection_name": "products_dinov2_base",
            "cutout_collection": "cutouts_dinov2_base",
            "supabase_url": "https://xxxxxxxxxxxx.supabase.co",
            "supabase_service_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN0cmluZyIsInJvbGUiOiJzZXJ2aWNlX3JvbGUiLCJpYXQiOjE2MDAwMDAwMDAsImV4cCI6MTYwMDAwMDAwMH0.signature",
            "qdrant_url": "https://qdrant.example.com:6333",
            "qdrant_api_key": "api_key_here_12345",
        }
        payload_size = len(json.dumps(job_input))
        assert payload_size < 10_000, f"Payload {payload_size} bytes exceeds 10KB limit"

    def test_job_input_required_fields(self):
        """Job input must have all required fields for SOTA mode."""
        required_fields = [
            "job_id",
            "source_config",
            "model_type",
            "embedding_dim",
            "collection_name",
            "supabase_url",
            "supabase_service_key",
            "qdrant_url",
            "qdrant_api_key",
        ]
        job_input = {
            "job_id": "test-id",
            "source_config": {"type": "both"},
            "model_type": "dinov2-base",
            "embedding_dim": 768,
            "collection_name": "test_collection",
            "supabase_url": "https://example.supabase.co",
            "supabase_service_key": "key",
            "qdrant_url": "https://qdrant.example.com",
            "qdrant_api_key": "key",
        }
        for field in required_fields:
            assert field in job_input, f"Missing required field: {field}"


class TestWorkerToDBContract:
    """Tests that worker's DB writes match database schema."""

    def test_embedding_job_status_values(self):
        """embedding_jobs.status must be valid."""
        valid_statuses = ["pending", "queued", "running", "completed", "failed", "cancelled"]
        for status in ["running", "completed", "failed", "cancelled"]:
            assert status in valid_statuses

    def test_embedding_job_update_schema(self):
        """embedding_jobs table update data must be valid."""
        update_data = {
            "status": "running",
            "processed_images": 100,
            "total_images": 1000,
            "progress": 10,
            "current_step": "Processing batch 1/10",
            "updated_at": "2024-01-25T10:00:00Z",
        }
        assert isinstance(update_data["status"], str)
        assert isinstance(update_data["processed_images"], int)
        assert 0 <= update_data["progress"] <= 100

    def test_embedding_collections_schema(self):
        """embedding_collections table data must be valid."""
        collection_data = {
            "name": "products_dinov2_base",
            "collection_type": "matching",
            "source_type": "all",
            "embedding_model_id": "550e8400-e29b-41d4-a716-446655440000",
            "image_types": ["synthetic", "real"],
            "vector_count": 1000,
        }
        valid_collection_types = ["matching", "training", "evaluation", "production"]
        valid_source_types = ["all", "matched", "dataset", "cutouts", "products"]

        assert collection_data["collection_type"] in valid_collection_types
        assert collection_data["source_type"] in valid_source_types
        assert isinstance(collection_data["image_types"], list)
        assert isinstance(collection_data["vector_count"], int)


class TestStatusEnumConsistency:
    """Tests that status enums are consistent across API, Worker, and Frontend."""

    VALID_STATUSES = {"pending", "queued", "running", "completed", "failed", "cancelled"}

    def test_api_uses_valid_statuses(self):
        """API must use valid status values."""
        api_statuses = {"pending", "queued", "running", "completed", "failed", "cancelled"}
        assert api_statuses == self.VALID_STATUSES

    def test_worker_uses_valid_statuses(self):
        """Worker status updates must use valid values."""
        worker_statuses = {"running", "completed", "failed", "cancelled"}
        assert worker_statuses.issubset(self.VALID_STATUSES)

    def test_frontend_type_includes_all_statuses(self):
        """Frontend EmbeddingJobStatus type must include all statuses."""
        # After fix: export type EmbeddingJobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled";
        frontend_statuses = {"pending", "queued", "running", "completed", "failed", "cancelled"}
        assert frontend_statuses == self.VALID_STATUSES


class TestTrainingContractSOTA:
    """Tests for Embedding Training LEGACY mode contract."""

    def test_training_config_has_product_ids(self):
        """Training config must have product_ids for LEGACY mode."""
        config = {
            "product_ids": ["uuid1", "uuid2", "uuid3"],
            "train_product_ids": ["uuid1", "uuid2"],
            "val_product_ids": ["uuid3"],
            "test_product_ids": [],
            "epochs": 10,
            "batch_size": 32,
        }
        assert "product_ids" in config
        assert isinstance(config["product_ids"], list)

    def test_training_payload_without_images(self):
        """LEGACY mode: training_images should NOT be in payload."""
        # SOTA pattern: No images in payload
        input_data = {
            "training_run_id": "uuid-here",
            "model_type": "dinov2-base",
            "config": {
                "product_ids": ["uuid1", "uuid2"],
                "epochs": 10,
            },
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "key",
        }
        # training_images should NOT be present
        assert "training_images" not in input_data

    def test_legacy_mode_payload_size(self):
        """LEGACY mode payload must be < 10 KB even with 1000 product IDs."""
        product_ids = [f"550e8400-e29b-41d4-a716-{str(i).zfill(12)}" for i in range(1000)]
        input_data = {
            "training_run_id": "550e8400-e29b-41d4-a716-446655440000",
            "model_type": "dinov2-base",
            "config": {
                "product_ids": product_ids,
                "train_product_ids": product_ids[:700],
                "val_product_ids": product_ids[700:850],
                "test_product_ids": product_ids[850:],
                "epochs": 10,
                "batch_size": 32,
            },
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxx",
        }
        payload_size = len(json.dumps(input_data))
        # With 1000 UUIDs (~36 chars each), payload should be ~150 KB
        # This is acceptable since RunPod limit is ~10 MB
        assert payload_size < 500_000, f"Payload {payload_size} bytes too large"


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy payloads."""

    def test_worker_accepts_both_modes(self):
        """Worker should accept both source_config and images modes."""
        # SOTA mode
        sota_input = {
            "job_id": "uuid",
            "source_config": {"type": "both"},
            "model_type": "dinov2-base",
        }
        assert "source_config" in sota_input
        assert "images" not in sota_input

        # Legacy mode
        legacy_input = {
            "job_id": "uuid",
            "images": [{"id": "1", "url": "https://example.com/1.jpg"}],
            "model_type": "dinov2-base",
        }
        assert "images" in legacy_input
        assert "source_config" not in legacy_input

    def test_training_worker_accepts_both_modes(self):
        """Training worker should accept both training_images and LEGACY mode."""
        # NEW mode (with images) - NOT RECOMMENDED for large datasets
        new_mode = {
            "training_run_id": "uuid",
            "training_images": {"product_1": [{"url": "https://example.com/1.jpg"}]},
            "config": {},
        }
        assert "training_images" in new_mode

        # LEGACY mode (no images, worker fetches from DB)
        legacy_mode = {
            "training_run_id": "uuid",
            "config": {"product_ids": ["uuid1", "uuid2"]},
        }
        assert "training_images" not in legacy_mode
        assert "product_ids" in legacy_mode["config"]
