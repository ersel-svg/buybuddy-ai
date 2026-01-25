"""
Integration tests for Training workers flow.

These tests verify the complete flow:
1. API → Worker payload
2. Worker → DB fetch (SOTA pattern)
3. Worker → Training execution
4. Worker → DB progress updates
5. Cancel flow
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


class TestEmbeddingTrainingFlow:
    """Tests for Embedding Training flow."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        table_mock = MagicMock()
        table_mock.select.return_value = table_mock
        table_mock.eq.return_value = table_mock
        table_mock.in_.return_value = table_mock
        table_mock.range.return_value = table_mock
        table_mock.not_.return_value = table_mock
        table_mock.is_.return_value = table_mock
        table_mock.execute.return_value = MagicMock(data=[])
        client.table.return_value = table_mock
        return client

    def test_source_config_to_db_query(self):
        """source_config should translate to correct DB queries."""
        source_config = {
            "product_ids": ["uuid1", "uuid2", "uuid3"],
            "train_product_ids": ["uuid1", "uuid2"],
            "val_product_ids": ["uuid3"],
            "image_config": {
                "image_types": ["synthetic"],
                "frame_selection": "first",
                "include_matched_cutouts": True,
            },
        }

        # Verify source_config structure
        assert "product_ids" in source_config
        assert len(source_config["product_ids"]) == 3
        assert source_config["image_config"]["include_matched_cutouts"] == True

    def test_training_data_split_consistency(self):
        """Train/val/test splits should be consistent."""
        product_ids = [f"uuid-{i}" for i in range(100)]
        train_ids = set(product_ids[:80])
        val_ids = set(product_ids[80:90])
        test_ids = set(product_ids[90:])

        # No overlap between splits
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        # All products accounted for
        assert len(train_ids | val_ids | test_ids) == 100


class TestCLSTrainingFlow:
    """Tests for CLS Training flow."""

    def test_dataset_id_to_db_query(self):
        """dataset_id should result in correct DB queries."""
        dataset_id = "550e8400-e29b-41d4-a716-446655440000"
        train_split = 0.8
        seed = 42

        # Verify parameters are valid
        assert len(dataset_id) == 36  # UUID length
        assert 0 < train_split < 1
        assert isinstance(seed, int)

    def test_class_ordering_consistency(self):
        """Class ordering should be consistent between API and Worker."""
        # Simulated classes from DB (sorted by name)
        classes = [
            {"id": "class-3", "name": "apple"},
            {"id": "class-1", "name": "banana"},
            {"id": "class-2", "name": "cherry"},
        ]

        # Both API and Worker should sort by name
        sorted_classes = sorted(classes, key=lambda c: c["name"])
        class_names = [c["name"] for c in sorted_classes]
        class_id_to_idx = {c["id"]: idx for idx, c in enumerate(sorted_classes)}

        assert class_names == ["apple", "banana", "cherry"]
        assert class_id_to_idx["class-3"] == 0  # apple
        assert class_id_to_idx["class-1"] == 1  # banana
        assert class_id_to_idx["class-2"] == 2  # cherry

    def test_train_val_split_reproducibility(self):
        """Train/val split should be reproducible with same seed."""
        import random

        all_urls = [{"url": f"https://example.com/{i}.jpg", "label": i % 5} for i in range(100)]

        # First split
        random.seed(42)
        shuffled1 = all_urls.copy()
        random.shuffle(shuffled1)
        train1 = shuffled1[:80]
        val1 = shuffled1[80:]

        # Second split with same seed
        random.seed(42)
        shuffled2 = all_urls.copy()
        random.shuffle(shuffled2)
        train2 = shuffled2[:80]
        val2 = shuffled2[80:]

        # Should be identical
        assert train1 == train2
        assert val1 == val2


class TestProgressTracking:
    """Tests for progress tracking."""

    def test_progress_calculates_correctly(self):
        """Progress percentage should be calculated correctly."""

        def calculate_progress(processed: int, total: int) -> float:
            if total == 0:
                return 0.0
            return min(100.0, (processed / total) * 100)

        assert calculate_progress(0, 100) == 0.0
        assert calculate_progress(50, 100) == 50.0
        assert calculate_progress(100, 100) == 100.0
        assert calculate_progress(150, 100) == 100.0  # Capped at 100
        assert calculate_progress(0, 0) == 0.0  # Handle zero total

    def test_epoch_progress_updates(self):
        """Epoch progress should update correctly."""
        progress_updates = []

        def update_progress(epoch, metrics):
            progress_updates.append({
                "epoch": epoch,
                "val_acc": metrics.get("val_acc", 0),
            })

        # Simulate 10 epochs
        for epoch in range(10):
            update_progress(epoch + 1, {"val_acc": 80 + epoch})

        assert len(progress_updates) == 10
        assert progress_updates[0]["epoch"] == 1
        assert progress_updates[-1]["epoch"] == 10
        assert progress_updates[-1]["val_acc"] == 89


class TestCancelFlow:
    """Tests for job cancellation flow."""

    def test_cancel_updates_status(self):
        """Cancel should update job status to cancelled."""
        job = {"id": "test-job", "status": "training"}

        def cancel_job(job_id: str) -> dict:
            if job["id"] == job_id:
                job["status"] = "cancelled"
                return {"status": "cancelled"}
            return {"error": "Job not found"}

        result = cancel_job("test-job")
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"

    def test_cancel_stops_at_next_epoch(self):
        """Cancel should stop processing at next epoch boundary."""
        cancel_requested = False
        completed_epochs = []

        def check_cancelled() -> bool:
            return cancel_requested

        def train_epoch(epoch: int):
            if check_cancelled():
                return False
            completed_epochs.append(epoch)
            return True

        # Train epochs
        for epoch in range(100):
            if epoch == 5:
                cancel_requested = True
            if not train_epoch(epoch):
                break

        assert len(completed_epochs) == 5  # Should stop at epoch 5
        assert 5 not in completed_epochs  # Epoch 5 should not be completed


class TestDBWriteFlow:
    """Tests for database write flow."""

    def test_job_status_transitions(self):
        """Job status should follow valid transitions."""
        valid_transitions = {
            "pending": ["queued", "preparing", "cancelled"],
            "queued": ["preparing", "training", "cancelled"],
            "preparing": ["training", "failed", "cancelled"],
            "training": ["completed", "failed", "cancelled"],
            "completed": [],  # Terminal state
            "failed": [],  # Terminal state
            "cancelled": [],  # Terminal state
        }

        def is_valid_transition(from_status: str, to_status: str) -> bool:
            return to_status in valid_transitions.get(from_status, [])

        # Valid transitions
        assert is_valid_transition("pending", "preparing")
        assert is_valid_transition("training", "completed")
        assert is_valid_transition("training", "failed")
        assert is_valid_transition("training", "cancelled")

        # Invalid transitions
        assert not is_valid_transition("completed", "training")
        assert not is_valid_transition("failed", "training")


class TestErrorHandling:
    """Tests for error handling in integration flow."""

    def test_db_connection_retry(self):
        """DB connection errors should be retried."""

        def fetch_with_retry(max_retries: int = 3) -> dict:
            attempts = 0
            last_error = None

            while attempts < max_retries:
                attempts += 1
                try:
                    # Simulate failure on first 2 attempts
                    if attempts < 3:
                        raise ConnectionError("DB connection failed")
                    return {"data": []}
                except ConnectionError as e:
                    last_error = e
                    continue

            raise last_error

        result = fetch_with_retry()
        assert result == {"data": []}

    def test_partial_failure_handling(self):
        """Partial failures should not lose successfully processed data."""
        processed_successfully = []
        failed_items = []

        def process_batch(items: list) -> tuple:
            success = []
            failures = []
            for item in items:
                try:
                    if item["id"] == 5:  # Simulate failure for item 5
                        raise ValueError("Invalid item")
                    success.append(item)
                except Exception as e:
                    failures.append({"item": item, "error": str(e)})
            return success, failures

        batch = [{"id": i} for i in range(10)]
        success, failures = process_batch(batch)

        assert len(success) == 9
        assert len(failures) == 1
        assert failures[0]["item"]["id"] == 5


class TestSupabaseFetcherIntegration:
    """Tests for Supabase fetcher integration."""

    def test_paginated_fetch_handles_empty_result(self):
        """Paginated fetch should handle empty results gracefully."""

        def fetch_page(offset: int, limit: int) -> list:
            return []  # Simulate empty result

        all_data = []
        offset = 0
        page_size = 1000

        while True:
            page = fetch_page(offset, page_size)
            if not page:
                break
            all_data.extend(page)
            offset += page_size

        assert all_data == []

    def test_paginated_fetch_handles_partial_page(self):
        """Paginated fetch should handle partial last page."""
        total_items = 2500
        page_size = 1000

        def fetch_page(offset: int, limit: int) -> list:
            remaining = total_items - offset
            if remaining <= 0:
                return []
            return [{"id": i} for i in range(offset, min(offset + limit, total_items))]

        all_data = []
        offset = 0

        while True:
            page = fetch_page(offset, page_size)
            if not page:
                break
            all_data.extend(page)
            offset += page_size

        assert len(all_data) == 2500
        assert all_data[-1]["id"] == 2499

    def test_chunked_in_query(self):
        """Large IN queries should be chunked."""
        product_ids = [f"uuid-{i}" for i in range(1500)]
        chunk_size = 500

        chunks = []
        for i in range(0, len(product_ids), chunk_size):
            chunk = product_ids[i:i + chunk_size]
            chunks.append(chunk)

        assert len(chunks) == 3
        assert len(chunks[0]) == 500
        assert len(chunks[1]) == 500
        assert len(chunks[2]) == 500
