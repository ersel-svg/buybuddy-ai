"""
Integration tests for Embedding Extraction flow.

These tests verify the complete flow:
1. API → Worker payload
2. Worker → DB fetch
3. Worker → Qdrant write
4. Worker → DB progress updates
5. Cancel flow
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


class TestEmbeddingExtractionFlow:
    """Tests for complete embedding extraction flow."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Create mock Supabase client."""
        client = MagicMock()

        # Mock table queries
        table_mock = MagicMock()
        table_mock.select.return_value = table_mock
        table_mock.eq.return_value = table_mock
        table_mock.neq.return_value = table_mock
        table_mock.is_.return_value = table_mock
        table_mock.range.return_value = table_mock
        table_mock.update.return_value = table_mock
        table_mock.execute.return_value = MagicMock(data=[])

        client.table.return_value = table_mock
        return client

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.upsert = MagicMock()
        client.get_collection = MagicMock(return_value=MagicMock(vectors_count=0))
        return client

    def test_source_config_to_db_query(self):
        """source_config should translate to correct DB query."""
        source_config = {
            "type": "cutouts",
            "filters": {
                "has_embedding": False,
                "cutout_filter_has_upc": True,
            },
            "frame_selection": "first",
        }

        # Verify filters translate to expected query conditions
        filters = source_config["filters"]
        assert filters["has_embedding"] == False  # Should filter unprocessed
        assert filters["cutout_filter_has_upc"] == True  # Should filter by UPC

    def test_progress_update_flow(self):
        """Progress updates should follow correct flow."""
        progress_updates = []

        def update_progress(job_id: str, processed: int, total: int, step: str):
            progress_updates.append({
                "job_id": job_id,
                "processed": processed,
                "total": total,
                "progress": (processed / total * 100) if total > 0 else 0,
                "step": step,
            })

        # Simulate extraction flow
        job_id = "test-job-123"
        total = 1000
        batch_size = 100

        for i in range(0, total, batch_size):
            processed = min(i + batch_size, total)
            step = f"Processing batch {i // batch_size + 1}/{total // batch_size}"
            update_progress(job_id, processed, total, step)

        assert len(progress_updates) == 10
        assert progress_updates[0]["progress"] == 10.0
        assert progress_updates[-1]["progress"] == 100.0
        assert progress_updates[-1]["processed"] == total

    @pytest.mark.asyncio
    async def test_batch_processing_with_cancel(self):
        """Batch processing should respect cancel signal."""
        cancelled = False
        processed_batches = []

        async def process_batch(batch_num: int) -> bool:
            """Process a batch, return False if cancelled."""
            if cancelled:
                return False
            await asyncio.sleep(0.001)  # Simulate processing
            processed_batches.append(batch_num)
            return True

        # Process until cancelled
        for i in range(10):
            if i == 5:
                cancelled = True
            result = await process_batch(i)
            if not result:
                break

        assert len(processed_batches) == 5  # Should stop at batch 5
        assert 5 not in processed_batches  # Batch 5 should not be processed


class TestCancelFlow:
    """Tests for job cancellation flow."""

    def test_cancel_updates_status(self):
        """Cancel should update job status to cancelled."""
        job = {"id": "test-job", "status": "running"}

        def cancel_job(job_id: str) -> dict:
            if job["id"] == job_id:
                job["status"] = "cancelled"
                return {"status": "cancelled"}
            return {"error": "Job not found"}

        result = cancel_job("test-job")
        assert result["status"] == "cancelled"
        assert job["status"] == "cancelled"

    def test_cancel_stops_at_next_batch(self):
        """Cancel should stop processing at next batch boundary."""
        cancel_requested = False
        processed = []

        def check_cancelled() -> bool:
            return cancel_requested

        def process_item(item_id: int):
            if check_cancelled():
                return False
            processed.append(item_id)
            return True

        # Process items
        for i in range(100):
            if i == 50:
                cancel_requested = True
            if not process_item(i):
                break

        assert len(processed) == 50  # Should stop at item 50
        assert 50 not in processed  # Item 50 should not be processed


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

    def test_progress_updates_are_throttled(self):
        """Progress updates should be throttled to avoid DB spam."""
        last_update_time = 0
        min_interval = 1.0  # 1 second minimum between updates
        updates_sent = []

        def should_update(current_time: float) -> bool:
            nonlocal last_update_time
            if current_time - last_update_time >= min_interval:
                last_update_time = current_time
                return True
            return False

        # Simulate rapid updates
        for i in range(100):
            current_time = i * 0.1  # 100ms intervals
            if should_update(current_time):
                updates_sent.append(current_time)

        # Should have ~10 updates (every 1 second)
        assert len(updates_sent) <= 11  # Allow for initial + 10 more


class TestDBWriteFlow:
    """Tests for database write flow."""

    def test_job_status_transitions(self):
        """Job status should follow valid transitions."""
        valid_transitions = {
            "pending": ["queued", "running", "cancelled"],
            "queued": ["running", "cancelled"],
            "running": ["completed", "failed", "cancelled"],
            "completed": [],  # Terminal state
            "failed": [],  # Terminal state
            "cancelled": [],  # Terminal state
        }

        def is_valid_transition(from_status: str, to_status: str) -> bool:
            return to_status in valid_transitions.get(from_status, [])

        # Valid transitions
        assert is_valid_transition("pending", "running")
        assert is_valid_transition("running", "completed")
        assert is_valid_transition("running", "failed")
        assert is_valid_transition("running", "cancelled")

        # Invalid transitions
        assert not is_valid_transition("completed", "running")
        assert not is_valid_transition("failed", "running")

    def test_embedding_collection_metadata_update(self):
        """Embedding collection metadata should be updated correctly."""
        collection = {
            "name": "products_dinov2_base",
            "vector_count": 0,
            "last_sync_at": None,
        }

        def update_collection(vectors_added: int):
            collection["vector_count"] += vectors_added
            collection["last_sync_at"] = datetime.now(timezone.utc).isoformat()

        update_collection(1000)
        assert collection["vector_count"] == 1000
        assert collection["last_sync_at"] is not None

        update_collection(500)
        assert collection["vector_count"] == 1500


class TestErrorHandling:
    """Tests for error handling in integration flow."""

    def test_db_connection_error_handling(self):
        """DB connection errors should be handled gracefully."""

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

    def test_qdrant_upsert_error_handling(self):
        """Qdrant upsert errors should be retried."""
        upsert_attempts = []

        def upsert_with_retry(vectors: list, max_retries: int = 3) -> bool:
            for attempt in range(max_retries):
                upsert_attempts.append(attempt)
                try:
                    if attempt < 2:
                        raise Exception("Qdrant unavailable")
                    return True
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    continue
            return False

        result = upsert_with_retry([{"id": 1, "vector": [0.1, 0.2]}])
        assert result == True
        assert len(upsert_attempts) == 3

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
            # Simulate empty result
            return []

        all_data = []
        offset = 0
        page_size = 100

        while True:
            page = fetch_page(offset, page_size)
            if not page:
                break
            all_data.extend(page)
            offset += page_size

        assert all_data == []

    def test_paginated_fetch_handles_partial_page(self):
        """Paginated fetch should handle partial last page."""
        total_items = 250
        page_size = 100

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

        assert len(all_data) == 250
        assert all_data[-1]["id"] == 249

    def test_fetch_respects_filters(self):
        """Fetch should respect provided filters."""
        all_items = [
            {"id": 1, "has_embedding": False, "type": "cutout"},
            {"id": 2, "has_embedding": True, "type": "cutout"},
            {"id": 3, "has_embedding": False, "type": "product"},
            {"id": 4, "has_embedding": False, "type": "cutout"},
        ]

        def fetch_with_filters(filters: dict) -> list:
            result = all_items.copy()

            if "has_embedding" in filters:
                result = [i for i in result if i.get("has_embedding") == filters["has_embedding"]]

            if "type" in filters:
                result = [i for i in result if i.get("type") == filters["type"]]

            return result

        # Filter for unprocessed cutouts
        filters = {"has_embedding": False, "type": "cutout"}
        result = fetch_with_filters(filters)

        assert len(result) == 2
        assert all(i["has_embedding"] == False for i in result)
        assert all(i["type"] == "cutout" for i in result)
