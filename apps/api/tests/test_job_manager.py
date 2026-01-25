"""
Unit tests for JobManager service.

Tests all 3 submission modes (ASYNC, SYNC, ASYNC_POLL) with mocked RunpodService.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.job_manager import (
    JobManager,
    JobMode,
    JobSubmissionError,
    JobTimeoutError,
    JobFailedError,
    JobCancelledError,
)
from services.runpod import EndpointType


class TestJobManager:
    """Test suite for JobManager."""

    @pytest.fixture
    def mock_runpod(self):
        """Create a mock RunpodService."""
        mock = MagicMock()
        mock.submit_job = AsyncMock()
        mock.submit_job_sync = AsyncMock()
        mock.submit_and_wait = AsyncMock()
        mock.cancel_job = AsyncMock()
        return mock

    @pytest.fixture
    def job_manager(self, mock_runpod):
        """Create JobManager with mocked RunpodService."""
        return JobManager(runpod_service=mock_runpod)

    # ==================== ASYNC MODE TESTS ====================

    @pytest.mark.asyncio
    async def test_async_mode_success(self, job_manager, mock_runpod):
        """Test ASYNC mode - successful job submission."""
        # Mock response
        mock_runpod.submit_job.return_value = {
            "id": "job-123",
            "status": "IN_QUEUE",
        }

        # Submit job
        result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data={"test": "data"},
            mode=JobMode.ASYNC,
            webhook_url="https://example.com/webhook",
        )

        # Verify
        assert result["job_id"] == "job-123"
        assert result["status"] == "submitted"
        assert result["result"] is None
        assert result["error"] is None

        # Verify RunpodService was called correctly
        mock_runpod.submit_job.assert_called_once_with(
            endpoint_type=EndpointType.EMBEDDING,
            input_data={"test": "data"},
            webhook_url="https://example.com/webhook",
        )

    @pytest.mark.asyncio
    async def test_async_mode_no_job_id(self, job_manager, mock_runpod):
        """Test ASYNC mode - missing job_id in response."""
        # Mock response without job_id
        mock_runpod.submit_job.return_value = {
            "status": "ERROR",
        }

        # Should raise JobSubmissionError
        with pytest.raises(JobSubmissionError, match="No job_id in response"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"test": "data"},
                mode=JobMode.ASYNC,
            )

    # ==================== SYNC MODE TESTS ====================

    @pytest.mark.asyncio
    async def test_sync_mode_success(self, job_manager, mock_runpod):
        """Test SYNC mode - successful job completion."""
        # Mock response
        mock_runpod.submit_job_sync.return_value = {
            "status": "COMPLETED",
            "output": {"result": "success", "embeddings": [1, 2, 3]},
        }

        # Submit job
        result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data={"image_url": "https://example.com/image.jpg"},
            mode=JobMode.SYNC,
            timeout=60,
        )

        # Verify
        assert result["job_id"] is None  # SYNC mode doesn't return job_id
        assert result["status"] == "completed"
        assert result["result"] == {"result": "success", "embeddings": [1, 2, 3]}
        assert result["error"] is None

        # Verify RunpodService was called correctly
        mock_runpod.submit_job_sync.assert_called_once_with(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data={"image_url": "https://example.com/image.jpg"},
            timeout=60,
        )

    @pytest.mark.asyncio
    async def test_sync_mode_job_failed(self, job_manager, mock_runpod):
        """Test SYNC mode - job fails on worker."""
        # Mock failed response
        mock_runpod.submit_job_sync.return_value = {
            "status": "FAILED",
            "error": "Out of memory",
        }

        # Should raise JobFailedError
        with pytest.raises(JobFailedError, match="Job failed: Out of memory"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.OD_ANNOTATION,
                input_data={"image_url": "https://example.com/image.jpg"},
                mode=JobMode.SYNC,
            )

    @pytest.mark.asyncio
    async def test_sync_mode_timeout(self, job_manager, mock_runpod):
        """Test SYNC mode - timeout error."""
        # Mock timeout
        mock_runpod.submit_job_sync.side_effect = TimeoutError("Request timed out after 60s")

        # Should raise JobTimeoutError
        with pytest.raises(JobTimeoutError, match="Request timed out after 60s"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.OD_ANNOTATION,
                input_data={"image_url": "https://example.com/image.jpg"},
                mode=JobMode.SYNC,
                timeout=60,
            )

    # ==================== ASYNC_POLL MODE TESTS ====================

    @pytest.mark.asyncio
    async def test_async_poll_mode_success(self, job_manager, mock_runpod):
        """Test ASYNC_POLL mode - successful job completion."""
        # Mock response
        mock_runpod.submit_and_wait.return_value = {
            "id": "job-456",
            "status": "COMPLETED",
            "output": {"processed": 100, "success": True},
        }

        # Submit job
        result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data={"batch_size": 50},
            mode=JobMode.ASYNC_POLL,
            timeout=300,
            poll_interval=1.0,
        )

        # Verify
        assert result["job_id"] == "job-456"
        assert result["status"] == "completed"
        assert result["result"] == {"processed": 100, "success": True}
        assert result["error"] is None

        # Verify RunpodService was called correctly
        mock_runpod.submit_and_wait.assert_called_once_with(
            endpoint_type=EndpointType.EMBEDDING,
            input_data={"batch_size": 50},
            timeout=300,
            poll_interval=1.0,
        )

    @pytest.mark.asyncio
    async def test_async_poll_mode_job_failed(self, job_manager, mock_runpod):
        """Test ASYNC_POLL mode - job fails on worker."""
        # Mock failed response
        mock_runpod.submit_and_wait.side_effect = RuntimeError("Job failed: Invalid input data")

        # Should raise JobFailedError
        with pytest.raises(JobFailedError, match="Job failed: Invalid input data"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"batch_size": 50},
                mode=JobMode.ASYNC_POLL,
            )

    @pytest.mark.asyncio
    async def test_async_poll_mode_cancelled(self, job_manager, mock_runpod):
        """Test ASYNC_POLL mode - job is cancelled."""
        # Mock cancelled response
        mock_runpod.submit_and_wait.side_effect = RuntimeError("Job was cancelled")

        # Should raise JobCancelledError
        with pytest.raises(JobCancelledError, match="Job was cancelled"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"batch_size": 50},
                mode=JobMode.ASYNC_POLL,
            )

    @pytest.mark.asyncio
    async def test_async_poll_mode_timeout(self, job_manager, mock_runpod):
        """Test ASYNC_POLL mode - timeout error."""
        # Mock timeout
        mock_runpod.submit_and_wait.side_effect = TimeoutError("Job job-789 timed out after 300s")

        # Should raise JobTimeoutError
        with pytest.raises(JobTimeoutError, match="Job job-789 timed out after 300s"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"batch_size": 50},
                mode=JobMode.ASYNC_POLL,
                timeout=300,
            )

    @pytest.mark.asyncio
    async def test_async_poll_mode_no_job_id(self, job_manager, mock_runpod):
        """Test ASYNC_POLL mode - missing job_id in response."""
        # Mock response without job_id
        mock_runpod.submit_and_wait.return_value = {
            "status": "COMPLETED",
            "output": {},
        }

        # Should raise JobSubmissionError
        with pytest.raises(JobSubmissionError, match="No job_id in response"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"batch_size": 50},
                mode=JobMode.ASYNC_POLL,
            )

    # ==================== CANCEL JOB TESTS ====================

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, job_manager, mock_runpod):
        """Test job cancellation."""
        # Mock response
        mock_runpod.cancel_job.return_value = {
            "id": "job-123",
            "status": "CANCELLED",
        }

        # Cancel job
        result = await job_manager.cancel_job(
            endpoint_type=EndpointType.EMBEDDING,
            job_id="job-123",
        )

        # Verify
        assert result["status"] == "CANCELLED"

        # Verify RunpodService was called
        mock_runpod.cancel_job.assert_called_once_with(
            endpoint_type=EndpointType.EMBEDDING,
            job_id="job-123",
        )

    @pytest.mark.asyncio
    async def test_cancel_job_no_job_id(self, job_manager, mock_runpod):
        """Test cancel job with None job_id."""
        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot cancel job without job_id"):
            await job_manager.cancel_job(
                endpoint_type=EndpointType.EMBEDDING,
                job_id=None,
            )

    # ==================== UTILITY METHOD TESTS ====================

    def test_is_cancellable_async(self, job_manager):
        """Test is_cancellable for ASYNC mode."""
        assert job_manager.is_cancellable(JobMode.ASYNC) is True

    def test_is_cancellable_async_poll(self, job_manager):
        """Test is_cancellable for ASYNC_POLL mode."""
        assert job_manager.is_cancellable(JobMode.ASYNC_POLL) is True

    def test_is_cancellable_sync(self, job_manager):
        """Test is_cancellable for SYNC mode."""
        assert job_manager.is_cancellable(JobMode.SYNC) is False

    # ==================== ERROR HANDLING TESTS ====================

    @pytest.mark.asyncio
    async def test_invalid_mode(self, job_manager):
        """Test with invalid job mode."""
        with pytest.raises(ValueError, match="Invalid job mode"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"test": "data"},
                mode="invalid_mode",
            )

    @pytest.mark.asyncio
    async def test_unexpected_error_wrapped(self, job_manager, mock_runpod):
        """Test that unexpected errors are wrapped in JobSubmissionError."""
        # Mock unexpected error
        mock_runpod.submit_job.side_effect = Exception("Unexpected network error")

        # Should wrap in JobSubmissionError
        with pytest.raises(JobSubmissionError, match="Failed to submit job: Unexpected network error"):
            await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={"test": "data"},
                mode=JobMode.ASYNC,
            )


# ==================== INTEGRATION TESTS ====================

class TestJobManagerIntegration:
    """
    Integration tests with real RunpodService (requires mocking at httpx level).

    These tests are skipped by default. To run:
    pytest tests/test_job_manager.py -k integration -v
    """

    @pytest.mark.skip(reason="Integration test - requires actual Runpod endpoint")
    @pytest.mark.asyncio
    async def test_real_async_submission(self):
        """Test real ASYNC job submission (manual test)."""
        from services.job_manager import job_manager

        result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data={
                "task": "detect",
                "model": "grounding_dino",
                "image_url": "https://example.com/test.jpg",
                "text_prompt": "cat",
            },
            mode=JobMode.ASYNC,
        )

        assert result["job_id"] is not None
        assert result["status"] == "submitted"
        print(f"Job submitted: {result['job_id']}")

    @pytest.mark.skip(reason="Integration test - requires actual Runpod endpoint")
    @pytest.mark.asyncio
    async def test_real_sync_submission(self):
        """Test real SYNC job submission (manual test)."""
        from services.job_manager import job_manager

        result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.OD_ANNOTATION,
            input_data={
                "task": "detect",
                "model": "grounding_dino",
                "image_url": "https://example.com/test.jpg",
                "text_prompt": "cat",
            },
            mode=JobMode.SYNC,
            timeout=60,
        )

        assert result["job_id"] is None
        assert result["status"] == "completed"
        assert result["result"] is not None
        print(f"Job completed: {result['result']}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
