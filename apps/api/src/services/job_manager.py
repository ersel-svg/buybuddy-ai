"""
JobManager Service - Unified Runpod job submission with consistent tracking.

This service provides a standardized interface for submitting jobs to Runpod
with proper job ID tracking and cancellation support.

Features:
- 3 submission modes: ASYNC, SYNC, ASYNC_POLL
- Consistent response format across all modes
- Proper job ID tracking for cancellation
- Error handling and timeout management
- Type-safe with full type hints

Usage:
    from services.job_manager import job_manager, JobMode

    # Async mode - for long-running jobs with webhooks
    result = await job_manager.submit_runpod_job(
        endpoint_type=EndpointType.TRAINING,
        input_data={"...": "..."},
        mode=JobMode.ASYNC,
    )
    # Returns: {"job_id": "xyz", "status": "submitted", "result": None}

    # Sync mode - for quick operations (avoid for long jobs!)
    result = await job_manager.submit_runpod_job(
        endpoint_type=EndpointType.OD_ANNOTATION,
        input_data={"...": "..."},
        mode=JobMode.SYNC,
        timeout=60,
    )
    # Returns: {"job_id": None, "status": "completed", "result": {...}}

    # Async poll mode - for reliable long-running jobs
    result = await job_manager.submit_runpod_job(
        endpoint_type=EndpointType.EMBEDDING,
        input_data={"...": "..."},
        mode=JobMode.ASYNC_POLL,
        timeout=300,
    )
    # Returns: {"job_id": "xyz", "status": "completed", "result": {...}}
"""

import asyncio
from enum import Enum
from typing import Any, Optional

from services.runpod import RunpodService, EndpointType


class JobMode(str, Enum):
    """
    Runpod job submission modes.

    ASYNC:
        - Uses Runpod /run endpoint
        - Returns immediately with job_id
        - Job runs in background
        - Best for: Long-running jobs with webhooks (training, video)
        - Cancellable: YES (via job_id)

    SYNC:
        - Uses Runpod /runsync endpoint
        - Waits for completion before returning
        - NO job_id returned
        - Best for: Quick operations (<30s, e.g., single image prediction)
        - Cancellable: NO (no job_id to cancel)
        - ⚠️ WARNING: Avoid for long jobs (>60s) - server timeout risk

    ASYNC_POLL:
        - Uses /run endpoint + client-side polling
        - Returns job_id AND waits for completion
        - Best for: Reliable long-running jobs without webhooks (embedding)
        - Cancellable: YES (via job_id)
        - More reliable than SYNC for long jobs
    """
    ASYNC = "async"
    SYNC = "sync"
    ASYNC_POLL = "async_poll"


class JobSubmissionError(Exception):
    """Raised when job submission fails."""
    pass


class JobTimeoutError(Exception):
    """Raised when job exceeds timeout."""
    pass


class JobFailedError(Exception):
    """Raised when job fails on worker."""
    pass


class JobCancelledError(Exception):
    """Raised when job is cancelled."""
    pass


class JobManager:
    """
    Unified job submission manager for Runpod.

    Provides consistent interface across all Runpod endpoint types
    with proper job tracking and error handling.
    """

    def __init__(self, runpod_service: Optional[RunpodService] = None):
        """
        Initialize JobManager.

        Args:
            runpod_service: Optional RunpodService instance (for testing/DI)
        """
        # Import singleton to avoid circular dependency
        if runpod_service is None:
            from services.runpod import runpod_service as default_service
            self.runpod = default_service
        else:
            self.runpod = runpod_service

    async def submit_runpod_job(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        mode: JobMode = JobMode.ASYNC,
        timeout: int = 300,
        poll_interval: float = 2.0,
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Submit a job to Runpod with specified mode.

        Args:
            endpoint_type: Runpod endpoint type
            input_data: Job input payload
            mode: Submission mode (ASYNC, SYNC, ASYNC_POLL)
            timeout: Maximum wait time in seconds (for SYNC/ASYNC_POLL)
            poll_interval: Polling interval in seconds (for ASYNC_POLL)
            webhook_url: Optional webhook URL (for ASYNC mode)

        Returns:
            Standardized response:
            {
                "job_id": str | None,  # Runpod job ID (None for SYNC)
                "status": str,         # "submitted", "completed", "failed"
                "result": dict | None, # Output data (only for SYNC/ASYNC_POLL)
                "error": str | None,   # Error message if failed
            }

        Raises:
            JobSubmissionError: If job submission fails
            JobTimeoutError: If job exceeds timeout (SYNC/ASYNC_POLL)
            JobFailedError: If job fails on worker
            JobCancelledError: If job is cancelled (ASYNC_POLL)
        """
        try:
            if mode == JobMode.ASYNC:
                return await self._submit_async(
                    endpoint_type, input_data, webhook_url
                )
            elif mode == JobMode.SYNC:
                return await self._submit_sync(
                    endpoint_type, input_data, timeout
                )
            elif mode == JobMode.ASYNC_POLL:
                return await self._submit_async_poll(
                    endpoint_type, input_data, timeout, poll_interval
                )
            else:
                raise ValueError(f"Invalid job mode: {mode}")

        except (JobSubmissionError, JobTimeoutError, JobFailedError, JobCancelledError):
            # Re-raise our custom exceptions
            raise
        except (ValueError, TypeError) as e:
            # Re-raise input validation errors as-is
            raise
        except TimeoutError as e:
            # Convert standard TimeoutError to our custom one
            raise JobTimeoutError(str(e)) from e
        except RuntimeError as e:
            # Convert RuntimeError (from worker failures) to our custom one
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                raise JobCancelledError(error_msg) from e
            else:
                raise JobFailedError(error_msg) from e
        except Exception as e:
            # Wrap unexpected errors
            raise JobSubmissionError(f"Failed to submit job: {str(e)}") from e

    async def _submit_async(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        webhook_url: Optional[str],
    ) -> dict[str, Any]:
        """
        Submit job in ASYNC mode (fire and forget with webhook).

        Returns immediately with job_id. Job runs in background.
        Worker should call webhook on completion.
        """
        response = await self.runpod.submit_job(
            endpoint_type=endpoint_type,
            input_data=input_data,
            webhook_url=webhook_url,
        )

        job_id = response.get("id")
        if not job_id:
            raise JobSubmissionError(f"No job_id in response: {response}")

        return {
            "job_id": job_id,
            "status": "submitted",
            "result": None,
            "error": None,
        }

    async def _submit_sync(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any]:
        """
        Submit job in SYNC mode (wait for result, no job_id).

        Waits for completion before returning.
        ⚠️ WARNING: No job_id means job is NOT cancellable!
        """
        response = await self.runpod.submit_job_sync(
            endpoint_type=endpoint_type,
            input_data=input_data,
            timeout=timeout,
        )

        # Check for error in response
        if response.get("status") == "FAILED":
            error_msg = response.get("error", "Unknown error")
            raise JobFailedError(f"Job failed: {error_msg}")

        # Extract output
        output = response.get("output")

        return {
            "job_id": None,  # SYNC mode doesn't return job_id
            "status": "completed",
            "result": output,
            "error": None,
        }

    async def _submit_async_poll(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        timeout: int,
        poll_interval: float,
    ) -> dict[str, Any]:
        """
        Submit job in ASYNC_POLL mode (get job_id + wait for result).

        More reliable than SYNC for long jobs.
        Returns job_id so job can be cancelled if needed.
        """
        response = await self.runpod.submit_and_wait(
            endpoint_type=endpoint_type,
            input_data=input_data,
            timeout=timeout,
            poll_interval=poll_interval,
        )

        job_id = response.get("id")
        status = response.get("status")
        output = response.get("output")

        if not job_id:
            raise JobSubmissionError(f"No job_id in response: {response}")

        # Check final status
        if status == "FAILED":
            error_msg = response.get("error", "Unknown error")
            raise JobFailedError(f"Job {job_id} failed: {error_msg}")

        if status == "CANCELLED":
            raise JobCancelledError(f"Job {job_id} was cancelled")

        return {
            "job_id": job_id,
            "status": "completed",
            "result": output,
            "error": None,
        }

    async def cancel_job(
        self,
        endpoint_type: EndpointType,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Cancel a running Runpod job.

        Args:
            endpoint_type: Runpod endpoint type
            job_id: Runpod job ID to cancel

        Returns:
            Cancellation response from Runpod

        Raises:
            ValueError: If job_id is None (can't cancel SYNC jobs)
        """
        if not job_id:
            raise ValueError("Cannot cancel job without job_id (SYNC mode jobs are not cancellable)")

        return await self.runpod.cancel_job(
            endpoint_type=endpoint_type,
            job_id=job_id,
        )

    def is_cancellable(self, mode: JobMode) -> bool:
        """
        Check if a job mode is cancellable.

        Args:
            mode: Job submission mode

        Returns:
            True if mode supports cancellation (has job_id)
        """
        return mode in (JobMode.ASYNC, JobMode.ASYNC_POLL)


# Singleton instance
job_manager = JobManager()
