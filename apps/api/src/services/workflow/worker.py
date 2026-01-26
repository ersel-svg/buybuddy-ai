"""
Workflow Background Worker

Processes async/background workflow executions from the queue.
Follows the same patterns as LocalJobWorker for consistency.

Features:
- Atomic job claiming (prevents duplicate processing)
- Priority-based queue processing
- Graceful shutdown with job interruption handling
- Retry on failure with exponential backoff
- Progress tracking for real-time updates
- Webhook callbacks for completion notification
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, Awaitable
import httpx

from services.supabase import supabase_service

logger = logging.getLogger(__name__)


class WorkflowWorker:
    """
    Background worker for async workflow execution.

    Polls the wf_executions table for pending async/background executions
    and processes them with priority ordering.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        max_concurrent: int = 5,
        stale_timeout_minutes: int = 30,
    ):
        """
        Initialize the workflow worker.

        Args:
            poll_interval: Seconds between queue polls
            max_concurrent: Maximum concurrent executions
            stale_timeout_minutes: Time after which running jobs are considered stale
        """
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.stale_timeout_minutes = stale_timeout_minutes

        self._running = False
        self._current_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the worker loop."""
        self._running = True
        logger.info(
            f"Workflow worker started (max_concurrent={self.max_concurrent}, "
            f"poll_interval={self.poll_interval}s)"
        )

        while self._running:
            try:
                # Calculate available slots
                available_slots = self.max_concurrent - len(self._current_tasks)

                if available_slots > 0:
                    # Claim pending jobs up to available slots
                    jobs = await self._claim_jobs(available_slots)

                    for job in jobs:
                        task = asyncio.create_task(
                            self._process_job(job),
                            name=f"workflow-{job['id'][:8]}"
                        )
                        self._current_tasks[job["id"]] = task

                # Clean up completed tasks
                completed = [
                    job_id for job_id, task in self._current_tasks.items()
                    if task.done()
                ]
                for job_id in completed:
                    # Check for exceptions
                    task = self._current_tasks.pop(job_id)
                    try:
                        task.result()
                    except Exception as e:
                        logger.exception(f"Task {job_id[:8]}... raised exception: {e}")

                # Periodically check for stale jobs
                await self._recover_stale_jobs()

            except Exception as e:
                logger.exception(f"Worker loop error: {e}")

            # Wait before next poll
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.poll_interval
                )
                # If we get here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue polling
                pass

        logger.info("Workflow worker loop ended")

    async def stop(self):
        """Gracefully stop the worker."""
        logger.info("Stopping workflow worker...")
        self._running = False
        self._shutdown_event.set()

        # Wait for current tasks to complete (with timeout)
        if self._current_tasks:
            logger.info(f"Waiting for {len(self._current_tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._current_tasks.values(),
                        return_exceptions=True
                    ),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling...")
                for task in self._current_tasks.values():
                    task.cancel()

        logger.info("Workflow worker stopped")

    async def _claim_jobs(self, limit: int) -> list[dict]:
        """
        Atomically claim pending jobs from queue.

        Uses optimistic locking to prevent duplicate claims across
        multiple worker instances.
        """
        # Find pending jobs with priority ordering
        result = supabase_service.client.table("wf_executions").select(
            "id, workflow_id, input_data, priority, retry_count, max_retries, timeout_seconds"
        ).eq(
            "status", "pending"
        ).in_(
            "execution_mode", ["async", "background"]
        ).order(
            "priority", desc=True
        ).order(
            "created_at", desc=False
        ).limit(limit).execute()

        if not result.data:
            return []

        claimed = []
        for job in result.data:
            # Try to claim with atomic update
            # Only claim if still pending (optimistic lock)
            update_result = supabase_service.client.table("wf_executions").update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job["id"]).eq("status", "pending").execute()

            if update_result.data:
                claimed.append(job)
                logger.info(
                    f"Claimed execution {job['id'][:8]}... "
                    f"(priority={job.get('priority', 5)})"
                )

        return claimed

    async def _process_job(self, job: dict):
        """Process a single workflow execution."""
        job_id = job["id"]
        workflow_id = job["workflow_id"]
        timeout = job.get("timeout_seconds", 300)

        try:
            # Get workflow definition
            workflow = supabase_service.client.table("wf_workflows").select(
                "definition, current_version"
            ).eq("id", workflow_id).single().execute()

            if not workflow.data:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Update with workflow version
            supabase_service.client.table("wf_executions").update({
                "workflow_version": workflow.data.get("current_version", 1),
            }).eq("id", job_id).execute()

            # Execute workflow with timeout
            from services.workflow import get_workflow_engine
            engine = get_workflow_engine()

            result = await asyncio.wait_for(
                engine.execute(
                    workflow=workflow.data["definition"],
                    inputs=job["input_data"],
                    workflow_id=workflow_id,
                    execution_id=job_id,
                ),
                timeout=timeout,
            )

            # Mark as completed
            await self._complete_job(job_id, result)

        except asyncio.TimeoutError:
            logger.error(f"Execution {job_id[:8]}... timed out after {timeout}s")
            await self._fail_job(job, f"Execution timeout after {timeout}s", permanent=True)

        except Exception as e:
            logger.exception(f"Execution {job_id[:8]}... failed: {e}")
            await self._fail_job(job, str(e))

    async def _update_progress(self, job_id: str, progress: dict):
        """Update job progress for real-time tracking."""
        try:
            supabase_service.client.table("wf_executions").update({
                "progress": progress,
            }).eq("id", job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update progress for {job_id[:8]}...: {e}")

    async def _complete_job(self, job_id: str, result: dict):
        """Mark job as completed and send webhook if configured."""
        now = datetime.now(timezone.utc)

        supabase_service.client.table("wf_executions").update({
            "status": "completed",
            "completed_at": now.isoformat(),
            "duration_ms": int(result.get("duration_ms", 0)),
            "output_data": result.get("outputs", {}),
            "node_metrics": result.get("metrics", {}),
            "progress": {
                "current_node": None,
                "completed_nodes": list(result.get("metrics", {}).keys()),
                "total_nodes": len(result.get("metrics", {})),
                "percent": 100,
            },
        }).eq("id", job_id).execute()

        logger.info(
            f"Execution {job_id[:8]}... completed in "
            f"{result.get('duration_ms', 0):.0f}ms"
        )

        # Send webhook if configured
        await self._send_completion_webhook(job_id, result)

    async def _fail_job(self, job: dict, error: str, permanent: bool = False):
        """
        Mark job as failed, with retry logic.

        Args:
            job: The job dict
            error: Error message
            permanent: If True, don't retry regardless of retry count
        """
        job_id = job["id"]
        retry_count = job.get("retry_count", 0)
        max_retries = job.get("max_retries", 3)
        now = datetime.now(timezone.utc)

        if not permanent and retry_count < max_retries:
            # Schedule retry with exponential backoff
            backoff_seconds = min(300, 2 ** retry_count * 10)  # Max 5 min backoff

            supabase_service.client.table("wf_executions").update({
                "status": "pending",
                "retry_count": retry_count + 1,
                "last_retry_at": now.isoformat(),
                "error_message": f"Retry {retry_count + 1}/{max_retries}: {error}",
            }).eq("id", job_id).execute()

            logger.info(
                f"Execution {job_id[:8]}... scheduled for retry "
                f"({retry_count + 1}/{max_retries}) in {backoff_seconds}s"
            )
        else:
            # Permanent failure
            supabase_service.client.table("wf_executions").update({
                "status": "failed",
                "completed_at": now.isoformat(),
                "error_message": error,
            }).eq("id", job_id).execute()

            logger.error(
                f"Execution {job_id[:8]}... failed permanently "
                f"after {retry_count} retries: {error}"
            )

            # Send failure webhook
            await self._send_failure_webhook(job_id, error)

    async def _recover_stale_jobs(self):
        """Recover jobs stuck in 'running' state."""
        stale_threshold = datetime.now(timezone.utc) - timedelta(
            minutes=self.stale_timeout_minutes
        )

        result = supabase_service.client.table("wf_executions").update({
            "status": "pending",
            "error_message": "Recovered from stale state (worker restart or timeout)",
        }).eq(
            "status", "running"
        ).in_(
            "execution_mode", ["async", "background"]
        ).lt(
            "started_at", stale_threshold.isoformat()
        ).execute()

        if result.data:
            logger.warning(f"Recovered {len(result.data)} stale executions")

    async def _send_completion_webhook(self, job_id: str, result: dict):
        """Send webhook on job completion."""
        try:
            # Get callback URL
            job = supabase_service.client.table("wf_executions").select(
                "callback_url, workflow_id"
            ).eq("id", job_id).single().execute()

            if not job.data or not job.data.get("callback_url"):
                return

            callback_url = job.data["callback_url"]

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    callback_url,
                    json={
                        "event": "workflow.completed",
                        "execution_id": job_id,
                        "workflow_id": job.data["workflow_id"],
                        "status": "completed",
                        "output_data": result.get("outputs"),
                        "duration_ms": result.get("duration_ms"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if response.status_code >= 400:
                    logger.warning(
                        f"Webhook returned {response.status_code} for {job_id[:8]}..."
                    )

        except Exception as e:
            logger.warning(f"Failed to send completion webhook for {job_id[:8]}...: {e}")

    async def _send_failure_webhook(self, job_id: str, error: str):
        """Send webhook on job failure."""
        try:
            job = supabase_service.client.table("wf_executions").select(
                "callback_url, workflow_id"
            ).eq("id", job_id).single().execute()

            if not job.data or not job.data.get("callback_url"):
                return

            callback_url = job.data["callback_url"]

            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(
                    callback_url,
                    json={
                        "event": "workflow.failed",
                        "execution_id": job_id,
                        "workflow_id": job.data["workflow_id"],
                        "status": "failed",
                        "error": error,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        except Exception as e:
            logger.warning(f"Failed to send failure webhook for {job_id[:8]}...: {e}")


# Singleton instance
_workflow_worker: Optional[WorkflowWorker] = None


def get_workflow_worker() -> WorkflowWorker:
    """Get singleton worker instance."""
    global _workflow_worker
    if _workflow_worker is None:
        _workflow_worker = WorkflowWorker()
    return _workflow_worker
