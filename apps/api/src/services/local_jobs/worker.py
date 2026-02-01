"""
Background worker that processes local jobs.

The worker runs as an asyncio task during the FastAPI application lifecycle.
It polls the database for pending jobs and processes them one at a time.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from services.supabase import supabase_service
from .registry import job_registry
from .base import JobProgress


class LocalJobWorker:
    """
    Background worker that processes local jobs.

    Features:
    - Polls database for pending jobs
    - Atomic job claiming (prevents duplicate processing)
    - Progress tracking
    - Graceful shutdown
    - Stale job detection
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        max_job_duration: int = 3600,
        stale_job_timeout: int = 300,
    ):
        """
        Initialize the worker.

        Args:
            poll_interval: Seconds between polling for new jobs
            max_job_duration: Maximum seconds a job can run before timeout
            stale_job_timeout: Seconds before a locked job is considered stale
        """
        self.worker_id = f"worker-{str(uuid.uuid4())[:8]}"
        self.running = False
        self.current_job_id: Optional[str] = None
        self.poll_interval = poll_interval
        self.max_job_duration = max_job_duration
        self.stale_job_timeout = stale_job_timeout
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the worker loop."""
        self.running = True
        print(f"[{self.worker_id}] Started local job worker")

        # Check for stale jobs on startup
        await self._recover_stale_jobs()

        while self.running:
            try:
                job = await self._claim_next_job()
                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                print(f"[{self.worker_id}] Worker cancelled")
                break
            except Exception as e:
                print(f"[{self.worker_id}] Error in worker loop: {e}")
                await asyncio.sleep(self.poll_interval)

        print(f"[{self.worker_id}] Stopped local job worker")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        print(f"[{self.worker_id}] Stopping worker...")
        self.running = False

        # If processing a job, mark it as interrupted
        if self.current_job_id:
            try:
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error": "Job interrupted by worker shutdown",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", self.current_job_id).execute()
                print(f"[{self.worker_id}] Marked job {self.current_job_id[:8]} as interrupted")
            except Exception as e:
                print(f"[{self.worker_id}] Failed to mark job as interrupted: {e}")

    async def _recover_stale_jobs(self) -> None:
        """Find and reset jobs that were left in 'running' state."""
        try:
            stale_threshold = datetime.now(timezone.utc) - timedelta(seconds=self.stale_job_timeout)

            result = supabase_service.client.table("jobs")\
                .select("id")\
                .like("type", "local_%")\
                .eq("status", "running")\
                .lt("locked_at", stale_threshold.isoformat())\
                .execute()

            if result.data:
                job_ids = [j["id"] for j in result.data]
                print(f"[{self.worker_id}] Found {len(job_ids)} stale jobs, resetting...")

                for job_id in job_ids:
                    supabase_service.client.table("jobs").update({
                        "status": "failed",
                        "error": "Job became stale (worker may have crashed)",
                        "worker_id": None,
                        "locked_at": None,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", job_id).execute()

                print(f"[{self.worker_id}] Reset {len(job_ids)} stale jobs")
        except Exception as e:
            print(f"[{self.worker_id}] Failed to recover stale jobs: {e}")

    async def _claim_next_job(self) -> Optional[dict]:
        """
        Atomically claim the next pending job.

        Uses optimistic locking to prevent race conditions when
        multiple workers try to claim the same job.

        Returns:
            Job dict if claimed, None if no jobs available
        """
        try:
            # Get next pending local job
            result = supabase_service.client.table("jobs")\
                .select("*")\
                .like("type", "local_%")\
                .eq("status", "pending")\
                .is_("worker_id", "null")\
                .order("created_at")\
                .limit(1)\
                .execute()

            if not result.data:
                return None

            job = result.data[0]

            # Try to claim it atomically
            now = datetime.now(timezone.utc).isoformat()
            update_result = supabase_service.client.table("jobs")\
                .update({
                    "status": "running",
                    "worker_id": self.worker_id,
                    "locked_at": now,
                    "started_at": now,
                    "updated_at": now,
                })\
                .eq("id", job["id"])\
                .eq("status", "pending")\
                .is_("worker_id", "null")\
                .execute()

            if update_result.data:
                return update_result.data[0]

            # Another worker claimed it
            return None

        except Exception as e:
            print(f"[{self.worker_id}] Failed to claim job: {e}")
            return None

    async def _process_job(self, job: dict) -> None:
        """Process a single job."""
        job_id = job["id"]
        job_type = job["type"]
        config = job.get("config", {})

        self.current_job_id = job_id
        short_id = job_id[:8]
        print(f"[{self.worker_id}] Processing {job_type} job {short_id}...")

        # Get handler
        handler_class = job_registry.get_handler(job_type)
        if not handler_class:
            await self._fail_job(job_id, f"Unknown job type: {job_type}")
            self.current_job_id = None
            return

        handler = handler_class()

        # Validate config
        validation_error = handler.validate_config(config)
        if validation_error:
            await self._fail_job(job_id, f"Invalid config: {validation_error}")
            self.current_job_id = None
            return

        try:
            # Execute with progress callback
            result = await handler.execute(
                job_id=job_id,
                config=config,
                update_progress=lambda p: self._update_progress(job_id, p),
            )

            # Mark as completed
            await self._complete_job(job_id, result)
            print(f"[{self.worker_id}] Completed job {short_id}")

        except asyncio.CancelledError:
            await self._fail_job(job_id, "Job was cancelled")
            raise
        except Exception as e:
            await self._fail_job(job_id, str(e))
            print(f"[{self.worker_id}] Failed job {short_id}: {e}")
        finally:
            self.current_job_id = None

    def _update_progress(self, job_id: str, progress: JobProgress) -> None:
        """Update job progress in database."""
        try:
            supabase_service.client.table("jobs").update({
                "status": "running",
                "progress": min(max(progress.progress, 0), 100),
                "current_step": progress.current_step,
                "result": {
                    "processed": progress.processed,
                    "total": progress.total,
                    "errors": progress.errors[-10:] if progress.errors else [],
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job_id).execute()
        except Exception as e:
            print(f"[{self.worker_id}] Failed to update progress: {e}")

    async def _complete_job(self, job_id: str, result: dict) -> None:
        """Mark job as completed."""
        now = datetime.now(timezone.utc).isoformat()
        supabase_service.client.table("jobs").update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": now,
            "updated_at": now,
            "worker_id": None,
            "locked_at": None,
        }).eq("id", job_id).execute()

    async def _fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        now = datetime.now(timezone.utc).isoformat()
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": error[:1000],  # Truncate long errors
            "completed_at": now,
            "updated_at": now,
            "worker_id": None,
            "locked_at": None,
        }).eq("id", job_id).execute()


# Singleton instance
local_job_worker = LocalJobWorker()
