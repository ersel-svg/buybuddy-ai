"""
Background Job Poller Service

SOTA approach for job completion:
1. PRIMARY: RunPod webhook callback (fastest, instant)
2. FALLBACK: Background poller (reliable, catches missed webhooks)
3. LAST RESORT: On-demand GET /jobs/{id} check (user-triggered)

This service implements the fallback mechanism - periodically checking
running jobs and completing them if RunPod has finished processing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from config import settings
from services.supabase import supabase_service
from services.runpod import runpod_service, EndpointType


class JobPollerService:
    """
    Background service that polls running jobs for completion.

    Ensures jobs complete even if:
    - Webhook fails/times out
    - Frontend closes before completion
    - Network issues during webhook delivery
    """

    def __init__(self):
        self.running = False
        self.poll_interval = 30  # Check every 30 seconds
        self.stale_job_threshold = timedelta(hours=2)  # Mark jobs older than 2h as potentially stale
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background polling service."""
        self.running = True
        print(f"[JobPoller] Started background job poller (interval: {self.poll_interval}s)")

        while self.running:
            try:
                await self._poll_running_jobs()
            except Exception as e:
                print(f"[JobPoller] Error during polling: {e}")

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop the background polling service."""
        self.running = False
        print("[JobPoller] Stopping background job poller...")

    async def _poll_running_jobs(self):
        """Check all running jobs and update their status if completed."""

        # Query running OD annotation jobs
        try:
            jobs_result = supabase_service.client.table("jobs").select(
                "id, runpod_job_id, config, started_at"
            ).eq("type", "od_annotation").eq("status", "running").execute()

            running_jobs = jobs_result.data or []

            if not running_jobs:
                return

            print(f"[JobPoller] Checking {len(running_jobs)} running OD annotation job(s)")

            for job in running_jobs:
                await self._check_and_complete_job(job)

        except Exception as e:
            print(f"[JobPoller] Failed to query running jobs: {e}")

    async def _check_and_complete_job(self, job: dict):
        """Check a single job's RunPod status and complete if finished."""
        job_id = job["id"]
        runpod_job_id = job.get("runpod_job_id")
        config = job.get("config", {})

        if not runpod_job_id:
            print(f"[JobPoller] Job {job_id[:8]}... has no RunPod ID, skipping")
            return

        try:
            # Check RunPod status
            runpod_status = await runpod_service.get_job_status(
                endpoint_type=EndpointType.OD_ANNOTATION,
                job_id=runpod_job_id,
            )

            status = runpod_status.get("status")

            if status == "COMPLETED":
                print(f"[JobPoller] Job {job_id[:8]}... completed on RunPod, saving results")
                await self._complete_job(job_id, config, runpod_status.get("output", {}))

            elif status == "FAILED":
                error = runpod_status.get("error", "Unknown error")
                print(f"[JobPoller] Job {job_id[:8]}... failed on RunPod: {error}")
                await self._fail_job(job_id, error)

            elif status == "IN_PROGRESS":
                # Still running, check if stale
                started_at = job.get("started_at")
                if started_at:
                    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    age = datetime.now(started.tzinfo) - started
                    if age > self.stale_job_threshold:
                        print(f"[JobPoller] Job {job_id[:8]}... is stale ({age}), marking as failed")
                        await self._fail_job(job_id, f"Job timed out after {age}")

            # Status might also be "IN_QUEUE" - just wait

        except Exception as e:
            error_str = str(e)

            # 404 means job expired from RunPod (usually after 24-48h)
            if "404" in error_str:
                print(f"[JobPoller] Job {job_id[:8]}... expired from RunPod (404)")
                await self._fail_job(
                    job_id,
                    "Job expired from RunPod. Results could not be retrieved. Please re-run the job."
                )
            else:
                print(f"[JobPoller] Error checking job {job_id[:8]}...: {e}")

    async def _complete_job(self, job_id: str, config: dict, output: dict):
        """Mark job as completed and save predictions as annotations."""
        from api.v1.od.ai import _save_predictions_as_annotations

        # Extract predictions from output
        results = output.get("results", [])
        predictions_count = sum(len(r.get("predictions", [])) for r in results)

        # Save predictions as annotations
        annotations_created = 0
        if results:
            predictions_by_image = {
                r["id"]: r.get("predictions", [])
                for r in results if r.get("id")
            }

            try:
                annotations_created = await _save_predictions_as_annotations(
                    dataset_id=config.get("dataset_id"),
                    predictions_by_image=predictions_by_image,
                    class_mapping=config.get("class_mapping"),
                    model=config.get("model"),
                    filter_classes=config.get("filter_classes"),
                )
                print(f"[JobPoller] Created {annotations_created} annotations for job {job_id[:8]}...")
            except Exception as e:
                print(f"[JobPoller] Failed to save annotations for job {job_id[:8]}...: {e}")

        # Update job status
        result_data = {
            **output,
            "total_predictions": predictions_count,
            "annotations_created": annotations_created,
            "completed_by": "background_poller",
        }

        supabase_service.client.table("jobs").update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result_data,
        }).eq("id", job_id).execute()

        print(f"[JobPoller] Job {job_id[:8]}... marked as completed ({predictions_count} predictions, {annotations_created} annotations)")

    async def _fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": error,
            "result": {"failed_by": "background_poller"},
        }).eq("id", job_id).execute()

        print(f"[JobPoller] Job {job_id[:8]}... marked as failed: {error}")


# Singleton instance
job_poller = JobPollerService()
