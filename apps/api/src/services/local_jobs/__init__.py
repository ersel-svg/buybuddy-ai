"""
Local Background Jobs Infrastructure.

This module provides a framework for running CPU-bound background jobs
without requiring external services like Runpod.

Usage:
    # In an API endpoint:
    from services.local_jobs import create_local_job

    job = await create_local_job(
        job_type="local_bulk_add_to_dataset",
        config={"dataset_id": "xxx", "filters": {...}}
    )
    return {"job_id": job["id"]}

    # Creating a new handler:
    from services.local_jobs import BaseJobHandler, JobProgress, job_registry

    @job_registry.register
    class MyHandler(BaseJobHandler):
        job_type = "local_my_operation"

        async def execute(self, job_id, config, update_progress):
            for i in range(100):
                # Do work...
                update_progress(JobProgress(
                    progress=i,
                    current_step=f"Step {i}",
                    processed=i,
                    total=100
                ))
            return {"success": True}
"""

from .base import BaseJobHandler, JobProgress
from .registry import job_registry, JobRegistry
from .worker import local_job_worker, LocalJobWorker
from .utils import chunks, calculate_progress

# Import handlers to register them
from . import handlers


async def create_local_job(job_type: str, config: dict) -> dict:
    """
    Create a new local background job.

    The job will be picked up by the worker and processed asynchronously.
    Poll GET /api/v1/jobs/{job_id} to track progress.

    Args:
        job_type: Type of job (must start with 'local_' and be registered)
        config: Job configuration dict

    Returns:
        Created job record from database

    Raises:
        ValueError: If job_type is invalid or not registered
    """
    from services.supabase import supabase_service

    # Validate job type
    if not job_type.startswith("local_"):
        raise ValueError(f"Local job type must start with 'local_': {job_type}")

    if not job_registry.is_registered(job_type):
        registered = job_registry.get_all_types()
        raise ValueError(
            f"Unknown job type: {job_type}. "
            f"Registered types: {registered}"
        )

    # Create job in database
    result = supabase_service.client.table("jobs").insert({
        "type": job_type,
        "status": "pending",
        "config": config,
        "progress": 0,
    }).execute()

    job = result.data[0]
    print(f"[LocalJobs] Created job {job['id'][:8]} ({job_type})")
    return job


__all__ = [
    # Base classes
    "BaseJobHandler",
    "JobProgress",
    # Registry
    "job_registry",
    "JobRegistry",
    # Worker
    "local_job_worker",
    "LocalJobWorker",
    # Utils
    "chunks",
    "calculate_progress",
    # API
    "create_local_job",
]
