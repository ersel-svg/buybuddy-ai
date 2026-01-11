"""Jobs API router for managing background jobs."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class Job(BaseModel):
    """Job schema."""

    id: str
    type: str
    status: str
    progress: int = 0
    config: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    runpod_job_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ===========================================
# Mock Data
# ===========================================

MOCK_JOBS: list[dict] = [
    {
        "id": str(uuid4()),
        "type": "video_processing",
        "status": "completed",
        "progress": 100,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
    {
        "id": str(uuid4()),
        "type": "training",
        "status": "running",
        "progress": 45,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
]


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_jobs(
    type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by job status"),
) -> list[Job]:
    """List all jobs with optional filters."""
    jobs = MOCK_JOBS.copy()

    if type:
        jobs = [j for j in jobs if j["type"] == type]
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    return [Job(**j) for j in jobs]


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str) -> Job:
    """Get job details."""
    job = next((j for j in MOCK_JOBS if j["id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return Job(**job)


@router.post("/{job_id}/cancel", response_model=Job)
async def cancel_job(job_id: str) -> Job:
    """Cancel a running job."""
    job = next((j for j in MOCK_JOBS if j["id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job['status']}")

    # TODO: Cancel Runpod job
    job["status"] = "cancelled"
    job["updated_at"] = datetime.now().isoformat()

    return Job(**job)
