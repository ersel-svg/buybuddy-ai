"""Jobs API router for managing background jobs."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class JobStatusResponse(BaseModel):
    """Job status response with Runpod status."""

    id: str
    type: str
    status: str
    progress: int
    runpod_job_id: Optional[str] = None
    runpod_status: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class JobCancel(BaseModel):
    """Job cancellation response."""

    status: str
    message: str


# ===========================================
# Dependencies
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


def get_runpod() -> RunpodService:
    """Get Runpod service instance."""
    return runpod_service


# ===========================================
# Helper Functions
# ===========================================


def get_endpoint_type_for_job(job_type: str) -> Optional[EndpointType]:
    """Map job type to Runpod endpoint type."""
    mapping = {
        "video_processing": EndpointType.VIDEO,
        "augmentation": EndpointType.AUGMENTATION,
        "training": EndpointType.TRAINING,
        "embedding_extraction": EndpointType.EMBEDDING,
    }
    return mapping.get(job_type)


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_jobs(
    type: Optional[str] = Query(None, alias="type", description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    db: SupabaseService = Depends(get_supabase),
):
    """List all jobs with optional filters."""
    jobs = await db.get_jobs(job_type=type)

    if status:
        jobs = [j for j in jobs if j.get("status") == status]

    return jobs


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
) -> JobStatusResponse:
    """Get job details with live Runpod status."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatusResponse(
        id=job["id"],
        type=job.get("type", "unknown"),
        status=job.get("status", "pending"),
        progress=job.get("progress", 0),
        runpod_job_id=job.get("runpod_job_id"),
        error=job.get("error"),
        result=job.get("result"),
    )

    # Fetch live status from Runpod if job is running
    runpod_job_id = job.get("runpod_job_id")
    if runpod_job_id and job.get("status") in ["running", "queued", "pending"]:
        endpoint_type = get_endpoint_type_for_job(job.get("type", ""))
        if endpoint_type and runpod.is_configured(endpoint_type):
            try:
                runpod_status = await runpod.get_job_status(endpoint_type, runpod_job_id)
                response.runpod_status = runpod_status.get("status")

                # Update local status based on Runpod status
                status_map = {
                    "COMPLETED": "completed",
                    "FAILED": "failed",
                    "IN_PROGRESS": "running",
                    "IN_QUEUE": "queued",
                    "CANCELLED": "cancelled",
                }
                new_status = status_map.get(response.runpod_status)
                if new_status and new_status != job.get("status"):
                    update_data = {"status": new_status}
                    if new_status == "completed":
                        update_data["progress"] = 100
                        update_data["result"] = runpod_status.get("output")
                    elif new_status == "failed":
                        update_data["error"] = runpod_status.get("error")
                    await db.update_job(job_id, update_data)
                    response.status = new_status

            except Exception as e:
                print(f"[Jobs] Failed to fetch Runpod status: {e}")

    return response


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Cancel a running job."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job['status']}",
        )

    # Cancel on Runpod if applicable
    runpod_job_id = job.get("runpod_job_id")
    if runpod_job_id:
        endpoint_type = get_endpoint_type_for_job(job.get("type", ""))
        if endpoint_type and runpod.is_configured(endpoint_type):
            try:
                await runpod.cancel_job(endpoint_type, runpod_job_id)
                print(f"[Jobs] Cancelled Runpod job {runpod_job_id}")
            except Exception as e:
                print(f"[Jobs] Failed to cancel Runpod job: {e}")

    # Update local job status
    updated_job = await db.update_job(job_id, {"status": "cancelled"})
    return updated_job


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get job logs (placeholder for future implementation)."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # TODO: Implement log fetching from Runpod or storage
    return {
        "job_id": job_id,
        "logs": [],
        "message": "Logs not yet implemented",
    }


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Retry a failed job."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed or cancelled jobs, current status: {job['status']}",
        )

    # Create a new job with the same config
    new_job = await db.create_job({
        "type": job.get("type"),
        "config": job.get("config", {}),
    })

    # TODO: Re-dispatch to Runpod based on job type

    return new_job
