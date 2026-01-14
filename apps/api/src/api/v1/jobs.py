"""Jobs API router for managing background jobs."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from auth.dependencies import get_current_user

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


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
# Schemas for Batch Operations
# ===========================================


class BatchCancelRequest(BaseModel):
    """Request to cancel multiple jobs."""

    job_ids: list[str] | None = None  # If None, cancel all active jobs of type
    job_type: str | None = None  # Filter by job type


class BatchCancelResponse(BaseModel):
    """Response from batch cancel."""

    cancelled_count: int
    failed_count: int
    errors: list[str]


# ===========================================
# Endpoints - IMPORTANT: Literal routes MUST come before parameterized routes
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


@router.get("/active/count")
async def get_active_jobs_count(
    job_type: str | None = Query(None, description="Filter by job type"),
    db: SupabaseService = Depends(get_supabase),
) -> dict:
    """Get count of active jobs (for header badge)."""
    count = await db.get_active_jobs_count(job_type=job_type)
    return {"count": count}


@router.post("/batch/cancel")
async def cancel_jobs_batch(
    request: BatchCancelRequest,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
) -> BatchCancelResponse:
    """Cancel multiple jobs at once."""
    cancelled_count = 0
    failed_count = 0
    errors = []

    # Get jobs to cancel
    if request.job_ids:
        # Cancel specific jobs
        jobs_to_cancel = []
        for job_id in request.job_ids:
            job = await db.get_job(job_id)
            if job and job.get("status") not in ["completed", "failed", "cancelled"]:
                jobs_to_cancel.append(job)
    else:
        # Cancel all active jobs (optionally filtered by type)
        all_jobs = await db.get_jobs(job_type=request.job_type, limit=5000)
        jobs_to_cancel = [
            j for j in all_jobs
            if j.get("status") in ["pending", "queued", "running"]
        ]

    print(f"[Jobs] Batch cancelling {len(jobs_to_cancel)} jobs")

    for job in jobs_to_cancel:
        try:
            # Cancel on Runpod if applicable
            runpod_job_id = job.get("runpod_job_id")
            if runpod_job_id:
                endpoint_type = get_endpoint_type_for_job(job.get("type", ""))
                if endpoint_type and runpod.is_configured(endpoint_type):
                    try:
                        await runpod.cancel_job(endpoint_type, runpod_job_id)
                    except Exception as e:
                        print(f"[Jobs] Failed to cancel Runpod job {runpod_job_id}: {e}")

            # Reset product and video status
            job_config = job.get("config", {})
            product_id = job_config.get("product_id")
            video_id = job_config.get("video_id")

            if product_id:
                try:
                    await db.update_product(product_id, {"status": "pending"})
                except Exception:
                    pass

            if video_id:
                try:
                    db.client.table("videos").update({"status": "pending"}).eq("id", video_id).execute()
                except Exception:
                    pass

            # Update job status
            await db.update_job(job["id"], {"status": "cancelled"})
            cancelled_count += 1

        except Exception as e:
            failed_count += 1
            errors.append(f"Job {job['id']}: {str(e)}")

    print(f"[Jobs] Batch cancel complete: {cancelled_count} cancelled, {failed_count} failed")

    return BatchCancelResponse(
        cancelled_count=cancelled_count,
        failed_count=failed_count,
        errors=errors[:10],  # Limit errors in response
    )


# ===========================================
# Parameterized Routes (must come after literal routes)
# ===========================================


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
    """Cancel a running job and reset associated product/video status."""
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

    # Reset product and video status
    job_config = job.get("config", {})
    product_id = job_config.get("product_id")
    video_id = job_config.get("video_id")

    if product_id:
        try:
            await db.update_product(product_id, {"status": "pending"})
            print(f"[Jobs] Reset product {product_id} status to pending")
        except Exception as e:
            print(f"[Jobs] Failed to reset product status: {e}")

    if video_id:
        try:
            db.client.table("videos").update({"status": "pending"}).eq("id", video_id).execute()
            print(f"[Jobs] Reset video {video_id} status to pending")
        except Exception as e:
            print(f"[Jobs] Failed to reset video status: {e}")

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
