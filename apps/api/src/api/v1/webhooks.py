"""Webhooks API router for handling external callbacks."""

from typing import Optional

from fastapi import APIRouter, Header, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class RunpodWebhookPayload(BaseModel):
    """Runpod webhook payload schema."""

    id: str
    status: str
    output: Optional[dict] = None
    error: Optional[str] = None


class WebhookResponse(BaseModel):
    """Webhook response schema."""

    received: bool
    job_id: str
    status: str


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Endpoints
# ===========================================


@router.post("/runpod", response_model=WebhookResponse)
async def handle_runpod_webhook(
    payload: RunpodWebhookPayload,
    x_runpod_signature: Optional[str] = Header(None),
    db: SupabaseService = Depends(get_supabase),
) -> WebhookResponse:
    """
    Handle Runpod job completion webhook.

    This endpoint is called by Runpod when a serverless job completes.
    """
    # TODO: Verify webhook signature

    print(f"[Webhook] Received Runpod webhook for job {payload.id}")
    print(f"          Status: {payload.status}")

    # Map Runpod status to our status
    status_map = {
        "COMPLETED": "completed",
        "FAILED": "failed",
        "IN_PROGRESS": "running",
        "IN_QUEUE": "queued",
        "CANCELLED": "cancelled",
    }
    job_status = status_map.get(payload.status, "pending")

    # Try to find and update the job
    try:
        # Find job by runpod_job_id
        jobs = await db.get_jobs()
        job = next(
            (j for j in jobs if j.get("runpod_job_id") == payload.id),
            None,
        )

        if job:
            update_data = {"status": job_status}

            if payload.status == "COMPLETED":
                update_data["progress"] = 100
                update_data["result"] = payload.output
                print(f"          Output: {payload.output}")

            elif payload.status == "FAILED":
                update_data["error"] = payload.error
                print(f"          Error: {payload.error}")

            await db.update_job(job["id"], update_data)
            print(f"          Job {job['id']} updated to {job_status}")

    except Exception as e:
        print(f"          Error updating job: {e}")

    return WebhookResponse(
        received=True,
        job_id=payload.id,
        status=payload.status,
    )


@router.post("/supabase")
async def handle_supabase_webhook(
    payload: dict,
    db: SupabaseService = Depends(get_supabase),
) -> dict:
    """
    Handle Supabase database webhooks.

    Can be used for real-time updates or triggers.
    """
    # TODO: Handle Supabase events
    print(f"[Webhook] Received Supabase webhook: {payload}")

    return {"received": True}
