"""Webhooks API router for handling external callbacks."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

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
# Endpoints
# ===========================================


@router.post("/runpod", response_model=WebhookResponse)
async def handle_runpod_webhook(
    payload: RunpodWebhookPayload,
    x_runpod_signature: Optional[str] = Header(None),
) -> WebhookResponse:
    """
    Handle Runpod job completion webhook.

    This endpoint is called by Runpod when a serverless job completes.
    """
    # TODO: Verify webhook signature
    # TODO: Update job status in database
    # TODO: Handle job output (save results, update related entities)

    print(f"[Webhook] Received Runpod webhook for job {payload.id}")
    print(f"          Status: {payload.status}")

    if payload.status == "COMPLETED":
        print(f"          Output: {payload.output}")
        # TODO: Process completed job
        # - For video_processing: Update product with frames
        # - For training: Save model artifact
        # - For embedding: Update index
        # - For augmentation: Update dataset

    elif payload.status == "FAILED":
        print(f"          Error: {payload.error}")
        # TODO: Handle failed job
        # - Update job status
        # - Send notification

    return WebhookResponse(
        received=True,
        job_id=payload.id,
        status=payload.status,
    )


@router.post("/supabase")
async def handle_supabase_webhook(payload: dict) -> dict:
    """
    Handle Supabase database webhooks.

    Can be used for real-time updates or triggers.
    """
    # TODO: Handle Supabase events
    print(f"[Webhook] Received Supabase webhook: {payload}")

    return {"received": True}
