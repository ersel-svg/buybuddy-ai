"""Webhooks API router for handling external callbacks."""

from typing import Optional, Any

from fastapi import APIRouter, Header, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


def extract_product_data_from_metadata(metadata: dict) -> dict[str, Any]:
    """
    Extract and map Gemini metadata to product database fields.

    Maps the nested Gemini output structure to flat product columns.
    """
    if not metadata:
        return {}

    product_data = {}

    # Brand info
    brand_info = metadata.get("brand_info", {})
    if brand_info.get("brand_name"):
        product_data["brand_name"] = brand_info["brand_name"]
    if brand_info.get("sub_brand"):
        product_data["sub_brand"] = brand_info["sub_brand"]
    if brand_info.get("manufacturer_country"):
        product_data["manufacturer_country"] = brand_info["manufacturer_country"]

    # Product identity
    product_identity = metadata.get("product_identity", {})
    if product_identity.get("product_name"):
        product_data["product_name"] = product_identity["product_name"]
    if product_identity.get("variant_flavor"):
        product_data["variant_flavor"] = product_identity["variant_flavor"]
    if product_identity.get("product_category"):
        product_data["category"] = product_identity["product_category"]
    if product_identity.get("container_type"):
        product_data["container_type"] = product_identity["container_type"]

    # Specifications
    specs = metadata.get("specifications", {})
    if specs.get("net_quantity_text"):
        product_data["net_quantity"] = specs["net_quantity_text"]
    if specs.get("pack_configuration"):
        product_data["pack_configuration"] = specs["pack_configuration"]
    if specs.get("identifiers"):
        product_data["identifiers"] = specs["identifiers"]

    # Marketing and claims
    marketing = metadata.get("marketing_and_claims", {})
    if marketing.get("claims_list"):
        product_data["claims"] = marketing["claims_list"]
    if marketing.get("marketing_description"):
        product_data["marketing_description"] = marketing["marketing_description"]

    # Nutrition facts (store as-is in JSONB)
    nutrition = metadata.get("nutrition_facts", {})
    if nutrition:
        product_data["nutrition_facts"] = nutrition

    # Visual grounding
    visual = metadata.get("visual_grounding", {})
    if visual.get("grounding_prompt"):
        product_data["grounding_prompt"] = visual["grounding_prompt"]

    # Extraction metadata
    extraction = metadata.get("extraction_metadata", {})
    if extraction.get("visibility_score") is not None:
        product_data["visibility_score"] = extraction["visibility_score"]
    if extraction.get("issues_detected"):
        product_data["issues_detected"] = extraction["issues_detected"]

    return product_data


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
    Updates both the job status AND the product metadata.
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
        # Find job by runpod_job_id (direct indexed query, not list scan)
        job = await db.get_job_by_runpod_id(payload.id)

        if not job:
            print(f"          WARNING: Job not found for runpod_job_id {payload.id}")
            return WebhookResponse(
                received=True,
                job_id=payload.id,
                status=payload.status,
            )

        job_config = job.get("config", {})
        product_id = job_config.get("product_id")
        video_id = job_config.get("video_id")

        update_data = {"status": job_status}

        if payload.status == "COMPLETED":
            update_data["progress"] = 100
            update_data["result"] = payload.output
            print(f"          Output keys: {list(payload.output.keys()) if payload.output else 'None'}")

            # Check if handler returned an error (RunPod sends COMPLETED even for caught errors)
            output_status = payload.output.get("status") if payload.output else None
            if output_status == "error":
                # Handler caught an error - reset product and video status
                update_data["status"] = "failed"
                update_data["error"] = payload.output.get("error", "Unknown error")
                await _reset_product_and_video_status(db, product_id, video_id, "pending")
            else:
                # Success - update product with metadata from pipeline
                await _update_product_from_result(db, payload.output)

        elif payload.status == "FAILED":
            update_data["error"] = payload.error
            print(f"          Error: {payload.error}")
            # Reset product and video status
            await _reset_product_and_video_status(db, product_id, video_id, "pending")

        elif payload.status == "CANCELLED":
            # Reset product and video status on cancellation
            await _reset_product_and_video_status(db, product_id, video_id, "pending")

        await db.update_job(job["id"], update_data)
        print(f"          Job {job['id']} updated to {job_status}")

    except Exception as e:
        print(f"          Error updating job: {e}")
        import traceback
        traceback.print_exc()

    return WebhookResponse(
        received=True,
        job_id=payload.id,
        status=payload.status,
    )


async def _reset_product_and_video_status(
    db: SupabaseService,
    product_id: str | None,
    video_id: int | None,
    status: str,
) -> None:
    """Reset product and video status on job failure/cancellation."""
    if product_id:
        try:
            await db.update_product(product_id, {"status": status})
            print(f"          Product {product_id} status reset to {status}")
        except Exception as e:
            print(f"          Error resetting product status: {e}")

    if video_id:
        try:
            db.client.table("videos").update({"status": status}).eq("id", video_id).execute()
            print(f"          Video {video_id} status reset to {status}")
        except Exception as e:
            print(f"          Error resetting video status: {e}")


async def _update_product_from_result(db: SupabaseService, result: dict) -> None:
    """
    Update product from pipeline result.

    Product is created when processing is triggered (in videos.py),
    so we only need to update it here with the pipeline results.

    The result contains:
    - product_id: UUID of existing product (required)
    - video_id: ID of the video that was processed
    - barcode: product barcode
    - metadata: Gemini extracted metadata
    - frame_count: number of frames extracted
    - frames_url: URL to frames in storage
    - primary_image_url: URL to primary product image
    """
    if not result:
        return

    product_id = result.get("product_id")
    video_id = result.get("video_id")
    metadata = result.get("metadata", {})

    if not product_id:
        print("[Webhook] No product_id in result, skipping product update")
        return

    # Extract product data from Gemini metadata
    product_data = extract_product_data_from_metadata(metadata)

    # Add pipeline output fields
    if result.get("frame_count"):
        product_data["frame_count"] = result["frame_count"]
    if result.get("frames_url"):
        product_data["frames_path"] = result["frames_url"]
    if result.get("primary_image_url"):
        product_data["primary_image_url"] = result["primary_image_url"]
    if result.get("video_id"):
        product_data["video_id"] = result["video_id"]

    # Set status to needs_matching (ready for real image matching)
    product_data["status"] = "needs_matching"

    try:
        await db.update_product(product_id, product_data)
        print(f"[Webhook] Updated product {product_id} with pipeline results")

        # Also update video status to completed
        if video_id:
            try:
                db.client.table("videos").update({
                    "status": "completed",
                    "product_id": product_id,
                }).eq("id", video_id).execute()
                print(f"[Webhook] Updated video {video_id} status to completed")
            except Exception as ve:
                print(f"[Webhook] Error updating video {video_id}: {ve}")

    except Exception as e:
        print(f"[Webhook] Error updating product {product_id}: {e}")


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
