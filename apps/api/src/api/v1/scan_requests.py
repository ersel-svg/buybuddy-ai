"""Scan Requests API router for product scan requests."""

import httpx
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
from pydantic import BaseModel, EmailStr

from services.supabase import supabase_service
from auth.dependencies import get_current_user
from config import settings

router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class ScanRequestCreate(BaseModel):
    """Schema for creating a scan request."""
    barcode: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    notes: Optional[str] = None
    requester_name: str
    requester_email: str
    reference_images: Optional[List[str]] = None  # Storage paths


class ScanRequestUpdate(BaseModel):
    """Schema for updating a scan request."""
    status: Optional[str] = None
    notes: Optional[str] = None


class ScanRequestResponse(BaseModel):
    """Schema for scan request response."""
    id: str
    barcode: str
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    reference_images: List[str] = []
    notes: Optional[str] = None
    requester_name: str
    requester_email: str
    status: str
    completed_at: Optional[datetime] = None
    completed_by_product_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ScanRequestsListResponse(BaseModel):
    """Schema for paginated scan requests list."""
    items: List[ScanRequestResponse]
    total: int
    page: int
    limit: int
    has_more: bool


class DuplicateCheckResponse(BaseModel):
    """Schema for duplicate check response."""
    has_duplicate: bool
    existing_requests: List[ScanRequestResponse] = []


# ===========================================
# Slack Notification
# ===========================================


async def send_slack_notification(scan_request: dict) -> bool:
    """Send Slack notification for new scan request."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"Attempting to send Slack notification for scan request: {scan_request.get('id')}")
    logger.info(f"Slack webhook URL configured: {bool(settings.slack_webhook_url)}")

    if not settings.slack_webhook_url:
        logger.warning("Slack webhook URL not configured, skipping notification")
        return False

    try:
        # Build product info text
        product_info = []
        if scan_request.get('brand_name'):
            product_info.append(scan_request['brand_name'])
        if scan_request.get('product_name'):
            product_info.append(scan_request['product_name'])
        product_text = " - ".join(product_info) if product_info else "No product info provided"

        # Image count
        images = scan_request.get('reference_images') or []
        image_text = f"{len(images)} reference image(s)" if images else "No images"

        message = {
            "username": "BuyBuddy AI",
            "icon_url": "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/scan-request-images/branding/bb-logomark.png",
            "attachments": [
                {
                    "color": "#6366f1",  # Indigo color
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*:package: New Scan Request*\n\nSomeone requested a scan for a product not in the system."
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Barcode:*\n`{scan_request['barcode']}`"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Product:*\n{product_text}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Requested by:*\n{scan_request['requester_name']}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Email:*\n{scan_request['requester_email']}"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        # Add notes if present
        if scan_request.get('notes'):
            message["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Notes:*\n_{scan_request['notes']}_"
                }
            })

        # Add reference images as clickable links
        if images:
            image_links = []
            storage_base_url = "https://qvyxpfcwfktxnaeavkxx.supabase.co/storage/v1/object/public/scan-request-images"
            for i, img_path in enumerate(images, 1):
                # Convert relative path to full URL if needed
                if img_path.startswith("http"):
                    full_url = img_path
                else:
                    full_url = f"{storage_base_url}/{img_path}"
                image_links.append(f"<{full_url}|Image {i}>")
            message["attachments"][0]["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reference Images:*\n{' • '.join(image_links)}"
                }
            })

        # Add context (images + timestamp)
        message["attachments"][0]["blocks"].append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":frame_with_picture: {image_text}  •  :clock1: {scan_request.get('created_at', 'Now')[:16].replace('T', ' ')}"
                }
            ]
        })

        async with httpx.AsyncClient() as client:
            logger.info(f"Sending Slack notification to webhook...")
            response = await client.post(
                settings.slack_webhook_url,
                json=message,
                timeout=10.0
            )
            logger.info(f"Slack response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Slack error response: {response.text}")
            return response.status_code == 200

    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}", exc_info=True)
        return False


# ===========================================
# Endpoints
# ===========================================


@router.get("/check-duplicate")
async def check_duplicate(barcode: str = Query(..., description="Barcode to check")) -> DuplicateCheckResponse:
    """Check if there's already a pending scan request for this barcode."""
    try:
        result = supabase_service.client.table("scan_requests").select("*").eq(
            "barcode", barcode
        ).in_("status", ["pending", "in_progress"]).execute()

        existing = result.data or []

        return DuplicateCheckResponse(
            has_duplicate=len(existing) > 0,
            existing_requests=existing
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check duplicate: {str(e)}")


@router.post("", response_model=ScanRequestResponse)
async def create_scan_request(request: ScanRequestCreate) -> ScanRequestResponse:
    """Create a new scan request."""
    try:
        # Insert into database
        data = {
            "barcode": request.barcode,
            "product_name": request.product_name,
            "brand_name": request.brand_name,
            "notes": request.notes,
            "requester_name": request.requester_name,
            "requester_email": request.requester_email,
            "reference_images": request.reference_images or [],
            "status": "pending"
        }

        result = supabase_service.client.table("scan_requests").insert(data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create scan request")

        scan_request = result.data[0]

        # Send Slack notification (async, don't wait for it to complete)
        await send_slack_notification(scan_request)

        return scan_request

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scan request: {str(e)}")


@router.get("", response_model=ScanRequestsListResponse)
async def list_scan_requests(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by barcode or product name")
) -> ScanRequestsListResponse:
    """List scan requests with pagination and filtering."""
    try:
        offset = (page - 1) * limit

        # Build query
        query = supabase_service.client.table("scan_requests").select("*", count="exact")

        # Apply filters
        if status:
            query = query.eq("status", status)

        if search:
            # Search in barcode or product_name
            query = query.or_(f"barcode.ilike.%{search}%,product_name.ilike.%{search}%")

        # Order and paginate
        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

        result = query.execute()

        items = result.data or []
        total = result.count or 0

        return ScanRequestsListResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            has_more=(offset + len(items)) < total
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list scan requests: {str(e)}")


@router.get("/{request_id}", response_model=ScanRequestResponse)
async def get_scan_request(request_id: str) -> ScanRequestResponse:
    """Get a specific scan request by ID."""
    try:
        result = supabase_service.client.table("scan_requests").select("*").eq(
            "id", request_id
        ).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Scan request not found")

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scan request: {str(e)}")


@router.patch("/{request_id}", response_model=ScanRequestResponse)
async def update_scan_request(request_id: str, update: ScanRequestUpdate) -> ScanRequestResponse:
    """Update a scan request (e.g., change status)."""
    try:
        # Build update data
        data = {}
        if update.status is not None:
            data["status"] = update.status
        if update.notes is not None:
            data["notes"] = update.notes

        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")

        result = supabase_service.client.table("scan_requests").update(data).eq(
            "id", request_id
        ).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Scan request not found")

        return result.data[0]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scan request: {str(e)}")


@router.delete("/{request_id}")
async def delete_scan_request(request_id: str) -> dict:
    """Delete (cancel) a scan request."""
    try:
        # Soft delete by setting status to cancelled
        result = supabase_service.client.table("scan_requests").update({
            "status": "cancelled"
        }).eq("id", request_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Scan request not found")

        return {"success": True, "message": "Scan request cancelled"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scan request: {str(e)}")


@router.post("/upload-image")
async def upload_reference_image(file: UploadFile = File(...)) -> dict:
    """Upload a reference image for a scan request."""
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: JPEG, PNG, WebP")

        # Validate file size (max 5MB)
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")

        # Generate unique filename
        import uuid
        ext = file.filename.split(".")[-1] if file.filename else "jpg"
        filename = f"{uuid.uuid4()}.{ext}"
        storage_path = f"scan-requests/{filename}"

        # Upload to Supabase Storage
        upload_response = supabase_service.client.storage.from_("scan-request-images").upload(
            storage_path,
            content,
            {"content-type": file.content_type}
        )

        # Check if upload had an error (supabase-py returns response with path on success)
        # The response is typically a dict with 'path' key on success
        if upload_response is None:
            raise HTTPException(status_code=500, detail="Storage upload returned no response")

        # Get public URL
        public_url = supabase_service.client.storage.from_("scan-request-images").get_public_url(storage_path)

        return {
            "success": True,
            "path": storage_path,
            "url": public_url
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")
