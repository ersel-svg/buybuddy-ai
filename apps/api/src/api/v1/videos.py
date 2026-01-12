"""Videos API router for syncing and processing videos."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from services.buybuddy import BuybuddyService, buybuddy_service
from config import settings

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class VideoSyncResponse(BaseModel):
    """Response from video sync."""

    synced_count: int


class ProcessVideoRequest(BaseModel):
    """Request to process a video."""

    video_id: int
    barcode: Optional[str] = None
    product_id: Optional[str] = None


class ProcessVideosRequest(BaseModel):
    """Request to process multiple videos."""

    video_ids: list[int]


class ProcessVideoByUrlRequest(BaseModel):
    """Request to process a video by URL directly."""

    video_url: str
    barcode: str
    product_id: Optional[str] = None


# ===========================================
# Dependencies
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


def get_runpod() -> RunpodService:
    """Get Runpod service instance."""
    return runpod_service


def get_buybuddy() -> BuybuddyService:
    """Get BuyBuddy service instance."""
    return buybuddy_service


# ===========================================
# Helper Functions
# ===========================================


def get_webhook_url(request: Request) -> str:
    """Build webhook URL for Runpod callbacks."""
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}{settings.api_prefix}/webhooks/runpod"


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_videos(
    db: SupabaseService = Depends(get_supabase),
):
    """List all videos."""
    return await db.get_videos()


@router.post("/sync")
async def sync_videos(
    limit: int = 50,
    db: SupabaseService = Depends(get_supabase),
    buybuddy: BuybuddyService = Depends(get_buybuddy),
) -> VideoSyncResponse:
    """Sync videos from Buybuddy API."""
    if not buybuddy.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured. Set BUYBUDDY_USERNAME and BUYBUDDY_PASSWORD.",
        )

    try:
        # Fetch products from BuyBuddy API
        products = await buybuddy.get_unprocessed_products(limit=limit)

        if not products:
            return VideoSyncResponse(synced_count=0)

        # Transform to video format
        videos = [
            {
                "barcode": p["barcode"],
                "video_url": p["video_url"],
                "video_id": p.get("video_id"),
                "status": "pending",
            }
            for p in products
        ]

        synced_count = await db.sync_videos_from_buybuddy(videos)
        return VideoSyncResponse(synced_count=synced_count)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync from BuyBuddy: {str(e)}",
        )


@router.post("/process")
async def process_video(
    request_data: ProcessVideoRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Process a single video through the pipeline."""
    # Get video details
    videos = await db.get_videos()
    video = next((v for v in videos if v.get("id") == request_data.video_id), None)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Create video processing job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_id": request_data.video_id,
            "video_url": video.get("video_url"),
            "barcode": request_data.barcode or video.get("barcode"),
            "product_id": request_data.product_id,
        },
    })

    # Dispatch to Runpod video-segmentation worker
    if runpod.is_configured(EndpointType.VIDEO):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.VIDEO,
                input_data={
                    "video_url": video.get("video_url"),
                    "barcode": request_data.barcode or video.get("barcode"),
                    "video_id": request_data.video_id,
                    "product_id": request_data.product_id,
                    "job_id": job["id"],
                },
                webhook_url=webhook_url,
            )

            # Update job with Runpod job ID
            await db.update_job(job["id"], {
                "status": "queued",
                "runpod_job_id": runpod_response.get("id"),
            })
            job["runpod_job_id"] = runpod_response.get("id")
            job["status"] = "queued"

            print(f"[Videos] Dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Videos] Failed to dispatch to Runpod: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Videos] Runpod not configured, job created but not dispatched")

    return job


@router.post("/process/url")
async def process_video_by_url(
    request_data: ProcessVideoByUrlRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Process a video directly by URL (without syncing first)."""
    # Create video processing job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_url": request_data.video_url,
            "barcode": request_data.barcode,
            "product_id": request_data.product_id,
        },
    })

    # Dispatch to Runpod video-segmentation worker
    if runpod.is_configured(EndpointType.VIDEO):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.VIDEO,
                input_data={
                    "video_url": request_data.video_url,
                    "barcode": request_data.barcode,
                    "product_id": request_data.product_id,
                    "job_id": job["id"],
                },
                webhook_url=webhook_url,
            )

            # Update job with Runpod job ID
            await db.update_job(job["id"], {
                "status": "queued",
                "runpod_job_id": runpod_response.get("id"),
            })
            job["runpod_job_id"] = runpod_response.get("id")
            job["status"] = "queued"

            print(f"[Videos] Dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Videos] Failed to dispatch to Runpod: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Videos] Runpod not configured, job created but not dispatched")

    return job


@router.post("/process/batch")
async def process_videos_batch(
    request_data: ProcessVideosRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Process multiple videos through the pipeline."""
    # Get all videos
    videos = await db.get_videos()
    video_map = {v.get("id"): v for v in videos}

    jobs = []
    webhook_url = get_webhook_url(request) if runpod.is_configured(EndpointType.VIDEO) else None

    for video_id in request_data.video_ids:
        video = video_map.get(video_id)
        if not video:
            continue

        # Create job
        job = await db.create_job({
            "type": "video_processing",
            "config": {
                "video_id": video_id,
                "video_url": video.get("video_url"),
                "barcode": video.get("barcode"),
            },
        })

        # Dispatch to Runpod
        if webhook_url and runpod.is_configured(EndpointType.VIDEO):
            try:
                runpod_response = await runpod.submit_job(
                    endpoint_type=EndpointType.VIDEO,
                    input_data={
                        "video_url": video.get("video_url"),
                        "barcode": video.get("barcode"),
                        "video_id": video_id,
                        "job_id": job["id"],
                    },
                    webhook_url=webhook_url,
                )

                await db.update_job(job["id"], {
                    "status": "queued",
                    "runpod_job_id": runpod_response.get("id"),
                })
                job["runpod_job_id"] = runpod_response.get("id")
                job["status"] = "queued"

            except Exception as e:
                print(f"[Videos] Failed to dispatch video {video_id}: {e}")
                await db.update_job(job["id"], {
                    "status": "failed",
                    "error": str(e),
                })

        jobs.append(job)

    return jobs


@router.get("/{video_id}")
async def get_video(
    video_id: int,
    db: SupabaseService = Depends(get_supabase),
):
    """Get video details."""
    # Get all videos and find the one with matching ID
    videos = await db.get_videos()
    video = next((v for v in videos if v.get("id") == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video
