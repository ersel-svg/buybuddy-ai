"""Videos API router for syncing and processing videos."""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from services.buybuddy import BuybuddyService, buybuddy_service
from auth.dependencies import get_current_user
from config import settings

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


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
    chunk_size: int = 25  # Process in chunks to avoid rate limits


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

    barcode = request_data.barcode or video.get("barcode")
    product_id = request_data.product_id

    # Create product first if not provided (so pipeline has product_id for storage)
    if not product_id:
        product = await db.create_product({
            "barcode": barcode,
            "video_id": request_data.video_id,
            "video_url": video.get("video_url"),
            "status": "processing",
        })
        product_id = product["id"]
        print(f"[Videos] Created product {product_id} for barcode {barcode}")

    # Create video processing job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_id": request_data.video_id,
            "video_url": video.get("video_url"),
            "barcode": barcode,
            "product_id": product_id,
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
                    "barcode": barcode,
                    "video_id": request_data.video_id,
                    "product_id": product_id,
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
            # Update product status to failed
            await db.update_product(product_id, {"status": "pending"})
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
    product_id = request_data.product_id

    # Create product first if not provided (so pipeline has product_id for storage)
    if not product_id:
        product = await db.create_product({
            "barcode": request_data.barcode,
            "video_url": request_data.video_url,
            "status": "processing",
        })
        product_id = product["id"]
        print(f"[Videos] Created product {product_id} for barcode {request_data.barcode}")

    # Create video processing job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_url": request_data.video_url,
            "barcode": request_data.barcode,
            "product_id": product_id,
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
                    "product_id": product_id,
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
            # Update product status to pending on failure
            await db.update_product(product_id, {"status": "pending"})
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
    """Process multiple videos through the pipeline with chunking."""
    # Get all videos
    videos = await db.get_videos()
    video_map = {v.get("id"): v for v in videos}

    jobs = []
    webhook_url = get_webhook_url(request) if runpod.is_configured(EndpointType.VIDEO) else None
    chunk_size = request_data.chunk_size
    video_ids = request_data.video_ids
    total = len(video_ids)

    print(f"[Videos] Processing {total} videos in chunks of {chunk_size}")

    # Process in chunks
    for chunk_idx in range(0, total, chunk_size):
        chunk = video_ids[chunk_idx:chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        total_chunks = (total + chunk_size - 1) // chunk_size

        print(f"[Videos] Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} videos)")

        for video_id in chunk:
            video = video_map.get(video_id)
            if not video:
                continue

            barcode = video.get("barcode")

            # Create product first (so pipeline has product_id for storage)
            product = await db.create_product({
                "barcode": barcode,
                "video_id": video_id,
                "video_url": video.get("video_url"),
                "status": "processing",
            })
            product_id = product["id"]
            print(f"[Videos] Created product {product_id} for barcode {barcode}")

            # Create job
            job = await db.create_job({
                "type": "video_processing",
                "config": {
                    "video_id": video_id,
                    "video_url": video.get("video_url"),
                    "barcode": barcode,
                    "product_id": product_id,
                },
            })

            # Dispatch to Runpod
            if webhook_url and runpod.is_configured(EndpointType.VIDEO):
                try:
                    runpod_response = await runpod.submit_job(
                        endpoint_type=EndpointType.VIDEO,
                        input_data={
                            "video_url": video.get("video_url"),
                            "barcode": barcode,
                            "video_id": video_id,
                            "product_id": product_id,
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
                    # Update product status to pending on failure
                    await db.update_product(product_id, {"status": "pending"})

            jobs.append(job)

        # Small delay between chunks to avoid rate limits
        if chunk_idx + chunk_size < total:
            print(f"[Videos] Waiting 1s before next chunk...")
            await asyncio.sleep(1)

    print(f"[Videos] Batch complete: {len(jobs)} jobs created")
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


@router.post("/{video_id}/reprocess")
async def reprocess_video(
    video_id: int,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """
    Reprocess a video that was previously processed.
    This will delete old frames and re-run the entire pipeline.
    """
    # Get video details
    videos = await db.get_videos()
    video = next((v for v in videos if v.get("id") == video_id), None)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    product_id = video.get("product_id")
    if not product_id:
        raise HTTPException(
            status_code=400,
            detail="Video has no associated product. Use /process instead."
        )

    barcode = video.get("barcode")
    video_url = video.get("video_url")

    if not video_url:
        raise HTTPException(status_code=400, detail="Video has no URL")

    print(f"[Videos] Reprocessing video {video_id} (product: {product_id})")

    # 1. Clean up old data (frames + storage)
    cleanup_result = await db.cleanup_product_for_reprocess(product_id)
    print(f"[Videos] Cleanup: {cleanup_result['frames_deleted']} frames, {cleanup_result['files_deleted']} files deleted")

    # 2. Update video status back to pending
    db.client.table("videos").update({
        "status": "processing"
    }).eq("id", video_id).execute()

    # 3. Create new job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_id": video_id,
            "video_url": video_url,
            "barcode": barcode,
            "product_id": product_id,
            "reprocess": True,
        },
    })

    # 4. Dispatch to Runpod
    if runpod.is_configured(EndpointType.VIDEO):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.VIDEO,
                input_data={
                    "video_url": video_url,
                    "barcode": barcode,
                    "video_id": video_id,
                    "product_id": product_id,
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

            print(f"[Videos] Reprocess dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Videos] Failed to dispatch reprocess to Runpod: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            # Reset product status to previous state
            await db.update_product(product_id, {"status": "pending"})
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Videos] Runpod not configured, reprocess job created but not dispatched")

    return {
        "job": job,
        "cleanup": cleanup_result,
        "message": f"Reprocessing video {video_id}",
    }
