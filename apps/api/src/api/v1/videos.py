"""Videos API router for syncing and processing videos."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class Video(BaseModel):
    """Video schema."""

    id: int
    barcode: str
    video_url: str
    status: str = "pending"
    product_id: Optional[str] = None
    created_at: datetime


class VideoSyncResponse(BaseModel):
    """Response from video sync."""

    synced_count: int
    new_videos: list[Video]


class ProcessVideoRequest(BaseModel):
    """Request to process a video."""

    video_id: int


class ProcessVideosRequest(BaseModel):
    """Request to process multiple videos."""

    video_ids: list[int]


class Job(BaseModel):
    """Job schema."""

    id: str
    type: str
    status: str
    progress: int = 0
    created_at: datetime


# ===========================================
# Mock Data
# ===========================================

MOCK_VIDEOS: list[dict] = []
MOCK_JOBS: list[dict] = []


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_videos() -> list[Video]:
    """List all videos."""
    return [Video(**v) for v in MOCK_VIDEOS]


@router.post("/sync", response_model=VideoSyncResponse)
async def sync_videos() -> VideoSyncResponse:
    """Sync videos from Buybuddy API."""
    # TODO: Implement actual sync with Buybuddy API
    # For now, return mock data
    new_videos = [
        Video(
            id=1001,
            barcode="0012345678905",
            video_url="https://example.com/video1.mp4",
            status="pending",
            created_at=datetime.now(),
        ),
        Video(
            id=1002,
            barcode="0012345678906",
            video_url="https://example.com/video2.mp4",
            status="pending",
            created_at=datetime.now(),
        ),
    ]

    return VideoSyncResponse(synced_count=len(new_videos), new_videos=new_videos)


@router.post("/process", response_model=Job)
async def process_video(request: ProcessVideoRequest) -> Job:
    """Process a single video through the pipeline."""
    # TODO: Dispatch to Runpod video-segmentation worker
    job = Job(
        id=str(uuid4()),
        type="video_processing",
        status="queued",
        progress=0,
        created_at=datetime.now(),
    )

    MOCK_JOBS.append(job.model_dump())
    return job


@router.post("/process/batch")
async def process_videos_batch(request: ProcessVideosRequest) -> list[Job]:
    """Process multiple videos through the pipeline."""
    jobs = []
    for video_id in request.video_ids:
        job = Job(
            id=str(uuid4()),
            type="video_processing",
            status="queued",
            progress=0,
            created_at=datetime.now(),
        )
        MOCK_JOBS.append(job.model_dump())
        jobs.append(job)

    return jobs


@router.get("/{video_id}")
async def get_video(video_id: int) -> Video:
    """Get video details."""
    video = next((v for v in MOCK_VIDEOS if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return Video(**video)
