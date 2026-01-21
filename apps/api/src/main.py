"""
BuyBuddy AI Platform - FastAPI Backend

Main application entry point.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings

# Import routers
from api.v1 import products, videos, datasets, jobs, training, triplets, matching, embeddings, webhooks, auth, locks, cutouts, scan_requests, product_matcher
from api.v1.od import router as od_router

# Import services for dashboard
from services.supabase import supabase_service


async def resume_interrupted_roboflow_imports():
    """Resume any Roboflow imports that were interrupted by API restart."""
    import asyncio
    import logging
    from services.import_checkpoint import checkpoint_service

    logger = logging.getLogger(__name__)

    try:
        interrupted_jobs = checkpoint_service.get_interrupted_jobs()

        if not interrupted_jobs:
            return

        print(f"   Found {len(interrupted_jobs)} interrupted Roboflow import(s)")

        for job in interrupted_jobs:
            job_id = job["id"]
            checkpoint = checkpoint_service.load_checkpoint(job_id)
            config = job.get("config", {})

            if checkpoint and checkpoint.can_resume() and config.get("api_key"):
                # Can resume - ZIP exists
                print(f"   Resuming job {job_id[:8]}... from stage: {checkpoint.stage}")

                # Prepare request data
                request_data = {
                    "api_key": config.get("api_key"),
                    "workspace": config.get("workspace"),
                    "project": config.get("project"),
                    "version": config.get("version"),
                    "dataset_id": config.get("dataset_id"),
                    "format": config.get("format", "coco"),
                    "skip_duplicates": config.get("skip_duplicates", True),
                    "merge_annotations": config.get("merge_annotations", False),
                    "class_mapping": config.get("class_mapping", []),
                }

                # Import here to avoid circular imports
                from api.v1.od.images import _run_roboflow_import

                # Start background task to resume
                asyncio.create_task(_run_roboflow_import(job_id, request_data, checkpoint))

            else:
                # Cannot resume - mark as failed
                print(f"   Cannot resume job {job_id[:8]}... - marking as failed")
                supabase_service.client.table("jobs").update({
                    "status": "failed",
                    "error": "API restarted before import could be resumed. No valid checkpoint found.",
                }).eq("id", job_id).execute()

    except Exception as e:
        logger.warning(f"Failed to resume interrupted imports: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print(f"🚀 Starting {settings.app_name}")
    print(f"   Debug mode: {settings.debug}")
    print(f"   API prefix: {settings.api_prefix}")

    # Resume interrupted Roboflow imports
    # NOTE: Disabled for now - was causing startup hang
    # TODO: Fix the underlying issue in resume_interrupted_roboflow_imports
    # await resume_interrupted_roboflow_imports()

    # Cleanup old import files
    try:
        from services.import_checkpoint import checkpoint_service
        cleaned = checkpoint_service.cleanup_old_imports()
        if cleaned > 0:
            print(f"   Cleaned up {cleaned} old import directories")
    except Exception as e:
        print(f"   Warning: Failed to cleanup old imports: {e}")

    yield

    # Shutdown
    print(f"👋 Shutting down {settings.app_name}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered product video processing and matching platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow Vercel domains and localhost
# Using allow_origin_regex to support all Vercel preview deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://buybuddy-ai-web.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.app_name, "version": "2026-01-16-v3"}


# Include API routers
app.include_router(
    products.router,
    prefix=f"{settings.api_prefix}/products",
    tags=["Products"],
)

app.include_router(
    videos.router,
    prefix=f"{settings.api_prefix}/videos",
    tags=["Videos"],
)

app.include_router(
    datasets.router,
    prefix=f"{settings.api_prefix}/datasets",
    tags=["Datasets"],
)

app.include_router(
    jobs.router,
    prefix=f"{settings.api_prefix}/jobs",
    tags=["Jobs"],
)

app.include_router(
    training.router,
    prefix=f"{settings.api_prefix}/training",
    tags=["Training"],
)

app.include_router(
    matching.router,
    prefix=f"{settings.api_prefix}/matching",
    tags=["Matching"],
)

app.include_router(
    embeddings.router,
    prefix=f"{settings.api_prefix}/embeddings",
    tags=["Embeddings"],
)

app.include_router(
    webhooks.router,
    prefix=f"{settings.api_prefix}/webhooks",
    tags=["Webhooks"],
)

app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}/auth",
    tags=["Authentication"],
)

app.include_router(
    locks.router,
    prefix=f"{settings.api_prefix}/locks",
    tags=["Resource Locks"],
)

app.include_router(
    cutouts.router,
    prefix=f"{settings.api_prefix}/cutouts",
    tags=["Cutout Images"],
)

app.include_router(
    triplets.router,
    prefix=f"{settings.api_prefix}/triplets",
    tags=["Triplet Mining"],
)

app.include_router(
    scan_requests.router,
    prefix=f"{settings.api_prefix}/scan-requests",
    tags=["Scan Requests"],
)

app.include_router(
    product_matcher.router,
    prefix=f"{settings.api_prefix}/product-matcher",
    tags=["Product Matcher"],
)

# Object Detection module
app.include_router(
    od_router,
    prefix=f"{settings.api_prefix}/od",
    tags=["Object Detection"],
)


# Dashboard stats endpoint
@app.get(f"{settings.api_prefix}/dashboard/stats")
async def get_dashboard_stats() -> dict:
    """Get dashboard statistics from Supabase."""
    from datetime import datetime, timezone

    try:
        # Get total products count (supabase-py is sync, no await needed)
        products_result = supabase_service.client.table("products").select("id, status", count="exact").execute()
        total_products = products_result.count or 0

        # Count products by status
        products_by_status = {
            "pending": 0,
            "processing": 0,
            "needs_matching": 0,
            "ready": 0,
            "rejected": 0,
        }
        if products_result.data:
            for product in products_result.data:
                status = product.get("status", "pending")
                if status in products_by_status:
                    products_by_status[status] += 1

        # Get total datasets count
        datasets_result = supabase_service.client.table("datasets").select("id", count="exact").execute()
        total_datasets = datasets_result.count or 0

        # Get active jobs count (pending, queued, running)
        active_jobs_result = supabase_service.client.table("jobs").select("id", count="exact").in_("status", ["pending", "queued", "running"]).execute()
        active_jobs = active_jobs_result.count or 0

        # Get completed jobs today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        completed_today_result = supabase_service.client.table("jobs").select("id", count="exact").eq("status", "completed").gte("updated_at", today_start).execute()
        completed_jobs_today = completed_today_result.count or 0

        # Get recent products (last 10)
        recent_products_result = supabase_service.client.table("products").select("*").order("created_at", desc=True).limit(10).execute()
        recent_products = recent_products_result.data or []

        # Get recent jobs (last 10)
        recent_jobs_result = supabase_service.client.table("jobs").select("*").order("created_at", desc=True).limit(10).execute()
        recent_jobs = recent_jobs_result.data or []

        return {
            "total_products": total_products,
            "products_by_status": products_by_status,
            "total_datasets": total_datasets,
            "active_jobs": active_jobs,
            "completed_jobs_today": completed_jobs_today,
            "recent_products": recent_products,
            "recent_jobs": recent_jobs,
        }

    except Exception as e:
        # Return empty stats on error (don't break the dashboard)
        print(f"Dashboard stats error: {e}")
        return {
            "total_products": 0,
            "products_by_status": {
                "pending": 0,
                "processing": 0,
                "needs_matching": 0,
                "ready": 0,
                "rejected": 0,
            },
            "total_datasets": 0,
            "active_jobs": 0,
            "completed_jobs_today": 0,
            "recent_products": [],
            "recent_jobs": [],
        }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": type(exc).__name__},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
