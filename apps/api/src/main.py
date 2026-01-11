"""
BuyBuddy AI Platform - FastAPI Backend

Main application entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings

# Import routers
from api.v1 import products, videos, datasets, jobs, training, matching, embeddings, webhooks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print(f"🚀 Starting {settings.app_name}")
    print(f"   Debug mode: {settings.debug}")
    print(f"   API prefix: {settings.api_prefix}")

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.app_name}


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


# Dashboard stats endpoint
@app.get(f"{settings.api_prefix}/dashboard/stats")
async def get_dashboard_stats() -> dict:
    """Get dashboard statistics."""
    # TODO: Implement with real data from Supabase
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
