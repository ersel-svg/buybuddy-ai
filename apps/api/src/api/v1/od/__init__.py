"""
Object Detection API Router

This module contains all endpoints for the object detection platform:
- Images: Upload, list, manage detection images
- Classes: Detection class management
- Datasets: Dataset CRUD and image organization
- Annotations: Bounding box annotation management
- Training: Detection model training (Phase 8)
- Models: Trained model registry (Phase 9)
"""

from fastapi import APIRouter

from services.supabase import supabase_service

# Import sub-routers
from .images import router as images_router
from .classes import router as classes_router
from .datasets import router as datasets_router
from .annotations import router as annotations_router
from .ai import router as ai_router
from .training import router as training_router
from .models import router as models_router

router = APIRouter()

# Include sub-routers
router.include_router(images_router, prefix="/images", tags=["OD Images"])
router.include_router(classes_router, prefix="/classes", tags=["OD Classes"])
router.include_router(datasets_router, prefix="/datasets", tags=["OD Datasets"])
router.include_router(annotations_router, prefix="/annotations", tags=["OD Annotations"])
router.include_router(ai_router, prefix="/ai", tags=["OD AI Annotation"])
router.include_router(training_router, prefix="/training", tags=["OD Training"])
router.include_router(models_router, prefix="/models", tags=["OD Models"])


# Health check endpoint for OD module
@router.get("/health")
async def od_health_check() -> dict[str, str]:
    """Object Detection module health check."""
    return {
        "status": "healthy",
        "module": "object-detection",
        "version": "0.1.0",
    }


# Stats endpoint for OD dashboard
@router.get("/stats")
async def get_od_stats() -> dict:
    """Get Object Detection dashboard statistics."""
    try:
        # Get total images
        images_result = supabase_service.client.table("od_images").select("id, status", count="exact").execute()
        total_images = images_result.count or 0

        # Count images by status
        images_by_status = {"pending": 0, "annotating": 0, "completed": 0, "skipped": 0}
        for img in images_result.data or []:
            status = img.get("status", "pending")
            if status in images_by_status:
                images_by_status[status] += 1

        # Get total datasets
        datasets_result = supabase_service.client.table("od_datasets").select("id", count="exact").execute()
        total_datasets = datasets_result.count or 0

        # Get total annotations
        annotations_result = supabase_service.client.table("od_annotations").select("id", count="exact").execute()
        total_annotations = annotations_result.count or 0

        # Get total classes
        classes_result = supabase_service.client.table("od_classes").select("id", count="exact").eq("is_active", True).execute()
        total_classes = classes_result.count or 0

        # Get total trained models
        models_result = supabase_service.client.table("od_trained_models").select("id", count="exact").execute()
        total_models = models_result.count or 0

        # Get recent datasets
        recent_datasets = supabase_service.client.table("od_datasets").select("*").order("created_at", desc=True).limit(5).execute()

        return {
            "total_images": total_images,
            "total_datasets": total_datasets,
            "total_annotations": total_annotations,
            "total_classes": total_classes,
            "total_models": total_models,
            "images_by_status": images_by_status,
            "recent_datasets": recent_datasets.data or [],
        }

    except Exception as e:
        print(f"OD Stats error: {e}")
        # Return zeros on error
        return {
            "total_images": 0,
            "total_datasets": 0,
            "total_annotations": 0,
            "total_classes": 0,
            "total_models": 0,
            "images_by_status": {"pending": 0, "annotating": 0, "completed": 0, "skipped": 0},
            "recent_datasets": [],
        }
