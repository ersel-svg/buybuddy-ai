"""
Classification API Router

This module contains all endpoints for the image classification platform:
- Images: Upload, list, manage classification images
- Classes: Classification class management
- Datasets: Dataset CRUD and image organization
- Labels: Image-class label management
- Labeling: Labeling workflow endpoints
- Training: Classification model training
- Models: Trained model registry
- Predictions: Model inference
"""

from fastapi import APIRouter

from services.supabase import supabase_service

# Import sub-routers
from .images import router as images_router
from .classes import router as classes_router
from .datasets import router as datasets_router
from .labels import router as labels_router
from .labeling import router as labeling_router
from .training import router as training_router
from .models import router as models_router
from .ai import router as ai_router

router = APIRouter()

# Include sub-routers
router.include_router(images_router, prefix="/images", tags=["CLS Images"])
router.include_router(classes_router, prefix="/classes", tags=["CLS Classes"])
router.include_router(datasets_router, prefix="/datasets", tags=["CLS Datasets"])
router.include_router(labels_router, prefix="/labels", tags=["CLS Labels"])
router.include_router(labeling_router, prefix="/labeling", tags=["CLS Labeling"])
router.include_router(training_router, prefix="/training", tags=["CLS Training"])
router.include_router(models_router, prefix="/models", tags=["CLS Models"])
router.include_router(ai_router, prefix="/ai", tags=["CLS AI"])


# Health check endpoint
@router.get("/health")
async def cls_health_check() -> dict[str, str]:
    """Classification module health check."""
    return {
        "status": "healthy",
        "module": "classification",
        "version": "0.1.0",
    }


# Stats endpoint for dashboard
@router.get("/stats")
async def get_cls_stats() -> dict:
    """Get Classification dashboard statistics."""
    try:
        # Get total images
        images_result = supabase_service.client.table("cls_images").select("id, status", count="exact").execute()
        total_images = images_result.count or 0

        # Count images by status
        images_by_status = {"pending": 0, "labeled": 0, "review": 0, "completed": 0, "skipped": 0}
        for img in images_result.data or []:
            status = img.get("status", "pending")
            if status in images_by_status:
                images_by_status[status] += 1

        # Get total datasets
        datasets_result = supabase_service.client.table("cls_datasets").select("id", count="exact").execute()
        total_datasets = datasets_result.count or 0

        # Get total labels
        labels_result = supabase_service.client.table("cls_labels").select("id", count="exact").execute()
        total_labels = labels_result.count or 0

        # Get total classes
        classes_result = supabase_service.client.table("cls_classes").select("id", count="exact").eq("is_active", True).execute()
        total_classes = classes_result.count or 0

        # Get total trained models
        models_result = supabase_service.client.table("cls_trained_models").select("id", count="exact").execute()
        total_models = models_result.count or 0

        # Get active training runs
        active_training = supabase_service.client.table("cls_training_runs").select("id", count="exact").in_("status", ["pending", "preparing", "queued", "training"]).execute()
        active_training_count = active_training.count or 0

        # Get recent datasets
        recent_datasets = supabase_service.client.table("cls_datasets").select("*").order("created_at", desc=True).limit(5).execute()

        return {
            "total_images": total_images,
            "total_datasets": total_datasets,
            "total_labels": total_labels,
            "total_classes": total_classes,
            "total_models": total_models,
            "active_training_runs": active_training_count,
            "images_by_status": images_by_status,
            "recent_datasets": recent_datasets.data or [],
        }

    except Exception as e:
        print(f"CLS Stats error: {e}")
        return {
            "total_images": 0,
            "total_datasets": 0,
            "total_labels": 0,
            "total_classes": 0,
            "total_models": 0,
            "active_training_runs": 0,
            "images_by_status": {"pending": 0, "labeled": 0, "review": 0, "completed": 0, "skipped": 0},
            "recent_datasets": [],
        }
