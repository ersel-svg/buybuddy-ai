"""Training API router for managing training jobs and models."""

from io import BytesIO
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    dataset_id: str
    model_name: str = "facebook/dinov2-large"
    proj_dim: int = 512
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    llrd_decay: float = 0.95
    domain_aware_ratio: float = 0.3
    hard_negative_pool_size: int = 10
    use_hardest_negatives: bool = True
    use_mixed_precision: bool = True
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    save_every: int = 5
    seed: int = 42


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Endpoints
# ===========================================


@router.get("/jobs")
async def list_training_jobs(
    db: SupabaseService = Depends(get_supabase),
):
    """List all training jobs."""
    return await db.get_training_jobs()


@router.get("/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get training job details."""
    job = await db.get_job(job_id)
    if not job or job.get("type") != "training":
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    db: SupabaseService = Depends(get_supabase),
):
    """Start a new training job."""
    # Validate ratios
    total_ratio = config.train_ratio + config.valid_ratio + config.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Train/valid/test ratios must sum to 1.0, got {total_ratio}",
        )

    # Create training job
    job = await db.create_training_job(config.model_dump())

    # TODO: Dispatch to Runpod training worker

    return job


@router.get("/models")
async def list_models(
    db: SupabaseService = Depends(get_supabase),
):
    """List all trained models."""
    return await db.get_models()


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get model details."""
    models = await db.get_models()
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/models/{model_id}/activate")
async def activate_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Set a model as the active model for matching."""
    models = await db.get_models()
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return await db.activate_model(model_id)


@router.get("/models/{model_id}/download")
async def download_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Download model checkpoint."""
    models = await db.get_models()
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # TODO: Return actual model file from Supabase Storage
    # For now, return empty placeholder
    buffer = BytesIO(b"Model checkpoint placeholder")
    buffer.seek(0)

    model_name = model.get("name", "model")
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model_name}.pth"},
    )
