"""Training API router for managing training jobs and models."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class TrainingConfig(BaseModel):
    """Training configuration schema."""

    dataset_id: str
    model_name: str = "facebook/dinov2-large"
    proj_dim: int = 1024
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    label_smoothing: float = 0.2
    warmup_epochs: int = 5
    grad_clip: float = 0.5
    llrd_decay: float = 0.9
    domain_aware_ratio: float = 0.57
    hard_negative_pool_size: int = 5
    use_hardest_negatives: bool = True
    use_mixed_precision: bool = False
    train_ratio: float = 0.80
    valid_ratio: float = 0.10
    test_ratio: float = 0.10
    save_every: int = 1
    seed: int = 1337


class TrainingJob(BaseModel):
    """Training job schema."""

    id: str
    type: str = "training"
    status: str
    progress: int = 0
    dataset_id: str
    dataset_name: Optional[str] = None
    epochs: int
    epochs_completed: int = 0
    batch_size: int
    learning_rate: float
    final_loss: Optional[float] = None
    checkpoint_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ModelArtifact(BaseModel):
    """Trained model artifact schema."""

    id: str
    name: str
    version: str
    training_job_id: str
    checkpoint_url: str
    embedding_dim: int = 1024
    num_classes: int = 0
    final_loss: float
    is_active: bool = False
    created_at: datetime


class Job(BaseModel):
    """Generic job schema."""

    id: str
    type: str
    status: str
    progress: int = 0
    created_at: datetime


# ===========================================
# Mock Data
# ===========================================

MOCK_TRAINING_JOBS: list[dict] = [
    {
        "id": str(uuid4()),
        "type": "training",
        "status": "completed",
        "progress": 100,
        "dataset_id": "ds-001",
        "dataset_name": "Beverages v1",
        "epochs": 30,
        "epochs_completed": 30,
        "batch_size": 16,
        "learning_rate": 0.00002,
        "final_loss": 0.0234,
        "checkpoint_url": "/models/model_001.pth",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
]

MOCK_MODELS: list[dict] = [
    {
        "id": str(uuid4()),
        "name": "DINOv2-Large Beverages",
        "version": "v1.0",
        "training_job_id": MOCK_TRAINING_JOBS[0]["id"],
        "checkpoint_url": "/models/model_001.pth",
        "embedding_dim": 1024,
        "num_classes": 150,
        "final_loss": 0.0234,
        "is_active": True,
        "created_at": datetime.now().isoformat(),
    }
]


# ===========================================
# Endpoints
# ===========================================


@router.get("/jobs")
async def list_training_jobs() -> list[TrainingJob]:
    """List all training jobs."""
    return [TrainingJob(**j) for j in MOCK_TRAINING_JOBS]


@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str) -> TrainingJob:
    """Get training job details."""
    job = next((j for j in MOCK_TRAINING_JOBS if j["id"] == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingJob(**job)


@router.post("/start", response_model=Job)
async def start_training(config: TrainingConfig) -> Job:
    """Start a new training job."""
    # Validate ratios
    total_ratio = config.train_ratio + config.valid_ratio + config.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"Train/valid/test ratios must sum to 1.0, got {total_ratio}",
        )

    # TODO: Dispatch to Runpod training worker
    job = Job(
        id=str(uuid4()),
        type="training",
        status="queued",
        progress=0,
        created_at=datetime.now(),
    )

    return job


@router.get("/models")
async def list_models() -> list[ModelArtifact]:
    """List all trained models."""
    return [ModelArtifact(**m) for m in MOCK_MODELS]


@router.get("/models/{model_id}", response_model=ModelArtifact)
async def get_model(model_id: str) -> ModelArtifact:
    """Get model details."""
    model = next((m for m in MOCK_MODELS if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelArtifact(**model)


@router.post("/models/{model_id}/activate", response_model=ModelArtifact)
async def activate_model(model_id: str) -> ModelArtifact:
    """Set a model as the active model for matching."""
    model = next((m for m in MOCK_MODELS if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Deactivate all other models
    for m in MOCK_MODELS:
        m["is_active"] = False

    # Activate this model
    model["is_active"] = True

    return ModelArtifact(**model)


@router.get("/models/{model_id}/download")
async def download_model(model_id: str) -> StreamingResponse:
    """Download model checkpoint."""
    model = next((m for m in MOCK_MODELS if m["id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # TODO: Return actual model file from Supabase Storage
    # For now, return empty placeholder
    buffer = BytesIO(b"Model checkpoint placeholder")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={model['name']}.pth"},
    )
