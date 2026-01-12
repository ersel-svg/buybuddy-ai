"""Datasets API router for managing training datasets."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from config import settings

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class DatasetBase(BaseModel):
    """Base dataset schema."""

    name: str
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    """Dataset creation schema."""

    product_ids: Optional[list[str]] = None
    filters: Optional[dict] = None


class DatasetUpdate(BaseModel):
    """Dataset update schema."""

    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[int] = None  # For optimistic locking


class AddProductsRequest(BaseModel):
    """Request to add products to dataset."""

    product_ids: list[str]


class AugmentRequest(BaseModel):
    """Request to start augmentation."""

    syn_target: int = 600
    real_target: int = 400


class TrainRequest(BaseModel):
    """Request to start training."""

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


class ExtractRequest(BaseModel):
    """Request to extract embeddings."""

    model_id: str


# ===========================================
# Dependencies
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


def get_runpod() -> RunpodService:
    """Get Runpod service instance."""
    return runpod_service


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
async def list_datasets(
    db: SupabaseService = Depends(get_supabase),
):
    """List all datasets."""
    return await db.get_datasets()


@router.post("")
async def create_dataset(
    data: DatasetCreate,
    db: SupabaseService = Depends(get_supabase),
):
    """Create a new dataset."""
    dataset = await db.create_dataset({
        "name": data.name,
        "description": data.description,
    })

    # Add products if provided
    if data.product_ids:
        await db.add_products_to_dataset(dataset["id"], data.product_ids)
        dataset = await db.get_dataset(dataset["id"])

    return dataset


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get dataset with products."""
    dataset = await db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.patch("/{dataset_id}")
async def update_dataset(
    dataset_id: str,
    data: DatasetUpdate,
    db: SupabaseService = Depends(get_supabase),
):
    """Update dataset with optimistic locking."""
    existing = await db.get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = data.model_dump(exclude_unset=True, exclude={"version"})
    return await db.update_dataset(dataset_id, update_data)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a dataset."""
    existing = await db.get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await db.delete_dataset(dataset_id)
    return {"status": "deleted"}


@router.post("/{dataset_id}/products")
async def add_products_to_dataset(
    dataset_id: str,
    request: AddProductsRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Add products to dataset."""
    existing = await db.get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")

    added_count = await db.add_products_to_dataset(dataset_id, request.product_ids)
    return {"added_count": added_count}


@router.delete("/{dataset_id}/products/{product_id}")
async def remove_product_from_dataset(
    dataset_id: str,
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Remove product from dataset."""
    existing = await db.get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await db.remove_product_from_dataset(dataset_id, product_id)
    return {"status": "removed"}


# ===========================================
# Dataset Actions (GPU Jobs)
# ===========================================


@router.post("/{dataset_id}/augment")
async def start_augmentation(
    dataset_id: str,
    request_data: AugmentRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Start augmentation job for dataset."""
    dataset = await db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.get("product_count", 0) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no products")

    # Create job
    job = await db.create_job({
        "type": "augmentation",
        "config": {
            "dataset_id": dataset_id,
            **request_data.model_dump(),
        },
    })

    # Dispatch to Runpod augmentation worker
    if runpod.is_configured(EndpointType.AUGMENTATION):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.AUGMENTATION,
                input_data={
                    "dataset_id": dataset_id,
                    "syn_target": request_data.syn_target,
                    "real_target": request_data.real_target,
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

            print(f"[Datasets] Augmentation dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Datasets] Failed to dispatch augmentation: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Datasets] Runpod augmentation not configured, job created but not dispatched")

    return job


@router.post("/{dataset_id}/train")
async def start_training(
    dataset_id: str,
    request_data: TrainRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Start training job for dataset."""
    dataset = await db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.get("product_count", 0) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no products")

    # Create training job
    job = await db.create_training_job({
        "dataset_id": dataset_id,
        **request_data.model_dump(),
    })

    # Dispatch to Runpod training worker
    if runpod.is_configured(EndpointType.TRAINING):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.TRAINING,
                input_data={
                    "dataset_id": dataset_id,
                    **request_data.model_dump(),
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

            print(f"[Datasets] Training dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Datasets] Failed to dispatch training: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Datasets] Runpod training not configured, job created but not dispatched")

    return job


@router.post("/{dataset_id}/extract")
async def start_embedding_extraction(
    dataset_id: str,
    request_data: ExtractRequest,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """Start embedding extraction job for dataset."""
    dataset = await db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.get("product_count", 0) == 0:
        raise HTTPException(status_code=400, detail="Dataset has no products")

    # Create job
    job = await db.create_job({
        "type": "embedding_extraction",
        "config": {
            "dataset_id": dataset_id,
            "model_id": request_data.model_id,
        },
    })

    # Dispatch to Runpod embedding worker
    if runpod.is_configured(EndpointType.EMBEDDING):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data={
                    "dataset_id": dataset_id,
                    "model_id": request_data.model_id,
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

            print(f"[Datasets] Embedding extraction dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Datasets] Failed to dispatch embedding extraction: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Datasets] Runpod embedding not configured, job created but not dispatched")

    return job
