"""Datasets API router for managing training datasets."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


class Dataset(DatasetBase):
    """Full dataset schema."""

    id: str
    product_count: int = 0
    version: int = 1
    created_at: datetime
    updated_at: datetime


class DatasetWithProducts(Dataset):
    """Dataset with products included."""

    products: list[dict] = []


class AddProductsRequest(BaseModel):
    """Request to add products to dataset."""

    product_ids: list[str]


class AugmentRequest(BaseModel):
    """Request to start augmentation."""

    syn_target_per_upc: int = 50
    real_target_per_upc: int = 20
    background_types: list[str] = ["solid", "gradient", "texture"]


class TrainRequest(BaseModel):
    """Request to start training."""

    model_name: str = "facebook/dinov2-large"
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.00002


class ExtractRequest(BaseModel):
    """Request to extract embeddings."""

    model_id: str


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

MOCK_DATASETS: list[dict] = [
    {
        "id": str(uuid4()),
        "name": "Beverages v1",
        "description": "All beverage products for training",
        "product_count": 150,
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "products": [],
    }
]


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_datasets() -> list[Dataset]:
    """List all datasets."""
    return [Dataset(**{k: v for k, v in d.items() if k != "products"}) for d in MOCK_DATASETS]


@router.post("", response_model=Dataset)
async def create_dataset(data: DatasetCreate) -> Dataset:
    """Create a new dataset."""
    dataset = {
        "id": str(uuid4()),
        "name": data.name,
        "description": data.description,
        "product_count": len(data.product_ids) if data.product_ids else 0,
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "products": [],
    }
    MOCK_DATASETS.append(dataset)
    return Dataset(**{k: v for k, v in dataset.items() if k != "products"})


@router.get("/{dataset_id}", response_model=DatasetWithProducts)
async def get_dataset(dataset_id: str) -> DatasetWithProducts:
    """Get dataset with products."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetWithProducts(**dataset)


@router.patch("/{dataset_id}", response_model=Dataset)
async def update_dataset(dataset_id: str, data: DatasetUpdate) -> Dataset:
    """Update dataset with optimistic locking."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Optimistic locking check
    if data.version is not None and data.version != dataset.get("version", 1):
        raise HTTPException(
            status_code=409,
            detail="Dataset was modified by another user. Please refresh and try again.",
        )

    update_data = data.model_dump(exclude_unset=True, exclude={"version"})
    dataset.update(update_data)
    dataset["updated_at"] = datetime.now().isoformat()
    dataset["version"] = dataset.get("version", 1) + 1

    return Dataset(**{k: v for k, v in dataset.items() if k != "products"})


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict[str, str]:
    """Delete a dataset."""
    global MOCK_DATASETS
    original_len = len(MOCK_DATASETS)
    MOCK_DATASETS = [d for d in MOCK_DATASETS if d["id"] != dataset_id]

    if len(MOCK_DATASETS) == original_len:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"status": "deleted"}


@router.post("/{dataset_id}/products")
async def add_products_to_dataset(dataset_id: str, request: AddProductsRequest) -> dict[str, int]:
    """Add products to dataset."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # TODO: Add actual products
    added_count = len(request.product_ids)
    dataset["product_count"] += added_count
    dataset["updated_at"] = datetime.now().isoformat()

    return {"added_count": added_count}


@router.delete("/{dataset_id}/products/{product_id}")
async def remove_product_from_dataset(dataset_id: str, product_id: str) -> dict[str, str]:
    """Remove product from dataset."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # TODO: Remove actual product
    dataset["product_count"] = max(0, dataset["product_count"] - 1)
    dataset["updated_at"] = datetime.now().isoformat()

    return {"status": "removed"}


# ===========================================
# Dataset Actions (GPU Jobs)
# ===========================================


@router.post("/{dataset_id}/augment", response_model=Job)
async def start_augmentation(dataset_id: str, request: AugmentRequest) -> Job:
    """Start augmentation job for dataset."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # TODO: Dispatch to Runpod augmentation worker
    job = Job(
        id=str(uuid4()),
        type="augmentation",
        status="queued",
        progress=0,
        created_at=datetime.now(),
    )

    return job


@router.post("/{dataset_id}/train", response_model=Job)
async def start_training(dataset_id: str, request: TrainRequest) -> Job:
    """Start training job for dataset."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # TODO: Dispatch to Runpod training worker
    job = Job(
        id=str(uuid4()),
        type="training",
        status="queued",
        progress=0,
        created_at=datetime.now(),
    )

    return job


@router.post("/{dataset_id}/extract", response_model=Job)
async def start_embedding_extraction(dataset_id: str, request: ExtractRequest) -> Job:
    """Start embedding extraction job for dataset."""
    dataset = next((d for d in MOCK_DATASETS if d["id"] == dataset_id), None)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # TODO: Dispatch to Runpod embedding worker
    job = Job(
        id=str(uuid4()),
        type="embedding_extraction",
        status="queued",
        progress=0,
        created_at=datetime.now(),
    )

    return job
