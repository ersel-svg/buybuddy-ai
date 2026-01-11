"""Embeddings API router for managing embedding indexes."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class EmbeddingIndex(BaseModel):
    """Embedding index schema."""

    id: str
    name: str
    model_artifact_id: str
    model_name: Optional[str] = None
    vector_count: int = 0
    index_path: str
    created_at: datetime


class CreateIndexRequest(BaseModel):
    """Request to create an embedding index."""

    name: str
    model_id: str


class AddEmbeddingsRequest(BaseModel):
    """Request to add embeddings to an index."""

    product_ids: list[str]


# ===========================================
# Mock Data
# ===========================================

MOCK_INDEXES: list[dict] = [
    {
        "id": str(uuid4()),
        "name": "Beverages Index v1",
        "model_artifact_id": "model-001",
        "model_name": "DINOv2-Large Beverages",
        "vector_count": 15000,
        "index_path": "/indexes/beverages_v1.faiss",
        "created_at": datetime.now().isoformat(),
    }
]


# ===========================================
# Endpoints
# ===========================================


@router.get("/indexes")
async def list_indexes() -> list[EmbeddingIndex]:
    """List all embedding indexes."""
    return [EmbeddingIndex(**idx) for idx in MOCK_INDEXES]


@router.post("/indexes", response_model=EmbeddingIndex)
async def create_index(request: CreateIndexRequest) -> EmbeddingIndex:
    """Create a new embedding index."""
    index = {
        "id": str(uuid4()),
        "name": request.name,
        "model_artifact_id": request.model_id,
        "model_name": None,  # TODO: Look up model name
        "vector_count": 0,
        "index_path": f"/indexes/{request.name.lower().replace(' ', '_')}.faiss",
        "created_at": datetime.now().isoformat(),
    }

    MOCK_INDEXES.append(index)
    return EmbeddingIndex(**index)


@router.get("/indexes/{index_id}", response_model=EmbeddingIndex)
async def get_index(index_id: str) -> EmbeddingIndex:
    """Get embedding index details."""
    index = next((idx for idx in MOCK_INDEXES if idx["id"] == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    return EmbeddingIndex(**index)


@router.post("/indexes/{index_id}/add")
async def add_embeddings_to_index(index_id: str, request: AddEmbeddingsRequest) -> dict:
    """Add embeddings for products to an index."""
    index = next((idx for idx in MOCK_INDEXES if idx["id"] == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    # TODO: Actually compute and add embeddings
    added_count = len(request.product_ids)
    index["vector_count"] += added_count

    return {"added_count": added_count, "total_count": index["vector_count"]}


@router.delete("/indexes/{index_id}")
async def delete_index(index_id: str) -> dict[str, str]:
    """Delete an embedding index."""
    global MOCK_INDEXES
    original_len = len(MOCK_INDEXES)
    MOCK_INDEXES = [idx for idx in MOCK_INDEXES if idx["id"] != index_id]

    if len(MOCK_INDEXES) == original_len:
        raise HTTPException(status_code=404, detail="Index not found")

    return {"status": "deleted"}
