"""Embeddings API router for managing embedding indexes."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class CreateIndexRequest(BaseModel):
    """Request to create an embedding index."""

    name: str
    model_id: str


class AddEmbeddingsRequest(BaseModel):
    """Request to add embeddings to an index."""

    product_ids: list[str]


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Endpoints
# ===========================================


@router.get("/indexes")
async def list_indexes(
    db: SupabaseService = Depends(get_supabase),
):
    """List all embedding indexes."""
    return await db.get_embedding_indexes()


@router.post("/indexes")
async def create_index(
    request: CreateIndexRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Create a new embedding index."""
    # Verify model exists
    models = await db.get_models()
    model = next((m for m in models if m.get("id") == request.model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return await db.create_embedding_index(request.name, request.model_id)


@router.get("/indexes/{index_id}")
async def get_index(
    index_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get embedding index details."""
    indexes = await db.get_embedding_indexes()
    index = next((idx for idx in indexes if idx.get("id") == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    return index


@router.post("/indexes/{index_id}/add")
async def add_embeddings_to_index(
    index_id: str,
    request: AddEmbeddingsRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Add embeddings for products to an index."""
    indexes = await db.get_embedding_indexes()
    index = next((idx for idx in indexes if idx.get("id") == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    # TODO: Actually compute and add embeddings via Runpod worker
    added_count = len(request.product_ids)
    new_total = (index.get("vector_count", 0) or 0) + added_count

    return {"added_count": added_count, "total_count": new_total}


@router.delete("/indexes/{index_id}")
async def delete_index(
    index_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete an embedding index."""
    indexes = await db.get_embedding_indexes()
    index = next((idx for idx in indexes if idx.get("id") == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    # TODO: Delete from Supabase
    # For now, just return success
    return {"status": "deleted"}
