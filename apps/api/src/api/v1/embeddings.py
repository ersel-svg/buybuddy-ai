"""Embeddings API router for models, jobs, and exports."""

from typing import Optional, Literal
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.qdrant import qdrant_service
from services.runpod import runpod_service, EndpointType
from services.export import export_service
from auth.dependencies import get_current_user
from auth.service import UserInfo
from config import settings

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class EmbeddingModelCreate(BaseModel):
    """Request to create an embedding model."""
    name: str
    model_type: Literal["dinov2-base", "dinov2-large", "custom"]
    model_path: Optional[str] = None
    checkpoint_url: Optional[str] = None
    embedding_dim: int
    config: Optional[dict] = None
    training_job_id: Optional[str] = None


class EmbeddingJobCreate(BaseModel):
    """Request to start an embedding job."""
    model_id: str
    job_type: Literal["full", "incremental"] = "full"
    source: Literal["cutouts", "products", "both"] = "cutouts"


class EmbeddingSyncRequest(BaseModel):
    """Request for synchronous embedding extraction."""
    model_id: Optional[str] = None  # If not provided, use active model
    source: Literal["cutouts", "products", "both"] = "both"
    batch_size: int = 50  # Max images per worker call
    limit: Optional[int] = None  # Max total images (None = all)


class ExportCreate(BaseModel):
    """Request to create an embedding export."""
    model_id: str
    format: Literal["json", "numpy", "faiss", "qdrant_snapshot"]


# Legacy schemas (for backward compatibility)
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
# Embedding Models Endpoints
# ===========================================


@router.get("/models")
async def list_embedding_models(
    db: SupabaseService = Depends(get_supabase),
):
    """List all embedding models."""
    return await db.get_embedding_models()


@router.get("/models/active")
async def get_active_model(
    db: SupabaseService = Depends(get_supabase),
):
    """Get the active embedding model for matching."""
    model = await db.get_active_embedding_model()
    if not model:
        raise HTTPException(status_code=404, detail="No active model found")
    return model


@router.get("/models/{model_id}")
async def get_embedding_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get embedding model details."""
    model = await db.get_embedding_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/models")
async def create_embedding_model(
    request: EmbeddingModelCreate,
    db: SupabaseService = Depends(get_supabase),
):
    """Create a new embedding model."""
    # Generate Qdrant collection name
    collection_name = f"embeddings_{request.name.lower().replace(' ', '_').replace('-', '_')}"

    model_data = {
        "name": request.name,
        "model_type": request.model_type,
        "model_path": request.model_path,
        "checkpoint_url": request.checkpoint_url,
        "embedding_dim": request.embedding_dim,
        "config": request.config or {},
        "qdrant_collection": collection_name,
        "training_job_id": request.training_job_id,
    }

    return await db.create_embedding_model(model_data)


@router.post("/models/{model_id}/activate")
async def activate_embedding_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Activate a model for matching (deactivates others)."""
    model = await db.get_embedding_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return await db.activate_embedding_model(model_id)


@router.delete("/models/{model_id}")
async def delete_embedding_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete an embedding model and its Qdrant collection."""
    model = await db.get_embedding_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Delete Qdrant collection if exists
    collection_name = model.get("qdrant_collection")
    if collection_name and qdrant_service.is_configured():
        try:
            await qdrant_service.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist

    await db.delete_embedding_model(model_id)
    return {"status": "deleted"}


# ===========================================
# Embedding Jobs Endpoints
# ===========================================


@router.get("/jobs")
async def list_embedding_jobs(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    db: SupabaseService = Depends(get_supabase),
):
    """List embedding jobs."""
    return await db.get_embedding_jobs(status=status, limit=limit)


@router.get("/jobs/{job_id}")
async def get_embedding_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get embedding job details."""
    job = await db.get_embedding_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs")
async def start_embedding_job(
    request: EmbeddingJobCreate,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start an embedding extraction job.

    This endpoint:
    1. Creates a job record for tracking
    2. Fetches images from Supabase
    3. Sends them to RunPod worker for embedding extraction
    4. Saves embeddings to Qdrant
    5. Updates job progress

    The processing happens synchronously but returns immediately with the job record.
    Poll the job status endpoint to check progress.
    """
    # Verify model exists
    model = await db.get_embedding_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_id = model["id"]
    model_type = model.get("model_type", "dinov2-base")
    embedding_dim = model.get("embedding_dim", 768)
    collection_name = model.get("qdrant_collection")

    if not collection_name:
        raise HTTPException(status_code=400, detail="Model has no Qdrant collection configured")

    # Check if there's already a running job for this model
    existing_jobs = await db.get_embedding_jobs(status="running")
    for job in existing_jobs:
        if job.get("embedding_model_id") == model_id:
            raise HTTPException(
                status_code=400,
                detail="There's already a running job for this model"
            )

    # Check services are configured
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Collect images to process
    images_to_process = []

    # Fetch cutouts
    if request.source in ["cutouts", "both"]:
        if request.job_type == "incremental":
            # Only cutouts without embeddings
            cutouts_result = db.client.table("cutout_images").select(
                "id, image_url, predicted_upc"
            ).eq("has_embedding", False).limit(10000).execute()
        else:
            # All cutouts
            cutouts_result = db.client.table("cutout_images").select(
                "id, image_url, predicted_upc"
            ).limit(10000).execute()

        for c in cutouts_result.data or []:
            images_to_process.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "predicted_upc": c.get("predicted_upc"),
                },
            })

    # Fetch products (frame_0000.png per product)
    if request.source in ["products", "both"]:
        products_result = db.client.table("products").select(
            "id, barcode, brand_name, product_name, frames_path, frame_count"
        ).gt("frame_count", 0).limit(10000).execute()

        for p in products_result.data or []:
            frames_path = p.get("frames_path", "")
            if not frames_path:
                continue

            frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
            images_to_process.append({
                "id": p["id"],
                "url": frame_url,
                "type": "product",
                "metadata": {
                    "source": "product",
                    "product_id": p["id"],
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                    "product_name": p.get("product_name"),
                },
            })

    total_images = len(images_to_process)

    if total_images == 0:
        raise HTTPException(status_code=400, detail="No images to process")

    # Create job record
    job_data = {
        "embedding_model_id": model_id,
        "job_type": request.job_type,
        "source": request.source,
        "status": "running",
        "total_images": total_images,
        "processed_images": 0,
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    # Ensure Qdrant collection exists
    await qdrant_service.create_collection(
        collection_name=collection_name,
        vector_size=embedding_dim,
        distance="Cosine",
    )

    # Process in batches
    batch_size = 50
    processed_count = 0
    failed_count = 0

    try:
        for i in range(0, total_images, batch_size):
            batch = images_to_process[i:i + batch_size]

            # Prepare worker input (matching worker's expected format)
            worker_input = {
                "images": [
                    {"id": img["id"], "url": img["url"], "type": img["type"]}
                    for img in batch
                ],
                "model_type": model_type,
                "batch_size": min(16, len(batch)),
            }

            try:
                # Call worker synchronously
                result = await runpod_service.submit_job_sync(
                    endpoint_type=EndpointType.EMBEDDING,
                    input_data=worker_input,
                    timeout=300,
                )

                # Check result
                output = result.get("output", {})
                if output.get("status") != "success":
                    error = output.get("error", "Unknown worker error")
                    print(f"Worker error: {error}")
                    failed_count += len(batch)
                    continue

                # Get embeddings from worker response
                embeddings = output.get("embeddings", [])
                embedding_map = {e["id"]: e for e in embeddings}

                # Prepare Qdrant points
                qdrant_points = []
                for img in batch:
                    emb_data = embedding_map.get(img["id"])
                    if emb_data and emb_data.get("vector"):
                        qdrant_points.append({
                            "id": img["id"],
                            "vector": emb_data["vector"],
                            "payload": img["metadata"],
                        })
                        processed_count += 1
                    else:
                        failed_count += 1

                # Upsert to Qdrant
                if qdrant_points:
                    await qdrant_service.upsert_points(collection_name, qdrant_points)

                # Update cutout records in Supabase
                for img in batch:
                    if img["type"] == "cutout" and img["id"] in embedding_map:
                        try:
                            db.client.table("cutout_images").update({
                                "has_embedding": True,
                                "embedding_model_id": model_id,
                                "qdrant_point_id": img["id"],
                            }).eq("id", img["id"]).execute()
                        except Exception as e:
                            print(f"Cutout update error: {e}")

            except Exception as e:
                print(f"Batch processing error: {e}")
                failed_count += len(batch)

            # Update job progress
            await db.update_embedding_job(job_id, {
                "processed_images": processed_count + failed_count,
            })

        # Mark job as completed
        await db.update_embedding_job(job_id, {
            "status": "completed",
            "processed_images": processed_count,
            "completed_at": datetime.utcnow().isoformat(),
        })

        # Update model's vector count
        try:
            info = await qdrant_service.get_collection_info(collection_name)
            qdrant_count = info.get("points_count", 0) if info else 0
            db.client.table("embedding_models").update({
                "qdrant_vector_count": qdrant_count,
            }).eq("id", model_id).execute()
        except Exception:
            pass

    except Exception as e:
        # Mark job as failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Job failed: {str(e)}")

    # Return updated job record
    return await db.get_embedding_job(job_id)


# ===========================================
# Embedding Exports Endpoints
# ===========================================


@router.get("/exports")
async def list_embedding_exports(
    limit: int = Query(50, ge=1, le=100),
    db: SupabaseService = Depends(get_supabase),
):
    """List embedding exports."""
    return await db.get_embedding_exports(limit=limit)


@router.get("/exports/{export_id}")
async def get_embedding_export(
    export_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get export details."""
    export = await db.get_embedding_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")
    return export


@router.post("/exports")
async def create_embedding_export(
    request: ExportCreate,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Create an embedding export.

    Exports all embeddings from a model's Qdrant collection in the specified format.
    Supported formats:
    - json: Human-readable JSON with vectors and metadata
    - numpy: Compressed .npz file with vectors, IDs, and payloads
    - faiss: FAISS index file + ID mapping for fast similarity search
    - qdrant_snapshot: Native Qdrant collection backup
    """
    # Verify model exists
    model = await db.get_embedding_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    collection_name = model.get("qdrant_collection")
    if not collection_name:
        raise HTTPException(
            status_code=400,
            detail="Model has no Qdrant collection"
        )

    # Check Qdrant is configured
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Create export record first
    export_data = {
        "embedding_model_id": request.model_id,
        "format": request.format,
        "vector_count": 0,
    }

    export = await db.create_embedding_export(export_data)
    export_id = export["id"]

    try:
        # Generate the export
        result = await export_service.export_embeddings(
            model_id=request.model_id,
            collection_name=collection_name,
            format=request.format,
            export_id=export_id,
        )

        # Update export record with results
        await db.update_embedding_export(export_id, {
            "file_url": result.get("file_url"),
            "file_size_bytes": result.get("file_size_bytes"),
            "vector_count": result.get("vector_count", 0),
        })

        # Return updated record
        export["file_url"] = result.get("file_url")
        export["file_size_bytes"] = result.get("file_size_bytes")
        export["vector_count"] = result.get("vector_count", 0)

        return export

    except Exception as e:
        # Clean up failed export
        print(f"Export error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export generation failed: {str(e)}"
        )


@router.get("/exports/{export_id}/download")
async def download_embedding_export(
    export_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get download URL for an embedding export.

    Returns a redirect to the file URL or the URL directly.
    """
    export = await db.get_embedding_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    file_url = export.get("file_url")
    if not file_url:
        raise HTTPException(status_code=400, detail="Export file not available")

    # Return the file URL - client can download directly
    return {
        "download_url": file_url,
        "format": export.get("format"),
        "vector_count": export.get("vector_count"),
        "file_size_bytes": export.get("file_size_bytes"),
    }


# ===========================================
# Qdrant Collection Stats
# ===========================================


@router.get("/models/{model_id}/qdrant-stats")
async def get_model_qdrant_stats(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get Qdrant collection stats for a model."""
    model = await db.get_embedding_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    collection_name = model.get("qdrant_collection")
    if not collection_name:
        return {"exists": False, "vector_count": 0}

    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        info = await qdrant_service.get_collection_info(collection_name)
        return {
            "exists": True,
            "collection_name": collection_name,
            "vector_count": info.get("vectors_count", 0),
            "points_count": info.get("points_count", 0),
            "status": info.get("status"),
        }
    except Exception:
        return {"exists": False, "vector_count": 0}


# ===========================================
# Sync Embedding Extraction (New Architecture)
# ===========================================


@router.post("/sync")
async def sync_embeddings(
    request: EmbeddingSyncRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Synchronously extract embeddings and save to Qdrant.

    This endpoint:
    1. Fetches images from Supabase (cutouts and/or products)
    2. Sends them to the RunPod worker for embedding extraction
    3. Receives embeddings and saves to local Qdrant
    4. Updates Supabase records

    Use this for smaller batches. For large-scale extraction, use the job-based flow.
    """
    # Get model (specified or active)
    if request.model_id:
        model = await db.get_embedding_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
    else:
        model = await db.get_active_embedding_model()
        if not model:
            raise HTTPException(status_code=404, detail="No active model found")

    model_id = model["id"]
    model_type = model.get("model_type", "dinov2-base")
    embedding_dim = model.get("embedding_dim", 768)
    collection_name = model.get("qdrant_collection")

    if not collection_name:
        raise HTTPException(status_code=400, detail="Model has no Qdrant collection configured")

    # Check RunPod is configured
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    # Check Qdrant is configured
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Ensure Qdrant collection exists
    await qdrant_service.create_collection(
        collection_name=collection_name,
        vector_size=embedding_dim,
        distance="Cosine",
    )

    # Collect images to process
    images_to_process = []

    # Fetch cutouts
    if request.source in ["cutouts", "both"]:
        cutouts_result = db.client.table("cutout_images").select(
            "id, image_url, predicted_upc"
        ).limit(request.limit or 10000).execute()

        for c in cutouts_result.data or []:
            images_to_process.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "predicted_upc": c.get("predicted_upc"),
                },
            })

    # Fetch products (frame_0000.png per product)
    if request.source in ["products", "both"]:
        products_result = db.client.table("products").select(
            "id, barcode, brand_name, product_name, frames_path, frame_count"
        ).gt("frame_count", 0).limit(request.limit or 10000).execute()

        for p in products_result.data or []:
            frames_path = p.get("frames_path", "")
            if not frames_path:
                continue

            frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
            images_to_process.append({
                "id": p["id"],
                "url": frame_url,
                "type": "product",
                "metadata": {
                    "source": "product",
                    "product_id": p["id"],
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                    "product_name": p.get("product_name"),
                },
            })

    if not images_to_process:
        return {
            "status": "success",
            "message": "No images to process",
            "processed_count": 0,
            "failed_count": 0,
            "qdrant_count": 0,
        }

    # Apply limit
    if request.limit:
        images_to_process = images_to_process[:request.limit]

    total_images = len(images_to_process)
    processed_count = 0
    failed_count = 0

    # Process in batches
    for i in range(0, total_images, request.batch_size):
        batch = images_to_process[i:i + request.batch_size]

        # Prepare worker input (new format)
        worker_input = {
            "images": [
                {"id": img["id"], "url": img["url"], "type": img["type"]}
                for img in batch
            ],
            "model_type": model_type,
            "batch_size": min(16, len(batch)),  # Worker internal batch size
        }

        try:
            # Call worker synchronously
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.EMBEDDING,
                input_data=worker_input,
                timeout=300,  # 5 minutes per batch
            )

            # Check result
            output = result.get("output", {})
            if output.get("status") != "success":
                error = output.get("error", "Unknown worker error")
                print(f"Worker error: {error}")
                failed_count += len(batch)
                continue

            # Get embeddings from worker response
            embeddings = output.get("embeddings", [])
            embedding_map = {e["id"]: e for e in embeddings}

            # Prepare Qdrant points
            qdrant_points = []
            for img in batch:
                emb_data = embedding_map.get(img["id"])
                if emb_data and emb_data.get("vector"):
                    qdrant_points.append({
                        "id": img["id"],
                        "vector": emb_data["vector"],
                        "payload": img["metadata"],
                    })
                    processed_count += 1
                else:
                    failed_count += 1

            # Upsert to Qdrant
            if qdrant_points:
                await qdrant_service.upsert_points(collection_name, qdrant_points)

            # Update cutout records in Supabase
            for img in batch:
                if img["type"] == "cutout" and img["id"] in embedding_map:
                    try:
                        db.client.table("cutout_images").update({
                            "has_embedding": True,
                            "embedding_model_id": model_id,
                            "qdrant_point_id": img["id"],
                        }).eq("id", img["id"]).execute()
                    except Exception as e:
                        print(f"Cutout update error: {e}")

        except Exception as e:
            print(f"Batch processing error: {e}")
            failed_count += len(batch)

    # Get final Qdrant count
    qdrant_count = 0
    try:
        info = await qdrant_service.get_collection_info(collection_name)
        qdrant_count = info.get("points_count", 0) if info else 0

        # Update model's vector count
        db.client.table("embedding_models").update({
            "qdrant_vector_count": qdrant_count,
        }).eq("id", model_id).execute()
    except Exception:
        pass

    return {
        "status": "success",
        "model_id": model_id,
        "collection_name": collection_name,
        "total_images": total_images,
        "processed_count": processed_count,
        "failed_count": failed_count,
        "qdrant_count": qdrant_count,
    }


# ===========================================
# Legacy Endpoints (for backward compatibility)
# ===========================================


@router.get("/indexes")
async def list_indexes(
    db: SupabaseService = Depends(get_supabase),
):
    """List all embedding indexes (legacy)."""
    return await db.get_embedding_indexes()


@router.post("/indexes")
async def create_index(
    request: CreateIndexRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Create a new embedding index (legacy)."""
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
    """Get embedding index details (legacy)."""
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
    """Add embeddings for products to an index (legacy)."""
    indexes = await db.get_embedding_indexes()
    index = next((idx for idx in indexes if idx.get("id") == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    added_count = len(request.product_ids)
    new_total = (index.get("vector_count", 0) or 0) + added_count

    return {"added_count": added_count, "total_count": new_total}


@router.delete("/indexes/{index_id}")
async def delete_index(
    index_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete an embedding index (legacy)."""
    indexes = await db.get_embedding_indexes()
    index = next((idx for idx in indexes if idx.get("id") == index_id), None)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    return {"status": "deleted"}
