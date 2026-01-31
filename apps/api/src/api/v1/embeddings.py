"""Embeddings API router for models, jobs, and exports."""

from typing import Optional, Literal
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.qdrant import qdrant_service
from services.runpod import runpod_service, EndpointType
from services.job_manager import job_manager, JobMode
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
    model_type: Literal[
        # DINOv2 family
        "dinov2-small", "dinov2-base", "dinov2-large",
        # DINOv3 family
        "dinov3-small", "dinov3-base", "dinov3-large",
        # CLIP family
        "clip-vit-l-14",
        # Custom fine-tuned
        "custom"
    ]
    model_family: Optional[Literal["dinov2", "dinov3", "clip", "custom"]] = None
    hf_model_id: Optional[str] = None  # HuggingFace model ID
    model_path: Optional[str] = None
    checkpoint_url: Optional[str] = None
    embedding_dim: int
    config: Optional[dict] = None
    training_run_id: Optional[str] = None  # Link to training run
    base_model_id: Optional[str] = None  # Base model for fine-tuned models
    is_pretrained: bool = True


# ===========================================
# Advanced Extraction Config Schemas
# ===========================================


class ProductExtractionConfig(BaseModel):
    """Product embedding extraction configuration."""
    frame_selection: Literal["first", "key_frames", "interval"] = "first"
    frame_interval: int = 5  # Every Nth frame when using interval
    max_frames: int = 10  # Maximum frames per product
    include_augmented: bool = False


class CutoutExtractionConfig(BaseModel):
    """Cutout embedding extraction configuration."""
    filter_has_upc: bool = False
    synced_after: Optional[str] = None  # ISO datetime string
    batch_size: int = 100


class CollectionStrategy(BaseModel):
    """Qdrant collection strategy."""
    separate_collections: bool = False  # If True, products and cutouts go to separate collections
    collection_prefix: Optional[str] = None  # Custom prefix override


class EmbeddingJobCreate(BaseModel):
    """Request to start an embedding job (basic)."""
    model_id: str
    job_type: Literal["full", "incremental"] = "full"
    source: Literal["cutouts", "products", "both"] = "cutouts"


class EmbeddingJobCreateAdvanced(BaseModel):
    """Request to start an advanced embedding job with full configuration."""
    model_id: str
    job_type: Literal["full", "incremental"] = "incremental"
    source: Literal["cutouts", "products", "both"] = "both"

    # Advanced configuration
    product_config: ProductExtractionConfig = ProductExtractionConfig()
    cutout_config: CutoutExtractionConfig = CutoutExtractionConfig()
    collection_strategy: CollectionStrategy = CollectionStrategy()


class CollectionInfo(BaseModel):
    """Response for collection information."""
    name: str
    vectors_count: int
    vector_size: int
    status: str  # "green", "yellow", "red"
    points_count: int = 0
    size_bytes: Optional[int] = None
    model_name: Optional[str] = None


# ===========================================
# New 3-Tab Extraction Schemas
# ===========================================


class MatchingExtractionRequest(BaseModel):
    """
    Request for Tab 1: Matching Extraction.
    Products + Cutouts → Qdrant for similarity search on matching page.
    """
    model_id: Optional[str] = None  # Use active model if not specified

    # Product source configuration
    product_source: Literal["all", "selected", "dataset", "filter", "new"] = "all"
    product_ids: Optional[list[str]] = None  # For "selected" source
    product_dataset_id: Optional[str] = None  # For "dataset" source
    product_filter: Optional[dict] = None  # For "filter" source

    # Cutout source
    include_cutouts: bool = True
    cutout_filter_has_upc: bool = False
    cutout_merchant_ids: Optional[list[int]] = None  # Filter cutouts by merchant IDs

    # Collection configuration
    collection_mode: Literal["create", "append"] = "create"
    product_collection_name: Optional[str] = None  # Auto-generated if not provided
    cutout_collection_name: Optional[str] = None  # Auto-generated if not provided

    # Frame selection for products
    frame_selection: Literal["first", "key_frames", "interval", "all"] = "first"
    frame_interval: int = 5
    max_frames: int = 10  # Ignored when frame_selection = "all"


class TrainingExtractionRequest(BaseModel):
    """
    Request for Tab 2: Training Extraction.
    Matched products (synthetic + real + augmented) → Qdrant or file for triplet mining.
    """
    model_id: Optional[str] = None  # Use active model if not specified

    # Image types to include
    image_types: list[Literal["synthetic", "real", "augmented"]] = ["synthetic", "real"]

    # Frame selection for synthetic images
    frame_selection: Literal["first", "key_frames", "interval", "all"] = "key_frames"
    frame_interval: int = 5
    max_frames: int = 10  # Ignored when frame_selection = "all"

    # Include matched cutouts as positives
    include_matched_cutouts: bool = True

    # Output configuration
    output_target: Literal["qdrant", "file"] = "qdrant"
    collection_mode: Literal["create", "append"] = "create"
    collection_name: Optional[str] = None  # Auto-generated if not provided

    # For file output
    file_format: Literal["npz", "json"] = "npz"


class EvaluationExtractionRequest(BaseModel):
    """
    Request for Tab 3: Evaluation Extraction.
    Dataset products → Qdrant with trained model for evaluation.
    """
    model_id: str  # Required - specific trained model to evaluate
    dataset_id: str  # Required - dataset to evaluate

    # Image types to include
    image_types: list[Literal["synthetic", "real", "augmented"]] = ["synthetic"]

    # Frame selection
    frame_selection: Literal["first", "key_frames", "interval", "all"] = "first"
    frame_interval: int = 5
    max_frames: int = 4

    # Collection configuration
    collection_mode: Literal["create", "append"] = "create"
    collection_name: Optional[str] = None  # Auto-generated if not provided


class ProductionExtractionRequest(BaseModel):
    """
    Request for Tab 4: Production Extraction.
    All products with ALL image types → Single Qdrant collection for inference.

    This creates a production-ready embedding index with:
    - Synthetic frames (360° video frames)
    - Real images (matched cutouts)
    - Augmented versions of both

    SOTA approach: Multi-view + augmentation at index time for better recall.
    """
    # Model selection - either base model OR trained model
    model_id: Optional[str] = None  # Base model ID
    training_run_id: Optional[str] = None  # For trained model
    checkpoint_id: Optional[str] = None  # Specific checkpoint (required if training_run_id set)

    # Product source
    product_source: Literal["all", "dataset", "selected"] = "all"
    product_dataset_id: Optional[str] = None  # For "dataset" source
    product_ids: Optional[list[str]] = None  # For "selected" source

    # Image types to include (default: ALL for production)
    image_types: list[Literal["synthetic", "real", "augmented"]] = ["synthetic", "real", "augmented"]

    # Frame selection
    frame_selection: Literal["first", "key_frames", "interval", "all"] = "key_frames"
    frame_interval: int = 5
    max_frames: int = 10  # Per image type

    # Collection configuration
    collection_mode: Literal["create", "replace"] = "create"
    collection_name: Optional[str] = None  # Auto-generated: production_{model_name}


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
    include_pretrained: bool = True,
    db: SupabaseService = Depends(get_supabase),
):
    """List all embedding models."""
    models = await db.get_embedding_models()
    if not include_pretrained:
        models = [m for m in models if not m.get("is_pretrained")]
    return models


@router.get("/models/presets")
async def get_model_presets():
    """
    Get available pretrained model configurations.

    Returns model presets that can be used for training or embedding extraction.
    """
    presets = [
        # DINOv2 family
        {
            "model_type": "dinov2-small",
            "model_family": "dinov2",
            "name": "DINOv2 Small",
            "hf_model_id": "facebook/dinov2-small",
            "embedding_dim": 384,
            "image_size": 518,
            "description": "Smallest DINOv2 model, fastest inference",
            "recommended_for": ["quick experiments", "limited compute"],
        },
        {
            "model_type": "dinov2-base",
            "model_family": "dinov2",
            "name": "DINOv2 Base",
            "hf_model_id": "facebook/dinov2-base",
            "embedding_dim": 768,
            "image_size": 518,
            "description": "Balanced DINOv2 model, good quality/speed tradeoff",
            "recommended_for": ["production", "fine-tuning"],
        },
        {
            "model_type": "dinov2-large",
            "model_family": "dinov2",
            "name": "DINOv2 Large",
            "hf_model_id": "facebook/dinov2-large",
            "embedding_dim": 1024,
            "image_size": 518,
            "description": "Largest DINOv2 model, best quality",
            "recommended_for": ["maximum quality", "evaluation"],
        },
        # DINOv3 family
        {
            "model_type": "dinov3-small",
            "model_family": "dinov3",
            "name": "DINOv3 Small",
            "hf_model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "embedding_dim": 384,
            "image_size": 518,
            "description": "Latest DINOv3 small model with improved features",
            "recommended_for": ["quick experiments", "limited compute"],
        },
        {
            "model_type": "dinov3-base",
            "model_family": "dinov3",
            "name": "DINOv3 Base",
            "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "embedding_dim": 768,
            "image_size": 518,
            "description": "DINOv3 base with state-of-the-art self-supervised features",
            "recommended_for": ["production", "fine-tuning"],
        },
        {
            "model_type": "dinov3-large",
            "model_family": "dinov3",
            "name": "DINOv3 Large",
            "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "embedding_dim": 1024,
            "image_size": 518,
            "description": "Largest DINOv3 model, highest quality features",
            "recommended_for": ["maximum quality", "evaluation"],
        },
        # CLIP family
        {
            "model_type": "clip-vit-b-16",
            "model_family": "clip",
            "name": "CLIP ViT-B/16",
            "hf_model_id": "openai/clip-vit-base-patch16",
            "embedding_dim": 512,
            "image_size": 224,
            "description": "CLIP base model with 16x16 patches, good for text-image tasks",
            "recommended_for": ["multi-modal", "text-aware matching"],
        },
        {
            "model_type": "clip-vit-b-32",
            "model_family": "clip",
            "name": "CLIP ViT-B/32",
            "hf_model_id": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "image_size": 224,
            "description": "CLIP base model with 32x32 patches, faster than ViT-B/16",
            "recommended_for": ["fast inference", "text-aware matching"],
        },
        {
            "model_type": "clip-vit-l-14",
            "model_family": "clip",
            "name": "CLIP ViT-L/14",
            "hf_model_id": "openai/clip-vit-large-patch14",
            "embedding_dim": 768,
            "image_size": 224,
            "description": "Largest CLIP model, best multi-modal performance",
            "recommended_for": ["maximum quality", "text-aware matching"],
        },
    ]

    return {
        "presets": presets,
        "families": [
            {
                "id": "dinov2",
                "name": "DINOv2",
                "description": "Self-supervised Vision Transformer by Meta AI",
            },
            {
                "id": "dinov3",
                "name": "DINOv3",
                "description": "Latest self-supervised model by Meta AI",
            },
            {
                "id": "clip",
                "name": "CLIP",
                "description": "Contrastive Language-Image Pretraining by OpenAI",
            },
        ],
    }


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
    model_name_slug = request.name.lower().replace(' ', '_').replace('-', '_')
    collection_name = f"embeddings_{model_name_slug}"

    # Auto-detect model family from model_type if not provided
    model_family = request.model_family
    if not model_family:
        if request.model_type.startswith("dinov2"):
            model_family = "dinov2"
        elif request.model_type.startswith("dinov3"):
            model_family = "dinov3"
        elif request.model_type.startswith("clip"):
            model_family = "clip"
        else:
            model_family = "custom"

    # Auto-detect HuggingFace model ID if not provided
    hf_model_id = request.hf_model_id
    if not hf_model_id and request.is_pretrained:
        HF_MODEL_MAP = {
            "dinov2-small": "facebook/dinov2-small",
            "dinov2-base": "facebook/dinov2-base",
            "dinov2-large": "facebook/dinov2-large",
            "dinov3-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "clip-vit-b-16": "openai/clip-vit-base-patch16",
            "clip-vit-b-32": "openai/clip-vit-base-patch32",
            "clip-vit-l-14": "openai/clip-vit-large-patch14",
        }
        hf_model_id = HF_MODEL_MAP.get(request.model_type)

    model_data = {
        "name": request.name,
        "model_type": request.model_type,
        "model_family": model_family,
        "hf_model_id": hf_model_id,
        "model_path": request.model_path,
        "checkpoint_url": request.checkpoint_url,
        "embedding_dim": request.embedding_dim,
        "config": request.config or {},
        "qdrant_collection": collection_name,
        "product_collection": f"products_{model_name_slug}",
        "cutout_collection": f"cutouts_{model_name_slug}",
        "is_pretrained": request.is_pretrained,
        "base_model_id": request.base_model_id,
    }

    # Link to training run if provided
    if request.training_run_id:
        model_data["training_run_id"] = request.training_run_id

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


@router.post("/jobs/{job_id}/cancel")
async def cancel_embedding_job(
    job_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Cancel a running embedding extraction job.

    Cancels the job on Runpod and updates status to 'cancelled'.
    Only jobs with status 'queued' or 'running' can be cancelled.
    """
    # Get job
    job = await db.get_embedding_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job can be cancelled
    status = job.get("status")
    if status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {status}",
        )

    # Cancel on Runpod if job_id exists
    runpod_job_id = job.get("runpod_job_id")
    if runpod_job_id:
        try:
            await runpod_service.cancel_job(
                endpoint_type=EndpointType.EMBEDDING,
                job_id=runpod_job_id,
            )
            print(f"[Embeddings] Cancelled Runpod job {runpod_job_id}")
        except Exception as e:
            print(f"[Embeddings] Failed to cancel Runpod job {runpod_job_id}: {e}")
            # Continue to update DB status even if Runpod cancel fails

    # Update job status to cancelled
    await db.update_embedding_job(job_id, {
        "status": "cancelled",
    })

    print(f"[Embeddings] Job {job_id} cancelled")

    return {
        "status": "cancelled",
        "message": "Job cancelled successfully",
        "job_id": job_id,
    }


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
        "status": "queued",  # Start as queued, worker will update to running
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

    # NEW ARCHITECTURE (Phase 2 Migration):
    # Submit single job to worker with all images
    # Worker will handle batch processing, Qdrant upserts, and progress updates
    try:
        # Prepare job input for worker
        job_input = {
            "job_id": job_id,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "collection_name": collection_name,
            # Send all images to worker
            "images": [
                {
                    "id": img["id"],
                    "url": img["url"],
                    "type": img["type"],
                    "metadata": img["metadata"],
                }
                for img in images_to_process
            ],
            # Worker credentials
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,
        }

        # Submit job using JobManager (ASYNC mode)
        # NOTE: Worker must be updated to handle this new format
        # TODO (Worker): Implement batch processing loop (50 images per batch)
        # TODO (Worker): Upsert embeddings to Qdrant
        # TODO (Worker): Update progress to Supabase (embedding_jobs table)
        # TODO (Worker): Update cutout_images.has_embedding = True
        # TODO (Worker): Set final status (completed/failed)
        job_result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=job_input,
            mode=JobMode.ASYNC,  # Fire and forget, worker updates DB
            webhook_url=None,  # Worker does direct Supabase writes (OD training pattern)
        )

        # Store runpod_job_id for cancellation support
        await db.update_embedding_job(job_id, {
            "runpod_job_id": job_result["job_id"],
            "status": "running",
        })

        print(f"[EMBEDDING] Job {job_id} submitted to Runpod: {job_result['job_id']}")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[EMBEDDING] Error submitting job {job_id}: {e}")
        print(f"[EMBEDDING] Traceback:\n{error_traceback}")

        # Update status to failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise

    # Return job record (job will be processed by worker)
    return job


# OLD IMPLEMENTATION (kept as reference, remove after worker migration):
# ==============================================================================
# This was the previous batch-by-batch processing approach.
# Problems:
# - Used submit_job_sync() which blocks API for up to 5 minutes per batch
# - No runpod_job_id tracking, so jobs were NOT cancellable
# - API handled batch processing instead of worker
# - For 10,000 images = 200 batches × 300s = 16 hours of blocking!
#
# Process in batches:
# for i in range(0, total_images, batch_size=50):
#     batch = images_to_process[i:i + batch_size]
#     result = await runpod_service.submit_job_sync(...)  # NO JOB ID!
#     # Process embeddings, upsert to Qdrant
#     # Update progress
#
# Mark job as completed
# Update model's vector count
# ==============================================================================


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

    # Create a job record for tracking (synchronous endpoints also use jobs now)
    job_data = {
        "embedding_model_id": model_id,
        "job_type": "sync",
        "source": request.source,
        "status": "queued",
        "total_images": total_images,
        "processed_images": 0,
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    # NEW ARCHITECTURE (Phase 2 Migration):
    # Submit single job to worker - same pattern as /jobs endpoint
    try:
        # Prepare job input for worker
        job_input = {
            "job_id": job_id,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "collection_name": collection_name,
            # Send all images to worker
            "images": [
                {
                    "id": img["id"],
                    "url": img["url"],
                    "type": img["type"],
                    "metadata": img["metadata"],
                }
                for img in images_to_process
            ],
            # Worker credentials
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,
        }

        # Submit job using JobManager (ASYNC mode)
        job_result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=job_input,
            mode=JobMode.ASYNC,
            webhook_url=None,  # Worker does direct Supabase writes
        )

        # Store runpod_job_id for cancellation
        await db.update_embedding_job(job_id, {
            "runpod_job_id": job_result["job_id"],
            "status": "running",
        })

        print(f"[EMBEDDING] Sync job {job_id} submitted to Runpod: {job_result['job_id']}")

        # Return immediately - job will be processed by worker
        return {
            "status": "submitted",
            "job_id": job_id,
            "runpod_job_id": job_result["job_id"],
            "model_id": model_id,
            "collection_name": collection_name,
            "total_images": total_images,
            "message": "Job submitted. Poll /embeddings/jobs/{job_id} for progress.",
        }

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[EMBEDDING] Error submitting sync job {job_id}: {e}")
        print(f"[EMBEDDING] Traceback:\n{error_traceback}")

        # Update status to failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

    # OLD IMPLEMENTATION (removed):
    # - Batch-by-batch processing with submit_job_sync()
    # - Blocked API for total_images / batch_size × 5 minutes
    # - Not cancellable
    # - API handled Qdrant upserts


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


# ===========================================
# Qdrant Collections Endpoints
# ===========================================


@router.get("/collections")
async def list_collections(
    db: SupabaseService = Depends(get_supabase),
):
    """
    List all Qdrant collections with their stats.

    Returns collection info including vector count, dimension, and status.
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        collections = await qdrant_service.list_collections()
        result = []

        for collection_name in collections:
            try:
                info = await qdrant_service.get_collection_info(collection_name)
                if info:
                    result.append({
                        "name": collection_name,
                        "vectors_count": info.get("vectors_count", 0),
                        "points_count": info.get("points_count", 0),
                        "vector_size": info.get("vector_size", 0),
                        "status": info.get("status", "unknown"),
                    })
            except Exception as e:
                print(f"Error getting info for collection {collection_name}: {e}")
                result.append({
                    "name": collection_name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "vector_size": 0,
                    "status": "error",
                })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get detailed stats for a specific Qdrant collection."""
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        info = await qdrant_service.get_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail="Collection not found")

        return {
            "name": collection_name,
            "vectors_count": info.get("vectors_count", 0),
            "points_count": info.get("points_count", 0),
            "vector_size": info.get("vector_size", 0),
            "status": info.get("status", "unknown"),
            "segments_count": info.get("segments_count", 0),
            "optimizer_status": info.get("optimizer_status"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection stats: {str(e)}")


@router.delete("/collections/{collection_name}")
async def delete_qdrant_collection(
    collection_name: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Delete a Qdrant collection.

    Warning: This will permanently delete all vectors in the collection.
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        await qdrant_service.delete_collection(collection_name)

        # Also update any embedding_models that reference this collection
        try:
            db.client.table("embedding_models").update({
                "qdrant_vector_count": 0,
            }).eq("qdrant_collection", collection_name).execute()

            db.client.table("embedding_models").update({
                "qdrant_vector_count": 0,
            }).eq("product_collection", collection_name).execute()

            db.client.table("embedding_models").update({
                "qdrant_vector_count": 0,
            }).eq("cutout_collection", collection_name).execute()
        except Exception:
            pass  # Ignore DB update errors

        return {"status": "deleted", "collection_name": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


# ===========================================
# Collection Index Management
# ===========================================


class CreateIndexRequest(BaseModel):
    """Request to create a payload index."""
    field_name: str
    field_type: str = "keyword"  # keyword, integer, float, bool, geo, text


class IndexInfo(BaseModel):
    """Info about a payload index."""
    field_name: str
    data_type: str
    points: int = 0


class EnsureIndexesResponse(BaseModel):
    """Response from ensure_indexes operation."""
    collection_name: str
    created: list[str]
    existing: list[str]
    failed: list[dict]


@router.get("/collections/{collection_name}/indexes")
async def get_collection_indexes(collection_name: str):
    """
    Get all payload indexes on a collection.
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        indexes = await qdrant_service.get_collection_indexes(collection_name)
        return {"collection_name": collection_name, "indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get indexes: {str(e)}")


@router.post("/collections/{collection_name}/indexes")
async def create_collection_index(
    collection_name: str,
    request: CreateIndexRequest,
):
    """
    Create a payload index on a collection for efficient filtering.

    Field types:
    - keyword: For string fields (source, product_id, barcode)
    - bool: For boolean fields (is_primary)
    - integer: For integer fields
    - float: For float fields
    - text: For full-text search
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        await qdrant_service.create_payload_index(
            collection_name=collection_name,
            field_name=request.field_name,
            field_type=request.field_type,
        )
        return {
            "status": "created",
            "collection_name": collection_name,
            "field_name": request.field_name,
            "field_type": request.field_type,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {str(e)}")


@router.post("/collections/{collection_name}/indexes/ensure")
async def ensure_collection_indexes(collection_name: str):
    """
    Ensure all required indexes exist on a collection.
    Creates missing indexes without affecting existing ones.

    Required indexes:
    - source (keyword): Filter by source type (product/cutout)
    - is_primary (bool): Filter primary frames
    - product_id (keyword): Filter by product
    - barcode (keyword): Filter by barcode
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        results = await qdrant_service.ensure_indexes(collection_name)
        return EnsureIndexesResponse(
            collection_name=collection_name,
            created=results["created"],
            existing=results["existing"],
            failed=results["failed"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ensure indexes: {str(e)}")


@router.post("/collections/ensure-all-indexes")
async def ensure_all_collections_indexes():
    """
    Ensure required indexes exist on ALL collections.
    Useful for migrating existing collections.
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        collections = await qdrant_service.list_collections()
        results = {}

        for collection_name in collections:
            try:
                result = await qdrant_service.ensure_indexes(collection_name)
                results[collection_name] = {
                    "status": "success",
                    "created": result["created"],
                    "existing": result["existing"],
                    "failed": result["failed"],
                }
            except Exception as e:
                results[collection_name] = {
                    "status": "error",
                    "error": str(e),
                }

        return {"collections": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ensure indexes: {str(e)}")


# ===========================================
# Collection Products Endpoint
# ===========================================


class CollectionProductsResponse(BaseModel):
    """Response for products in a collection."""

    products: list[dict]
    total_count: int
    page: int
    limit: int
    collection_name: str


@router.get("/collections/{collection_name}/products")
async def get_collection_products(
    collection_name: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None, description="Search by product name or barcode"),
    # Product filters (comma-separated for multi-select)
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    sub_brand: Optional[str] = Query(None),
    product_name: Optional[str] = Query(None),
    variant_flavor: Optional[str] = Query(None),
    container_type: Optional[str] = Query(None),
    net_quantity: Optional[str] = Query(None),
    pack_type: Optional[str] = Query(None),
    manufacturer_country: Optional[str] = Query(None),
    claims: Optional[str] = Query(None),
    # Boolean filters
    has_video: Optional[bool] = Query(None),
    has_image: Optional[bool] = Query(None),
    has_nutrition: Optional[bool] = Query(None),
    has_description: Optional[bool] = Query(None),
    has_prompt: Optional[bool] = Query(None),
    has_issues: Optional[bool] = Query(None),
    # Range filters
    frame_count_min: Optional[int] = Query(None),
    frame_count_max: Optional[int] = Query(None),
    visibility_score_min: Optional[float] = Query(None),
    visibility_score_max: Optional[float] = Query(None),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get products that have embeddings in a specific Qdrant collection.

    Strategy:
    1. Get product_ids from Qdrant collection
    2. Apply filters via DB query
    3. Return paginated results with primary_image_url
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    try:
        # Try optimized path first: is_primary=True filter
        filter_conditions = {"source": "product", "is_primary": True}
        all_points, _ = await qdrant_service.scroll(
            collection_name=collection_name,
            filter_conditions=filter_conditions,
            limit=50000,
            with_vectors=False,
        )

        # Fallback: if no is_primary points, get all and deduplicate
        if not all_points:
            filter_conditions = {"source": "product"}
            all_points, _ = await qdrant_service.scroll(
                collection_name=collection_name,
                filter_conditions=filter_conditions,
                limit=50000,
                with_vectors=False,
            )

        if not all_points:
            return CollectionProductsResponse(
                products=[],
                total_count=0,
                page=page,
                limit=limit,
                collection_name=collection_name,
            )

        # Extract unique product_ids from Qdrant
        product_ids_in_collection = set()
        for point in all_points:
            payload = point.get("payload", {})
            product_id = payload.get("product_id")
            if product_id:
                product_ids_in_collection.add(product_id)

        # Check if any filters are applied
        has_filters = any([
            search, status, category, brand, sub_brand, product_name,
            variant_flavor, container_type, net_quantity, pack_type,
            manufacturer_country, claims, has_video is not None,
            has_image is not None, has_nutrition is not None,
            has_description is not None, has_issues is not None,
            frame_count_min is not None, frame_count_max is not None,
            visibility_score_min is not None, visibility_score_max is not None
        ])

        if has_filters:
            # Strategy: Query DB with filters first, then intersect with Qdrant IDs
            query = db.client.table("products").select(
                "id, barcode, brand_name, product_name, primary_image_url, frames_path, frame_count",
                count="exact"
            )

            # Apply text search
            if search:
                query = query.or_(f"product_name.ilike.%{search}%,barcode.ilike.%{search}%,brand_name.ilike.%{search}%")

            # Apply comma-separated filters (multi-select)
            if status:
                query = query.in_("status", status.split(","))
            if category:
                query = query.in_("category", category.split(","))
            if brand:
                query = query.in_("brand_name", brand.split(","))
            if sub_brand:
                query = query.in_("sub_brand", sub_brand.split(","))
            if product_name:
                query = query.in_("product_name", product_name.split(","))
            if variant_flavor:
                query = query.in_("variant_flavor", variant_flavor.split(","))
            if container_type:
                query = query.in_("container_type", container_type.split(","))
            if net_quantity:
                query = query.in_("net_quantity", net_quantity.split(","))
            if pack_type:
                query = query.in_("pack_type", pack_type.split(","))
            if manufacturer_country:
                query = query.in_("manufacturer_country", manufacturer_country.split(","))
            if claims:
                for claim in claims.split(","):
                    query = query.ilike("claims", f"%{claim}%")

            # Apply boolean filters
            if has_video is not None:
                if has_video:
                    query = query.not_.is_("video_url", "null")
                else:
                    query = query.is_("video_url", "null")
            if has_image is not None:
                if has_image:
                    query = query.not_.is_("primary_image_url", "null")
                else:
                    query = query.is_("primary_image_url", "null")
            if has_nutrition is not None:
                if has_nutrition:
                    query = query.not_.is_("nutrition_info", "null")
                else:
                    query = query.is_("nutrition_info", "null")
            if has_description is not None:
                if has_description:
                    query = query.not_.is_("description", "null")
                else:
                    query = query.is_("description", "null")
            if has_issues is not None:
                if has_issues:
                    query = query.not_.is_("issues", "null").neq("issues", "[]")
                else:
                    query = query.or_("issues.is.null,issues.eq.[]")

            # Apply range filters
            if frame_count_min is not None:
                query = query.gte("frame_count", frame_count_min)
            if frame_count_max is not None:
                query = query.lte("frame_count", frame_count_max)
            if visibility_score_min is not None:
                query = query.gte("visibility_score", visibility_score_min)
            if visibility_score_max is not None:
                query = query.lte("visibility_score", visibility_score_max)

            # Get all filtered products (limit high enough)
            result = query.limit(50000).execute()
            filtered_products = result.data or []

            # Intersect with Qdrant collection
            products = [p for p in filtered_products if p["id"] in product_ids_in_collection]
            total_count = len(products)

            # Paginate in memory
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated = products[start_idx:end_idx]

        else:
            # No filters: paginate directly from Qdrant IDs, fetch from DB
            product_ids_list = list(product_ids_in_collection)
            total_count = len(product_ids_list)

            # Paginate IDs
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            paginated_ids = product_ids_list[start_idx:end_idx]

            # Fetch only paginated products from DB
            if paginated_ids:
                result = db.client.table("products").select(
                    "id, barcode, brand_name, product_name, primary_image_url, frames_path, frame_count"
                ).in_("id", paginated_ids).execute()

                # Maintain order
                details_map = {p["id"]: p for p in (result.data or [])}
                paginated = [details_map.get(pid, {"id": pid}) for pid in paginated_ids if pid in details_map]
            else:
                paginated = []

        # Add frame_counts for paginated products
        if paginated:
            paginated_product_ids = [p["id"] for p in paginated]
            legacy_counts = {p["id"]: p.get("frame_count", 0) for p in paginated}
            frame_counts_map = await db.get_batch_frame_counts(
                paginated_product_ids,
                check_storage=False,
                legacy_frame_counts=legacy_counts
            )
            for product in paginated:
                product["frame_counts"] = frame_counts_map.get(
                    product["id"],
                    {"synthetic": 0, "real": 0, "augmented": 0}
                )

        return CollectionProductsResponse(
            products=paginated,
            total_count=total_count,
            page=page,
            limit=limit,
            collection_name=collection_name,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection products: {str(e)}"
        )


class CollectionExportRequest(BaseModel):
    """Request to export a Qdrant collection."""
    format: Literal["json", "numpy", "faiss"] = "json"


@router.post("/collections/{collection_name}/export")
async def export_collection(
    collection_name: str,
    request: CollectionExportRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Export all embeddings from a Qdrant collection.

    Supported formats:
    - json: Human-readable JSON with vectors and metadata
    - numpy: Compressed .npz file with vectors, IDs, and payloads
    - faiss: FAISS index file + ID mapping for fast similarity search

    Returns a download URL for the exported file.
    """
    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Check collection exists
    try:
        stats = await qdrant_service.get_collection_info(collection_name)
        if not stats:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Create export record
    export_data = {
        "embedding_model_id": None,  # Not model-specific, collection-based
        "format": request.format,
        "vector_count": 0,
    }

    export = await db.create_embedding_export(export_data)
    export_id = export["id"]

    try:
        # Generate the export
        result = await export_service.export_embeddings(
            model_id=None,
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

        return {
            "export_id": export_id,
            "collection_name": collection_name,
            "format": request.format,
            "vector_count": result.get("vector_count", 0),
            "file_url": result.get("file_url"),
            "file_size_bytes": result.get("file_size_bytes"),
        }

    except Exception as e:
        print(f"Collection export error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Export generation failed: {str(e)}"
        )


# ===========================================
# Advanced Embedding Job Endpoint
# ===========================================


@router.post("/jobs/advanced")
async def start_advanced_embedding_job(
    request: EmbeddingJobCreateAdvanced,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start an advanced embedding extraction job with full configuration options.

    This endpoint supports:
    - Multi-view product embeddings (multiple frames per product)
    - Separate collections for products and cutouts
    - Configurable frame selection (first, key_frames, interval)
    - Custom filtering for cutouts

    Each product frame is stored as a separate embedding point with metadata:
    - Point ID format: {product_id}_{frame_index}
    - Payload includes: product_id, frame_index, is_primary, barcode, etc.
    """
    # Verify model exists
    model = await db.get_embedding_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_id = model["id"]
    model_type = model.get("model_type", "dinov2-base")
    model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
    embedding_dim = model.get("embedding_dim", 768)

    # Determine collection names based on strategy
    if request.collection_strategy.separate_collections:
        prefix = request.collection_strategy.collection_prefix or model_name
        product_collection = f"products_{prefix}"
        cutout_collection = f"cutouts_{prefix}"
    else:
        # Use existing single collection or create new one
        product_collection = model.get("qdrant_collection") or f"embeddings_{model_name}"
        cutout_collection = product_collection

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
    product_count = 0
    cutout_count = 0

    # Fetch products with multi-view support
    if request.source in ["products", "both"]:
        products_result = db.client.table("products").select(
            "id, barcode, brand_name, product_name, frames_path, frame_count"
        ).gt("frame_count", 0).limit(10000).execute()

        for p in products_result.data or []:
            frames_path = p.get("frames_path", "")
            frame_count = p.get("frame_count", 0)
            if not frames_path or frame_count == 0:
                continue

            # Determine which frames to use based on config
            frame_selection = request.product_config.frame_selection

            if frame_selection == "first":
                frame_indices = [0]
            elif frame_selection == "key_frames":
                # Select frames at 0°, 90°, 180°, 270° (approximately)
                step = max(1, frame_count // 4)
                frame_indices = [0]
                for i in range(1, 4):
                    idx = i * step
                    if idx < frame_count:
                        frame_indices.append(idx)
            elif frame_selection == "interval":
                interval = request.product_config.frame_interval
                frame_indices = list(range(0, frame_count, interval))
            elif frame_selection == "all":
                # All frames - no limit
                frame_indices = list(range(frame_count))
            else:
                frame_indices = [0]

            # Limit frames ONLY if not "all"
            if frame_selection != "all":
                frame_indices = frame_indices[:request.product_config.max_frames]

            # Add each frame as a separate image
            for idx in frame_indices:
                frame_url = f"{frames_path.rstrip('/')}/frame_{idx:04d}.png"
                images_to_process.append({
                    "id": f"{p['id']}_{idx}",  # Unique ID: productId_frameIndex
                    "url": frame_url,
                    "type": "product",
                    "collection": product_collection,
                    "metadata": {
                        "source": "product",
                        "product_id": p["id"],
                        "frame_index": idx,
                        "is_primary": idx == 0,
                        "domain": "synthetic",  # Video frames are synthetic
                        "barcode": p.get("barcode"),
                        "brand_name": p.get("brand_name"),
                        "product_name": p.get("product_name"),
                    },
                })

            product_count += 1

    # Fetch cutouts
    if request.source in ["cutouts", "both"]:
        query = db.client.table("cutout_images").select("id, image_url, predicted_upc")

        # Apply incremental filter
        if request.job_type == "incremental":
            query = query.eq("has_embedding", False)

        # Apply UPC filter if requested
        if request.cutout_config.filter_has_upc:
            query = query.not_.is_("predicted_upc", "null")

        # Apply date filter if provided
        if request.cutout_config.synced_after:
            query = query.gte("synced_at", request.cutout_config.synced_after)

        cutouts_result = query.limit(10000).execute()

        for c in cutouts_result.data or []:
            images_to_process.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "collection": cutout_collection,
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "domain": "real",  # Cutouts are real images
                    "predicted_upc": c.get("predicted_upc"),
                },
            })
            cutout_count += 1

    total_images = len(images_to_process)

    if total_images == 0:
        raise HTTPException(status_code=400, detail="No images to process")

    # Create job record with extraction config
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

    # Ensure Qdrant collections exist
    collections_to_create = set()
    for img in images_to_process:
        collections_to_create.add(img["collection"])

    for coll_name in collections_to_create:
        await qdrant_service.create_collection(
            collection_name=coll_name,
            vector_size=embedding_dim,
            distance="Cosine",
        )

    # Process in batches
    batch_size = request.cutout_config.batch_size
    processed_count = 0
    failed_count = 0

    try:
        for i in range(0, total_images, batch_size):
            batch = images_to_process[i:i + batch_size]

            # Prepare worker input
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

                # Group points by collection
                collection_points = {}
                for img in batch:
                    emb_data = embedding_map.get(img["id"])
                    if emb_data and emb_data.get("vector"):
                        coll = img["collection"]
                        if coll not in collection_points:
                            collection_points[coll] = []
                        collection_points[coll].append({
                            "id": img["id"],
                            "vector": emb_data["vector"],
                            "payload": img["metadata"],
                        })
                        processed_count += 1
                    else:
                        failed_count += 1

                # Upsert to each collection
                for coll_name, points in collection_points.items():
                    if points:
                        await qdrant_service.upsert_points(coll_name, points)

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

        # Update model's collection references and vector counts
        try:
            update_data = {"qdrant_vector_count": processed_count}
            if request.collection_strategy.separate_collections:
                update_data["product_collection"] = product_collection
                update_data["cutout_collection"] = cutout_collection
            else:
                update_data["qdrant_collection"] = product_collection

            db.client.table("embedding_models").update(update_data).eq("id", model_id).execute()
        except Exception as e:
            print(f"Model update error: {e}")

    except Exception as e:
        # Mark job as failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Job failed: {str(e)}")

    # Return job with summary
    updated_job = await db.get_embedding_job(job_id)
    return {
        **updated_job,
        "summary": {
            "product_count": product_count,
            "cutout_count": cutout_count,
            "total_embeddings": processed_count,
            "failed_count": failed_count,
            "collections": list(collections_to_create),
            "separate_collections": request.collection_strategy.separate_collections,
        }
    }


# ===========================================
# Tab 1: Matching Extraction Endpoint
# ===========================================


@router.post("/jobs/matching")
async def start_matching_extraction(
    request: MatchingExtractionRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start a matching extraction job (Tab 1).

    Creates embeddings for products and cutouts to be used in the matching page.
    Products are stored in a separate collection from cutouts for efficient search.

    Product sources:
    - all: All products with frames
    - selected: Specific product IDs
    - dataset: Products from a dataset
    - filter: Products matching filter criteria
    - new: Only products without existing embeddings
    """
    # Get model (specified or active)
    checkpoint_url = None
    is_trained_model = False

    if request.model_id and request.model_id.startswith("trained:"):
        # Trained model selected - get from trained_models table
        trained_model_id = request.model_id.replace("trained:", "")
        trained_model = await db.get_trained_model(trained_model_id)
        if not trained_model:
            raise HTTPException(status_code=404, detail="Trained model not found")

        # Get the checkpoint URL
        checkpoint = await db.get_training_checkpoint(trained_model["checkpoint_id"])
        if not checkpoint or not checkpoint.get("checkpoint_url"):
            raise HTTPException(status_code=400, detail="Trained model has no checkpoint URL")

        checkpoint_url = checkpoint["checkpoint_url"]
        is_trained_model = True

        # Get the base model from the training run for model_type
        training_run = await db.get_training_run(trained_model["training_run_id"])
        model_type = training_run.get("base_model_type", "dinov2-base") if training_run else "dinov2-base"
        model_name = trained_model["name"].lower().replace(" ", "_").replace("-", "_")
        embedding_dim = 512  # Fine-tuned models typically use 512
        model_id = trained_model_id

    elif request.model_id:
        model = await db.get_embedding_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        model_id = model["id"]
        model_type = model.get("model_type", "dinov2-base")
        model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
        embedding_dim = model.get("embedding_dim", 768)
    else:
        model = await db.get_active_embedding_model()
        if not model:
            raise HTTPException(status_code=404, detail="No active model found")
        model_id = model["id"]
        model_type = model.get("model_type", "dinov2-base")
        model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
        embedding_dim = model.get("embedding_dim", 768)

    # Check services are configured
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Generate collection names
    product_collection = request.product_collection_name or f"products_{model_name}"
    cutout_collection = request.cutout_collection_name or f"cutouts_{model_name}"

    # Append mode: validate existing collections and vector dimensions
    if request.collection_mode == "append":
        if not request.product_collection_name:
            raise HTTPException(status_code=400, detail="product_collection_name is required for append mode")
        if request.include_cutouts and not request.cutout_collection_name:
            raise HTTPException(status_code=400, detail="cutout_collection_name is required for append mode when include_cutouts is true")

        collections_to_check = [product_collection]
        if request.include_cutouts:
            collections_to_check.append(cutout_collection)

        for coll in collections_to_check:
            info = await qdrant_service.get_collection_info(coll)
            if not info:
                raise HTTPException(status_code=400, detail=f"Collection not found: {coll}")
            if info.get("vector_size") and info["vector_size"] != embedding_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection dimension mismatch for {coll}: expected {embedding_dim}, got {info['vector_size']}",
                )

    # Get products based on source
    products = await db.get_products_for_extraction(
        source_type=request.product_source,
        source_product_ids=request.product_ids,
        source_dataset_id=request.product_dataset_id,
        source_filter=request.product_filter,
    )

    # Collect images to process
    images_to_process = []

    # Process products
    for p in products:
        frames_path = p.get("frames_path", "")
        frame_count = p.get("frame_count", 0)
        if not frames_path or frame_count == 0:
            continue

        # Determine frame indices
        if request.frame_selection == "first":
            frame_indices = [0]
        elif request.frame_selection == "key_frames":
            step = max(1, frame_count // 4)
            frame_indices = [0] + [i * step for i in range(1, 4) if i * step < frame_count]
        elif request.frame_selection == "interval":
            frame_indices = list(range(0, frame_count, request.frame_interval))
        elif request.frame_selection == "all":
            # All frames - no limit
            frame_indices = list(range(frame_count))
        else:
            # Fallback to first frame
            frame_indices = [0]

        # Apply max_frames limit ONLY if not "all"
        if request.frame_selection != "all":
            frame_indices = frame_indices[:request.max_frames]

        for idx in frame_indices:
            frame_url = f"{frames_path.rstrip('/')}/frame_{idx:04d}.png"
            images_to_process.append({
                "id": f"{p['id']}_{idx}",
                "url": frame_url,
                "type": "product",
                "collection": product_collection,
                "metadata": {
                    "source": "product",
                    "product_id": p["id"],
                    "frame_index": idx,
                    "is_primary": idx == 0,
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                    "product_name": p.get("product_name"),
                },
            })

    product_count = len(products)

    # Process cutouts with pagination
    cutout_count = 0
    if request.include_cutouts:
        # Use pagination to get ALL cutouts (not limited to 1000 or 10000)
        filters = {}
        if request.cutout_filter_has_upc:
            filters["not_null"] = ["predicted_upc"]

        # Add merchant filter if specified
        if request.cutout_merchant_ids:
            filters["in"] = {"merchant_id": request.cutout_merchant_ids}

        cutouts = await db.paginated_query(
            table_name="cutout_images",
            select="id, image_url, predicted_upc, merchant, merchant_id",
            filters=filters if filters else None,
            max_records=None  # No limit - get ALL cutouts
        )

        for c in cutouts:
            if not c.get("image_url"):
                continue  # Skip cutouts without image URL

            images_to_process.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "collection": cutout_collection,
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "domain": "real",  # Cutouts are real images
                    "predicted_upc": c.get("predicted_upc"),
                    "merchant": c.get("merchant"),
                    "merchant_id": c.get("merchant_id"),
                },
            })
            cutout_count += 1

    total_images = len(images_to_process)
    failed_count = 0  # Will be updated by worker

    if total_images == 0:
        raise HTTPException(status_code=400, detail="No images to process")

    # Create job record
    job_data = {
        "embedding_model_id": model_id,
        "job_type": "full" if request.collection_mode == "create" else "incremental",
        "source": "both" if request.include_cutouts else "products",
        "status": "queued",  # Start as queued, worker will update to running
        "total_images": total_images,
        "processed_images": 0,
        "purpose": "matching",
        "source_config": {
            "product_source": request.product_source,
            "include_cutouts": request.include_cutouts,
            "cutout_merchant_ids": request.cutout_merchant_ids,
            "frame_selection": request.frame_selection,
            "product_collection": product_collection,
            "cutout_collection": cutout_collection if request.include_cutouts else None,
        },
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    # Create or ensure collections exist
    collections = {product_collection, cutout_collection} if request.include_cutouts else {product_collection}

    if request.collection_mode == "create":
        for coll in collections:
            try:
                await qdrant_service.delete_collection(coll)
            except Exception:
                pass

        for coll in collections:
            await qdrant_service.create_collection(
                collection_name=coll,
                vector_size=embedding_dim,
                distance="Cosine",
            )

    # SOTA ARCHITECTURE (Phase 3):
    # Send source_config instead of images - worker fetches from DB
    # This avoids 22.5 MB payload issue (60K images * 370 bytes = 22.5 MB)
    try:
        # Prepare job input for worker - SOTA pattern (~1 KB payload)
        job_input = {
            "job_id": job_id,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "collection_name": product_collection,  # Primary collection
            "purpose": "matching",
            "checkpoint_url": checkpoint_url,  # For trained models

            # SOTA: source_config tells worker what to fetch from DB
            "source_config": {
                "type": "both" if request.include_cutouts else "products",
                "filters": {
                    "has_embedding": False if request.collection_mode == "create" else None,
                    "product_source": request.product_source,
                    "product_ids": request.product_ids,  # For "selected" source
                    "product_dataset_id": request.product_dataset_id,  # For "dataset" source
                    "product_filter": request.product_filter,  # For "filter" source
                    "cutout_filter_has_upc": request.cutout_filter_has_upc,
                    "cutout_merchant_ids": request.cutout_merchant_ids,  # For merchant filtering
                },
                "append_only_new": request.collection_mode == "append",
                "frame_selection": request.frame_selection,
                "frame_interval": request.frame_interval,
                "max_frames": request.max_frames,
                "product_collection": product_collection,
                "cutout_collection": cutout_collection if request.include_cutouts else None,
            },

            # Worker credentials for Supabase and Qdrant
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,

            # Collection metadata to create after completion
            "collection_metadata": {
                "collection_type": "matching",
                "source_type": request.product_source,
                "embedding_model_id": model_id,
                "product_collection": product_collection,
                "cutout_collection": cutout_collection if request.include_cutouts else None,
                "frame_selection": request.frame_selection,
            },
        }

        # Submit job using JobManager (ASYNC mode)
        job_result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=job_input,
            mode=JobMode.ASYNC,
            webhook_url=None,  # Worker does direct Supabase writes
        )

        # Store runpod_job_id for cancellation
        await db.update_embedding_job(job_id, {
            "runpod_job_id": job_result["job_id"],
            "status": "running",
        })

        print(f"[EMBEDDING] Matching job {job_id} submitted to Runpod: {job_result['job_id']}")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[EMBEDDING] Error submitting matching job {job_id}: {e}")
        print(f"[EMBEDDING] Traceback:\n{error_traceback}")

        # Update status to failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

    # Return job record immediately (job will be processed by worker)
    return {
        "job_id": job_id,
        "status": "running",
        "product_collection": product_collection,
        "cutout_collection": cutout_collection if request.include_cutouts else None,
        "product_count": product_count,
        "cutout_count": cutout_count,
        "total_images": total_images,
        "failed_count": failed_count,
    }


# ===========================================
# Tab 2: Training Extraction Endpoint
# ===========================================


@router.post("/jobs/training")
async def start_training_extraction(
    request: TrainingExtractionRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start a training extraction job (Tab 2).

    Creates embeddings from matched products (synthetic, real, augmented images)
    for triplet mining before model training.

    Only products with matched cutouts are included, as they have verified
    positive pairs (product images ↔ real cutout images).
    """
    # Get model
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
    model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
    embedding_dim = model.get("embedding_dim", 768)

    # Check services
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    if request.output_target == "qdrant" and not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Generate collection name
    collection_name = request.collection_name or f"training_{model_name}"

    # Get matched products
    matched_products = await db.get_products_for_extraction(source_type="matched")

    if not matched_products:
        raise HTTPException(
            status_code=400,
            detail="No matched products found. Match some cutouts first."
        )

    # Collect images
    images_to_process = []
    product_image_counts = {}

    for p in matched_products:
        product_id = p["id"]
        product_image_counts[product_id] = {"synthetic": 0, "real": 0, "augmented": 0, "cutout": 0}

        # Get product images by type
        product_images = await db.get_product_images_by_types(
            product_id=product_id,
            image_types=request.image_types,
            frame_selection=request.frame_selection,
            frame_interval=request.frame_interval,
            max_frames=request.max_frames,
        )

        for img in product_images:
            img_type = img.get("image_type", "synthetic")
            frame_idx = img.get("frame_index", 0)
            img_url = img.get("image_url") or img.get("image_path")

            if not img_url:
                continue

            images_to_process.append({
                "id": f"{product_id}_{img_type}_{frame_idx}",
                "url": img_url,
                "type": "product",
                "collection": collection_name,
                "metadata": {
                    "source": "product",
                    "product_id": product_id,
                    "image_type": img_type,
                    "frame_index": frame_idx,
                    "domain": img_type if img_type in ("synthetic", "real", "augmented") else "synthetic",
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                },
            })
            product_image_counts[product_id][img_type] += 1

        # Include matched cutouts as positive examples
        if request.include_matched_cutouts:
            matched_cutouts = await db.get_matched_cutouts_for_product(product_id)
            for cutout in matched_cutouts:
                images_to_process.append({
                    "id": cutout["id"],
                    "url": cutout["image_url"],
                    "type": "cutout",
                    "collection": collection_name,
                    "metadata": {
                        "source": "cutout",
                        "product_id": product_id,  # Link to product for triplet mining
                        "cutout_id": cutout["id"],
                        "domain": "real",  # Cutouts are real images
                        "is_positive_pair": True,
                    },
                })
                product_image_counts[product_id]["cutout"] += 1

    total_images = len(images_to_process)

    if total_images == 0:
        raise HTTPException(status_code=400, detail="No images found for matched products")

    # Create job record
    job_data = {
        "embedding_model_id": model_id,
        "job_type": "full",
        "source": "both",
        "status": "queued",  # Start as queued, worker will update to running
        "total_images": total_images,
        "processed_images": 0,
        "purpose": "training",
        "image_types": request.image_types,
        "output_target": request.output_target,
        "source_config": {
            "matched_products": len(matched_products),
            "image_types": request.image_types,
            "include_matched_cutouts": request.include_matched_cutouts,
            "collection_name": collection_name,
        },
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    if request.output_target == "qdrant":
        # Create collection
        if request.collection_mode == "create":
            try:
                await qdrant_service.delete_collection(collection_name)
            except Exception:
                pass

        await qdrant_service.create_collection(
            collection_name=collection_name,
            vector_size=embedding_dim,
            distance="Cosine",
        )

        # SOTA ARCHITECTURE (Phase 3):
        # Send source_config instead of images - worker fetches from DB
        try:
            # Prepare job input for worker - SOTA pattern (~1 KB payload)
            job_input = {
                "job_id": job_id,
                "model_type": model_type,
                "embedding_dim": embedding_dim,
                "collection_name": collection_name,
                "purpose": "training",

                # SOTA: source_config tells worker what to fetch from DB
                "source_config": {
                    "type": "both" if request.include_matched_cutouts else "products",
                    "filters": {
                        "product_source": "matched",  # Only matched products for training
                        "image_types": request.image_types,
                        "include_matched_cutouts": request.include_matched_cutouts,
                    },
                    "frame_selection": request.frame_selection,
                    "frame_interval": request.frame_interval,
                    "max_frames": request.max_frames,
                    "product_collection": collection_name,
                    "cutout_collection": collection_name,  # Same collection for training
                },

                # Worker credentials
                "supabase_url": settings.supabase_url,
                "supabase_service_key": settings.supabase_service_role_key,
                "qdrant_url": settings.qdrant_url,
                "qdrant_api_key": settings.qdrant_api_key,

                # Collection metadata to create after completion
                "collection_metadata": {
                    "collection_type": "training",
                    "source_type": "matched",
                    "embedding_model_id": model_id,
                    "image_types": request.image_types,
                },
            }

            # Submit job using JobManager (ASYNC mode)
            job_result = await job_manager.submit_runpod_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data=job_input,
                mode=JobMode.ASYNC,
                webhook_url=None,  # Worker does direct Supabase writes
            )

            # Store runpod_job_id for cancellation
            await db.update_embedding_job(job_id, {
                "runpod_job_id": job_result["job_id"],
                "status": "running",
            })

            print(f"[EMBEDDING] Training job {job_id} submitted to Runpod: {job_result['job_id']}")

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"[EMBEDDING] Error submitting training job {job_id}: {e}")
            print(f"[EMBEDDING] Traceback:\n{error_traceback}")

            # Update status to failed
            await db.update_embedding_job(job_id, {
                "status": "failed",
                "error_message": str(e),
            })
            raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

        # Return job record immediately (job will be processed by worker)
        return {
            "job_id": job_id,
            "status": "running",
            "collection_name": collection_name,
            "matched_product_count": len(matched_products),
            "total_images": total_images,
            "output_target": "qdrant",
        }

    else:
        # File export (to be implemented with export service)
        # For now, process embeddings and return URL
        raise HTTPException(
            status_code=501,
            detail="File export for training not yet implemented. Use output_target='qdrant'"
        )


# ===========================================
# Tab 3: Evaluation Extraction Endpoint
# ===========================================


@router.post("/jobs/evaluation")
async def start_evaluation_extraction(
    request: EvaluationExtractionRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start an evaluation extraction job (Tab 3).

    Creates embeddings for a dataset using a specific trained model.
    Used to evaluate model performance on a held-out test set.
    """
    # Get model (required for evaluation)
    model = await db.get_embedding_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_id = model["id"]
    model_type = model.get("model_type", "dinov2-base")
    model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
    embedding_dim = model.get("embedding_dim", 768)

    # Get dataset
    dataset = await db.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_name = dataset.get("name", "unnamed").lower().replace(" ", "_")

    # Check services
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Generate collection name
    collection_name = request.collection_name or f"eval_{dataset_name}_{model_name}"

    # Get products from dataset
    products = await db.get_products_for_extraction(
        source_type="dataset",
        source_dataset_id=request.dataset_id,
    )

    if not products:
        raise HTTPException(status_code=400, detail="Dataset has no products")

    # Collect images
    images_to_process = []

    for p in products:
        product_id = p["id"]

        # Get product images by type
        product_images = await db.get_product_images_by_types(
            product_id=product_id,
            image_types=request.image_types,
            frame_selection=request.frame_selection,
            frame_interval=request.frame_interval,
            max_frames=request.max_frames,
        )

        for img in product_images:
            img_type = img.get("image_type", "synthetic")
            frame_idx = img.get("frame_index", 0)
            img_url = img.get("image_url") or img.get("image_path")

            if not img_url:
                continue

            images_to_process.append({
                "id": f"{product_id}_{img_type}_{frame_idx}",
                "url": img_url,
                "type": "product",
                "collection": collection_name,
                "metadata": {
                    "source": "product",
                    "product_id": product_id,
                    "image_type": img_type,
                    "frame_index": frame_idx,
                    "dataset_id": request.dataset_id,
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                },
            })

    total_images = len(images_to_process)

    if total_images == 0:
        raise HTTPException(status_code=400, detail="No images found in dataset")

    # Create job record
    job_data = {
        "embedding_model_id": model_id,
        "job_type": "full",
        "source": "products",
        "status": "queued",  # Start as queued, worker will update
        "total_images": total_images,
        "processed_images": 0,
        "purpose": "evaluation",
        "image_types": request.image_types,
        "source_config": {
            "dataset_id": request.dataset_id,
            "dataset_name": dataset_name,
            "product_count": len(products),
        },
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    # Create collection
    if request.collection_mode == "create":
        try:
            await qdrant_service.delete_collection(collection_name)
        except Exception:
            pass

    await qdrant_service.create_collection(
        collection_name=collection_name,
        vector_size=embedding_dim,
        distance="Cosine",
    )

    # NEW ARCHITECTURE (Phase 2 Migration):
    # Submit single job to worker for evaluation extraction
    try:
        # Prepare job input for worker
        job_input = {
            "job_id": job_id,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "collection_name": collection_name,
            "purpose": "evaluation",
            # Send all images to worker
            "images": [
                {
                    "id": img["id"],
                    "url": img["url"],
                    "type": img["type"],
                    "collection": img["collection"],
                    "metadata": img["metadata"],
                }
                for img in images_to_process
            ],
            # Worker credentials
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,
            # Collection metadata to create after completion
            "collection_metadata": {
                "collection_type": "evaluation",
                "source_type": "dataset",
                "source_dataset_id": request.dataset_id,
                "embedding_model_id": model_id,
                "image_types": request.image_types,
            },
        }

        # Submit job using JobManager (ASYNC mode)
        job_result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=job_input,
            mode=JobMode.ASYNC,
            webhook_url=None,  # Worker does direct Supabase writes
        )

        # Store runpod_job_id for cancellation
        await db.update_embedding_job(job_id, {
            "runpod_job_id": job_result["job_id"],
            "status": "running",
        })

        print(f"[EMBEDDING] Evaluation job {job_id} submitted to Runpod: {job_result['job_id']}")

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[EMBEDDING] Error submitting evaluation job {job_id}: {e}")
        print(f"[EMBEDDING] Traceback:\n{error_traceback}")

        # Update status to failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise

    # Return job record (job will be processed by worker)
    return job

    # OLD IMPLEMENTATION (removed):
    # - Used _process_embedding_batch() which called submit_job_sync() in loops
    # - Not cancellable
    # - API handled batch processing

    # Mark completed
    await db.update_embedding_job(job_id, {
        "status": "completed",
        "processed_images": processed_count,
        "completed_at": datetime.utcnow().isoformat(),
    })

    return {
        "job_id": job_id,
        "status": "completed",
        "collection_name": collection_name,
        "dataset_id": request.dataset_id,
        "dataset_name": dataset_name,
        "product_count": len(products),
        "total_embeddings": processed_count,
        "failed_count": failed_count,
    }


# ===========================================
# Tab 4: Production Extraction Endpoint
# ===========================================


@router.post("/jobs/production")
async def start_production_extraction(
    request: ProductionExtractionRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Start a production extraction job (Tab 4).

    Creates a production-ready embedding collection with ALL image types
    (synthetic, real, augmented) for similarity search / inference.

    SOTA approach: Multi-view + augmentation at index time improves recall
    because query images can match any angle or variation.
    """
    # Determine model source and get checkpoint URL
    checkpoint_url = None
    model_id = None
    model_type = "dinov2-base"
    model_name = "dinov2_base"
    embedding_dim = 768

    if request.training_run_id and request.checkpoint_id:
        # Trained model - get checkpoint from training system
        checkpoint = await db.get_training_checkpoint(request.checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        checkpoint_url = checkpoint.get("checkpoint_url")
        if not checkpoint_url:
            raise HTTPException(status_code=400, detail="Checkpoint has no URL")

        # Get training run for model info
        training_run = await db.get_training_run(request.training_run_id)
        if not training_run:
            raise HTTPException(status_code=404, detail="Training run not found")

        model_type = training_run.get("base_model_type", "dinov2-base")
        model_name = training_run.get("name", "trained").lower().replace(" ", "_").replace("-", "_")
        embedding_dim = training_run.get("training_config", {}).get("embedding_dim", 512)
        model_id = request.training_run_id  # Use training run ID as model reference

    elif request.model_id:
        # Base model
        model = await db.get_embedding_model(request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        model_id = model["id"]
        model_type = model.get("model_type", "dinov2-base")
        model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
        embedding_dim = model.get("embedding_dim", 768)
    else:
        # Use active model
        model = await db.get_active_embedding_model()
        if not model:
            raise HTTPException(status_code=404, detail="No active model found")

        model_id = model["id"]
        model_type = model.get("model_type", "dinov2-base")
        model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")
        embedding_dim = model.get("embedding_dim", 768)

    # Check services
    if not runpod_service.is_configured(EndpointType.EMBEDDING):
        raise HTTPException(status_code=500, detail="Embedding endpoint not configured")

    if not qdrant_service.is_configured():
        raise HTTPException(status_code=500, detail="Qdrant not configured")

    # Generate collection name
    collection_name = request.collection_name or f"production_{model_name}"

    # Get products based on source
    if request.product_source == "all":
        products = await db.get_products_for_extraction(source_type="all")
    elif request.product_source == "dataset":
        if not request.product_dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID required for dataset source")
        products = await db.get_products_for_extraction(
            source_type="dataset",
            source_dataset_id=request.product_dataset_id,
        )
    elif request.product_source == "selected":
        if not request.product_ids:
            raise HTTPException(status_code=400, detail="Product IDs required for selected source")
        products = await db.get_products_for_extraction(
            source_type="selected",
            source_product_ids=request.product_ids,
        )
    else:
        products = await db.get_products_for_extraction(source_type="all")

    if not products:
        raise HTTPException(status_code=400, detail="No products found for extraction")

    # Collect ALL images for each product
    images_to_process = []
    product_stats = {"synthetic": 0, "real": 0, "augmented": 0}

    for p in products:
        product_id = p["id"]

        # Get images by type using the existing method
        product_images = await db.get_product_images_by_types(
            product_id=product_id,
            image_types=request.image_types,
            frame_selection=request.frame_selection,
            frame_interval=request.frame_interval,
            max_frames=request.max_frames,
        )

        for img in product_images:
            img_type = img.get("image_type", "synthetic")
            frame_idx = img.get("frame_index", 0)
            img_url = img.get("image_url") or img.get("image_path")

            if not img_url:
                continue

            # Create unique ID for each image
            img_id = f"{product_id}_{img_type}_{frame_idx}"

            images_to_process.append({
                "id": img_id,
                "url": img_url,
                "type": "product",
                "collection": collection_name,
                "metadata": {
                    "source": "product",
                    "product_id": product_id,
                    "image_type": img_type,
                    "frame_index": frame_idx,
                    "domain": img_type,  # synthetic/real/augmented
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                    "product_name": p.get("product_name"),
                    "is_production": True,
                },
            })
            product_stats[img_type] = product_stats.get(img_type, 0) + 1

    total_images = len(images_to_process)

    if total_images == 0:
        raise HTTPException(
            status_code=400,
            detail="No images found. Check that products have images in product_images table."
        )

    # Create job record
    job_data = {
        "embedding_model_id": model_id,
        "job_type": "full",
        "source": "products",
        "status": "queued",  # Start as queued, worker will update
        "total_images": total_images,
        "processed_images": 0,
        "purpose": "production",
        "image_types": request.image_types,
        "source_config": {
            "product_source": request.product_source,
            "product_count": len(products),
            "image_stats": product_stats,
            "has_checkpoint": checkpoint_url is not None,
        },
    }

    job = await db.create_embedding_job(job_data)
    job_id = job["id"]

    # Create or replace collection
    if request.collection_mode == "replace":
        try:
            await qdrant_service.delete_collection(collection_name)
        except Exception:
            pass

    await qdrant_service.create_collection(
        collection_name=collection_name,
        vector_size=embedding_dim,
        distance="Cosine",
    )

    # NEW ARCHITECTURE (Phase 2 Migration):
    # Submit single job to worker for production extraction
    try:
        # Prepare job input for worker
        job_input = {
            "job_id": job_id,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "collection_name": collection_name,
            "purpose": "production",
            "checkpoint_url": checkpoint_url,  # For trained models
            # Send all images to worker
            "images": [
                {
                    "id": img["id"],
                    "url": img["url"],
                    "type": img["type"],
                    "collection": img["collection"],
                    "metadata": img["metadata"],
                }
                for img in images_to_process
            ],
            # Worker credentials
            "supabase_url": settings.supabase_url,
            "supabase_service_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,
            # Collection metadata to create after completion
            "collection_metadata": {
                "collection_type": "production",
                "source_type": request.product_source,
                "embedding_model_id": model_id,
                "image_types": request.image_types,
            },
        }

        # Submit job using JobManager (ASYNC mode)
        job_result = await job_manager.submit_runpod_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=job_input,
            mode=JobMode.ASYNC,
            webhook_url=None,  # Worker does direct Supabase writes
        )

        # Store runpod_job_id for cancellation
        await db.update_embedding_job(job_id, {
            "runpod_job_id": job_result["job_id"],
            "status": "running",
        })

        print(f"[EMBEDDING] Production job {job_id} submitted to Runpod: {job_result['job_id']}")

        # Return immediately - job will be processed by worker
        return {
            "job_id": job_id,
            "runpod_job_id": job_result["job_id"],
            "status": "submitted",
            "collection_name": collection_name,
            "product_count": len(products),
            "image_stats": product_stats,
            "total_images": total_images,
            "model_type": model_type,
            "embedding_dim": embedding_dim,
            "has_trained_model": checkpoint_url is not None,
            "message": "Job submitted. Poll /embeddings/jobs/{job_id} for progress.",
        }

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[EMBEDDING] Error submitting production job {job_id}: {e}")
        print(f"[EMBEDDING] Traceback:\n{error_traceback}")

        # Update status to failed
        await db.update_embedding_job(job_id, {
            "status": "failed",
            "error_message": str(e),
        })
        raise

    # OLD IMPLEMENTATION (removed):
    # - Used _process_embedding_batch() which called submit_job_sync() in loops
    # - Not cancellable
    # - Blocked API for hours on large datasets


# ===========================================
# Helper Function for Batch Processing
# ===========================================


async def _process_embedding_batch(
    db: SupabaseService,
    job_id: str,
    model_id: str,
    model_type: str,
    images: list[dict],
    batch_size: int = 50,
    checkpoint_url: Optional[str] = None,
) -> tuple[int, int]:
    """
    Process images in batches, extract embeddings, and save to Qdrant.

    Args:
        checkpoint_url: Optional URL to fine-tuned model checkpoint for trained models.

    Returns (processed_count, failed_count).
    """
    total = len(images)
    processed_count = 0
    failed_count = 0

    for i in range(0, total, batch_size):
        batch = images[i:i + batch_size]

        worker_input = {
            "images": [
                {"id": img["id"], "url": img["url"], "type": img["type"]}
                for img in batch
            ],
            "model_type": model_type,
            "batch_size": min(16, len(batch)),
        }

        # Add checkpoint URL for fine-tuned models
        if checkpoint_url:
            worker_input["checkpoint_url"] = checkpoint_url

        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.EMBEDDING,
                input_data=worker_input,
                timeout=300,
            )

            output = result.get("output", {})
            if output.get("status") != "success":
                failed_count += len(batch)
                continue

            embeddings = output.get("embeddings", [])
            embedding_map = {e["id"]: e for e in embeddings}

            # Group by collection
            collection_points = {}
            for img in batch:
                emb_data = embedding_map.get(img["id"])
                if emb_data and emb_data.get("vector"):
                    coll = img["collection"]
                    if coll not in collection_points:
                        collection_points[coll] = []
                    collection_points[coll].append({
                        "id": img["id"],
                        "vector": emb_data["vector"],
                        "payload": img["metadata"],
                    })
                    processed_count += 1
                else:
                    failed_count += 1

            # Upsert to each collection
            for coll_name, points in collection_points.items():
                if points:
                    await qdrant_service.upsert_points(coll_name, points)

            # Update cutout records if needed
            for img in batch:
                if img["type"] == "cutout" and img["id"] in embedding_map:
                    try:
                        db.client.table("cutout_images").update({
                            "has_embedding": True,
                            "embedding_model_id": model_id,
                            "qdrant_point_id": img["id"],
                        }).eq("id", img["id"]).execute()
                    except Exception:
                        pass

        except Exception as e:
            print(f"Batch error: {e}")
            failed_count += len(batch)

        # Update job progress
        await db.update_embedding_job(job_id, {
            "processed_images": processed_count + failed_count,
        })

    return processed_count, failed_count


# ===========================================
# Matched Products Stats Endpoint
# ===========================================


@router.get("/matched-products/stats")
async def get_matched_products_stats(
    db: SupabaseService = Depends(get_supabase),
):
    """Get statistics about products with matched cutouts (for training tab)."""
    count = await db.get_matched_products_count()

    # Get sample of matched products with counts
    matched = await db.get_matched_products(page=1, limit=10)

    return {
        "total_matched_products": count,
        "sample": matched.get("items", []),
    }
