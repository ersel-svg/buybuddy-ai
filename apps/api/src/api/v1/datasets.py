"""Datasets API router for managing training datasets."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from auth.dependencies import get_current_user
from config import settings

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


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
    """Request to add products to dataset.

    Either product_ids OR filters can be provided:
    - product_ids: Add specific products by ID
    - filters: Add all products matching the filter criteria
    """

    product_ids: Optional[list[str]] = None

    # Filter-based selection (alternative to product_ids)
    filters: Optional[dict] = None  # Contains: search, status, category, brand, etc.


class AugmentationConfigRequest(BaseModel):
    """Configuration for augmentation effects."""

    # Preset: 'clean', 'normal', 'realistic', 'extreme', 'custom'
    preset: str = "normal"

    # Transform probabilities (0.0 - 1.0)
    PROB_HEAVY_AUGMENTATION: Optional[float] = None
    PROB_NEIGHBORING_PRODUCTS: Optional[float] = None
    PROB_TIPPED_OVER_NEIGHBOR: Optional[float] = None

    # Shelf elements
    PROB_PRICE_TAG: Optional[float] = None
    PROB_SHELF_RAIL: Optional[float] = None
    PROB_CAMPAIGN_STICKER: Optional[float] = None

    # Lighting effects
    PROB_FLUORESCENT_BANDING: Optional[float] = None
    PROB_COLOR_TRANSFER: Optional[float] = None
    PROB_SHELF_REFLECTION: Optional[float] = None
    PROB_SHADOW: Optional[float] = None

    # Camera effects
    PROB_PERSPECTIVE_CHANGE: Optional[float] = None
    PROB_LENS_DISTORTION: Optional[float] = None
    PROB_CHROMATIC_ABERRATION: Optional[float] = None
    PROB_CAMERA_NOISE: Optional[float] = None

    # Refrigerator effects
    PROB_CONDENSATION: Optional[float] = None
    PROB_FROST_CRYSTALS: Optional[float] = None
    PROB_COLD_COLOR_FILTER: Optional[float] = None
    PROB_WIRE_RACK: Optional[float] = None

    # Color adjustments
    PROB_HSV_SHIFT: Optional[float] = None
    PROB_RGB_SHIFT: Optional[float] = None
    PROB_MEDIAN_BLUR: Optional[float] = None
    PROB_ISO_NOISE: Optional[float] = None
    PROB_CLAHE: Optional[float] = None
    PROB_SHARPEN: Optional[float] = None
    PROB_HORIZONTAL_FLIP: Optional[float] = None

    # Neighbor settings
    MIN_NEIGHBORS: Optional[int] = None
    MAX_NEIGHBORS: Optional[int] = None

    # Color shift limits
    HSV_HUE_LIMIT: Optional[int] = None
    HSV_SAT_LIMIT: Optional[int] = None
    HSV_VAL_LIMIT: Optional[int] = None
    RGB_SHIFT_LIMIT: Optional[int] = None


class AugmentRequest(BaseModel):
    """Request to start augmentation."""

    syn_target: int = 600
    real_target: int = 400

    # Use diversity pyramid for random level selection
    use_diversity_pyramid: bool = True

    # Include neighbor products in shelf composition
    include_neighbors: bool = True

    # Frame interval for angle diversity (1 = all frames, 20 = every 20th frame)
    # Higher values select fewer frames but more diverse angles from 360° videos
    frame_interval: int = 1

    # Augmentation config (optional - uses preset defaults if not provided)
    augmentation_config: Optional[AugmentationConfigRequest] = None


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
        # Refresh dataset to get updated product_count
        result = db.client.table("datasets").select("*").eq("id", dataset["id"]).limit(1).execute()
        if result.data:
            dataset = result.data[0]

    return dataset


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    # Pagination
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
    # Search
    search: Optional[str] = Query(None),
    # Sorting
    sort_by: Optional[str] = Query(None),
    sort_order: Optional[str] = Query("desc"),
    # Comma-separated list filters
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
    visibility_score_min: Optional[int] = Query(None),
    visibility_score_max: Optional[int] = Query(None),
    # Options
    include_frame_counts: bool = Query(True),
    db: SupabaseService = Depends(get_supabase),
):
    """Get dataset with filtered products.

    Uses the same filtering approach as the products page for consistency.
    Filters are applied to the products table, then intersected with dataset membership.
    """
    # First, get the dataset metadata (without loading all products)
    dataset_result = db.client.table("datasets").select("*").eq("id", dataset_id).limit(1).execute()
    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_info = dataset_result.data[0]

    # Use get_products with include_dataset_id to get filtered products
    # This uses the same filtering logic as the products page
    products_result = await db.get_products(
        page=page,
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        status=status,
        category=category,
        brand=brand,
        sub_brand=sub_brand,
        product_name=product_name,
        variant_flavor=variant_flavor,
        container_type=container_type,
        net_quantity=net_quantity,
        pack_type=pack_type,
        manufacturer_country=manufacturer_country,
        claims=claims,
        has_video=has_video,
        has_image=has_image,
        has_nutrition=has_nutrition,
        has_description=has_description,
        has_prompt=has_prompt,
        has_issues=has_issues,
        frame_count_min=frame_count_min,
        frame_count_max=frame_count_max,
        visibility_score_min=visibility_score_min,
        visibility_score_max=visibility_score_max,
        include_dataset_id=dataset_id,
        include_frame_counts=include_frame_counts,
    )

    # Calculate total frame counts from products
    products = products_result.get("items", [])
    total_synthetic = 0
    total_real = 0
    total_augmented = 0

    for product in products:
        frame_counts = product.get("frame_counts", {})
        total_synthetic += frame_counts.get("synthetic", 0) or 0
        total_real += frame_counts.get("real", 0) or 0
        total_augmented += frame_counts.get("augmented", 0) or 0

    # Note: For accurate totals across ALL products (not just current page),
    # we need to fetch totals separately if pagination is active
    products_total = products_result.get("total", 0)

    # If we're on page 1 and have all products, the above totals are accurate
    # Otherwise, we need to get totals from all products in dataset
    if products_total > len(products):
        # Fetch frame count totals for all products in dataset
        totals = await db.get_dataset_frame_totals(dataset_id)
        total_synthetic = totals.get("synthetic", 0)
        total_real = totals.get("real", 0)
        total_augmented = totals.get("augmented", 0)

    # Combine dataset info with products
    return {
        **dataset_info,
        "products": products,
        "products_total": products_total,
        "total": products_total,  # Keep for backwards compatibility
        "page": products_result.get("page", page),
        "limit": products_result.get("limit", limit),
        "total_synthetic": total_synthetic,
        "total_real": total_real,
        "total_augmented": total_augmented,
    }


@router.patch("/{dataset_id}")
async def update_dataset(
    dataset_id: str,
    data: DatasetUpdate,
    db: SupabaseService = Depends(get_supabase),
):
    """Update dataset with optimistic locking."""
    # Simple existence check
    existing = db.client.table("datasets").select("id").eq("id", dataset_id).limit(1).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = data.model_dump(exclude_unset=True, exclude={"version"})
    return await db.update_dataset(dataset_id, update_data)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a dataset."""
    # Simple existence check
    existing = db.client.table("datasets").select("id").eq("id", dataset_id).limit(1).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await db.delete_dataset(dataset_id)
    return {"status": "deleted"}


@router.post("/{dataset_id}/products")
async def add_products_to_dataset(
    dataset_id: str,
    request: AddProductsRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Add products to dataset.

    Supports two modes:
    1. product_ids: Add specific products by their IDs (sync <50, async 50+)
    2. filters: Add all products matching filter criteria (always async)

    For large batches (50+ products), returns job_id for progress tracking.
    For small batches (<50), returns added_count immediately.
    """
    # Simple existence check without loading all products
    dataset_check = db.client.table("datasets").select("id").eq("id", dataset_id).limit(1).execute()
    if not dataset_check.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate request - must have either product_ids or filters
    if not request.product_ids and not request.filters:
        raise HTTPException(
            status_code=400,
            detail="Either product_ids or filters must be provided"
        )

    from services.local_jobs import create_local_job

    # Mode 1: Add specific products by ID (sync for small batches)
    if request.product_ids:
        # For small batches (<50), process synchronously
        if len(request.product_ids) < 50:
            added_count = await db.add_products_to_dataset(dataset_id, request.product_ids)
            return {"added_count": added_count}

        # For large batches (50+), use background job
        job = await create_local_job(
            job_type="local_bulk_add_products_to_dataset",
            config={
                "dataset_id": dataset_id,
                "product_ids": request.product_ids,
            }
        )
        return {"job_id": job["id"], "message": "Background job started"}

    # Mode 2: Add all products matching filters (always async)
    if request.filters:
        # Create background job for filtered products
        job = await create_local_job(
            job_type="local_bulk_add_products_to_dataset",
            config={
                "dataset_id": dataset_id,
                "filters": request.filters,
            }
        )
        return {"job_id": job["id"], "message": "Background job started"}

    return {"added_count": 0}


@router.delete("/{dataset_id}/products/{product_id}")
async def remove_product_from_dataset(
    dataset_id: str,
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Remove product from dataset."""
    # Simple existence check
    existing = db.client.table("datasets").select("id").eq("id", dataset_id).limit(1).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await db.remove_product_from_dataset(dataset_id, product_id)
    return {"status": "removed"}


@router.get("/{dataset_id}/filter-options")
async def get_dataset_filter_options(
    dataset_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get available filter options for products in this dataset."""
    options = await db.get_dataset_filter_options(dataset_id)
    if options is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return options


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
    # Get dataset with product_count
    dataset_result = db.client.table("datasets").select("id, product_count").eq("id", dataset_id).limit(1).execute()
    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = dataset_result.data[0]
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

            # Build input data with config
            input_data = {
                "dataset_id": dataset_id,
                "syn_target": request_data.syn_target,
                "real_target": request_data.real_target,
                "use_diversity_pyramid": request_data.use_diversity_pyramid,
                "include_neighbors": request_data.include_neighbors,
                "frame_interval": request_data.frame_interval,
            }

            # Add augmentation config if provided
            if request_data.augmentation_config:
                input_data["augmentation_config"] = request_data.augmentation_config.model_dump(
                    exclude_unset=True
                )

            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.AUGMENTATION,
                input_data=input_data,
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
    # Get dataset with product_count
    dataset_result = db.client.table("datasets").select("id, product_count").eq("id", dataset_id).limit(1).execute()
    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = dataset_result.data[0]
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
    # Get dataset with product_count
    dataset_result = db.client.table("datasets").select("id, product_count").eq("id", dataset_id).limit(1).execute()
    if not dataset_result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = dataset_result.data[0]
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
