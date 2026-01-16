"""Products API router with CRUD, download, and export functionality."""

import csv
import json
import zipfile
from io import BytesIO
from typing import Optional, List, Literal
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service, EndpointType
from auth.dependencies import get_current_user
from auth.service import UserInfo
from config import settings

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class ProductBase(BaseModel):
    """Base product schema."""

    barcode: str
    brand_name: Optional[str] = None
    sub_brand: Optional[str] = None
    manufacturer_country: Optional[str] = None
    product_name: Optional[str] = None
    variant_flavor: Optional[str] = None
    category: Optional[str] = None
    container_type: Optional[str] = None
    net_quantity: Optional[str] = None
    pack_configuration: Optional[dict] = None
    identifiers: Optional[dict] = None
    nutrition_facts: Optional[dict] = None
    claims: Optional[list[str]] = None
    marketing_description: Optional[str] = None
    grounding_prompt: Optional[str] = None
    visibility_score: Optional[int] = None
    issues_detected: Optional[list[str]] = None


class ProductCreate(ProductBase):
    """Product creation schema."""

    video_id: Optional[int] = None
    video_url: Optional[str] = None


class ProductUpdate(BaseModel):
    """Product update schema."""

    brand_name: Optional[str] = None
    sub_brand: Optional[str] = None
    manufacturer_country: Optional[str] = None
    product_name: Optional[str] = None
    variant_flavor: Optional[str] = None
    category: Optional[str] = None
    container_type: Optional[str] = None
    net_quantity: Optional[str] = None
    pack_configuration: Optional[dict] = None
    identifiers: Optional[dict] = None
    nutrition_facts: Optional[dict] = None
    claims: Optional[list[str]] = None
    marketing_description: Optional[str] = None
    grounding_prompt: Optional[str] = None
    visibility_score: Optional[int] = None
    issues_detected: Optional[list[str]] = None
    status: Optional[str] = None
    primary_image_url: Optional[str] = None  # For setting primary frame/thumbnail
    version: Optional[int] = None  # For optimistic locking


class ProductIdsRequest(BaseModel):
    """Request with product IDs."""

    product_ids: Optional[list[str]] = None


# ===========================================
# Product Identifier Schemas
# ===========================================


IdentifierType = Literal["barcode", "short_code", "sku", "upc", "ean", "custom"]


class ProductIdentifier(BaseModel):
    """Product identifier schema."""

    id: Optional[str] = None
    identifier_type: IdentifierType
    identifier_value: str
    custom_label: Optional[str] = None
    is_primary: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ProductIdentifierCreate(BaseModel):
    """Create product identifier."""

    identifier_type: IdentifierType
    identifier_value: str
    custom_label: Optional[str] = None
    is_primary: bool = False


class ProductIdentifierUpdate(BaseModel):
    """Update product identifier."""

    identifier_value: Optional[str] = None
    custom_label: Optional[str] = None
    is_primary: Optional[bool] = None


class ProductIdentifiersUpdate(BaseModel):
    """Bulk update identifiers for a product."""

    identifiers: List[ProductIdentifierCreate]


class CustomFieldsUpdate(BaseModel):
    """Update custom fields."""

    custom_fields: dict[str, str]


class ExportRequest(BaseModel):
    """Request for export with filters."""

    product_ids: Optional[list[str]] = None
    # Filters (when product_ids is not provided)
    search: Optional[str] = None
    status: Optional[List[str]] = None
    category: Optional[List[str]] = None
    brand: Optional[List[str]] = None
    sub_brand: Optional[List[str]] = None
    product_name: Optional[List[str]] = None
    variant_flavor: Optional[List[str]] = None
    container_type: Optional[List[str]] = None
    net_quantity: Optional[List[str]] = None
    pack_type: Optional[List[str]] = None
    manufacturer_country: Optional[List[str]] = None
    claims: Optional[List[str]] = None
    has_video: Optional[bool] = None
    has_image: Optional[bool] = None
    has_nutrition: Optional[bool] = None
    has_description: Optional[bool] = None
    has_prompt: Optional[bool] = None
    has_issues: Optional[bool] = None
    frame_count_min: Optional[int] = None
    frame_count_max: Optional[int] = None
    visibility_score_min: Optional[int] = None
    visibility_score_max: Optional[int] = None


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


def get_runpod() -> RunpodService:
    """Get Runpod service instance."""
    return runpod_service


def get_webhook_url(request: Request) -> str:
    """Build webhook URL for Runpod callbacks."""
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}{settings.api_prefix}/webhooks/runpod"


# ===========================================
# Helper Functions
# ===========================================


async def get_all_filtered_products(
    db: SupabaseService,
    request: ExportRequest,
) -> list[dict]:
    """Get all products matching filters, handling pagination."""
    if request.product_ids:
        # Fetch specific products by ID
        products = []
        for pid in request.product_ids:
            product = await db.get_product(pid)
            if product:
                products.append(product)
        return products

    # Fetch all products with server-side filters, then apply client-side filters
    all_products = []
    page = 1
    page_size = 1000  # Max per page

    while True:
        result = await db.get_products(
            page=page,
            limit=page_size,
            search=request.search,
        )
        items = result.get("items", [])
        if not items:
            break

        all_products.extend(items)

        # Check if we got all items
        total = result.get("total", 0)
        if len(all_products) >= total:
            break

        page += 1

    # Apply client-side filters
    def matches_filter(product: dict) -> bool:
        # Status filter
        if request.status and product.get("status") not in request.status:
            return False

        # Category filter
        if request.category and product.get("category") not in request.category:
            return False

        # Brand filter
        if request.brand and product.get("brand_name") not in request.brand:
            return False

        # Sub Brand filter
        if request.sub_brand and product.get("sub_brand") not in request.sub_brand:
            return False

        # Product Name filter
        if request.product_name and product.get("product_name") not in request.product_name:
            return False

        # Variant/Flavor filter
        if request.variant_flavor and product.get("variant_flavor") not in request.variant_flavor:
            return False

        # Container Type filter
        if request.container_type and product.get("container_type") not in request.container_type:
            return False

        # Net Quantity filter
        if request.net_quantity and product.get("net_quantity") not in request.net_quantity:
            return False

        # Pack Type filter
        if request.pack_type:
            pack_config = product.get("pack_configuration") or {}
            if pack_config.get("type") not in request.pack_type:
                return False

        # Manufacturer Country filter
        if request.manufacturer_country and product.get("manufacturer_country") not in request.manufacturer_country:
            return False

        # Claims filter (product must have at least one of the selected claims)
        if request.claims:
            product_claims = product.get("claims") or []
            if not any(c in request.claims for c in product_claims):
                return False

        # Boolean filters
        if request.has_video is not None:
            has_video = bool(product.get("video_url"))
            if request.has_video != has_video:
                return False

        if request.has_image is not None:
            has_image = bool(product.get("primary_image_url"))
            if request.has_image != has_image:
                return False

        if request.has_nutrition is not None:
            nutrition = product.get("nutrition_facts") or {}
            has_nutrition = len(nutrition) > 0
            if request.has_nutrition != has_nutrition:
                return False

        if request.has_description is not None:
            has_desc = bool(product.get("marketing_description"))
            if request.has_description != has_desc:
                return False

        if request.has_prompt is not None:
            has_prompt = bool(product.get("grounding_prompt"))
            if request.has_prompt != has_prompt:
                return False

        if request.has_issues is not None:
            issues = product.get("issues_detected") or []
            has_issues = len(issues) > 0
            if request.has_issues != has_issues:
                return False

        # Range filters
        if request.frame_count_min is not None:
            if product.get("frame_count", 0) < request.frame_count_min:
                return False

        if request.frame_count_max is not None:
            if product.get("frame_count", 0) > request.frame_count_max:
                return False

        if request.visibility_score_min is not None:
            score = product.get("visibility_score")
            if score is None or score < request.visibility_score_min:
                return False

        if request.visibility_score_max is not None:
            score = product.get("visibility_score")
            if score is None or score > request.visibility_score_max:
                return False

        return True

    return [p for p in all_products if matches_filter(p)]


def product_to_full_dict(product: dict) -> dict:
    """Convert product to full dictionary with all fields."""
    pack_config = product.get("pack_configuration") or {}
    identifiers = product.get("identifiers") or {}
    nutrition = product.get("nutrition_facts") or {}

    return {
        "id": product.get("id"),
        "barcode": product.get("barcode"),
        "video_id": product.get("video_id"),
        "video_url": product.get("video_url"),
        "brand_name": product.get("brand_name"),
        "sub_brand": product.get("sub_brand"),
        "manufacturer_country": product.get("manufacturer_country"),
        "product_name": product.get("product_name"),
        "variant_flavor": product.get("variant_flavor"),
        "category": product.get("category"),
        "container_type": product.get("container_type"),
        "net_quantity": product.get("net_quantity"),
        "pack_type": pack_config.get("type"),
        "pack_item_count": pack_config.get("item_count"),
        "sku_model_code": identifiers.get("sku_model_code"),
        "secondary_barcode": identifiers.get("barcode"),
        "claims": ", ".join(product.get("claims") or []),
        "marketing_description": product.get("marketing_description"),
        "grounding_prompt": product.get("grounding_prompt"),
        "visibility_score": product.get("visibility_score"),
        "issues_detected": ", ".join(product.get("issues_detected") or []),
        "frame_count": product.get("frame_count", 0),
        "frames_path": product.get("frames_path"),
        "primary_image_url": product.get("primary_image_url"),
        "status": product.get("status"),
        "version": product.get("version"),
        "created_at": product.get("created_at"),
        "updated_at": product.get("updated_at"),
        # Nutrition facts as separate columns
        "nutrition_serving_size": nutrition.get("serving_size"),
        "nutrition_calories": nutrition.get("calories"),
        "nutrition_total_fat": nutrition.get("total_fat"),
        "nutrition_protein": nutrition.get("protein"),
        "nutrition_carbohydrates": nutrition.get("carbohydrates"),
        "nutrition_sugar": nutrition.get("sugar"),
        "nutrition_fiber": nutrition.get("fiber"),
        "nutrition_sodium": nutrition.get("sodium"),
    }


# All CSV/JSON field names
ALL_FIELD_NAMES = [
    "id",
    "barcode",
    "video_id",
    "video_url",
    "brand_name",
    "sub_brand",
    "manufacturer_country",
    "product_name",
    "variant_flavor",
    "category",
    "container_type",
    "net_quantity",
    "pack_type",
    "pack_item_count",
    "sku_model_code",
    "secondary_barcode",
    "claims",
    "marketing_description",
    "grounding_prompt",
    "visibility_score",
    "issues_detected",
    "frame_count",
    "frames_path",
    "primary_image_url",
    "status",
    "version",
    "created_at",
    "updated_at",
    "nutrition_serving_size",
    "nutrition_calories",
    "nutrition_total_fat",
    "nutrition_protein",
    "nutrition_carbohydrates",
    "nutrition_sugar",
    "nutrition_fiber",
    "nutrition_sodium",
]


# ===========================================
# CRUD Endpoints
# ===========================================


@router.get("")
async def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=1000),
    search: Optional[str] = None,
    # Sorting parameters
    sort_by: Optional[str] = Query(None, description="Column to sort by"),
    sort_order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
    # All filter parameters
    status: Optional[str] = Query(None, description="Comma-separated status values"),
    category: Optional[str] = Query(None, description="Comma-separated category values"),
    brand: Optional[str] = Query(None, description="Comma-separated brand names"),
    sub_brand: Optional[str] = Query(None, description="Comma-separated sub-brand names"),
    product_name: Optional[str] = Query(None, description="Comma-separated product names"),
    variant_flavor: Optional[str] = Query(None, description="Comma-separated variant/flavor values"),
    container_type: Optional[str] = Query(None, description="Comma-separated container types"),
    net_quantity: Optional[str] = Query(None, description="Comma-separated net quantity values"),
    pack_type: Optional[str] = Query(None, description="Comma-separated pack types"),
    manufacturer_country: Optional[str] = Query(None, description="Comma-separated countries"),
    claims: Optional[str] = Query(None, description="Comma-separated claims"),
    # Boolean filters
    has_video: Optional[bool] = None,
    has_image: Optional[bool] = None,
    has_nutrition: Optional[bool] = None,
    has_description: Optional[bool] = None,
    has_prompt: Optional[bool] = None,
    has_issues: Optional[bool] = None,
    # Range filters
    frame_count_min: Optional[int] = None,
    frame_count_max: Optional[int] = None,
    visibility_score_min: Optional[int] = None,
    visibility_score_max: Optional[int] = None,
    # Exclusion filters
    exclude_dataset_id: Optional[str] = Query(None, description="Exclude products that are in this dataset"),
    include_frame_counts: bool = Query(False, description="Include synthetic/real/augmented frame counts"),
    db: SupabaseService = Depends(get_supabase),
):
    """List products with pagination and comprehensive filters."""
    result = await db.get_products(
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
        exclude_dataset_id=exclude_dataset_id,
        include_frame_counts=include_frame_counts,
    )
    return result


@router.get("/categories")
async def get_product_categories(
    db: SupabaseService = Depends(get_supabase),
) -> list[str]:
    """Get unique product categories."""
    return await db.get_product_categories()


@router.get("/filter-options")
async def get_filter_options(
    # Current filter selections (for cascading filters)
    status: Optional[str] = Query(None, description="Comma-separated status values"),
    category: Optional[str] = Query(None, description="Comma-separated category values"),
    brand: Optional[str] = Query(None, description="Comma-separated brand names"),
    sub_brand: Optional[str] = Query(None, description="Comma-separated sub-brand names"),
    product_name: Optional[str] = Query(None, description="Comma-separated product names"),
    variant_flavor: Optional[str] = Query(None, description="Comma-separated variant/flavor values"),
    container_type: Optional[str] = Query(None, description="Comma-separated container types"),
    net_quantity: Optional[str] = Query(None, description="Comma-separated net quantity values"),
    pack_type: Optional[str] = Query(None, description="Comma-separated pack types"),
    manufacturer_country: Optional[str] = Query(None, description="Comma-separated countries"),
    claims: Optional[str] = Query(None, description="Comma-separated claims"),
    # Boolean filters
    has_video: Optional[bool] = None,
    has_image: Optional[bool] = None,
    has_nutrition: Optional[bool] = None,
    has_description: Optional[bool] = None,
    has_prompt: Optional[bool] = None,
    has_issues: Optional[bool] = None,
    # Range filters
    frame_count_min: Optional[int] = None,
    frame_count_max: Optional[int] = None,
    visibility_score_min: Optional[int] = None,
    visibility_score_max: Optional[int] = None,
    # Exclusion filters
    exclude_dataset_id: Optional[str] = Query(None, description="Exclude products in this dataset"),
    db: SupabaseService = Depends(get_supabase),
) -> dict:
    """Get filter options, optionally filtered by current selections (cascading filters)."""
    return await db.get_product_filter_options(
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
        claims_filter=claims,
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
        exclude_dataset_id=exclude_dataset_id,
    )


@router.get("/{product_id}")
async def get_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get product details with identifiers."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Fetch identifiers and add to response
    identifiers = await db.get_product_identifiers(product_id)
    product["identifiers_list"] = identifiers

    return product


@router.post("")
async def create_product(
    data: ProductCreate,
    db: SupabaseService = Depends(get_supabase),
):
    """Create a new product."""
    product_data = data.model_dump(exclude_unset=True)
    return await db.create_product(product_data)


@router.patch("/{product_id}")
async def update_product(
    product_id: str,
    data: ProductUpdate,
    db: SupabaseService = Depends(get_supabase),
):
    """Update product metadata with optimistic locking."""
    # Check if product exists
    existing = await db.get_product(product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = data.model_dump(exclude_unset=True, exclude={"version"})

    try:
        return await db.update_product(
            product_id,
            update_data,
            expected_version=data.version,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=409,
            detail="Product was modified by another user. Please refresh and try again.",
        ) from e


@router.delete("/{product_id}")
async def delete_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Delete a product and ALL related data.

    This includes:
    - All frames (product_images table)
    - All storage files (Supabase Storage)
    - Dataset references
    - Product identifiers
    """
    existing = await db.get_product(product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await db.delete_product_cascade(product_id)
    return {
        "status": "deleted",
        "details": result,
    }


@router.get("/{product_id}/frames")
async def get_product_frames(
    product_id: str,
    image_type: Optional[str] = Query(None, description="Filter by type: synthetic, real, augmented"),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get product frames from database.

    Returns all frames by default, or filter by image_type.
    Supports: synthetic (from video), real (from matching), augmented (from augmentation)
    """
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Validate image_type if provided
    if image_type and image_type not in ["synthetic", "real", "augmented"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image_type. Must be 'synthetic', 'real', or 'augmented'"
        )

    # Get frames from database
    db_frames = await db.get_product_frames(product_id, image_type=image_type)

    # If no frames in DB, fallback to storage-based lookup for synthetic frames
    # This handles legacy products that don't have DB records yet
    if not db_frames and (image_type is None or image_type == "synthetic"):
        frame_count = product.get("frame_count", 0)
        for i in range(frame_count):
            db_frames.append({
                "id": None,  # No DB record
                "url": db.get_public_url("frames", f"{product_id}/frame_{i:04d}.png"),
                "image_url": db.get_public_url("frames", f"{product_id}/frame_{i:04d}.png"),
                "index": i,
                "frame_index": i,
                "image_type": "synthetic",
                "source": "video_frame",
            })

    # Format response
    frames = []
    for frame in db_frames:
        frames.append({
            "id": frame.get("id"),
            "url": frame.get("image_url") or db.get_public_url("frames", frame.get("image_path", "")),
            "index": frame.get("frame_index", 0),
            "image_type": frame.get("image_type", "synthetic"),
            "source": frame.get("source", "video_frame"),
        })

    # Get counts
    counts = await db.get_product_frame_counts(product_id)

    return {
        "frames": frames,
        "counts": counts,
        "total": len(frames),
    }


@router.delete("/{product_id}/frames")
async def delete_product_frames(
    product_id: str,
    frame_ids: List[str] = Query(..., description="List of frame IDs to delete"),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Delete specific frames from a product.

    Also deletes the corresponding files from storage.
    """
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if not frame_ids:
        raise HTTPException(status_code=400, detail="No frame IDs provided")

    # Get frame details before deletion (for storage cleanup)
    frames_to_delete = await db.get_product_frames(product_id)
    frames_to_delete = [f for f in frames_to_delete if f.get("id") in frame_ids]

    # Delete from database
    deleted_count = await db.delete_product_frames(product_id, frame_ids)

    # Delete from storage
    storage_deleted = 0
    for frame in frames_to_delete:
        image_path = frame.get("image_path")
        if image_path:
            try:
                await db.delete_file("frames", image_path)
                storage_deleted += 1
            except Exception:
                pass  # File might not exist

    return {
        "status": "deleted",
        "deleted_count": deleted_count,
        "storage_deleted": storage_deleted,
    }


@router.get("/{product_id}/debug-frames")
async def debug_product_frames(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Debug endpoint to check frame data in database and storage.

    Helps identify discrepancies between storage files and database records.
    """
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Get all records from product_images table
    try:
        response = (
            db.client.table("product_images")
            .select("*")
            .eq("product_id", product_id)
            .execute()
        )
        db_records = response.data
    except Exception as e:
        db_records = []

    # Count by type
    db_counts = {"synthetic": 0, "real": 0, "augmented": 0}
    for record in db_records:
        img_type = record.get("image_type")
        if img_type in db_counts:
            db_counts[img_type] += 1

    # List storage files in augmented folder
    augmented_files = []
    try:
        files = db.client.storage.from_("frames").list(f"{product_id}/augmented")
        augmented_files = [f.get("name") for f in files if f.get("name") and f.get("name").endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    except Exception as e:
        augmented_files = []

    # List storage files in root folder (synthetic frames)
    synthetic_files = []
    try:
        files = db.client.storage.from_("frames").list(product_id)
        synthetic_files = [f.get("name") for f in files if f.get("name") and f.get("name").endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    except Exception:
        synthetic_files = []

    return {
        "product_id": product_id,
        "barcode": product.get("barcode"),
        "frame_count_field": product.get("frame_count", 0),
        "database": {
            "total_records": len(db_records),
            "counts": db_counts,
            "sample_records": db_records[:5] if db_records else [],
        },
        "storage": {
            "synthetic_files_count": len(synthetic_files),
            "augmented_files_count": len(augmented_files),
            "augmented_files_sample": augmented_files[:10] if augmented_files else [],
        },
        "diagnosis": {
            "augmented_in_db": db_counts["augmented"],
            "augmented_in_storage": len(augmented_files),
            "mismatch": db_counts["augmented"] != len(augmented_files),
        }
    }


@router.post("/{product_id}/sync-augmented")
async def sync_augmented_from_storage(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Sync augmented images from storage to database.

    Registers any augmented files in storage that aren't tracked in the database.
    This fixes products where augmentation ran but DB records weren't created.
    """
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    result = await db.sync_augmented_from_storage(product_id)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/{product_id}/reprocess")
async def reprocess_product(
    product_id: str,
    request: Request,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """
    Reprocess a product that was previously processed.
    This will delete old frames and re-run the entire pipeline.
    Uses the product's video_id or video_url to reprocess.
    """
    # Get product details
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    video_id = product.get("video_id")
    video_url = product.get("video_url")
    barcode = product.get("barcode")

    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="Product has no video URL. Cannot reprocess."
        )

    print(f"[Products] Reprocessing product {product_id} (barcode: {barcode})")

    # 1. Clean up old data (frames + storage)
    cleanup_result = await db.cleanup_product_for_reprocess(product_id)
    print(f"[Products] Cleanup: {cleanup_result['frames_deleted']} frames, {cleanup_result['files_deleted']} files deleted")

    # 2. Update product status to processing
    await db.update_product(product_id, {"status": "processing"})

    # 3. If product has video_id, update video status too
    if video_id:
        try:
            db.client.table("videos").update({
                "status": "processing"
            }).eq("id", video_id).execute()
        except Exception as e:
            print(f"[Products] Warning: Could not update video status: {e}")

    # 4. Create new job
    job = await db.create_job({
        "type": "video_processing",
        "config": {
            "video_id": video_id,
            "video_url": video_url,
            "barcode": barcode,
            "product_id": product_id,
            "reprocess": True,
        },
    })

    # 5. Dispatch to Runpod
    if runpod.is_configured(EndpointType.VIDEO):
        try:
            webhook_url = get_webhook_url(request)
            runpod_response = await runpod.submit_job(
                endpoint_type=EndpointType.VIDEO,
                input_data={
                    "video_url": video_url,
                    "barcode": barcode,
                    "video_id": video_id,
                    "product_id": product_id,
                    "job_id": job["id"],
                },
                webhook_url=webhook_url,
            )

            await db.update_job(job["id"], {
                "status": "queued",
                "runpod_job_id": runpod_response.get("id"),
            })
            job["runpod_job_id"] = runpod_response.get("id")
            job["status"] = "queued"

            print(f"[Products] Reprocess dispatched to Runpod: {runpod_response.get('id')}")

        except Exception as e:
            print(f"[Products] Failed to dispatch reprocess to Runpod: {e}")
            await db.update_job(job["id"], {
                "status": "failed",
                "error": f"Failed to dispatch to Runpod: {str(e)}",
            })
            # Reset product status
            await db.update_product(product_id, {"status": "pending"})
            raise HTTPException(
                status_code=500,
                detail=f"Failed to dispatch to Runpod: {str(e)}",
            )
    else:
        print("[Products] Runpod not configured, reprocess job created but not dispatched")

    return {
        "job": job,
        "cleanup": cleanup_result,
        "message": f"Reprocessing product {product_id}",
    }


# ===========================================
# Download & Export Endpoints
# ===========================================


@router.post("/download")
async def download_products(
    request: ExportRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download products as ZIP with all metadata. Supports all filters."""
    products = await get_all_filtered_products(db, request)

    if not products:
        raise HTTPException(status_code=404, detail="No products found matching filters")

    # Create ZIP in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for product in products:
            folder_name = product.get("barcode", product["id"])

            # Add full metadata.json with ALL fields
            metadata = {
                "id": product.get("id"),
                "barcode": product.get("barcode"),
                "video_id": product.get("video_id"),
                "video_url": product.get("video_url"),
                "brand_name": product.get("brand_name"),
                "sub_brand": product.get("sub_brand"),
                "manufacturer_country": product.get("manufacturer_country"),
                "product_name": product.get("product_name"),
                "variant_flavor": product.get("variant_flavor"),
                "category": product.get("category"),
                "container_type": product.get("container_type"),
                "net_quantity": product.get("net_quantity"),
                "pack_configuration": product.get("pack_configuration"),
                "identifiers": product.get("identifiers"),
                "nutrition_facts": product.get("nutrition_facts"),
                "claims": product.get("claims"),
                "marketing_description": product.get("marketing_description"),
                "grounding_prompt": product.get("grounding_prompt"),
                "visibility_score": product.get("visibility_score"),
                "issues_detected": product.get("issues_detected"),
                "frame_count": product.get("frame_count", 0),
                "frames_path": product.get("frames_path"),
                "primary_image_url": product.get("primary_image_url"),
                "status": product.get("status"),
                "version": product.get("version"),
                "created_at": product.get("created_at"),
                "updated_at": product.get("updated_at"),
            }
            zf.writestr(f"{folder_name}/metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False, default=str))

    zip_buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=products_{len(products)}_{timestamp}.zip"},
    )


@router.get("/download/all")
async def download_all_products(
    status: Optional[str] = None,
    category: Optional[str] = None,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download ALL products as ZIP (legacy endpoint, use POST /download instead)."""
    request = ExportRequest(
        status=[status] if status else None,
        category=[category] if category else None,
    )
    return await download_products(request, db)


@router.get("/{product_id}/download")
async def download_single_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download single product as ZIP with all frames and metadata."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Full metadata
        metadata = {
            "id": product.get("id"),
            "barcode": product.get("barcode"),
            "video_id": product.get("video_id"),
            "video_url": product.get("video_url"),
            "brand_name": product.get("brand_name"),
            "sub_brand": product.get("sub_brand"),
            "manufacturer_country": product.get("manufacturer_country"),
            "product_name": product.get("product_name"),
            "variant_flavor": product.get("variant_flavor"),
            "category": product.get("category"),
            "container_type": product.get("container_type"),
            "net_quantity": product.get("net_quantity"),
            "pack_configuration": product.get("pack_configuration"),
            "identifiers": product.get("identifiers"),
            "nutrition_facts": product.get("nutrition_facts"),
            "claims": product.get("claims"),
            "marketing_description": product.get("marketing_description"),
            "grounding_prompt": product.get("grounding_prompt"),
            "visibility_score": product.get("visibility_score"),
            "issues_detected": product.get("issues_detected"),
            "frame_count": product.get("frame_count", 0),
            "frames_path": product.get("frames_path"),
            "primary_image_url": product.get("primary_image_url"),
            "status": product.get("status"),
            "version": product.get("version"),
            "created_at": product.get("created_at"),
            "updated_at": product.get("updated_at"),
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False, default=str))

    zip_buffer.seek(0)
    barcode = product.get("barcode", product_id)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={barcode}.zip"},
    )


@router.post("/export/csv")
async def export_products_csv(
    request: ExportRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Export products as CSV file with ALL fields. Supports all filters."""
    products = await get_all_filtered_products(db, request)

    if not products:
        raise HTTPException(status_code=404, detail="No products found matching filters")

    output = BytesIO()
    # Write UTF-8 BOM for Excel compatibility
    output.write(b"\xef\xbb\xbf")

    # Use TextIOWrapper for CSV writer
    import io

    text_output = io.TextIOWrapper(output, encoding="utf-8", newline="", write_through=True)

    writer = csv.DictWriter(text_output, fieldnames=ALL_FIELD_NAMES, extrasaction="ignore")
    writer.writeheader()

    for product in products:
        row = product_to_full_dict(product)
        writer.writerow(row)

    text_output.detach()  # Detach to prevent closing BytesIO
    output.seek(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=products_{len(products)}_{timestamp}.csv"},
    )


@router.post("/export/json")
async def export_products_json(
    request: ExportRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Export products as JSON file with ALL fields. Supports all filters."""
    products = await get_all_filtered_products(db, request)

    if not products:
        raise HTTPException(status_code=404, detail="No products found matching filters")

    # Full product data for JSON
    export_data = []
    for product in products:
        export_data.append({
            "id": product.get("id"),
            "barcode": product.get("barcode"),
            "video_id": product.get("video_id"),
            "video_url": product.get("video_url"),
            "brand_name": product.get("brand_name"),
            "sub_brand": product.get("sub_brand"),
            "manufacturer_country": product.get("manufacturer_country"),
            "product_name": product.get("product_name"),
            "variant_flavor": product.get("variant_flavor"),
            "category": product.get("category"),
            "container_type": product.get("container_type"),
            "net_quantity": product.get("net_quantity"),
            "pack_configuration": product.get("pack_configuration"),
            "identifiers": product.get("identifiers"),
            "nutrition_facts": product.get("nutrition_facts"),
            "claims": product.get("claims"),
            "marketing_description": product.get("marketing_description"),
            "grounding_prompt": product.get("grounding_prompt"),
            "visibility_score": product.get("visibility_score"),
            "issues_detected": product.get("issues_detected"),
            "frame_count": product.get("frame_count", 0),
            "frames_path": product.get("frames_path"),
            "primary_image_url": product.get("primary_image_url"),
            "status": product.get("status"),
            "version": product.get("version"),
            "created_at": product.get("created_at"),
            "updated_at": product.get("updated_at"),
        })

    output = BytesIO()
    output.write(json.dumps(export_data, indent=2, ensure_ascii=False, default=str).encode("utf-8"))
    output.seek(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        output,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=products_{len(products)}_{timestamp}.json"},
    )


@router.post("/bulk-delete")
async def bulk_delete_products(
    request: ProductIdsRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Delete multiple products and ALL related data at once.

    This includes for each product:
    - All frames (product_images table)
    - All storage files (Supabase Storage)
    - Dataset references
    - Product identifiers
    """
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    try:
        result = await db.delete_products_cascade(request.product_ids)
        return {
            "deleted_count": result["products_deleted"],
            "details": result,
        }
    except Exception as e:
        error_msg = str(e)
        if "invalid input syntax for type uuid" in error_msg:
            raise HTTPException(status_code=400, detail="Invalid product ID format")
        raise HTTPException(status_code=500, detail=f"Delete failed: {error_msg}")


# ===========================================
# Product Identifiers Endpoints
# ===========================================


@router.get("/{product_id}/identifiers")
async def get_product_identifiers(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
) -> List[ProductIdentifier]:
    """Get all identifiers for a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    identifiers = await db.get_product_identifiers(product_id)
    return identifiers


@router.post("/{product_id}/identifiers")
async def add_product_identifier(
    product_id: str,
    data: ProductIdentifierCreate,
    db: SupabaseService = Depends(get_supabase),
) -> ProductIdentifier:
    """Add a new identifier to a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        identifier = await db.add_product_identifier(product_id, data.model_dump())
        return identifier
    except Exception as e:
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "unique" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="This identifier already exists for this product"
            )
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{product_id}/identifiers")
async def replace_product_identifiers(
    product_id: str,
    data: ProductIdentifiersUpdate,
    db: SupabaseService = Depends(get_supabase),
) -> List[ProductIdentifier]:
    """Replace all identifiers for a product (bulk update)."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    identifiers = await db.replace_product_identifiers(
        product_id,
        [i.model_dump() for i in data.identifiers]
    )
    return identifiers


@router.patch("/{product_id}/identifiers/{identifier_id}")
async def update_product_identifier(
    product_id: str,
    identifier_id: str,
    data: ProductIdentifierUpdate,
    db: SupabaseService = Depends(get_supabase),
) -> ProductIdentifier:
    """Update a specific identifier."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = data.model_dump(exclude_unset=True)
    identifier = await db.update_product_identifier(identifier_id, update_data)
    if not identifier:
        raise HTTPException(status_code=404, detail="Identifier not found")
    return identifier


@router.delete("/{product_id}/identifiers/{identifier_id}")
async def delete_product_identifier(
    product_id: str,
    identifier_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a specific identifier."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    await db.delete_product_identifier(identifier_id)
    return {"status": "deleted"}


@router.post("/{product_id}/identifiers/{identifier_id}/set-primary")
async def set_primary_identifier(
    product_id: str,
    identifier_id: str,
    db: SupabaseService = Depends(get_supabase),
) -> ProductIdentifier:
    """Set an identifier as the primary identifier."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    identifier = await db.set_primary_identifier(product_id, identifier_id)
    if not identifier:
        raise HTTPException(status_code=404, detail="Identifier not found")
    return identifier


# ===========================================
# Custom Fields Endpoints
# ===========================================


@router.get("/{product_id}/custom-fields")
async def get_custom_fields(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
) -> dict[str, str]:
    """Get custom fields for a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    return product.get("custom_fields") or {}


@router.put("/{product_id}/custom-fields")
async def update_custom_fields(
    product_id: str,
    data: CustomFieldsUpdate,
    db: SupabaseService = Depends(get_supabase),
) -> dict[str, str]:
    """Replace all custom fields for a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        updated = await db.update_product(
            product_id,
            {"custom_fields": data.custom_fields},
            expected_version=product.get("version"),
        )
        return updated.get("custom_fields") or {}
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail="Product was modified by another user. Please refresh and try again.",
        )


@router.patch("/{product_id}/custom-fields")
async def patch_custom_fields(
    product_id: str,
    data: dict[str, Optional[str]],
    db: SupabaseService = Depends(get_supabase),
) -> dict[str, str]:
    """Patch custom fields (add/update/delete individual fields). Set value to null to delete a field."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    current_fields = product.get("custom_fields") or {}

    # Apply updates
    for key, value in data.items():
        if value is None:
            current_fields.pop(key, None)
        else:
            current_fields[key] = value

    try:
        updated = await db.update_product(
            product_id,
            {"custom_fields": current_fields},
            expected_version=product.get("version"),
        )
        return updated.get("custom_fields") or {}
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail="Product was modified by another user. Please refresh and try again.",
        )
