"""Products API router with CRUD, download, and export functionality."""

import csv
import json
import zipfile
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class ProductBase(BaseModel):
    """Base product schema."""

    barcode: str
    brand_name: Optional[str] = None
    sub_brand: Optional[str] = None
    product_name: Optional[str] = None
    variant_flavor: Optional[str] = None
    category: Optional[str] = None
    container_type: Optional[str] = None
    net_quantity: Optional[str] = None


class ProductCreate(ProductBase):
    """Product creation schema."""

    video_id: Optional[int] = None
    video_url: Optional[str] = None


class ProductUpdate(BaseModel):
    """Product update schema."""

    brand_name: Optional[str] = None
    sub_brand: Optional[str] = None
    product_name: Optional[str] = None
    variant_flavor: Optional[str] = None
    category: Optional[str] = None
    container_type: Optional[str] = None
    net_quantity: Optional[str] = None
    status: Optional[str] = None
    version: Optional[int] = None  # For optimistic locking


class ProductIdsRequest(BaseModel):
    """Request with product IDs."""

    product_ids: Optional[list[str]] = None


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# CRUD Endpoints
# ===========================================


@router.get("")
async def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=1000),
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    db: SupabaseService = Depends(get_supabase),
):
    """List products with pagination and filters."""
    result = await db.get_products(
        page=page,
        limit=limit,
        search=search,
        status=status,
        category=category,
    )
    return result


@router.get("/categories")
async def get_product_categories(
    db: SupabaseService = Depends(get_supabase),
) -> list[str]:
    """Get unique product categories."""
    return await db.get_product_categories()


@router.get("/{product_id}")
async def get_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get product details."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
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
    """Delete a product."""
    existing = await db.get_product(product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")

    await db.delete_product(product_id)
    return {"status": "deleted"}


@router.get("/{product_id}/frames")
async def get_product_frames(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get product frames."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Get frames from Supabase Storage
    # Pipeline uploads using product_id as folder name
    frame_count = product.get("frame_count", 0)

    frames = []
    for i in range(frame_count):
        frames.append({
            "url": db.get_public_url("frames", f"{product_id}/frame_{i:04d}.png"),
            "index": i,
        })

    return {"frames": frames}


# ===========================================
# Download & Export Endpoints
# ===========================================


@router.post("/download")
async def download_products(
    request: ProductIdsRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download selected products as ZIP with frames and metadata."""
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    # Fetch products
    products = []
    for pid in request.product_ids:
        product = await db.get_product(pid)
        if product:
            products.append(product)

    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    # Create ZIP in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for product in products:
            folder_name = product.get("barcode", product["id"])

            # Add metadata.json
            metadata = {
                "id": product["id"],
                "barcode": product.get("barcode"),
                "brand_name": product.get("brand_name"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "status": product.get("status"),
                "frame_count": product.get("frame_count", 0),
            }
            zf.writestr(f"{folder_name}/metadata.json", json.dumps(metadata, indent=2))

            # TODO: Add frame images from Supabase Storage

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=products_{len(products)}.zip"},
    )


@router.get("/download/all")
async def download_all_products(
    status: Optional[str] = None,
    category: Optional[str] = None,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download ALL products as ZIP (filtered by status/category)."""
    result = await db.get_products(page=1, limit=1000, status=status, category=category)
    products = result.get("items", [])

    if not products:
        raise HTTPException(status_code=404, detail="No products found")

    # Create ZIP in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for product in products:
            folder_name = product.get("barcode", product["id"])
            metadata = {
                "id": product["id"],
                "barcode": product.get("barcode"),
                "brand_name": product.get("brand_name"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "status": product.get("status"),
            }
            zf.writestr(f"{folder_name}/metadata.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=all_products_{len(products)}.zip"},
    )


@router.get("/{product_id}/download")
async def download_single_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Download single product as ZIP with all frames."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        metadata = {
            "id": product["id"],
            "barcode": product.get("barcode"),
            "brand_name": product.get("brand_name"),
            "product_name": product.get("product_name"),
            "category": product.get("category"),
            "status": product.get("status"),
        }
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    barcode = product.get("barcode", product_id)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={barcode}.zip"},
    )


@router.post("/export/csv")
async def export_products_csv(
    request: ProductIdsRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Export products as CSV file."""
    if request.product_ids:
        products = []
        for pid in request.product_ids:
            product = await db.get_product(pid)
            if product:
                products.append(product)
    else:
        result = await db.get_products(page=1, limit=10000)
        products = result.get("items", [])

    output = BytesIO()
    # Write UTF-8 BOM for Excel compatibility
    output.write(b"\xef\xbb\xbf")

    # Use TextIOWrapper for CSV writer
    import io

    text_output = io.TextIOWrapper(output, encoding="utf-8", newline="", write_through=True)

    fieldnames = [
        "id",
        "barcode",
        "brand_name",
        "product_name",
        "category",
        "status",
        "frame_count",
        "created_at",
    ]
    writer = csv.DictWriter(text_output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for product in products:
        writer.writerow(product)

    text_output.detach()  # Detach to prevent closing BytesIO
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=products.csv"},
    )


@router.post("/export/json")
async def export_products_json(
    request: ProductIdsRequest,
    db: SupabaseService = Depends(get_supabase),
) -> StreamingResponse:
    """Export products as JSON file."""
    if request.product_ids:
        products = []
        for pid in request.product_ids:
            product = await db.get_product(pid)
            if product:
                products.append(product)
    else:
        result = await db.get_products(page=1, limit=10000)
        products = result.get("items", [])

    # Clean up products for export
    export_data = []
    for p in products:
        export_data.append(
            {
                "id": p["id"],
                "barcode": p.get("barcode"),
                "brand_name": p.get("brand_name"),
                "product_name": p.get("product_name"),
                "category": p.get("category"),
                "status": p.get("status"),
                "frame_count": p.get("frame_count", 0),
                "created_at": p.get("created_at"),
            }
        )

    output = BytesIO()
    output.write(json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8"))
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=products.json"},
    )


@router.post("/bulk-delete")
async def bulk_delete_products(
    request: ProductIdsRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete multiple products at once."""
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    deleted_count = await db.delete_products(request.product_ids)
    return {"deleted_count": deleted_count}
