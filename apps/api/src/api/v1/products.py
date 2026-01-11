"""Products API router with CRUD, download, and export functionality."""

import csv
import json
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


class Product(ProductBase):
    """Full product schema."""

    id: str
    video_id: Optional[int] = None
    video_url: Optional[str] = None
    status: str = "pending"
    frame_count: int = 0
    frames_path: Optional[str] = None
    primary_image_url: Optional[str] = None
    version: int = 1
    created_at: datetime
    updated_at: datetime


class ProductsResponse(BaseModel):
    """Paginated products response."""

    items: list[Product]
    total: int
    page: int
    limit: int


class ProductIdsRequest(BaseModel):
    """Request with product IDs."""

    product_ids: Optional[list[str]] = None


# ===========================================
# Mock Data (will be replaced with Supabase)
# ===========================================

MOCK_PRODUCTS: list[dict] = [
    {
        "id": str(uuid4()),
        "barcode": "0012345678901",
        "brand_name": "Coca-Cola",
        "product_name": "Classic",
        "category": "Beverages",
        "container_type": "Can",
        "status": "ready",
        "frame_count": 24,
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
    {
        "id": str(uuid4()),
        "barcode": "0012345678902",
        "brand_name": "Pepsi",
        "product_name": "Max",
        "category": "Beverages",
        "container_type": "Bottle",
        "status": "ready",
        "frame_count": 18,
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
]


# ===========================================
# CRUD Endpoints
# ===========================================


@router.get("", response_model=ProductsResponse)
async def list_products(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
) -> ProductsResponse:
    """List products with pagination and filters."""
    items = MOCK_PRODUCTS.copy()

    # Apply filters
    if search:
        search_lower = search.lower()
        items = [
            p
            for p in items
            if search_lower in p.get("barcode", "").lower()
            or search_lower in (p.get("product_name") or "").lower()
            or search_lower in (p.get("brand_name") or "").lower()
        ]
    if status:
        items = [p for p in items if p.get("status") == status]
    if category:
        items = [p for p in items if p.get("category") == category]

    total = len(items)
    start = (page - 1) * limit
    end = start + limit

    return ProductsResponse(
        items=[Product(**p) for p in items[start:end]],
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/categories")
async def get_product_categories() -> list[str]:
    """Get unique product categories."""
    categories = list(set(p.get("category") for p in MOCK_PRODUCTS if p.get("category")))
    return sorted(categories)


@router.get("/{product_id}", response_model=Product)
async def get_product(product_id: str) -> Product:
    """Get product details."""
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return Product(**product)


@router.patch("/{product_id}", response_model=Product)
async def update_product(product_id: str, data: ProductUpdate) -> Product:
    """Update product metadata with optimistic locking."""
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Optimistic locking check
    if data.version is not None and data.version != product.get("version", 1):
        raise HTTPException(
            status_code=409,
            detail="Product was modified by another user. Please refresh and try again.",
        )

    update_data = data.model_dump(exclude_unset=True, exclude={"version"})
    product.update(update_data)
    product["updated_at"] = datetime.now().isoformat()
    product["version"] = product.get("version", 1) + 1

    return Product(**product)


@router.delete("/{product_id}")
async def delete_product(product_id: str) -> dict[str, str]:
    """Delete a product."""
    global MOCK_PRODUCTS
    original_len = len(MOCK_PRODUCTS)
    MOCK_PRODUCTS = [p for p in MOCK_PRODUCTS if p["id"] != product_id]

    if len(MOCK_PRODUCTS) == original_len:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"status": "deleted"}


@router.get("/{product_id}/frames")
async def get_product_frames(product_id: str) -> dict:
    """Get product frames."""
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # TODO: Return frame URLs from Supabase Storage
    frame_count = product.get("frame_count", 0)
    frames = [{"url": f"/frames/{product_id}/frame_{i:04d}.png", "index": i} for i in range(frame_count)]

    return {"frames": frames}


# ===========================================
# Download & Export Endpoints
# ===========================================


@router.post("/download")
async def download_products(request: ProductIdsRequest) -> StreamingResponse:
    """Download selected products as ZIP with frames and metadata."""
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]

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
) -> StreamingResponse:
    """Download ALL products as ZIP (filtered by status/category)."""
    products = MOCK_PRODUCTS.copy()

    if status:
        products = [p for p in products if p.get("status") == status]
    if category:
        products = [p for p in products if p.get("category") == category]

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
async def download_single_product(product_id: str) -> StreamingResponse:
    """Download single product as ZIP with all frames."""
    product = next((p for p in MOCK_PRODUCTS if p["id"] == product_id), None)
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
async def export_products_csv(request: ProductIdsRequest) -> StreamingResponse:
    """Export products as CSV file."""
    if request.product_ids:
        products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]
    else:
        products = MOCK_PRODUCTS

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
async def export_products_json(request: ProductIdsRequest) -> StreamingResponse:
    """Export products as JSON file."""
    if request.product_ids:
        products = [p for p in MOCK_PRODUCTS if p["id"] in request.product_ids]
    else:
        products = MOCK_PRODUCTS

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
async def bulk_delete_products(request: ProductIdsRequest) -> dict[str, int]:
    """Delete multiple products at once."""
    if not request.product_ids:
        raise HTTPException(status_code=400, detail="No product IDs provided")

    global MOCK_PRODUCTS
    deleted_count = len([p for p in MOCK_PRODUCTS if p["id"] in request.product_ids])
    MOCK_PRODUCTS = [p for p in MOCK_PRODUCTS if p["id"] not in request.product_ids]

    return {"deleted_count": deleted_count}
