"""Matching API router for product matching with FAISS."""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service

router = APIRouter()


# ===========================================
# Schemas
# ===========================================


class MatchCandidate(BaseModel):
    """Match candidate schema."""

    id: str
    image_path: str
    image_url: str
    similarity: float
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """Search request schema."""

    top_k: int = 50
    min_similarity: float = 0.7


class SearchResponse(BaseModel):
    """Search response schema."""

    candidates: list[MatchCandidate]


class ApproveMatchRequest(BaseModel):
    """Request to approve/reject a match."""

    product_id: str
    candidate_paths: list[str]
    is_approved: bool


class ProductSummary(BaseModel):
    """Product summary for matching list."""

    id: str
    barcode: Optional[str] = None
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    primary_image_url: Optional[str] = None
    frame_count: int = 0
    real_image_count: int = 0
    status: Optional[str] = None


class RealImageUpload(BaseModel):
    """Request to add real images to a product."""

    product_id: str
    image_urls: list[str]


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Endpoints
# ===========================================


@router.get("/products", response_model=List[ProductSummary])
async def list_matching_products(
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=500),
    search: Optional[str] = Query(None, description="Search by barcode, name, or brand"),
    db: SupabaseService = Depends(get_supabase),
):
    """List all products for matching (with frame counts)."""
    # Get all products (not filtered by status)
    result = await db.get_products(page=page, limit=limit, search=search)

    products = []
    for p in result.get("items", []):
        # Get real image count from product_real_images table
        real_count = await db.get_real_image_count(p["id"])

        products.append(ProductSummary(
            id=p["id"],
            barcode=p.get("barcode"),
            brand_name=p.get("brand_name"),
            product_name=p.get("product_name"),
            primary_image_url=p.get("primary_image_url"),
            frame_count=p.get("frame_count", 0),
            real_image_count=real_count,
            status=p.get("status"),
        ))

    return products


@router.get("/products/{product_id}")
async def get_matching_product(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get product details for matching."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Get synthetic frames
    frame_count = product.get("frame_count", 0)
    frames = []
    for i in range(frame_count):
        frames.append({
            "url": db.get_public_url("frames", f"{product_id}/frame_{i:04d}.png"),
            "index": i,
            "type": "synthetic",
        })

    # Get real images
    real_images = await db.get_real_images(product_id)

    return {
        "product": product,
        "synthetic_frames": frames,
        "real_images": real_images,
    }


@router.get("/products/{product_id}/real-images")
async def get_product_real_images(
    product_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get real images for a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    real_images = await db.get_real_images(product_id)
    return {"images": real_images}


@router.post("/products/{product_id}/real-images")
async def add_real_images(
    product_id: str,
    image_urls: list[str],
    db: SupabaseService = Depends(get_supabase),
):
    """Add real images to a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    added = await db.add_real_images(product_id, image_urls)
    return {"added": added}


@router.delete("/products/{product_id}/real-images")
async def remove_real_images(
    product_id: str,
    image_ids: list[str],
    db: SupabaseService = Depends(get_supabase),
):
    """Remove real images from a product."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    await db.remove_real_images(product_id, image_ids)
    return {"status": "removed"}


# Legacy endpoints (keep for compatibility)

@router.get("/upcs")
async def list_upcs(
    db: SupabaseService = Depends(get_supabase),
):
    """List all UPCs that need matching (legacy)."""
    result = await db.get_products(page=1, limit=1000, status="needs_matching")
    return [p["barcode"] for p in result.get("items", []) if p.get("barcode")]


@router.post("/upcs/{upc}/search")
async def search_matches(
    upc: str,
    request: SearchRequest,
    db: SupabaseService = Depends(get_supabase),
) -> SearchResponse:
    """Search for matching candidates for a UPC."""
    result = await db.get_products(page=1, limit=1, search=upc)
    products = result.get("items", [])

    if not products:
        raise HTTPException(status_code=404, detail="UPC not found")

    # TODO: Implement FAISS search for real candidates
    # For now, return mock candidates
    mock_candidates = [
        {
            "id": f"match-{upc}-1",
            "image_path": f"/real/{upc}/image1.jpg",
            "image_url": db.get_public_url("real-images", f"{upc}/image1.jpg"),
            "similarity": 0.95,
            "metadata": {"source": "retail_dataset"},
        },
        {
            "id": f"match-{upc}-2",
            "image_path": f"/real/{upc}/image2.jpg",
            "image_url": db.get_public_url("real-images", f"{upc}/image2.jpg"),
            "similarity": 0.89,
            "metadata": {"source": "retail_dataset"},
        },
    ]

    sorted_candidates = sorted(mock_candidates, key=lambda x: x["similarity"], reverse=True)
    limited = sorted_candidates[: request.top_k]

    return SearchResponse(candidates=[MatchCandidate(**c) for c in limited])


@router.post("/approve")
async def approve_match(
    request: ApproveMatchRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Approve or reject a match (add real images to product)."""
    if request.is_approved:
        added = await db.add_real_images(request.product_id, request.candidate_paths)
        return {"status": "approved", "added": added}
    return {"status": "rejected"}


@router.get("/matches")
async def list_matches(
    upc: Optional[str] = Query(None, description="Filter by UPC"),
    is_approved: Optional[bool] = Query(None, description="Filter by approval status"),
    db: SupabaseService = Depends(get_supabase),
):
    """List all product matches."""
    # TODO: Implement product_matches table query
    return []
