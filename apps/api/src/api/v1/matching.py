"""Matching API router for product matching with Qdrant similarity search."""

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.qdrant import qdrant_service
from auth.dependencies import get_current_user
from auth.service import UserInfo

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


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


class CutoutCandidate(BaseModel):
    """Cutout candidate for matching."""

    id: str
    external_id: int
    image_url: str
    predicted_upc: Optional[str] = None
    similarity: Optional[float] = None
    match_type: str  # "barcode" | "similarity" | "both"
    has_embedding: bool = False
    is_matched: bool = False


class ProductCandidatesResponse(BaseModel):
    """Response for product candidates."""

    product: dict
    candidates: List[CutoutCandidate]
    barcode_match_count: int
    similarity_match_count: int
    total_count: int
    has_product_embedding: bool = False


class BulkMatchRequest(BaseModel):
    """Request to match multiple cutouts to a product."""

    cutout_ids: List[str]
    similarity_scores: Optional[List[float]] = None


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
    """Get product details for matching with all frame types."""
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Get all frames from database
    all_frames = await db.get_product_frames(product_id)

    # If no frames in DB, fallback to storage-based lookup for synthetic frames
    if not all_frames:
        frame_count = product.get("frame_count", 0)
        for i in range(frame_count):
            all_frames.append({
                "id": None,
                "image_url": db.get_public_url("frames", f"{product_id}/frame_{i:04d}.png"),
                "frame_index": i,
                "image_type": "synthetic",
                "source": "video_frame",
            })

    # Separate frames by type
    synthetic_frames = []
    real_images = []
    augmented_frames = []

    for frame in all_frames:
        frame_data = {
            "id": frame.get("id"),
            "url": frame.get("image_url") or db.get_public_url("frames", frame.get("image_path", "")),
            "index": frame.get("frame_index", 0),
            "type": frame.get("image_type", "synthetic"),
            "source": frame.get("source"),
        }

        img_type = frame.get("image_type", "synthetic")
        if img_type == "synthetic":
            synthetic_frames.append(frame_data)
        elif img_type == "real":
            real_images.append(frame_data)
        elif img_type == "augmented":
            augmented_frames.append(frame_data)

    # Get counts
    counts = await db.get_product_frame_counts(product_id)

    return {
        "product": product,
        "synthetic_frames": synthetic_frames,
        "real_images": real_images,
        "augmented_frames": augmented_frames,
        "frame_counts": counts,
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


# ===========================================
# New Matching Endpoints (Product-Centric)
# ===========================================


@router.get("/products/{product_id}/candidates", response_model=ProductCandidatesResponse)
async def get_product_candidates(
    product_id: str,
    min_similarity: float = Query(0.7, ge=0.0, le=1.0),
    include_matched: bool = Query(False, description="Include already matched cutouts"),
    limit: int = Query(100, ge=1, le=500),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get candidate cutouts for a product.

    Returns cutouts that match by:
    1. Barcode match: cutout.predicted_upc == product.barcode
    2. Similarity match: embedding similarity >= min_similarity (if embeddings exist)

    Candidates are sorted by: barcode matches first, then by similarity.
    """
    # Get product
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    product_barcode = product.get("barcode")
    candidates = []
    barcode_match_ids = set()
    similarity_match_ids = set()
    has_product_embedding = False

    # 1. Get barcode matches
    if product_barcode:
        barcode_cutouts = await db.get_cutouts(
            page=1,
            limit=limit,
            predicted_upc=product_barcode,
            is_matched=None if include_matched else False,
        )

        for cutout in barcode_cutouts.get("items", []):
            barcode_match_ids.add(cutout["id"])
            candidates.append(CutoutCandidate(
                id=cutout["id"],
                external_id=cutout["external_id"],
                image_url=cutout["image_url"],
                predicted_upc=cutout.get("predicted_upc"),
                similarity=1.0,  # Barcode match = perfect match
                match_type="barcode",
                has_embedding=cutout.get("has_embedding", False),
                is_matched=cutout.get("matched_product_id") is not None,
            ))

    # 2. Get similarity matches from Qdrant (if configured)
    if qdrant_service.is_configured():
        try:
            # Get active embedding model
            active_model = await db.get_active_embedding_model()
            if active_model and active_model.get("qdrant_collection"):
                collection_name = active_model["qdrant_collection"]

                # Get product embedding from Qdrant (product_id is the point ID)
                product_point = await qdrant_service.get_point(
                    collection_name=collection_name,
                    point_id=product_id,
                    with_vector=True,
                )

                if product_point and product_point.get("vector"):
                    has_product_embedding = True
                    # Search for similar cutouts using product's embedding
                    similar_results = await qdrant_service.search(
                        collection_name=collection_name,
                        query_vector=product_point["vector"],
                        limit=limit * 2,  # Get more to filter
                        score_threshold=min_similarity,
                        filter_conditions={"source": "cutout"},  # Only cutouts
                        with_payload=True,
                    )

                    # Process similarity results
                    for result in similar_results:
                        cutout_id = result["payload"].get("cutout_id")
                        if not cutout_id:
                            continue

                        # Skip if already in barcode matches
                        if cutout_id in barcode_match_ids:
                            # Update existing candidate with similarity score
                            for c in candidates:
                                if c.id == cutout_id:
                                    c.similarity = max(c.similarity or 0, result["score"])
                            similarity_match_ids.add(cutout_id)
                            continue

                        # Skip if already matched to another product
                        if not include_matched:
                            cutout = await db.get_cutout(cutout_id)
                            if cutout and cutout.get("matched_product_id"):
                                continue

                        similarity_match_ids.add(cutout_id)

                        # Get cutout details
                        cutout = await db.get_cutout(cutout_id)
                        if cutout:
                            candidates.append(CutoutCandidate(
                                id=cutout["id"],
                                external_id=cutout["external_id"],
                                image_url=cutout["image_url"],
                                predicted_upc=cutout.get("predicted_upc"),
                                similarity=result["score"],
                                match_type="similarity",
                                has_embedding=cutout.get("has_embedding", False),
                                is_matched=cutout.get("matched_product_id") is not None,
                            ))
                else:
                    print(f"No embedding found for product {product_id}")
        except Exception as e:
            # Log but don't fail if Qdrant search fails
            print(f"Qdrant similarity search failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Mark candidates that are both barcode AND similarity matches
    for candidate in candidates:
        if candidate.id in barcode_match_ids and candidate.id in similarity_match_ids:
            candidate.match_type = "both"

    # Sort: barcode matches first, then by similarity
    candidates.sort(key=lambda x: (
        0 if x.match_type in ["barcode", "both"] else 1,
        -(x.similarity or 0)
    ))

    return ProductCandidatesResponse(
        product=product,
        candidates=candidates[:limit],
        barcode_match_count=len(barcode_match_ids),
        similarity_match_count=len(similarity_match_ids),
        total_count=len(candidates),
        has_product_embedding=has_product_embedding,
    )


@router.post("/products/{product_id}/match")
async def match_cutouts_to_product(
    product_id: str,
    request: BulkMatchRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Match multiple cutouts to a product.

    This will:
    1. Update each cutout's matched_product_id
    2. Add cutout images as real images to the product
    """
    # Verify product exists
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    matched_count = 0
    image_urls = []

    for i, cutout_id in enumerate(request.cutout_ids):
        # Get cutout
        cutout = await db.get_cutout(cutout_id)
        if not cutout:
            continue

        # Get similarity score if provided
        similarity = None
        if request.similarity_scores and i < len(request.similarity_scores):
            similarity = request.similarity_scores[i]

        # Update cutout with match
        updated = await db.match_cutout_to_product(
            cutout_id=cutout_id,
            product_id=product_id,
            similarity=similarity,
            matched_by=current_user.username if current_user else None,
        )

        if updated:
            matched_count += 1
            image_urls.append(cutout["image_url"])

    # Add cutout images as real images to product
    if image_urls:
        await db.add_real_images(product_id, image_urls)

    return {
        "status": "success",
        "matched_count": matched_count,
        "total_requested": len(request.cutout_ids),
        "images_added": len(image_urls),
    }


@router.post("/products/{product_id}/unmatch")
async def unmatch_cutouts_from_product(
    product_id: str,
    cutout_ids: List[str],
    db: SupabaseService = Depends(get_supabase),
):
    """
    Remove match from cutouts (unmatch them from the product).
    """
    # Verify product exists
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    unmatched_count = 0

    for cutout_id in cutout_ids:
        # Update cutout to remove match
        updated = await db.update_cutout(cutout_id, {
            "matched_product_id": None,
            "match_similarity": None,
            "matched_by": None,
            "matched_at": None,
        })

        if updated:
            unmatched_count += 1

    return {
        "status": "success",
        "unmatched_count": unmatched_count,
        "total_requested": len(cutout_ids),
    }


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
