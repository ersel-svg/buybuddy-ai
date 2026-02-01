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
    match_type_filter: Optional[str] = Query(None, description="Filter by match type: 'barcode', 'similarity', 'both', or None for all"),
    product_collection: Optional[str] = Query(None, description="Product embeddings collection (defaults to active model)"),
    cutout_collection: Optional[str] = Query(None, description="Cutout embeddings collection (defaults to active model)"),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get candidate cutouts for a product.

    Returns cutouts that match by:
    1. Identifier match: cutout.predicted_upc OR cutout.annotated_upc matches ANY product identifier
       (barcode, upc, ean, short_code, sku from product_identifiers table)
    2. Similarity match: embedding similarity >= min_similarity (if embeddings exist)

    Supports multi-view products: if a product has multiple embeddings (different frames),
    searches using all of them and returns the best match for each cutout.

    Collections can be manually specified or will default to the active embedding model's collections.

    Candidates are sorted by: identifier matches first, then by similarity.
    """
    # Get product
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    candidates = []
    barcode_match_ids = set()
    similarity_match_ids = set()
    has_product_embedding = False

    # 1. Get barcode matches using ALL product identifiers
    # This searches product_identifiers table (barcode, upc, ean, short_code, sku)
    # plus the legacy products.barcode field
    all_identifiers = await db.get_all_product_identifier_values(product_id)

    if all_identifiers:
        # Query cutouts where predicted_upc OR annotated_upc matches ANY of the identifiers
        def _escape_identifier(value: str) -> str:
            return value.replace("\\", "\\\\").replace("\"", "\\\"")

        identifiers = [f"\"{_escape_identifier(str(v))}\"" for v in all_identifiers if v]
        if identifiers:
            identifiers_csv = ",".join(identifiers)
            cutouts_query = db.client.table("cutout_images").select(
                "id, external_id, image_url, predicted_upc, has_embedding, matched_product_id"
            ).or_(
                f"predicted_upc.in.({identifiers_csv}),"
                f"annotated_upc.in.({identifiers_csv})"
            )
        else:
            cutouts_query = None

        if cutouts_query is not None:
            if not include_matched:
                cutouts_query = cutouts_query.is_("matched_product_id", "null")

            cutouts_query = cutouts_query.limit(limit)
            barcode_cutouts_response = cutouts_query.execute()

            for cutout in barcode_cutouts_response.data or []:
                barcode_match_ids.add(cutout["id"])
                candidates.append(CutoutCandidate(
                    id=cutout["id"],
                    external_id=cutout["external_id"],
                    image_url=cutout["image_url"],
                    predicted_upc=cutout.get("predicted_upc"),
                    similarity=1.0,  # Barcode/identifier match = perfect match
                    match_type="barcode",
                    has_embedding=cutout.get("has_embedding", False),
                    is_matched=cutout.get("matched_product_id") is not None,
                ))

    # 2. Get similarity matches from Qdrant (if configured)
    if qdrant_service.is_configured():
        try:
            # Get active embedding model for default collections
            active_model = await db.get_active_embedding_model()

            # Determine collections to use
            # Priority: explicit parameter > model's separate collections > model's single collection
            actual_product_collection = product_collection
            actual_cutout_collection = cutout_collection

            if active_model:
                if not actual_product_collection:
                    actual_product_collection = (
                        active_model.get("product_collection") or
                        active_model.get("qdrant_collection")
                    )
                if not actual_cutout_collection:
                    actual_cutout_collection = active_model.get("cutout_collection")

                    # Auto-derive cutout collection from product collection name
                    # products_dinov2_base -> cutouts_dinov2_base
                    if not actual_cutout_collection and actual_product_collection:
                        if actual_product_collection.startswith("products_"):
                            actual_cutout_collection = actual_product_collection.replace("products_", "cutouts_", 1)
                        else:
                            actual_cutout_collection = active_model.get("qdrant_collection")

            if actual_product_collection:
                # Check if we're using separate collections
                use_separate_collections = actual_product_collection != actual_cutout_collection

                # Get product embeddings (multi-view support)
                product_embeddings = []

                if use_separate_collections:
                    # Separate collections: get all product embeddings by product_id
                    product_embeddings = await qdrant_service.get_product_embeddings(
                        collection_name=actual_product_collection,
                        product_id=product_id,
                        with_vectors=True,
                    )
                else:
                    # Single collection: try multi-view first, then fallback to single point
                    product_embeddings = await qdrant_service.get_product_embeddings(
                        collection_name=actual_product_collection,
                        product_id=product_id,
                        with_vectors=True,
                    )

                    # Fallback: try getting point directly by ID (legacy single embedding)
                    if not product_embeddings:
                        product_point = await qdrant_service.get_point(
                            collection_name=actual_product_collection,
                            point_id=product_id,
                            with_vector=True,
                        )
                        if product_point and product_point.get("vector"):
                            product_embeddings = [product_point]

                if product_embeddings:
                    has_product_embedding = True

                    # Collect all similarity results from all product views
                    # Key: cutout_id -> best score
                    cutout_best_scores: dict[str, float] = {}

                    for emb in product_embeddings:
                        vector = emb.get("vector")
                        if not vector:
                            continue

                        # Determine search collection and filter
                        search_collection = actual_cutout_collection or actual_product_collection
                        filter_conditions = None if use_separate_collections else {"source": "cutout"}

                        similar_results = await qdrant_service.search(
                            collection_name=search_collection,
                            query_vector=vector,
                            limit=limit,  # Optimized: don't fetch extra
                            score_threshold=min_similarity,
                            filter_conditions=filter_conditions,
                            with_payload=True,
                        )

                        # Track best score per cutout across all product views
                        for result in similar_results:
                            cutout_id = result["payload"].get("cutout_id")
                            if not cutout_id:
                                continue

                            score = result["score"]
                            if cutout_id not in cutout_best_scores or score > cutout_best_scores[cutout_id]:
                                cutout_best_scores[cutout_id] = score

                    # Process deduplicated results with best scores
                    # First, update barcode matches with similarity scores
                    for cutout_id, best_score in cutout_best_scores.items():
                        if cutout_id in barcode_match_ids:
                            for c in candidates:
                                if c.id == cutout_id:
                                    c.similarity = max(c.similarity or 0, best_score)
                            similarity_match_ids.add(cutout_id)

                    # Get IDs that need DB lookup (not in barcode matches)
                    cutout_ids_to_fetch = [
                        cid for cid in cutout_best_scores.keys()
                        if cid not in barcode_match_ids
                    ]

                    # BATCH FETCH: Chunk IDs to avoid overly long IN() queries
                    if cutout_ids_to_fetch:
                        cutouts_map: dict[str, dict] = {}
                        batch_size = 200
                        for i in range(0, len(cutout_ids_to_fetch), batch_size):
                            batch = cutout_ids_to_fetch[i:i + batch_size]
                            batch_result = db.client.table("cutout_images").select(
                                "id, external_id, image_url, predicted_upc, has_embedding, matched_product_id"
                            ).in_("id", batch).execute()
                            for c in (batch_result.data or []):
                                cutouts_map[c["id"]] = c

                        for cutout_id in cutout_ids_to_fetch:
                            cutout = cutouts_map.get(cutout_id)
                            if not cutout:
                                continue

                            # Skip if already matched to another product
                            if not include_matched and cutout.get("matched_product_id"):
                                continue

                            similarity_match_ids.add(cutout_id)
                            candidates.append(CutoutCandidate(
                                id=cutout["id"],
                                external_id=cutout["external_id"],
                                image_url=cutout["image_url"],
                                predicted_upc=cutout.get("predicted_upc"),
                                similarity=cutout_best_scores[cutout_id],
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

    # 4. Apply match_type filter if specified
    if match_type_filter:
        if match_type_filter == "both":
            candidates = [c for c in candidates if c.match_type == "both"]
        elif match_type_filter == "barcode":
            candidates = [c for c in candidates if c.match_type in ["barcode", "both"]]
        elif match_type_filter == "similarity":
            candidates = [c for c in candidates if c.match_type in ["similarity", "both"]]

    # 5. Sort: "both" first, then "barcode", then "similarity" - all by similarity descending
    def sort_key(x):
        type_priority = {"both": 0, "barcode": 1, "similarity": 2}.get(x.match_type, 3)
        return (type_priority, -(x.similarity or 0))

    candidates.sort(key=sort_key)

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
    Match multiple cutouts to a product (bulk operation).

    This will:
    1. Update all cutouts' matched_product_id in bulk
    2. Add cutout images as real images to the product
    """
    # Verify product exists
    product = await db.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Bulk match all cutouts at once
    matched_count, image_urls = await db.bulk_match_cutouts_to_product(
        cutout_ids=request.cutout_ids,
        product_id=product_id,
        similarity_scores=request.similarity_scores,
        matched_by=current_user.username if current_user else None,
    )

    # Add cutout images as real images to product (already bulk)
    images_added = 0
    if image_urls:
        images_added = await db.add_real_images(product_id, image_urls)

    return {
        "status": "success",
        "matched_count": matched_count,
        "total_requested": len(request.cutout_ids),
        "images_added": images_added,
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


# ===========================================
# Reverse Matching (Cutout → Products)
# ===========================================


class ProductMatchCandidate(BaseModel):
    """Product candidate for reverse matching."""

    id: str
    barcode: Optional[str] = None
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    primary_image_url: Optional[str] = None
    similarity: float
    match_type: str  # "barcode" | "similarity" | "both"


class CutoutProductsResponse(BaseModel):
    """Response for cutout's product candidates."""

    cutout: dict
    candidates: List[ProductMatchCandidate]
    barcode_match_count: int
    similarity_match_count: int
    total_count: int
    has_cutout_embedding: bool = False


@router.get("/cutouts/{cutout_id}/products", response_model=CutoutProductsResponse)
async def get_cutout_product_candidates(
    cutout_id: str,
    min_similarity: float = Query(0.7, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    product_collection: Optional[str] = Query(None, description="Product embeddings collection"),
    cutout_collection: Optional[str] = Query(None, description="Cutout embeddings collection"),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get product candidates for a cutout (reverse matching).

    Given a cutout image, find matching products by:
    1. Identifier match: cutout.predicted_upc matches ANY product identifier
       (searches product_identifiers table + legacy products.barcode field)
    2. Similarity match: embedding similarity >= min_similarity

    Useful for:
    - Verifying a cutout's match
    - Finding alternative products for a cutout
    - Quality checking existing matches
    """
    # Get cutout
    cutout = await db.get_cutout(cutout_id)
    if not cutout:
        raise HTTPException(status_code=404, detail="Cutout not found")

    predicted_upc = cutout.get("predicted_upc")
    candidates = []
    barcode_match_ids = set()
    similarity_match_ids = set()
    has_cutout_embedding = cutout.get("has_embedding", False)

    # 1. Get barcode/identifier matches
    # Search both product_identifiers table AND legacy products.barcode field
    if predicted_upc:
        # Use the new method that searches product_identifiers + legacy barcode
        barcode_products = await db.get_products_by_identifier_value(predicted_upc)

        for p in barcode_products:
            barcode_match_ids.add(p["id"])
            candidates.append(ProductMatchCandidate(
                id=p["id"],
                barcode=p.get("barcode"),
                brand_name=p.get("brand_name"),
                product_name=p.get("product_name"),
                primary_image_url=p.get("primary_image_url"),
                similarity=1.0,  # Barcode/identifier match = perfect match
                match_type="barcode",
            ))

    # 2. Get similarity matches from Qdrant
    if qdrant_service.is_configured() and has_cutout_embedding:
        try:
            # Get active model for default collections
            active_model = await db.get_active_embedding_model()

            # Determine collections
            actual_cutout_collection = cutout_collection
            actual_product_collection = product_collection

            if active_model:
                if not actual_cutout_collection:
                    actual_cutout_collection = (
                        active_model.get("cutout_collection") or
                        active_model.get("qdrant_collection")
                    )
                if not actual_product_collection:
                    actual_product_collection = (
                        active_model.get("product_collection") or
                        active_model.get("qdrant_collection")
                    )

            if actual_cutout_collection:
                # Get cutout embedding
                cutout_point = await qdrant_service.get_point(
                    collection_name=actual_cutout_collection,
                    point_id=cutout_id,
                    with_vector=True,
                )

                if cutout_point and cutout_point.get("vector"):
                    # Search in products collection
                    search_collection = actual_product_collection or actual_cutout_collection
                    use_separate = actual_product_collection != actual_cutout_collection

                    # Filter for products if using single collection
                    filter_conditions = None if use_separate else {"source": "product"}

                    similar_results = await qdrant_service.search(
                        collection_name=search_collection,
                        query_vector=cutout_point["vector"],
                        limit=limit * 3,  # Get more to handle multi-view dedup
                        score_threshold=min_similarity,
                        filter_conditions=filter_conditions,
                        with_payload=True,
                    )

                    # Deduplicate by product_id (multi-view support)
                    # Keep best score per product
                    product_best_scores: dict[str, tuple[float, dict]] = {}

                    for result in similar_results:
                        product_id = result["payload"].get("product_id")
                        if not product_id:
                            continue

                        score = result["score"]
                        if product_id not in product_best_scores or score > product_best_scores[product_id][0]:
                            product_best_scores[product_id] = (score, result["payload"])

                    # Build candidates from deduplicated results
                    for product_id, (score, payload) in product_best_scores.items():
                        similarity_match_ids.add(product_id)

                        # Skip if already in barcode matches
                        if product_id in barcode_match_ids:
                            # Update existing candidate
                            for c in candidates:
                                if c.id == product_id:
                                    c.similarity = max(c.similarity, score)
                            continue

                        # Get full product details
                        product = await db.get_product(product_id)
                        if product:
                            candidates.append(ProductMatchCandidate(
                                id=product_id,
                                barcode=product.get("barcode"),
                                brand_name=product.get("brand_name"),
                                product_name=product.get("product_name"),
                                primary_image_url=product.get("primary_image_url"),
                                similarity=score,
                                match_type="similarity",
                            ))

        except Exception as e:
            print(f"Qdrant reverse search failed: {e}")
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

    return CutoutProductsResponse(
        cutout=cutout,
        candidates=candidates[:limit],
        barcode_match_count=len(barcode_match_ids),
        similarity_match_count=len(similarity_match_ids),
        total_count=len(candidates),
        has_cutout_embedding=has_cutout_embedding,
    )


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
