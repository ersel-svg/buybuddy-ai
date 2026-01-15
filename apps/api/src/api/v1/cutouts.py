"""Cutout Images API router for embedding matching system."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from services.supabase import SupabaseService, supabase_service
from services.buybuddy import buybuddy_service
from auth.dependencies import get_current_user
from auth.service import UserInfo


# Router with authentication required
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================


class CutoutMatchRequest(BaseModel):
    """Request to match cutout to product."""
    product_id: str
    similarity: Optional[float] = None


class SyncRequest(BaseModel):
    """Request to sync cutouts from BuyBuddy."""
    max_pages: Optional[int] = None
    page_size: int = 100
    sort_order: str = "desc"  # "desc" for newest first, "asc" for oldest first


class SyncNewRequest(BaseModel):
    """Request to sync new cutouts (newest first until existing found)."""
    max_items: int = 10000  # Safety limit
    page_size: int = 100


class BackfillRequest(BaseModel):
    """Request to backfill old cutouts."""
    max_items: int = 10000  # Items per backfill batch
    page_size: int = 100
    start_page: int = 1  # Start from this page number (useful for filling gaps)


class SyncResponse(BaseModel):
    """Response from sync operation."""
    synced_count: int
    skipped_count: int
    total_fetched: int
    highest_external_id: Optional[int] = None
    last_page: Optional[int] = None  # Last page processed (for backfill continuation)
    lowest_external_id: Optional[int] = None
    stopped_early: bool = False


class SyncStateResponse(BaseModel):
    """Current sync state."""
    min_synced_external_id: Optional[int] = None
    max_synced_external_id: Optional[int] = None
    total_synced: int = 0
    backfill_completed: bool = False
    last_sync_new_at: Optional[datetime] = None
    last_backfill_at: Optional[datetime] = None
    last_backfill_page: int = 1  # Last page processed in backfill
    # Computed fields
    buybuddy_max_id: Optional[int] = None
    estimated_remaining: Optional[int] = None


class CutoutStatsResponse(BaseModel):
    """Cutout statistics response."""
    total: int
    with_embedding: int
    without_embedding: int
    matched: int
    unmatched: int


# ===========================================
# Dependency
# ===========================================


def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


# ===========================================
# Endpoints
# ===========================================


@router.get("")
async def list_cutouts(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    has_embedding: Optional[bool] = None,
    is_matched: Optional[bool] = None,
    predicted_upc: Optional[str] = None,
    db: SupabaseService = Depends(get_supabase),
):
    """
    List cutout images with pagination and filters.

    Filters:
    - has_embedding: Filter by whether cutout has embedding vector
    - is_matched: Filter by whether cutout has been matched to a product
    - predicted_upc: Filter by predicted UPC code
    """
    result = await db.get_cutouts(
        page=page,
        limit=limit,
        has_embedding=has_embedding,
        is_matched=is_matched,
        predicted_upc=predicted_upc,
    )
    return result


@router.get("/stats")
async def get_cutout_stats(
    db: SupabaseService = Depends(get_supabase),
) -> CutoutStatsResponse:
    """Get cutout statistics."""
    stats = await db.get_cutout_stats()
    return CutoutStatsResponse(**stats)


@router.get("/sync/state")
async def get_sync_state(
    db: SupabaseService = Depends(get_supabase),
) -> SyncStateResponse:
    """
    Get current cutout sync state.

    Returns min/max synced external IDs and progress information.
    """
    state = await db.get_cutout_sync_state()

    if not state:
        return SyncStateResponse()

    # Get BuyBuddy max ID for progress calculation
    buybuddy_max_id = None
    estimated_remaining = None

    try:
        if buybuddy_service.is_configured():
            result = await buybuddy_service.get_cutout_images(
                page=1, page_size=1, sort_field="id", sort_order="desc"
            )
            if result["items"]:
                buybuddy_max_id = result["items"][0]["external_id"]

                # Estimate remaining
                min_synced = state.get("min_synced_external_id")
                if min_synced and buybuddy_max_id:
                    # This is rough - assumes sequential IDs
                    estimated_remaining = min_synced - 1  # IDs below min
    except Exception:
        pass  # Ignore errors getting BuyBuddy max

    return SyncStateResponse(
        min_synced_external_id=state.get("min_synced_external_id"),
        max_synced_external_id=state.get("max_synced_external_id"),
        total_synced=state.get("total_synced", 0),
        backfill_completed=state.get("backfill_completed", False),
        last_sync_new_at=state.get("last_sync_new_at"),
        last_backfill_at=state.get("last_backfill_at"),
        last_backfill_page=state.get("last_backfill_page", 1),
        buybuddy_max_id=buybuddy_max_id,
        estimated_remaining=estimated_remaining,
    )


@router.post("/sync/new")
async def sync_new_cutouts(
    request: SyncNewRequest = SyncNewRequest(),
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> SyncResponse:
    """
    Sync NEW cutouts from BuyBuddy (newest first).

    Fetches from newest to oldest and stops when it hits already-synced cutouts.
    Use this for regular updates to get new cutouts.
    """
    if not buybuddy_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured"
        )

    try:
        import asyncio
        max_pages = request.max_items // request.page_size

        # Get current sync state ONCE at the beginning
        min_synced, max_synced = await db.get_synced_external_id_range()

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        page = 1

        # Phase 1: Fetch all pages from BuyBuddy
        while page <= max_pages:
            result = await buybuddy_service.get_cutout_images(
                page=page,
                page_size=request.page_size,
                sort_field="id",
                sort_order="desc",
            )

            cutouts = result["items"]
            if not cutouts:
                break

            total_fetched += len(cutouts)

            # Track IDs
            batch_ids = [c["external_id"] for c in cutouts if c.get("external_id")]
            if batch_ids:
                if highest_id is None:
                    highest_id = max(batch_ids)
                lowest_id = min(batch_ids)

            # Filter: only keep cutouts newer than max_synced
            if max_synced is not None:
                new_cutouts = [c for c in cutouts if c["external_id"] > max_synced]
                if len(new_cutouts) < len(cutouts):
                    # Some cutouts were already synced, we've reached the boundary
                    all_cutouts.extend(new_cutouts)
                    stopped_early = True
                    break
                all_cutouts.extend(new_cutouts)
            else:
                # First sync - add all
                all_cutouts.extend(cutouts)

            if not result["has_more"]:
                break

            page += 1
            await asyncio.sleep(0.1)

        if not all_cutouts:
            return SyncResponse(
                synced_count=0,
                skipped_count=total_fetched,
                total_fetched=total_fetched,
                highest_external_id=highest_id,
                lowest_external_id=lowest_id,
                stopped_early=stopped_early,
            )

        # Phase 2: Check for duplicates in DB and insert new ones
        # Check in batches to avoid query size limits
        BATCH_SIZE = 500
        existing_ids = set()
        external_ids = [c["external_id"] for c in all_cutouts]

        for i in range(0, len(external_ids), BATCH_SIZE):
            batch_ids = external_ids[i:i + BATCH_SIZE]
            existing = (
                db.client.table("cutout_images")
                .select("external_id")
                .in_("external_id", batch_ids)
                .execute()
            )
            existing_ids.update(item["external_id"] for item in existing.data)

        new_cutouts = [
            {
                "external_id": c["external_id"],
                "image_url": c["image_url"],
                "predicted_upc": c.get("predicted_upc"),
            }
            for c in all_cutouts
            if c["external_id"] not in existing_ids
        ]

        if new_cutouts:
            # Upsert in batches to avoid Supabase limits
            # Using upsert to handle race conditions and duplicate key errors gracefully
            inserted_count = 0
            for i in range(0, len(new_cutouts), BATCH_SIZE):
                batch = new_cutouts[i:i + BATCH_SIZE]
                result = db.client.table("cutout_images").upsert(
                    batch,
                    on_conflict="external_id",
                    ignore_duplicates=True
                ).execute()
                inserted_count += len(result.data)

            # Update sync state with actually inserted count
            new_ids = [c["external_id"] for c in new_cutouts]
            await db.update_sync_state_for_new(max(new_ids), inserted_count)

        synced_count = len(new_cutouts)
        skipped_count = total_fetched - synced_count

        return SyncResponse(
            synced_count=synced_count,
            skipped_count=skipped_count,
            total_fetched=total_fetched,
            highest_external_id=highest_id,
            lowest_external_id=lowest_id,
            stopped_early=stopped_early,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync new cutouts: {str(e)}"
        )


@router.post("/sync/backfill")
async def backfill_cutouts(
    request: BackfillRequest = BackfillRequest(),
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> SyncResponse:
    """
    Backfill cutouts from BuyBuddy (oldest first).

    Fetches from oldest to newest and syncs any cutouts not already in DB.
    Use this to fill in gaps and historical cutouts.
    """
    if not buybuddy_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured"
        )

    try:
        import asyncio
        pages_to_fetch = request.max_items // request.page_size
        start_page = request.start_page

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        page = start_page
        end_page = start_page + pages_to_fetch
        consecutive_all_exists = 0

        # Phase 1: Fetch pages from BuyBuddy (oldest first)
        while page < end_page:
            result = await buybuddy_service.get_cutout_images(
                page=page,
                page_size=request.page_size,
                sort_field="id",
                sort_order="asc",
            )

            cutouts = result["items"]
            if not cutouts:
                break

            total_fetched += len(cutouts)

            # Track IDs
            batch_ids = [c["external_id"] for c in cutouts if c.get("external_id")]
            if batch_ids:
                if lowest_id is None:
                    lowest_id = min(batch_ids)
                highest_id = max(batch_ids)

            # Check if this batch already exists in DB
            external_ids = [c["external_id"] for c in cutouts]
            existing = (
                db.client.table("cutout_images")
                .select("external_id")
                .in_("external_id", external_ids)
                .execute()
            )
            existing_ids = {item["external_id"] for item in existing.data}

            new_in_batch = [c for c in cutouts if c["external_id"] not in existing_ids]

            if not new_in_batch:
                consecutive_all_exists += 1
                # Stop if we've seen 5 consecutive pages all existing (increased from 3)
                if consecutive_all_exists >= 5:
                    stopped_early = True
                    break
            else:
                consecutive_all_exists = 0
                all_cutouts.extend(new_in_batch)

            if not result["has_more"]:
                break

            page += 1
            await asyncio.sleep(0.1)

        if not all_cutouts:
            # Still update last_backfill_page even if no new cutouts
            await db.update_last_backfill_page(page)
            return SyncResponse(
                synced_count=0,
                skipped_count=total_fetched,
                total_fetched=total_fetched,
                highest_external_id=highest_id,
                lowest_external_id=lowest_id,
                stopped_early=stopped_early,
                last_page=page,
            )

        # Phase 2: Insert new cutouts
        new_cutouts = [
            {
                "external_id": c["external_id"],
                "image_url": c["image_url"],
                "predicted_upc": c.get("predicted_upc"),
            }
            for c in all_cutouts
        ]

        if new_cutouts:
            # Upsert in batches of 500 to avoid Supabase limits
            # Using upsert to handle race conditions gracefully
            BATCH_SIZE = 500
            inserted_count = 0
            for i in range(0, len(new_cutouts), BATCH_SIZE):
                batch = new_cutouts[i:i + BATCH_SIZE]
                result = db.client.table("cutout_images").upsert(
                    batch,
                    on_conflict="external_id",
                    ignore_duplicates=True
                ).execute()
                inserted_count += len(result.data)

            # Update sync state with actually inserted count and last page
            new_ids = [c["external_id"] for c in new_cutouts]
            await db.update_sync_state_for_backfill(min(new_ids), inserted_count, last_page=page)

        synced_count = len(new_cutouts)
        skipped_count = total_fetched - synced_count

        return SyncResponse(
            synced_count=synced_count,
            skipped_count=skipped_count,
            total_fetched=total_fetched,
            highest_external_id=highest_id,
            lowest_external_id=lowest_id,
            stopped_early=stopped_early,
            last_page=page,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to backfill cutouts: {str(e)}"
        )


@router.post("/sync")
async def sync_cutouts(
    request: SyncRequest = SyncRequest(),
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
) -> SyncResponse:
    """
    Legacy sync endpoint - syncs cutouts from BuyBuddy.

    For new implementations, prefer /sync/new (for updates) or /sync/backfill (for historical data).
    """
    if not buybuddy_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="BuyBuddy API credentials not configured"
        )

    # Validate sort_order
    if request.sort_order not in ("asc", "desc"):
        raise HTTPException(
            status_code=400,
            detail="sort_order must be 'asc' or 'desc'"
        )

    try:
        # Fetch from BuyBuddy API with sort order
        cutouts = await buybuddy_service.get_all_cutout_images(
            max_pages=request.max_pages,
            page_size=request.page_size,
            sort_order=request.sort_order,
        )

        if not cutouts:
            return SyncResponse(synced_count=0, skipped_count=0, total_fetched=0)

        # Calculate watermarks from fetched data
        external_ids = [c["external_id"] for c in cutouts if c.get("external_id")]
        highest_id = max(external_ids) if external_ids else None
        lowest_id = min(external_ids) if external_ids else None

        # Sync to database (using new incremental method)
        mode = "new" if request.sort_order == "desc" else "backfill"
        result = await db.sync_cutouts_incremental(cutouts, mode=mode)

        return SyncResponse(
            synced_count=result["synced_count"],
            skipped_count=result["skipped_count"],
            total_fetched=len(cutouts),
            highest_external_id=highest_id,
            lowest_external_id=lowest_id,
            stopped_early=result.get("stopped_early", False),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync cutouts: {str(e)}"
        )


@router.get("/{cutout_id}")
async def get_cutout(
    cutout_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get single cutout by ID."""
    cutout = await db.get_cutout(cutout_id)
    if not cutout:
        raise HTTPException(status_code=404, detail="Cutout not found")
    return cutout


@router.post("/{cutout_id}/match")
async def match_cutout(
    cutout_id: str,
    request: CutoutMatchRequest,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Match a cutout image to a product.

    This associates the cutout with a product_id and optionally records
    the similarity score. The cutout image will be added as a real image
    to the product.
    """
    # Verify cutout exists
    cutout = await db.get_cutout(cutout_id)
    if not cutout:
        raise HTTPException(status_code=404, detail="Cutout not found")

    # Verify product exists
    product = await db.get_product(request.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Update cutout with match
    updated = await db.match_cutout_to_product(
        cutout_id=cutout_id,
        product_id=request.product_id,
        similarity=request.similarity,
        matched_by=current_user.username if current_user else None,
    )

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update cutout")

    # Add cutout image as real image to product
    await db.add_real_images(request.product_id, [cutout["image_url"]])

    return updated


@router.post("/{cutout_id}/unmatch")
async def unmatch_cutout(
    cutout_id: str,
    current_user: UserInfo = Depends(get_current_user),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Remove match from a cutout image.

    Clears the product association from the cutout. Note: This does NOT
    remove the image from the product's real images automatically.
    """
    # Verify cutout exists
    cutout = await db.get_cutout(cutout_id)
    if not cutout:
        raise HTTPException(status_code=404, detail="Cutout not found")

    # Clear match data
    updated = await db.update_cutout(cutout_id, {
        "matched_product_id": None,
        "match_similarity": None,
        "matched_by": None,
        "matched_at": None,
    })

    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update cutout")

    return updated


@router.delete("/{cutout_id}")
async def delete_cutout(
    cutout_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a cutout image."""
    cutout = await db.get_cutout(cutout_id)
    if not cutout:
        raise HTTPException(status_code=404, detail="Cutout not found")

    # Delete from database
    from supabase import PostgrestAPIError
    try:
        db.client.table("cutout_images").delete().eq("id", cutout_id).execute()
        return {"status": "deleted"}
    except PostgrestAPIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cutout: {str(e)}")
