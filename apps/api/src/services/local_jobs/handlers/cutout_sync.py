"""
Handler for background cutout sync operations.

Syncs cutout images from BuyBuddy API with progress tracking.
Supports both "sync_new" (newest first) and "backfill" (oldest first) modes.
"""

import asyncio
from typing import Callable

from services.supabase import supabase_service
from services.buybuddy import buybuddy_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import calculate_progress


@job_registry.register
class CutoutSyncJobHandler(BaseJobHandler):
    """
    Handler for background cutout sync from BuyBuddy.

    Config:
        mode: str - "sync_new" or "backfill"
        merchant_ids: list[int] - Required merchant IDs to filter
        max_items: int - Maximum items to sync (default 10000)
        page_size: int - Items per page (default 100)
        start_page: int - Start page for backfill (default 1)

    Result:
        synced_count: int - Number of cutouts synced
        skipped_count: int - Number already existing
        total_fetched: int - Total fetched from API
        highest_external_id: int - Highest ID synced
        lowest_external_id: int - Lowest ID synced
        last_page: int - Last page processed
        stopped_early: bool - Whether stopped before completion
    """

    job_type = "local_cutout_sync"
    BATCH_SIZE = 500  # DB insert batch size

    def validate_config(self, config: dict) -> str | None:
        if not config.get("merchant_ids"):
            return "merchant_ids is required"
        if not isinstance(config.get("merchant_ids"), list):
            return "merchant_ids must be a list"
        if len(config["merchant_ids"]) == 0:
            return "merchant_ids must contain at least one ID"
        mode = config.get("mode", "sync_new")
        if mode not in ("sync_new", "backfill"):
            return "mode must be 'sync_new' or 'backfill'"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        mode = config.get("mode", "sync_new")
        merchant_ids = config["merchant_ids"]
        max_items = config.get("max_items", 10000)
        page_size = config.get("page_size", 100)
        start_page = config.get("start_page", 1)

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing sync...",
            processed=0,
            total=0,
        ))

        if mode == "sync_new":
            return await self._sync_new(
                job_id, merchant_ids, max_items, page_size, update_progress
            )
        else:
            return await self._backfill(
                job_id, merchant_ids, max_items, page_size, start_page, update_progress
            )

    async def _sync_new(
        self,
        job_id: str,
        merchant_ids: list[int],
        max_items: int,
        page_size: int,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        """Sync new cutouts (newest first, stop when hitting existing)."""
        max_pages = max_items // page_size

        # Get current sync state
        update_progress(JobProgress(
            progress=2,
            current_step="Checking sync state...",
            processed=0,
            total=0,
        ))

        min_synced, max_synced = await self._get_synced_id_range()

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        page = 1
        errors = []

        # Phase 1: Fetch from BuyBuddy API
        while page <= max_pages:
            update_progress(JobProgress(
                progress=5 + calculate_progress(page, max_pages) * 0.4,
                current_step=f"Fetching page {page}...",
                processed=total_fetched,
                total=max_items,
            ))

            try:
                result = await buybuddy_service.get_cutout_images(
                    page=page,
                    page_size=page_size,
                    sort_field="id",
                    sort_order="desc",
                    merchant_ids=merchant_ids,
                )
            except Exception as e:
                errors.append(f"Page {page}: {str(e)}")
                break

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
                    all_cutouts.extend(new_cutouts)
                    stopped_early = True
                    break
                all_cutouts.extend(new_cutouts)
            else:
                all_cutouts.extend(cutouts)

            if not result["has_more"]:
                break

            page += 1
            await asyncio.sleep(0.1)

        if not all_cutouts:
            return {
                "synced_count": 0,
                "skipped_count": total_fetched,
                "total_fetched": total_fetched,
                "highest_external_id": highest_id,
                "lowest_external_id": lowest_id,
                "last_page": page,
                "stopped_early": stopped_early,
                "errors": errors,
                "message": "No new cutouts to sync",
            }

        # Phase 2: Insert into database
        update_progress(JobProgress(
            progress=50,
            current_step="Checking for duplicates...",
            processed=total_fetched,
            total=len(all_cutouts),
        ))

        # Check existing
        existing_ids = await self._get_existing_ids(
            [c["external_id"] for c in all_cutouts]
        )

        new_cutouts = [
            self._transform_cutout(c)
            for c in all_cutouts
            if c["external_id"] not in existing_ids
        ]

        synced_count = 0
        if new_cutouts:
            update_progress(JobProgress(
                progress=60,
                current_step=f"Inserting {len(new_cutouts)} cutouts...",
                processed=0,
                total=len(new_cutouts),
            ))

            synced_count = await self._insert_cutouts(
                new_cutouts, update_progress, 60, 95
            )

        return {
            "synced_count": synced_count,
            "skipped_count": total_fetched - synced_count,
            "total_fetched": total_fetched,
            "highest_external_id": highest_id,
            "lowest_external_id": lowest_id,
            "last_page": page,
            "stopped_early": stopped_early,
            "errors": errors[:10] if errors else [],
            "message": f"Synced {synced_count} new cutouts",
        }

    async def _backfill(
        self,
        job_id: str,
        merchant_ids: list[int],
        max_items: int,
        page_size: int,
        start_page: int,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        """Backfill old cutouts (oldest first)."""
        pages_to_fetch = max_items // page_size
        end_page = start_page + pages_to_fetch

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        page = start_page
        consecutive_all_exists = 0
        errors = []

        # Phase 1: Fetch from BuyBuddy API
        while page < end_page:
            update_progress(JobProgress(
                progress=5 + calculate_progress(page - start_page, pages_to_fetch) * 0.4,
                current_step=f"Fetching page {page}...",
                processed=total_fetched,
                total=max_items,
            ))

            try:
                result = await buybuddy_service.get_cutout_images(
                    page=page,
                    page_size=page_size,
                    sort_field="id",
                    sort_order="asc",
                    merchant_ids=merchant_ids,
                )
            except Exception as e:
                errors.append(f"Page {page}: {str(e)}")
                break

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

            # Check if batch already exists
            external_ids = [c["external_id"] for c in cutouts]
            existing_ids = await self._get_existing_ids(external_ids)

            new_in_batch = [c for c in cutouts if c["external_id"] not in existing_ids]

            if not new_in_batch:
                consecutive_all_exists += 1
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
            return {
                "synced_count": 0,
                "skipped_count": total_fetched,
                "total_fetched": total_fetched,
                "highest_external_id": highest_id,
                "lowest_external_id": lowest_id,
                "last_page": page,
                "stopped_early": stopped_early,
                "errors": errors,
                "message": f"No new cutouts found. Last page: {page}",
            }

        # Phase 2: Insert into database
        update_progress(JobProgress(
            progress=50,
            current_step=f"Inserting {len(all_cutouts)} cutouts...",
            processed=0,
            total=len(all_cutouts),
        ))

        new_cutouts = [self._transform_cutout(c) for c in all_cutouts]
        synced_count = await self._insert_cutouts(
            new_cutouts, update_progress, 50, 95
        )

        return {
            "synced_count": synced_count,
            "skipped_count": total_fetched - synced_count,
            "total_fetched": total_fetched,
            "highest_external_id": highest_id,
            "lowest_external_id": lowest_id,
            "last_page": page,
            "stopped_early": stopped_early,
            "errors": errors[:10] if errors else [],
            "message": f"Backfilled {synced_count} cutouts. Last page: {page}",
        }

    def _transform_cutout(self, cutout: dict) -> dict:
        """Transform BuyBuddy cutout to database format."""
        return {
            "external_id": cutout["external_id"],
            "image_url": cutout["image_url"],
            "predicted_upc": cutout.get("predicted_upc"),
            "merchant": cutout.get("merchant"),
            "row_index": cutout.get("row_index"),
            "column_index": cutout.get("column_index"),
            "annotated_upc": cutout.get("annotated_upc"),
        }

    async def _get_synced_id_range(self) -> tuple[int | None, int | None]:
        """Get min and max synced external IDs."""
        # Query directly (simple and reliable)
        min_result = supabase_service.client.table("cutout_images") \
            .select("external_id") \
            .order("external_id", desc=False) \
            .limit(1) \
            .execute()

        max_result = supabase_service.client.table("cutout_images") \
            .select("external_id") \
            .order("external_id", desc=True) \
            .limit(1) \
            .execute()

        min_id = min_result.data[0]["external_id"] if min_result.data else None
        max_id = max_result.data[0]["external_id"] if max_result.data else None

        return min_id, max_id

    async def _get_existing_ids(self, external_ids: list[int]) -> set[int]:
        """Check which external IDs already exist in database."""
        existing_ids = set()

        for i in range(0, len(external_ids), self.BATCH_SIZE):
            batch = external_ids[i:i + self.BATCH_SIZE]
            result = supabase_service.client.table("cutout_images") \
                .select("external_id") \
                .in_("external_id", batch) \
                .execute()
            existing_ids.update(item["external_id"] for item in result.data)

        return existing_ids

    async def _insert_cutouts(
        self,
        cutouts: list[dict],
        update_progress: Callable[[JobProgress], None],
        progress_start: int,
        progress_end: int,
    ) -> int:
        """Insert cutouts in batches with progress updates."""
        total = len(cutouts)
        inserted = 0
        progress_range = progress_end - progress_start

        for i in range(0, total, self.BATCH_SIZE):
            batch = cutouts[i:i + self.BATCH_SIZE]

            try:
                result = supabase_service.client.table("cutout_images").upsert(
                    batch,
                    on_conflict="external_id",
                    ignore_duplicates=True
                ).execute()
                inserted += len(result.data)
            except Exception as e:
                print(f"[CutoutSync] Batch insert error: {e}")

            processed = min(i + len(batch), total)
            progress = progress_start + calculate_progress(processed, total) * progress_range / 100

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Inserting cutouts... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        return inserted
