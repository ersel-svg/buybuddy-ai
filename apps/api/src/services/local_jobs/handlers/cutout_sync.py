"""
Handler for background cutout sync operations.

SOTA Features:
- Concurrent page fetching with semaphore control
- RPC batch insert for optimal database performance
- Streaming progress updates
- Graceful error handling with partial results
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
    SOTA Handler for background cutout sync from BuyBuddy.

    Config:
        mode: str - "sync_new" or "backfill"
        merchant_ids: list[int] - Required merchant IDs to filter
        max_items: int - Maximum items to sync (default 10000)
        page_size: int - Items per page (default 100)
        start_page: int - Start page for backfill (default 1)
        concurrency: int - Concurrent page fetches (default 5)

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
    DB_BATCH_SIZE = 500  # DB insert batch size
    DEFAULT_CONCURRENCY = 5  # Concurrent API requests

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
        concurrency = config.get("concurrency", self.DEFAULT_CONCURRENCY)
        inserted_at = config.get("inserted_at")
        updated_at = config.get("updated_at")

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing sync...",
            processed=0,
            total=0,
        ))

        if mode == "sync_new":
            return await self._sync_new(
                job_id, merchant_ids, max_items, page_size, concurrency,
                update_progress, inserted_at=inserted_at, updated_at=updated_at
            )
        else:
            return await self._backfill(
                job_id, merchant_ids, max_items, page_size, start_page,
                concurrency, update_progress, inserted_at=inserted_at, updated_at=updated_at
            )

    async def _fetch_page(
        self,
        page: int,
        page_size: int,
        sort_order: str,
        merchant_ids: list[int],
        semaphore: asyncio.Semaphore,
        inserted_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict:
        """Fetch a single page with semaphore control."""
        async with semaphore:
            try:
                result = await buybuddy_service.get_cutout_images(
                    page=page,
                    page_size=page_size,
                    sort_field="id",
                    sort_order=sort_order,
                    merchant_ids=merchant_ids,
                    inserted_at=inserted_at,
                    updated_at=updated_at,
                )
                return {"page": page, "success": True, "data": result}
            except Exception as e:
                return {"page": page, "success": False, "error": str(e)}

    async def _sync_new(
        self,
        job_id: str,
        merchant_ids: list[int],
        max_items: int,
        page_size: int,
        concurrency: int,
        update_progress: Callable[[JobProgress], None],
        inserted_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict:
        """Sync new cutouts with concurrent fetching (newest first)."""
        max_pages = max_items // page_size

        # Get current sync state
        update_progress(JobProgress(
            progress=2,
            current_step="Checking sync state...",
            processed=0,
            total=0,
        ))

        _, max_synced = await self._get_synced_id_range()

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        errors = []
        last_page = 0

        # Semaphore for concurrent requests
        semaphore = asyncio.Semaphore(concurrency)

        # Fetch pages in batches for better control
        page = 1
        while page <= max_pages:
            # Determine batch of pages to fetch concurrently
            batch_end = min(page + concurrency, max_pages + 1)
            pages_to_fetch = list(range(page, batch_end))

            update_progress(JobProgress(
                progress=5 + calculate_progress(page, max_pages) * 0.4,
                current_step=f"Fetching pages {page}-{batch_end - 1}...",
                processed=total_fetched,
                total=max_items,
            ))

            # Concurrent fetch
            tasks = [
                self._fetch_page(p, page_size, "desc", merchant_ids, semaphore,
                                 inserted_at, updated_at)
                for p in pages_to_fetch
            ]
            results = await asyncio.gather(*tasks)

            # Process results in order
            should_stop = False
            for result in sorted(results, key=lambda r: r["page"]):
                if not result["success"]:
                    errors.append(f"Page {result['page']}: {result['error']}")
                    continue

                cutouts = result["data"]["items"]
                last_page = result["page"]

                if not cutouts:
                    should_stop = True
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
                        should_stop = True
                        break
                    all_cutouts.extend(new_cutouts)
                else:
                    all_cutouts.extend(cutouts)

                if not result["data"]["has_more"]:
                    should_stop = True
                    break

            if should_stop:
                break

            page = batch_end
            await asyncio.sleep(0.05)  # Small delay between batches

        if not all_cutouts:
            return {
                "synced_count": 0,
                "skipped_count": total_fetched,
                "total_fetched": total_fetched,
                "highest_external_id": highest_id,
                "lowest_external_id": lowest_id,
                "last_page": last_page,
                "stopped_early": stopped_early,
                "errors": errors[:10],
                "message": "No new cutouts to sync",
            }

        # Phase 2: Insert into database
        update_progress(JobProgress(
            progress=50,
            current_step="Checking for duplicates...",
            processed=total_fetched,
            total=len(all_cutouts),
        ))

        # Check existing in parallel batches
        existing_ids = await self._get_existing_ids_parallel(
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

            synced_count = await self._insert_cutouts_batch(
                new_cutouts, update_progress, 60, 95
            )

        return {
            "synced_count": synced_count,
            "skipped_count": total_fetched - synced_count,
            "total_fetched": total_fetched,
            "highest_external_id": highest_id,
            "lowest_external_id": lowest_id,
            "last_page": last_page,
            "stopped_early": stopped_early,
            "errors": errors[:10] if errors else [],
            "message": f"Synced {synced_count:,} new cutouts",
        }

    async def _backfill(
        self,
        job_id: str,
        merchant_ids: list[int],
        max_items: int,
        page_size: int,
        start_page: int,
        concurrency: int,
        update_progress: Callable[[JobProgress], None],
        inserted_at: str | None = None,
        updated_at: str | None = None,
    ) -> dict:
        """Backfill old cutouts with concurrent fetching (oldest first)."""
        pages_to_fetch = max_items // page_size
        end_page = start_page + pages_to_fetch

        all_cutouts = []
        total_fetched = 0
        highest_id = None
        lowest_id = None
        stopped_early = False
        errors = []
        consecutive_all_exists = 0
        last_page = start_page

        # Semaphore for concurrent requests
        semaphore = asyncio.Semaphore(concurrency)

        # Fetch pages sequentially for backfill (need to track consecutive exists)
        page = start_page
        while page < end_page:
            update_progress(JobProgress(
                progress=5 + calculate_progress(page - start_page, pages_to_fetch) * 0.4,
                current_step=f"Fetching page {page}...",
                processed=total_fetched,
                total=max_items,
            ))

            result = await self._fetch_page(
                page, page_size, "asc", merchant_ids, semaphore,
                inserted_at, updated_at
            )

            if not result["success"]:
                errors.append(f"Page {page}: {result['error']}")
                break

            cutouts = result["data"]["items"]
            last_page = page

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
            existing_ids = await self._get_existing_ids_parallel(external_ids)

            new_in_batch = [c for c in cutouts if c["external_id"] not in existing_ids]

            if not new_in_batch:
                consecutive_all_exists += 1
                if consecutive_all_exists >= 5:
                    stopped_early = True
                    break
            else:
                consecutive_all_exists = 0
                all_cutouts.extend(new_in_batch)

            if not result["data"]["has_more"]:
                break

            page += 1
            await asyncio.sleep(0.05)

        if not all_cutouts:
            return {
                "synced_count": 0,
                "skipped_count": total_fetched,
                "total_fetched": total_fetched,
                "highest_external_id": highest_id,
                "lowest_external_id": lowest_id,
                "last_page": last_page,
                "stopped_early": stopped_early,
                "errors": errors[:10],
                "message": f"No new cutouts found. Last page: {last_page}",
            }

        # Phase 2: Insert into database
        update_progress(JobProgress(
            progress=50,
            current_step=f"Inserting {len(all_cutouts)} cutouts...",
            processed=0,
            total=len(all_cutouts),
        ))

        new_cutouts = [self._transform_cutout(c) for c in all_cutouts]
        synced_count = await self._insert_cutouts_batch(
            new_cutouts, update_progress, 50, 95
        )

        return {
            "synced_count": synced_count,
            "skipped_count": total_fetched - synced_count,
            "total_fetched": total_fetched,
            "highest_external_id": highest_id,
            "lowest_external_id": lowest_id,
            "last_page": last_page,
            "stopped_early": stopped_early,
            "errors": errors[:10] if errors else [],
            "message": f"Backfilled {synced_count:,} cutouts. Last page: {last_page}",
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

    async def _get_existing_ids_parallel(self, external_ids: list[int]) -> set[int]:
        """Check existing IDs with parallel batch queries."""
        if not external_ids:
            return set()

        existing_ids = set()
        batch_size = self.DB_BATCH_SIZE

        # Split into batches and query concurrently
        batches = [
            external_ids[i:i + batch_size]
            for i in range(0, len(external_ids), batch_size)
        ]

        async def check_batch(batch: list[int]) -> set[int]:
            result = supabase_service.client.table("cutout_images") \
                .select("external_id") \
                .in_("external_id", batch) \
                .execute()
            return {item["external_id"] for item in result.data}

        # Run batches concurrently (limit to 5 concurrent)
        semaphore = asyncio.Semaphore(5)

        async def check_with_semaphore(batch: list[int]) -> set[int]:
            async with semaphore:
                return await check_batch(batch)

        results = await asyncio.gather(*[check_with_semaphore(b) for b in batches])
        for result_set in results:
            existing_ids.update(result_set)

        return existing_ids

    async def _insert_cutouts_batch(
        self,
        cutouts: list[dict],
        update_progress: Callable[[JobProgress], None],
        progress_start: int,
        progress_end: int,
    ) -> int:
        """Insert cutouts using optimized batch upsert."""
        total = len(cutouts)
        if total == 0:
            return 0

        inserted = 0
        progress_range = progress_end - progress_start

        # Use larger batches for better throughput
        batch_size = self.DB_BATCH_SIZE

        for i in range(0, total, batch_size):
            batch = cutouts[i:i + batch_size]

            try:
                result = supabase_service.client.table("cutout_images").upsert(
                    batch,
                    on_conflict="external_id",
                    ignore_duplicates=True
                ).execute()
                inserted += len(result.data)
            except Exception as e:
                print(f"[CutoutSync] Batch insert error: {e}")
                # Try smaller batches on failure
                for item in batch:
                    try:
                        supabase_service.client.table("cutout_images").upsert(
                            item,
                            on_conflict="external_id",
                            ignore_duplicates=True
                        ).execute()
                        inserted += 1
                    except Exception as inner_e:
                        print(f"[CutoutSync] Single insert error: {inner_e}")

            processed = min(i + len(batch), total)
            progress = progress_start + calculate_progress(processed, total) * progress_range / 100

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Inserting cutouts... ({processed:,}/{total:,})",
                processed=processed,
                total=total,
            ))

        return inserted
