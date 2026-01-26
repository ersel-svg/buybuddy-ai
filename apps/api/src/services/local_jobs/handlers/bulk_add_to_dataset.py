"""
Handler for bulk adding images to a dataset.

This handler processes the operation in batches with progress tracking,
handling Supabase's row limits and providing real-time progress updates.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkAddToDatasetHandler(BaseJobHandler):
    """
    Handler for bulk adding images to an OD dataset.

    Config:
        dataset_id: str - Target dataset ID
        product_ids: list[str] - Optional, specific product IDs to add
        filters: dict - Optional, filter criteria for images
            - statuses: str (comma-separated)
            - sources: str (comma-separated)
            - folders: str (comma-separated)
            - search: str
            - merchant_ids: str (comma-separated)
            - store_ids: str (comma-separated)

    Result:
        added: int - Number of images added
        skipped: int - Number of images already in dataset
        total: int - Total matching images
        errors: list[str] - Any errors encountered
    """

    job_type = "local_bulk_add_to_dataset"
    BATCH_SIZE = 100
    PAGE_SIZE = 1000  # Supabase pagination

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        dataset_id = config["dataset_id"]
        product_ids = config.get("product_ids")
        filters = config.get("filters", {})

        # Update initial progress
        update_progress(JobProgress(
            progress=0,
            current_step="Fetching matching images...",
            processed=0,
            total=0,
        ))

        # Verify dataset exists
        dataset = supabase_service.client.table("od_datasets")\
            .select("id")\
            .eq("id", dataset_id)\
            .execute()

        if not dataset.data or len(dataset.data) == 0:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get all matching image IDs (with pagination)
        update_progress(JobProgress(
            progress=5,
            current_step="Collecting image IDs...",
            processed=0,
            total=0,
        ))

        # Get image IDs either from product_ids or filters
        if product_ids:
            image_ids = product_ids
        else:
            image_ids = self._get_all_image_ids(filters)

        total = len(image_ids)

        if total == 0:
            return {
                "added": 0,
                "skipped": 0,
                "total": 0,
                "message": "No images match the filters",
            }

        # Get existing images in dataset
        update_progress(JobProgress(
            progress=10,
            current_step="Checking existing images...",
            processed=0,
            total=total,
        ))

        existing_ids = self._get_existing_ids(dataset_id)

        # Process in batches
        added = 0
        skipped = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            # Filter out existing
            new_links = []
            for img_id in batch:
                if img_id in existing_ids:
                    skipped += 1
                else:
                    new_links.append({
                        "dataset_id": dataset_id,
                        "image_id": img_id,
                        "status": "pending",
                    })

            # Insert batch
            if new_links:
                try:
                    supabase_service.client.table("od_dataset_images")\
                        .insert(new_links)\
                        .execute()
                    added += len(new_links)
                    # Add to existing to prevent duplicates in subsequent batches
                    existing_ids.update(link["image_id"] for link in new_links)
                except Exception as e:
                    error_msg = f"Batch {batch_num + 1}: {str(e)}"
                    errors.append(error_msg)
                    print(f"[BulkAddToDataset] {error_msg}")

            # Update progress (10-95% range for processing)
            processed = (batch_num + 1) * self.BATCH_SIZE
            processed = min(processed, total)
            progress = 10 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Adding images... ({processed}/{total})",
                processed=processed,
                total=total,
                errors=errors[-5:] if errors else [],
            ))

        # Update dataset counts
        update_progress(JobProgress(
            progress=95,
            current_step="Updating dataset counts...",
            processed=total,
            total=total,
        ))

        self._update_dataset_count(dataset_id)

        return {
            "added": added,
            "skipped": skipped,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Added {added} images, skipped {skipped} (already in dataset)",
        }

    def _get_all_image_ids(self, filters: dict) -> list[str]:
        """Get all image IDs matching filters with pagination."""
        all_ids = []
        offset = 0

        while True:
            query = supabase_service.client.table("od_images").select("id")

            # Apply filters
            if filters.get("statuses"):
                status_list = [s.strip() for s in filters["statuses"].split(",") if s.strip()]
                if status_list:
                    query = query.in_("status", status_list)

            if filters.get("sources"):
                source_list = [s.strip() for s in filters["sources"].split(",") if s.strip()]
                if source_list:
                    query = query.in_("source", source_list)

            if filters.get("folders"):
                folder_list = [f.strip() for f in filters["folders"].split(",") if f.strip()]
                if folder_list:
                    query = query.in_("folder", folder_list)

            if filters.get("search"):
                query = query.ilike("filename", f"%{filters['search']}%")

            if filters.get("merchant_ids"):
                merchant_list = [
                    int(m.strip())
                    for m in filters["merchant_ids"].split(",")
                    if m.strip().isdigit()
                ]
                if merchant_list:
                    query = query.in_("merchant_id", merchant_list)

            if filters.get("store_ids"):
                store_list = [
                    int(s.strip())
                    for s in filters["store_ids"].split(",")
                    if s.strip().isdigit()
                ]
                if store_list:
                    query = query.in_("store_id", store_list)

            result = query.range(offset, offset + self.PAGE_SIZE - 1).execute()

            if not result.data:
                break

            all_ids.extend(img["id"] for img in result.data)

            if len(result.data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        return all_ids

    def _get_existing_ids(self, dataset_id: str) -> set[str]:
        """Get existing image IDs in dataset (with pagination)."""
        existing_ids = set()
        offset = 0

        while True:
            result = supabase_service.client.table("od_dataset_images")\
                .select("image_id")\
                .eq("dataset_id", dataset_id)\
                .range(offset, offset + self.PAGE_SIZE - 1)\
                .execute()

            if not result.data:
                break

            existing_ids.update(r["image_id"] for r in result.data)

            if len(result.data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        return existing_ids

    def _update_dataset_count(self, dataset_id: str) -> None:
        """Update dataset image count."""
        count_result = supabase_service.client.table("od_dataset_images")\
            .select("id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .execute()

        supabase_service.client.table("od_datasets").update({
            "image_count": count_result.count or 0,
        }).eq("id", dataset_id).execute()
