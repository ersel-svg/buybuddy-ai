"""
Handler for bulk deleting images from the system.

This handler processes image deletions in batches with progress tracking,
handling storage deletion and database cleanup.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkDeleteImagesHandler(BaseJobHandler):
    """
    Handler for bulk deleting images from the OD system.

    Config:
        image_ids: list[str] (optional) - Specific image IDs to delete
        filters: dict (optional) - Filter criteria to select images
            - statuses: str (comma-separated)
            - sources: str (comma-separated)
            - folders: str (comma-separated)
            - search: str
            - merchant_ids: str (comma-separated)
            - store_ids: str (comma-separated)
        skip_in_datasets: bool - Skip images that are in any dataset (default: True)

    Result:
        deleted: int - Number of images deleted
        skipped: int - Number of images skipped (in datasets)
        total: int - Total images matched
        errors: list[str] - Any errors encountered
    """

    job_type = "local_bulk_delete_images"
    BATCH_SIZE = 100
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("image_ids") and not config.get("filters"):
            return "Either image_ids or filters is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        image_ids = config.get("image_ids", [])
        filters = config.get("filters", {})
        skip_in_datasets = config.get("skip_in_datasets", True)

        # Update initial progress
        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=0,
        ))

        # Get target image IDs
        if image_ids:
            target_ids = image_ids
        else:
            update_progress(JobProgress(
                progress=5,
                current_step="Collecting image IDs...",
                processed=0,
                total=0,
            ))
            target_ids = self._get_filtered_image_ids(filters)

        total = len(target_ids)

        if total == 0:
            return {
                "deleted": 0,
                "skipped": 0,
                "total": 0,
                "message": "No images to delete",
            }

        # Get image filenames for storage deletion
        update_progress(JobProgress(
            progress=10,
            current_step="Fetching image data...",
            processed=0,
            total=total,
        ))

        images_data = self._get_image_data(target_ids)

        # Check which images are in datasets
        if skip_in_datasets:
            update_progress(JobProgress(
                progress=15,
                current_step="Checking dataset memberships...",
                processed=0,
                total=total,
            ))
            images_in_datasets = self._get_images_in_datasets(target_ids)
        else:
            images_in_datasets = set()

        # Process deletions
        deleted = 0
        skipped = 0
        errors = []

        deletable_ids = []
        for img_id in target_ids:
            if img_id in images_in_datasets:
                skipped += 1
            elif img_id in images_data:
                deletable_ids.append(img_id)
            else:
                errors.append(f"{img_id}: Not found")

        if not deletable_ids:
            return {
                "deleted": 0,
                "skipped": skipped,
                "total": total,
                "errors": errors[:10] if errors else [],
                "message": f"No images could be deleted ({skipped} in datasets)",
            }

        # Delete in batches
        for batch_num, batch in enumerate(chunks(deletable_ids, self.BATCH_SIZE)):
            # Delete from storage
            filenames = [images_data[img_id] for img_id in batch if images_data.get(img_id)]
            if filenames:
                try:
                    supabase_service.client.storage.from_("od-images").remove(filenames)
                except Exception as e:
                    errors.append(f"Storage batch {batch_num + 1}: {str(e)}")

            # Delete from database
            try:
                supabase_service.client.table("od_images")\
                    .delete()\
                    .in_("id", batch)\
                    .execute()
                deleted += len(batch)
            except Exception as e:
                errors.append(f"Database batch {batch_num + 1}: {str(e)}")

            # Update progress (15-95% range)
            processed = (batch_num + 1) * self.BATCH_SIZE
            processed = min(processed, len(deletable_ids))
            progress = 15 + calculate_progress(processed, len(deletable_ids)) * 0.80

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Deleting images... ({processed}/{len(deletable_ids)})",
                processed=processed,
                total=total,
            ))

        return {
            "deleted": deleted,
            "skipped": skipped,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Deleted {deleted} images" + (f", skipped {skipped} (in datasets)" if skipped else ""),
        }

    def _get_filtered_image_ids(self, filters: dict) -> list[str]:
        """Get all image IDs matching filters."""
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

    def _get_image_data(self, image_ids: list[str]) -> dict[str, str]:
        """Get image filenames for deletion."""
        data = {}

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("od_images")\
                .select("id, filename")\
                .in_("id", batch)\
                .execute()

            for img in result.data or []:
                if img.get("filename"):
                    data[img["id"]] = img["filename"]

        return data

    def _get_images_in_datasets(self, image_ids: list[str]) -> set[str]:
        """Get set of image IDs that are in any dataset."""
        in_datasets = set()

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("od_dataset_images")\
                .select("image_id")\
                .in_("image_id", batch)\
                .execute()

            in_datasets.update(r["image_id"] for r in result.data or [])

        return in_datasets
