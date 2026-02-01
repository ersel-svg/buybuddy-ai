"""
Handler for bulk updating image status in a dataset.

This handler processes status updates in batches with progress tracking.
"""

from datetime import datetime, timezone
from typing import Callable

from services.supabase import supabase_service
from services.od_sync import update_dataset_annotated_image_count
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkUpdateStatusHandler(BaseJobHandler):
    """
    Handler for bulk updating image status in an OD dataset.

    Config:
        dataset_id: str - Target dataset ID
        new_status: str - Status to set (pending, annotating, completed, skipped)
        image_ids: list[str] (optional) - Specific image IDs to update
        filters: dict (optional) - Filter criteria to select images
            - current_status: str
            - has_annotations: bool

    Result:
        updated: int - Number of images updated
        total: int - Total images processed
        message: str - Summary message
    """

    job_type = "local_bulk_update_status"
    BATCH_SIZE = 200  # Larger batch for updates (cheaper operation)
    PAGE_SIZE = 1000

    VALID_STATUSES = ["pending", "annotating", "completed", "skipped"]

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        if not config.get("new_status"):
            return "new_status is required"
        if config["new_status"] not in self.VALID_STATUSES:
            return f"Invalid status. Must be one of: {self.VALID_STATUSES}"
        if not config.get("image_ids") and not config.get("filters"):
            return "Either image_ids or filters is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        dataset_id = config["dataset_id"]
        new_status = config["new_status"]
        image_ids = config.get("image_ids", [])
        filters = config.get("filters", {})

        # Update initial progress
        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=0,
        ))

        # Verify dataset exists
        dataset = supabase_service.client.table("od_datasets")\
            .select("id")\
            .eq("id", dataset_id)\
            .single()\
            .execute()

        if not dataset.data:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get image IDs to update
        if image_ids:
            target_ids = image_ids
        else:
            update_progress(JobProgress(
                progress=5,
                current_step="Collecting image IDs...",
                processed=0,
                total=0,
            ))
            target_ids = self._get_filtered_image_ids(dataset_id, filters)

        total = len(target_ids)

        if total == 0:
            return {
                "updated": 0,
                "total": 0,
                "message": "No images to update",
            }

        # Prepare update data
        update_data = {"status": new_status}
        if new_status == "completed":
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Process updates in batches
        updated = 0

        for batch_num, batch in enumerate(chunks(target_ids, self.BATCH_SIZE)):
            try:
                result = supabase_service.client.table("od_dataset_images")\
                    .update(update_data)\
                    .eq("dataset_id", dataset_id)\
                    .in_("image_id", batch)\
                    .execute()
                updated += len(result.data) if result.data else 0
            except Exception as e:
                print(f"[BulkUpdateStatus] Batch {batch_num + 1} error: {e}")

            # Update progress (5-95% range)
            processed = (batch_num + 1) * self.BATCH_SIZE
            processed = min(processed, total)
            progress = 5 + calculate_progress(processed, total) * 0.90

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Updating status... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        # Update dataset annotated_image_count
        update_progress(JobProgress(
            progress=95,
            current_step="Updating dataset counts...",
            processed=total,
            total=total,
        ))

        update_dataset_annotated_image_count(dataset_id)

        return {
            "updated": updated,
            "total": total,
            "status": new_status,
            "message": f"Updated {updated} images to '{new_status}'",
        }

    def _get_filtered_image_ids(self, dataset_id: str, filters: dict) -> list[str]:
        """Get image IDs in dataset matching filters."""
        all_ids = []
        offset = 0

        while True:
            query = supabase_service.client.table("od_dataset_images")\
                .select("image_id, annotation_count")\
                .eq("dataset_id", dataset_id)

            # Filter by current status
            if filters.get("current_status"):
                query = query.eq("status", filters["current_status"])

            result = query.range(offset, offset + self.PAGE_SIZE - 1).execute()

            if not result.data:
                break

            # Filter by has_annotations in Python (more flexible)
            for item in result.data:
                has_ann = (item.get("annotation_count") or 0) > 0
                if filters.get("has_annotations") is None:
                    all_ids.append(item["image_id"])
                elif filters["has_annotations"] == has_ann:
                    all_ids.append(item["image_id"])

            if len(result.data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        return all_ids
