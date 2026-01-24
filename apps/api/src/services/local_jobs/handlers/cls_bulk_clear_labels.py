"""
Handler for bulk clearing labels from classification images.

This handler processes label removals in batches with progress tracking.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkClearCLSLabelsHandler(BaseJobHandler):
    """
    Handler for bulk clearing labels from classification images.

    Config:
        dataset_id: str - Target dataset ID
        image_ids: list[str] - Image IDs to clear labels from
        class_ids: list[str] (optional) - Only clear specific classes

    Result:
        cleared: int - Number of labels cleared
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_clear_labels"
    BATCH_SIZE = 200
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        if not config.get("image_ids"):
            return "image_ids is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        dataset_id = config["dataset_id"]
        image_ids = config["image_ids"]
        class_ids = config.get("class_ids")  # Optional: only clear specific classes
        total = len(image_ids)

        if total == 0:
            return {
                "cleared": 0,
                "total": 0,
                "message": "No images to clear labels from",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=total,
        ))

        # Verify dataset exists
        dataset = supabase_service.client.table("cls_datasets")\
            .select("id, name")\
            .eq("id", dataset_id)\
            .single()\
            .execute()

        if not dataset.data:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get affected class IDs for stats update
        update_progress(JobProgress(
            progress=5,
            current_step="Collecting affected classes...",
            processed=0,
            total=total,
        ))

        affected_class_ids = self._get_affected_class_ids(dataset_id, image_ids, class_ids)

        # Process label deletions in batches
        cleared = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            try:
                query = supabase_service.client.table("cls_labels")\
                    .delete()\
                    .eq("dataset_id", dataset_id)\
                    .in_("image_id", batch)

                if class_ids:
                    query = query.in_("class_id", class_ids)

                result = query.execute()
                cleared += len(result.data) if result.data else 0

                # Update dataset image statuses to pending (if all labels cleared)
                if not class_ids:
                    supabase_service.client.table("cls_dataset_images")\
                        .update({
                            "status": "pending",
                            "labeled_at": None,
                        })\
                        .eq("dataset_id", dataset_id)\
                        .in_("image_id", batch)\
                        .execute()

            except Exception as e:
                errors.append(f"Batch {batch_num + 1}: {str(e)}")

            # Update progress (5-90% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 5 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Clearing labels... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        # Update stats
        update_progress(JobProgress(
            progress=95,
            current_step="Updating stats...",
            processed=total,
            total=total,
        ))

        try:
            supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()

            for cid in affected_class_ids:
                supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": cid}).execute()
        except Exception as e:
            errors.append(f"Stats update: {str(e)}")

        return {
            "cleared": cleared,
            "total": total,
            "affected_classes": len(affected_class_ids),
            "errors": errors[:10] if errors else [],
            "message": f"Cleared {cleared} labels from {total} images",
        }

    def _get_affected_class_ids(
        self,
        dataset_id: str,
        image_ids: list[str],
        class_ids: list[str] | None,
    ) -> set[str]:
        """Get set of class IDs that will be affected."""
        affected = set()

        for batch in chunks(image_ids, self.PAGE_SIZE):
            query = supabase_service.client.table("cls_labels")\
                .select("class_id")\
                .eq("dataset_id", dataset_id)\
                .in_("image_id", batch)

            if class_ids:
                query = query.in_("class_id", class_ids)

            result = query.execute()
            affected.update(r["class_id"] for r in result.data or [])

        return affected
