"""
Handler for bulk adding images to a classification dataset.

This handler processes the operation in batches with progress tracking,
checking for duplicates and updating dataset stats.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkAddToCLSDatasetHandler(BaseJobHandler):
    """
    Handler for bulk adding images to a classification dataset.

    Config:
        dataset_id: str - Target dataset ID
        image_ids: list[str] - Image IDs to add

    Result:
        added: int - Number of images added
        skipped: int - Number of images skipped (already in dataset)
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_add_to_dataset"
    BATCH_SIZE = 100
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
        total = len(image_ids)

        if total == 0:
            return {
                "added": 0,
                "skipped": 0,
                "total": 0,
                "message": "No images to add",
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

        # Get existing images in dataset
        update_progress(JobProgress(
            progress=5,
            current_step="Checking existing images...",
            processed=0,
            total=total,
        ))

        existing_ids = self._get_existing_image_ids(dataset_id, image_ids)

        # Filter out existing images
        new_image_ids = [img_id for img_id in image_ids if img_id not in existing_ids]
        skipped = len(existing_ids)

        if not new_image_ids:
            return {
                "added": 0,
                "skipped": skipped,
                "total": total,
                "message": f"All {skipped} images already in dataset",
            }

        # Process additions in batches
        added = 0
        errors = []

        for batch_num, batch in enumerate(chunks(new_image_ids, self.BATCH_SIZE)):
            try:
                records = [
                    {
                        "dataset_id": dataset_id,
                        "image_id": img_id,
                        "status": "pending",
                    }
                    for img_id in batch
                ]
                supabase_service.client.table("cls_dataset_images")\
                    .insert(records)\
                    .execute()
                added += len(batch)
            except Exception as e:
                errors.append(f"Batch {batch_num + 1}: {str(e)}")

            # Update progress (10-90% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, len(new_image_ids))
            progress = 10 + calculate_progress(processed, len(new_image_ids)) * 0.80

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Adding images... ({processed}/{len(new_image_ids)})",
                processed=processed + skipped,
                total=total,
            ))

        # Update dataset stats
        update_progress(JobProgress(
            progress=95,
            current_step="Updating dataset stats...",
            processed=total,
            total=total,
        ))

        try:
            supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()
        except Exception as e:
            errors.append(f"Stats update: {str(e)}")

        return {
            "added": added,
            "skipped": skipped,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Added {added} images to dataset" + (f", {skipped} already existed" if skipped else ""),
        }

    def _get_existing_image_ids(self, dataset_id: str, image_ids: list[str]) -> set[str]:
        """Get set of image IDs already in the dataset."""
        existing = set()

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("cls_dataset_images")\
                .select("image_id")\
                .eq("dataset_id", dataset_id)\
                .in_("image_id", batch)\
                .execute()

            existing.update(r["image_id"] for r in result.data or [])

        return existing
