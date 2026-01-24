"""
Handler for bulk setting labels on classification images.

This handler processes label assignments in batches with progress tracking,
supporting both single-label and multi-label datasets.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkSetCLSLabelsHandler(BaseJobHandler):
    """
    Handler for bulk setting labels on classification images.

    Config:
        dataset_id: str - Target dataset ID
        class_id: str - Class ID to assign
        image_ids: list[str] - Image IDs to label

    Result:
        created: int - Number of new labels created
        updated: int - Number of existing labels updated
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_set_labels"
    BATCH_SIZE = 200
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        if not config.get("class_id"):
            return "class_id is required"
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
        class_id = config["class_id"]
        image_ids = config["image_ids"]
        total = len(image_ids)

        if total == 0:
            return {
                "created": 0,
                "updated": 0,
                "total": 0,
                "message": "No images to label",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=total,
        ))

        # Get dataset task type
        dataset = supabase_service.client.table("cls_datasets")\
            .select("id, name, task_type")\
            .eq("id", dataset_id)\
            .single()\
            .execute()

        if not dataset.data:
            raise ValueError(f"Dataset not found: {dataset_id}")

        task_type = dataset.data.get("task_type", "single_label")

        # Get existing labels
        update_progress(JobProgress(
            progress=5,
            current_step="Checking existing labels...",
            processed=0,
            total=total,
        ))

        existing_labels = self._get_existing_labels(dataset_id, image_ids, class_id, task_type)

        # Process label assignments in batches
        created = 0
        updated = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            for image_id in batch:
                try:
                    if task_type == "single_label":
                        if image_id in existing_labels:
                            # Update existing label
                            supabase_service.client.table("cls_labels")\
                                .update({"class_id": class_id})\
                                .eq("dataset_id", dataset_id)\
                                .eq("image_id", image_id)\
                                .execute()
                            updated += 1
                        else:
                            # Create new label
                            supabase_service.client.table("cls_labels")\
                                .insert({
                                    "dataset_id": dataset_id,
                                    "image_id": image_id,
                                    "class_id": class_id,
                                })\
                                .execute()
                            created += 1
                    else:
                        # Multi-label: add if not exists
                        if image_id not in existing_labels:
                            supabase_service.client.table("cls_labels")\
                                .insert({
                                    "dataset_id": dataset_id,
                                    "image_id": image_id,
                                    "class_id": class_id,
                                })\
                                .execute()
                            created += 1

                    # Update dataset image status
                    supabase_service.client.table("cls_dataset_images")\
                        .update({
                            "status": "labeled",
                            "labeled_at": "now()",
                        })\
                        .eq("dataset_id", dataset_id)\
                        .eq("image_id", image_id)\
                        .execute()

                except Exception as e:
                    errors.append(f"{image_id}: {str(e)}")

            # Update progress (5-90% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 5 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Assigning labels... ({processed}/{total})",
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
            supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": class_id}).execute()
        except Exception as e:
            errors.append(f"Stats update: {str(e)}")

        return {
            "created": created,
            "updated": updated,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Labeled {created + updated} images ({created} new, {updated} updated)",
        }

    def _get_existing_labels(
        self,
        dataset_id: str,
        image_ids: list[str],
        class_id: str,
        task_type: str,
    ) -> set[str]:
        """Get set of image IDs that already have labels."""
        existing = set()

        for batch in chunks(image_ids, self.PAGE_SIZE):
            query = supabase_service.client.table("cls_labels")\
                .select("image_id")\
                .eq("dataset_id", dataset_id)\
                .in_("image_id", batch)

            # For multi-label, check specific class
            if task_type != "single_label":
                query = query.eq("class_id", class_id)

            result = query.execute()
            existing.update(r["image_id"] for r in result.data or [])

        return existing
