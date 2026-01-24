"""
Handler for bulk removing images from a dataset.

This handler processes the operation in batches with progress tracking,
handling deletion of annotations, dataset links, and optionally images.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkRemoveFromDatasetHandler(BaseJobHandler):
    """
    Handler for bulk removing images from an OD dataset.

    Config:
        dataset_id: str - Target dataset ID
        image_ids: list[str] (optional) - Specific image IDs to remove
        filters: dict (optional) - Filter criteria to select images
            - statuses: str (comma-separated)
            - has_annotations: bool
        delete_completely: bool - If True, delete images from system entirely (default: True)

    Result:
        removed: int - Number of images removed from dataset
        deleted_from_storage: int - Number of images deleted from storage
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_bulk_remove_from_dataset"
    BATCH_SIZE = 50  # Smaller batch for deletions (more expensive)
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
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
        image_ids = config.get("image_ids", [])
        filters = config.get("filters", {})
        delete_completely = config.get("delete_completely", True)

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

        # Get image IDs to remove
        if image_ids:
            # Use provided IDs
            target_ids = image_ids
        else:
            # Get IDs from filters
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
                "removed": 0,
                "deleted_from_storage": 0,
                "total": 0,
                "message": "No images to remove",
            }

        # Get image filenames for storage deletion
        image_filenames = {}
        if delete_completely:
            update_progress(JobProgress(
                progress=10,
                current_step="Fetching image data...",
                processed=0,
                total=total,
            ))
            image_filenames = self._get_image_filenames(target_ids)

        # Process removals in batches
        removed = 0
        deleted_from_storage = 0
        errors = []

        for batch_num, batch in enumerate(chunks(target_ids, self.BATCH_SIZE)):
            # Delete annotations for this batch
            try:
                for image_id in batch:
                    supabase_service.client.table("od_annotations")\
                        .delete()\
                        .eq("dataset_id", dataset_id)\
                        .eq("image_id", image_id)\
                        .execute()
            except Exception as e:
                errors.append(f"Annotations batch {batch_num + 1}: {str(e)}")

            # Delete dataset_image records
            try:
                result = supabase_service.client.table("od_dataset_images")\
                    .delete()\
                    .eq("dataset_id", dataset_id)\
                    .in_("image_id", batch)\
                    .execute()
                removed += len(result.data) if result.data else 0
            except Exception as e:
                errors.append(f"Dataset links batch {batch_num + 1}: {str(e)}")
                continue

            # If delete_completely, delete images not used elsewhere
            if delete_completely:
                for image_id in batch:
                    try:
                        # Check if image is used in other datasets
                        other_datasets = supabase_service.client.table("od_dataset_images")\
                            .select("id")\
                            .eq("image_id", image_id)\
                            .limit(1)\
                            .execute()

                        if not other_datasets.data:
                            # Not used elsewhere, safe to delete
                            filename = image_filenames.get(image_id)
                            if filename:
                                try:
                                    supabase_service.client.storage\
                                        .from_("od-images")\
                                        .remove([filename])
                                    deleted_from_storage += 1
                                except Exception:
                                    pass  # Continue even if storage delete fails

                            # Delete image record
                            supabase_service.client.table("od_images")\
                                .delete()\
                                .eq("id", image_id)\
                                .execute()
                    except Exception as e:
                        errors.append(f"Image {image_id}: {str(e)}")

            # Update progress (10-95% range for processing)
            processed = (batch_num + 1) * self.BATCH_SIZE
            processed = min(processed, total)
            progress = 10 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Removing images... ({processed}/{total})",
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
            "removed": removed,
            "deleted_from_storage": deleted_from_storage,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Removed {removed} images from dataset" + (
                f", {deleted_from_storage} deleted from storage" if delete_completely else ""
            ),
        }

    def _get_filtered_image_ids(self, dataset_id: str, filters: dict) -> list[str]:
        """Get image IDs in dataset matching filters."""
        all_ids = []
        offset = 0

        while True:
            query = supabase_service.client.table("od_dataset_images")\
                .select("image_id")\
                .eq("dataset_id", dataset_id)

            # Apply filters
            if filters.get("statuses"):
                status_list = [s.strip() for s in filters["statuses"].split(",") if s.strip()]
                if status_list:
                    query = query.in_("status", status_list)

            if filters.get("has_annotations") is not None:
                if filters["has_annotations"]:
                    query = query.gt("annotation_count", 0)
                else:
                    query = query.eq("annotation_count", 0)

            result = query.range(offset, offset + self.PAGE_SIZE - 1).execute()

            if not result.data:
                break

            all_ids.extend(r["image_id"] for r in result.data)

            if len(result.data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        return all_ids

    def _get_image_filenames(self, image_ids: list[str]) -> dict[str, str]:
        """Get filenames for images (for storage deletion)."""
        filenames = {}

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("od_images")\
                .select("id, filename")\
                .in_("id", batch)\
                .execute()

            for img in result.data or []:
                if img.get("filename"):
                    filenames[img["id"]] = img["filename"]

        return filenames

    def _update_dataset_count(self, dataset_id: str) -> None:
        """Update dataset image and annotation counts."""
        # Image count
        img_count = supabase_service.client.table("od_dataset_images")\
            .select("id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .execute()

        # Annotation count
        ann_count = supabase_service.client.table("od_annotations")\
            .select("id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .execute()

        # Annotated image count
        annotated_count = supabase_service.client.table("od_dataset_images")\
            .select("id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .eq("status", "completed")\
            .execute()

        supabase_service.client.table("od_datasets").update({
            "image_count": img_count.count or 0,
            "annotation_count": ann_count.count or 0,
            "annotated_image_count": annotated_count.count or 0,
        }).eq("id", dataset_id).execute()
