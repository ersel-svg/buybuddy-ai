"""
Handler for bulk removing images from a classification dataset.

This handler processes the operation in batches with progress tracking,
removing labels and optionally deleting images entirely.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


CLS_IMAGES_BUCKET = "cls-images"


@job_registry.register
class BulkRemoveFromCLSDatasetHandler(BaseJobHandler):
    """
    Handler for bulk removing images from a classification dataset.

    Config:
        dataset_id: str - Target dataset ID
        image_ids: list[str] - Image IDs to remove
        delete_completely: bool - If True, delete images from system entirely (default: False)

    Result:
        removed: int - Number of images removed from dataset
        deleted: int - Number of images deleted from system (if delete_completely)
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_remove_from_dataset"
    BATCH_SIZE = 50
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
        delete_completely = config.get("delete_completely", False)
        total = len(image_ids)

        if total == 0:
            return {
                "removed": 0,
                "deleted": 0,
                "total": 0,
                "message": "No images to remove",
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

        # Get storage paths if we need to delete completely
        storage_paths = {}
        if delete_completely:
            update_progress(JobProgress(
                progress=5,
                current_step="Fetching image data...",
                processed=0,
                total=total,
            ))
            storage_paths = self._get_storage_paths(image_ids)

        # Process removals in batches
        removed = 0
        deleted = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            # Remove from dataset (cascade will delete labels)
            try:
                result = supabase_service.client.table("cls_dataset_images")\
                    .delete()\
                    .eq("dataset_id", dataset_id)\
                    .in_("image_id", batch)\
                    .execute()
                removed += len(result.data) if result.data else 0
            except Exception as e:
                errors.append(f"Dataset removal batch {batch_num + 1}: {str(e)}")

            # If delete_completely, also delete images not in other datasets
            if delete_completely:
                for img_id in batch:
                    try:
                        # Check if image is in other datasets
                        other_datasets = supabase_service.client.table("cls_dataset_images")\
                            .select("id")\
                            .eq("image_id", img_id)\
                            .limit(1)\
                            .execute()

                        if not other_datasets.data:
                            # Not used elsewhere, safe to delete
                            path = storage_paths.get(img_id)
                            if path:
                                try:
                                    supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).remove([path])
                                except:
                                    pass

                            supabase_service.client.table("cls_images")\
                                .delete()\
                                .eq("id", img_id)\
                                .execute()
                            deleted += 1
                    except Exception as e:
                        errors.append(f"Image {img_id}: {str(e)}")

            # Update progress (5-90% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 5 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Removing images... ({processed}/{total})",
                processed=processed,
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

        message = f"Removed {removed} images from dataset"
        if delete_completely and deleted > 0:
            message += f", deleted {deleted} from system"

        return {
            "removed": removed,
            "deleted": deleted,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": message,
        }

    def _get_storage_paths(self, image_ids: list[str]) -> dict[str, str]:
        """Get storage paths for images."""
        paths = {}

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("cls_images")\
                .select("id, storage_path")\
                .in_("id", batch)\
                .execute()

            for img in result.data or []:
                if img.get("storage_path"):
                    paths[img["id"]] = img["storage_path"]

        return paths
