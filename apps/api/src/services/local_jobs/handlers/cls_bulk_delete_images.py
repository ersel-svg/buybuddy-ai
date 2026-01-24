"""
Handler for bulk deleting classification images.

This handler processes image deletions in batches with progress tracking,
handling storage deletion and database cleanup.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


CLS_IMAGES_BUCKET = "cls-images"


@job_registry.register
class BulkDeleteCLSImagesHandler(BaseJobHandler):
    """
    Handler for bulk deleting classification images.

    Config:
        image_ids: list[str] - Image IDs to delete

    Result:
        deleted: int - Number of images deleted
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_delete_images"
    BATCH_SIZE = 100
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("image_ids"):
            return "image_ids is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        image_ids = config["image_ids"]
        total = len(image_ids)

        if total == 0:
            return {
                "deleted": 0,
                "total": 0,
                "message": "No images to delete",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=total,
        ))

        # Get storage paths for all images
        update_progress(JobProgress(
            progress=5,
            current_step="Fetching image data...",
            processed=0,
            total=total,
        ))

        storage_paths = self._get_storage_paths(image_ids)

        # Process deletions in batches
        deleted = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            # Delete from storage
            batch_paths = [storage_paths[img_id] for img_id in batch if storage_paths.get(img_id)]
            if batch_paths:
                try:
                    supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).remove(batch_paths)
                except Exception as e:
                    errors.append(f"Storage batch {batch_num + 1}: {str(e)}")

            # Delete from database (cascade will handle dataset_images and labels)
            try:
                supabase_service.client.table("cls_images")\
                    .delete()\
                    .in_("id", batch)\
                    .execute()
                deleted += len(batch)
            except Exception as e:
                errors.append(f"Database batch {batch_num + 1}: {str(e)}")

            # Update progress (5-95% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 5 + calculate_progress(processed, total) * 0.90

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Deleting images... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        return {
            "deleted": deleted,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Deleted {deleted} images",
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
