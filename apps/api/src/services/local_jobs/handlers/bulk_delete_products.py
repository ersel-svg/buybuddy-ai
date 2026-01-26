"""
Handler for bulk deleting products with cascade.

This handler processes product deletions in batches with progress tracking,
handling cascade deletions for frames, storage files, and dataset references.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkDeleteProductsHandler(BaseJobHandler):
    """
    Handler for bulk deleting products with cascade.

    Config:
        product_ids: list[str] - Product IDs to delete

    Result:
        deleted: int - Number of products deleted
        frames_deleted: int - Number of frames deleted
        storage_deleted: int - Number of storage files deleted
        dataset_refs_deleted: int - Number of dataset references deleted
        failed: int - Number of products that failed to delete
        total: int - Total products processed
        errors: list[dict] - Failed deletions with product_id and error
    """

    job_type = "local_bulk_delete_products"
    BATCH_SIZE = 20  # Smaller batch for delete operations

    def validate_config(self, config: dict) -> str | None:
        if not config.get("product_ids"):
            return "product_ids is required"
        if not isinstance(config["product_ids"], list):
            return "product_ids must be a list"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        product_ids = config["product_ids"]
        total = len(product_ids)

        if total == 0:
            return {
                "deleted": 0,
                "frames_deleted": 0,
                "storage_deleted": 0,
                "dataset_refs_deleted": 0,
                "failed": 0,
                "total": 0,
                "errors": [],
                "message": "No products to delete",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing deletion...",
            processed=0,
            total=total,
        ))

        # Counters
        deleted = 0
        failed = 0
        errors = []
        total_frames_deleted = 0
        total_storage_deleted = 0
        total_dataset_refs_deleted = 0

        # Process deletions in batches
        for batch_num, batch in enumerate(chunks(product_ids, self.BATCH_SIZE)):
            for product_id in batch:
                try:
                    # Use cascade delete from supabase service
                    result = await supabase_service.delete_products_cascade([product_id])

                    deleted += result.get("products_deleted", 0)
                    total_frames_deleted += result.get("frames_deleted", 0)
                    total_storage_deleted += result.get("storage_deleted", 0)
                    total_dataset_refs_deleted += result.get("dataset_refs_deleted", 0)

                except Exception as e:
                    failed += 1
                    errors.append({
                        "product_id": product_id,
                        "error": str(e),
                    })

            # Update progress (0-95% for deletions)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = calculate_progress(processed, total) * 0.95

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Deleting products... ({processed}/{total})",
                processed=processed,
                total=total,
                errors=errors[-5:] if errors else [],
            ))

        # Final progress
        update_progress(JobProgress(
            progress=100,
            current_step="Deletion complete",
            processed=total,
            total=total,
        ))

        return {
            "deleted": deleted,
            "frames_deleted": total_frames_deleted,
            "storage_deleted": total_storage_deleted,
            "dataset_refs_deleted": total_dataset_refs_deleted,
            "failed": failed,
            "total": total,
            "errors": errors[:20] if errors else [],  # Limit errors to 20
            "message": f"Deleted {deleted} products ({total_frames_deleted} frames, {total_storage_deleted} storage files)" + (f", {failed} failed" if failed else ""),
        }
