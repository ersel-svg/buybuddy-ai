"""
Local job to synchronize OD annotation counts and statuses.
"""

from services.supabase import supabase_service
from services.local_jobs import BaseJobHandler, JobProgress, job_registry
from services.od_sync import fallback_sync_annotation_counts


@job_registry.register
class ODSyncCountsHandler(BaseJobHandler):
    """Sync annotation counts for a dataset (fallback path)."""

    job_type = "local_od_sync_counts"

    async def execute(self, job_id: str, config: dict, update_progress):
        dataset_id = config.get("dataset_id")
        image_ids = config.get("image_ids") or []
        class_ids = config.get("class_ids") or []

        if not dataset_id:
            raise ValueError("dataset_id is required")

        update_progress(JobProgress(
            progress=5,
            current_step="Starting resync...",
            processed=0,
            total=0,
        ))

        # Fast path: single SQL RPC (avoids per-image queries)
        try:
            rpc_result = supabase_service.client.rpc(
                "resync_od_annotation_counts",
                {"p_dataset_id": dataset_id},
            ).execute()
            payload = rpc_result.data or {}
            total_images = payload.get("total_images", 0)
            updated_images = payload.get("updated_images", 0)
            update_progress(JobProgress(
                progress=100,
                current_step="Done",
                processed=total_images,
                total=total_images,
            ))
            return {
                **payload,
                "processed": total_images,
                "total": total_images,
                "message": f"Resync complete (updated {updated_images} images)",
            }
        except Exception as e:
            update_progress(JobProgress(
                progress=10,
                current_step=f"RPC failed, falling back: {e}",
                processed=0,
                total=0,
            ))

        # Fallback: slower per-image sync
        if not image_ids:
            update_progress(JobProgress(
                progress=12,
                current_step="Fetching dataset images...",
                processed=0,
                total=0,
            ))
            rows = await supabase_service.paginated_query(
                "od_dataset_images",
                select="image_id",
                filters={"eq": {"dataset_id": dataset_id}},
                batch_size=1000,
            )
            image_ids = [row.get("image_id") for row in rows if row.get("image_id")]

        if not class_ids:
            class_rows = supabase_service.client.table("od_classes").select(
                "id"
            ).eq("dataset_id", dataset_id).execute()
            class_ids = [row.get("id") for row in class_rows.data or [] if row.get("id")]

        total = len(image_ids)
        update_progress(JobProgress(
            progress=15,
            current_step="Initializing fallback...",
            processed=0,
            total=total,
        ))

        if total == 0:
            return {
                "updated_images": 0,
                "total_images": 0,
                "message": "No images to sync",
            }

        def progress_cb(processed: int, total_images: int, message: str) -> None:
            pct = int((processed / max(total_images, 1)) * 90)
            update_progress(JobProgress(
                progress=pct,
                current_step=message,
                processed=processed,
                total=total_images,
            ))

        result = fallback_sync_annotation_counts(
            dataset_id,
            image_ids,
            class_ids,
            progress_cb=progress_cb,
        )
        updated_images = result.get("updated_images", 0)
        total_images = result.get("total_images", total)
        result.update({
            "processed": total_images,
            "total": total_images,
            "message": f"Resync complete (updated {updated_images} images)",
        })

        update_progress(JobProgress(
            progress=100,
            current_step="Done",
            processed=total,
            total=total,
        ))

        return result
