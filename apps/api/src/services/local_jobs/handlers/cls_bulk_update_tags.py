"""
Handler for bulk updating tags on classification images.

This handler processes tag updates in batches with progress tracking,
supporting add, remove, and replace operations.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkUpdateCLSTagsHandler(BaseJobHandler):
    """
    Handler for bulk updating tags on classification images.

    Config:
        image_ids: list[str] - Image IDs to update
        action: str - "add" | "remove" | "replace"
        tags: list[str] - Tags to add/remove/replace with

    Result:
        updated: int - Number of images updated
        total: int - Total images processed
        errors: list[str] - Any errors encountered
    """

    job_type = "local_cls_bulk_update_tags"
    BATCH_SIZE = 200
    PAGE_SIZE = 1000

    def validate_config(self, config: dict) -> str | None:
        if not config.get("image_ids"):
            return "image_ids is required"
        if not config.get("action"):
            return "action is required"
        if config["action"] not in ["add", "remove", "replace"]:
            return "action must be 'add', 'remove', or 'replace'"
        if not config.get("tags"):
            return "tags is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        image_ids = config["image_ids"]
        action = config["action"]
        tags = config["tags"]
        total = len(image_ids)

        if total == 0:
            return {
                "updated": 0,
                "total": 0,
                "message": "No images to update",
            }

        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=total,
        ))

        # For replace action, we can do a batch update directly
        if action == "replace":
            return await self._bulk_replace_tags(image_ids, tags, update_progress, total)

        # For add/remove, we need to process individually
        return await self._process_tag_updates(image_ids, action, tags, update_progress, total)

    async def _bulk_replace_tags(
        self,
        image_ids: list[str],
        tags: list[str],
        update_progress: Callable[[JobProgress], None],
        total: int,
    ) -> dict:
        """Replace tags for all images in bulk."""
        updated = 0
        errors = []

        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            try:
                result = supabase_service.client.table("cls_images")\
                    .update({"tags": tags})\
                    .in_("id", batch)\
                    .execute()
                updated += len(result.data) if result.data else 0
            except Exception as e:
                errors.append(f"Batch {batch_num + 1}: {str(e)}")

            # Update progress
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 5 + calculate_progress(processed, total) * 0.90

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Replacing tags... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        return {
            "updated": updated,
            "total": total,
            "action": "replace",
            "errors": errors[:10] if errors else [],
            "message": f"Replaced tags on {updated} images",
        }

    async def _process_tag_updates(
        self,
        image_ids: list[str],
        action: str,
        tags: list[str],
        update_progress: Callable[[JobProgress], None],
        total: int,
    ) -> dict:
        """Process add/remove tag updates."""
        updated = 0
        errors = []

        # Get current tags for all images
        update_progress(JobProgress(
            progress=5,
            current_step="Fetching current tags...",
            processed=0,
            total=total,
        ))

        current_tags_map = self._get_current_tags(image_ids)

        # Process updates
        for batch_num, batch in enumerate(chunks(image_ids, self.BATCH_SIZE)):
            for image_id in batch:
                try:
                    current = current_tags_map.get(image_id, [])

                    if action == "add":
                        new_tags = list(set(current + tags))
                    else:  # remove
                        new_tags = [t for t in current if t not in tags]

                    # Only update if changed
                    if set(new_tags) != set(current):
                        supabase_service.client.table("cls_images")\
                            .update({"tags": new_tags})\
                            .eq("id", image_id)\
                            .execute()
                        updated += 1

                except Exception as e:
                    errors.append(f"{image_id}: {str(e)}")

            # Update progress (10-95% range)
            processed = min((batch_num + 1) * self.BATCH_SIZE, total)
            progress = 10 + calculate_progress(processed, total) * 0.85

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Updating tags... ({processed}/{total})",
                processed=processed,
                total=total,
            ))

        action_verb = "Added tags to" if action == "add" else "Removed tags from"

        return {
            "updated": updated,
            "total": total,
            "action": action,
            "errors": errors[:10] if errors else [],
            "message": f"{action_verb} {updated} images",
        }

    def _get_current_tags(self, image_ids: list[str]) -> dict[str, list[str]]:
        """Get current tags for all images."""
        tags_map = {}

        for batch in chunks(image_ids, self.PAGE_SIZE):
            result = supabase_service.client.table("cls_images")\
                .select("id, tags")\
                .in_("id", batch)\
                .execute()

            for img in result.data or []:
                tags_map[img["id"]] = img.get("tags") or []

        return tags_map
