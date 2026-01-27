"""
Handler for background class merge operations.

Merges multiple OD classes into a target class with progress tracking.
Useful for large merges that would timeout as synchronous operations.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry


@job_registry.register
class ClassMergeJobHandler(BaseJobHandler):
    """
    Handler for background class merge operations.

    Config:
        target_class_id: str - ID of the class to merge into
        source_class_ids: list[str] - IDs of classes to merge from
        dataset_id: str - Dataset ID (for updating counts)

    Result:
        merged_count: int - Number of source classes merged
        annotations_moved: int - Total annotations moved
        errors: list[str] - Any errors encountered
    """

    job_type = "local_class_merge"
    BATCH_SIZE = 1000  # Annotations per batch

    def validate_config(self, config: dict) -> str | None:
        if not config.get("target_class_id"):
            return "target_class_id is required"
        if not config.get("source_class_ids"):
            return "source_class_ids is required"
        if not isinstance(config.get("source_class_ids"), list):
            return "source_class_ids must be a list"
        if len(config["source_class_ids"]) == 0:
            return "source_class_ids must contain at least one ID"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        target_class_id = config["target_class_id"]
        source_class_ids = config["source_class_ids"]

        update_progress(JobProgress(
            progress=0,
            current_step="Validating classes...",
            processed=0,
            total=len(source_class_ids),
        ))

        # Validate target class exists
        target = supabase_service.client.table("od_classes") \
            .select("*") \
            .eq("id", target_class_id) \
            .single() \
            .execute()

        if not target.data:
            return {
                "merged_count": 0,
                "annotations_moved": 0,
                "errors": ["Target class not found"],
                "message": "Failed: Target class not found",
            }

        # Validate source classes exist
        sources = supabase_service.client.table("od_classes") \
            .select("*") \
            .in_("id", source_class_ids) \
            .execute()

        if len(sources.data or []) != len(source_class_ids):
            return {
                "merged_count": 0,
                "annotations_moved": 0,
                "errors": ["One or more source classes not found"],
                "message": "Failed: One or more source classes not found",
            }

        total_moved = 0
        merged_count = 0
        errors = []

        # Calculate total annotations to move for progress tracking
        total_annotations = sum(
            (s.get("annotation_count", 0) or 0)
            for s in sources.data
            if s["id"] != target_class_id
        )

        update_progress(JobProgress(
            progress=5,
            current_step=f"Merging {len(source_class_ids)} classes ({total_annotations:,} annotations)...",
            processed=0,
            total=total_annotations,
        ))

        # Move annotations from source classes to target
        for idx, source_id in enumerate(source_class_ids):
            if source_id == target_class_id:
                continue

            source_class = next((s for s in sources.data if s["id"] == source_id), None)
            source_name = source_class["name"] if source_class else source_id
            source_annotation_count = source_class.get("annotation_count", 0) or 0 if source_class else 0

            update_progress(JobProgress(
                progress=5 + int((idx / len(source_class_ids)) * 85),
                current_step=f"Moving annotations from '{source_name}'...",
                processed=total_moved,
                total=total_annotations,
            ))

            # Move annotations in batches
            moved_for_source = 0
            try:
                while True:
                    # Get batch of annotation IDs
                    batch_result = supabase_service.client.table("od_annotations") \
                        .select("id") \
                        .eq("class_id", source_id) \
                        .limit(self.BATCH_SIZE) \
                        .execute()

                    if not batch_result.data:
                        break

                    batch_ids = [ann["id"] for ann in batch_result.data]

                    # Update batch
                    supabase_service.client.table("od_annotations").update({
                        "class_id": target_class_id
                    }).in_("id", batch_ids).execute()

                    moved_for_source += len(batch_ids)
                    total_moved += len(batch_ids)

                    update_progress(JobProgress(
                        progress=5 + int((idx / len(source_class_ids)) * 85) + int((moved_for_source / max(source_annotation_count, 1)) * (85 / len(source_class_ids))),
                        current_step=f"Moving annotations from '{source_name}' ({moved_for_source:,}/{source_annotation_count:,})...",
                        processed=total_moved,
                        total=total_annotations,
                    ))

                    # If less than batch size, we're done with this source
                    if len(batch_ids) < self.BATCH_SIZE:
                        break

            except Exception as e:
                error_msg = f"Error moving annotations from '{source_name}': {str(e)}"
                errors.append(error_msg)
                print(f"[ClassMerge] {error_msg}")
                continue

            # Log the merge
            try:
                supabase_service.client.table("od_class_changes").insert({
                    "change_type": "merge",
                    "source_class_ids": [source_id],
                    "target_class_id": target_class_id,
                    "old_name": source_name,
                    "new_name": target.data["name"],
                    "affected_annotation_count": moved_for_source,
                }).execute()
            except Exception as e:
                print(f"[ClassMerge] Failed to log merge: {e}")

            # Delete source class
            try:
                supabase_service.client.table("od_classes").delete() \
                    .eq("id", source_id) \
                    .execute()
                merged_count += 1
                print(f"[ClassMerge] Merged '{source_name}' into '{target.data['name']}' ({moved_for_source:,} annotations)")
            except Exception as e:
                error_msg = f"Failed to delete source class '{source_name}': {str(e)}"
                errors.append(error_msg)
                print(f"[ClassMerge] {error_msg}")

        # Update target class annotation count
        update_progress(JobProgress(
            progress=95,
            current_step="Updating target class count...",
            processed=total_moved,
            total=total_annotations,
        ))

        try:
            new_count = (target.data.get("annotation_count", 0) or 0) + total_moved
            supabase_service.client.table("od_classes").update({
                "annotation_count": new_count
            }).eq("id", target_class_id).execute()
        except Exception as e:
            errors.append(f"Failed to update target count: {str(e)}")

        return {
            "merged_count": merged_count,
            "annotations_moved": total_moved,
            "target_class_id": target_class_id,
            "target_class_name": target.data["name"],
            "errors": errors[:10] if errors else [],
            "message": f"Merged {merged_count} classes, moved {total_moved:,} annotations to '{target.data['name']}'",
        }
