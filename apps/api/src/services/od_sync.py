"""Helpers to keep OD annotation counts and statuses consistent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from services.supabase import supabase_service


ANNOTATED_STATUSES = ("annotated", "completed")


def update_dataset_annotated_image_count(dataset_id: str) -> int:
    """Recalculate annotated_image_count based on status (annotated + completed)."""
    count = (
        supabase_service.client.table("od_dataset_images")
        .select("id", count="exact")
        .eq("dataset_id", dataset_id)
        .in_("status", list(ANNOTATED_STATUSES))
        .execute()
    )
    annotated = count.count or 0
    supabase_service.client.table("od_datasets").update(
        {"annotated_image_count": annotated}
    ).eq("id", dataset_id).execute()
    return annotated


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _rpc_update_annotation_counts_batch(
    dataset_id: str,
    image_ids: Sequence[str],
    class_ids: Sequence[str],
    batch_size: int = 2000,
) -> None:
    if not image_ids:
        return

    unique_class_ids = list({cid for cid in (class_ids or []) if cid})

    for batch in _chunks(list(image_ids), batch_size):
        supabase_service.client.rpc(
            "update_annotation_counts_batch",
            {
                "p_dataset_id": dataset_id,
                "p_image_ids": list(batch),
                "p_class_ids": unique_class_ids,
            },
        ).execute()


def fallback_sync_annotation_counts(
    dataset_id: str,
    image_ids: Sequence[str],
    class_ids: Sequence[str] | None = None,
    progress_cb=None,
) -> dict:
    """
    Slow but safe fallback: per-image counts + status fix.

    Preserves completed/skipped statuses.
    """
    unique_image_ids = list(dict.fromkeys([img_id for img_id in image_ids if img_id]))
    total = len(unique_image_ids)

    if total == 0:
        return {"updated_images": 0, "total_images": 0}

    # Fetch current status/annotation_count in batches
    current = {}
    for batch in _chunks(unique_image_ids, 500):
        result = (
            supabase_service.client.table("od_dataset_images")
            .select("image_id, status, annotation_count")
            .eq("dataset_id", dataset_id)
            .in_("image_id", list(batch))
            .execute()
        )
        for row in result.data or []:
            current[row["image_id"]] = {
                "status": row.get("status") or "pending",
                "annotation_count": row.get("annotation_count") or 0,
            }

    now = datetime.now(timezone.utc).isoformat()
    updated = 0

    for idx, image_id in enumerate(unique_image_ids):
        ann_count_result = (
            supabase_service.client.table("od_annotations")
            .select("id", count="exact")
            .eq("dataset_id", dataset_id)
            .eq("image_id", image_id)
            .execute()
        )
        ann_count = ann_count_result.count or 0

        existing = current.get(image_id, {})
        current_status = existing.get("status") or "pending"
        current_count = existing.get("annotation_count") or 0

        new_status = current_status
        if current_status not in ("completed", "skipped"):
            new_status = "annotated" if ann_count > 0 else "pending"

        update_data = {}
        if ann_count != current_count:
            update_data["annotation_count"] = ann_count
        if new_status != current_status:
            update_data["status"] = new_status
        if ann_count > 0:
            update_data["last_annotated_at"] = now

        if update_data:
            supabase_service.client.table("od_dataset_images").update(
                update_data
            ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
            updated += 1

        if progress_cb and idx % 100 == 0:
            progress_cb(idx, total, "Updating image counts...")

    # Update class annotation counts
    unique_class_ids = list({cid for cid in (class_ids or []) if cid})
    for class_id in unique_class_ids:
        class_count = (
            supabase_service.client.table("od_annotations")
            .select("id", count="exact")
            .eq("dataset_id", dataset_id)
            .eq("class_id", class_id)
            .execute()
        )
        supabase_service.client.table("od_classes").update(
            {"annotation_count": class_count.count or 0}
        ).eq("id", class_id).execute()

    # Update dataset annotation_count
    dataset_ann_count = (
        supabase_service.client.table("od_annotations")
        .select("id", count="exact")
        .eq("dataset_id", dataset_id)
        .execute()
    )
    supabase_service.client.table("od_datasets").update(
        {"annotation_count": dataset_ann_count.count or 0}
    ).eq("id", dataset_id).execute()

    # Update annotated_image_count based on status
    annotated_count = update_dataset_annotated_image_count(dataset_id)

    if progress_cb:
        progress_cb(total, total, "Finalizing dataset counts...")

    return {
        "updated_images": updated,
        "total_images": total,
        "annotated_image_count": annotated_count,
    }


async def sync_annotation_counts_after_write(
    dataset_id: str,
    image_ids: Sequence[str],
    class_ids: Sequence[str] | None = None,
    source: str = "",
) -> dict:
    """
    Preferred path: RPC batch update. Fallback to slow sync or background job.
    """
    unique_image_ids = list(dict.fromkeys([img_id for img_id in image_ids if img_id]))
    unique_class_ids = list({cid for cid in (class_ids or []) if cid})

    if not unique_image_ids:
        return {"method": "noop", "source": source, "total_images": 0}

    try:
        _rpc_update_annotation_counts_batch(
            dataset_id,
            unique_image_ids,
            unique_class_ids,
        )
        # Ensure annotated_image_count follows status semantics
        annotated_count = update_dataset_annotated_image_count(dataset_id)
        return {
            "method": "rpc",
            "source": source,
            "total_images": len(unique_image_ids),
            "annotated_image_count": annotated_count,
        }
    except Exception as e:
        print(f"[OD Sync] RPC failed ({source}): {e}")

    if len(unique_image_ids) <= 1000:
        result = fallback_sync_annotation_counts(
            dataset_id,
            unique_image_ids,
            unique_class_ids,
        )
        result.update({"method": "fallback_sync", "source": source})
        return result

    # Large batch fallback -> local job
    try:
        from services.local_jobs import create_local_job

        job = await create_local_job(
            job_type="local_od_sync_counts",
            config={
                "dataset_id": dataset_id,
                "image_ids": unique_image_ids,
                "class_ids": unique_class_ids,
                "source": source,
            },
        )
        return {
            "method": "fallback_job",
            "source": source,
            "job_id": job.get("id"),
            "total_images": len(unique_image_ids),
        }
    except Exception as job_error:
        print(f"[OD Sync] Failed to create local job ({source}): {job_error}")
        result = fallback_sync_annotation_counts(
            dataset_id,
            unique_image_ids,
            unique_class_ids,
        )
        result.update({"method": "fallback_sync", "source": source})
        return result

