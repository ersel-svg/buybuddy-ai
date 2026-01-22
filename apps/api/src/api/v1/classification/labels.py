"""
Classification - Labels Router

Endpoints for managing image-class labels.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSLabelCreate,
    CLSLabelUpdate,
    CLSLabelResponse,
    CLSBulkLabelRequest,
    CLSBulkClearLabelsRequest,
)

router = APIRouter()


@router.get("/datasets/{dataset_id}")
async def list_labels_in_dataset(
    dataset_id: str,
    page: int = 1,
    limit: int = 100,
    class_id: Optional[str] = None,
    is_ai_generated: Optional[bool] = None,
    is_reviewed: Optional[bool] = None,
):
    """List all labels in a dataset."""
    offset = (page - 1) * limit

    query = supabase_service.client.table("cls_labels").select(
        "*, class:cls_classes(name, color)",
        count="exact"
    ).eq("dataset_id", dataset_id)

    if class_id:
        query = query.eq("class_id", class_id)
    if is_ai_generated is not None:
        query = query.eq("is_ai_generated", is_ai_generated)
    if is_reviewed is not None:
        query = query.eq("is_reviewed", is_reviewed)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    # Format response with class info
    labels = []
    for label in result.data or []:
        labels.append({
            **label,
            "class_name": label.get("class", {}).get("name") if label.get("class") else None,
            "class_color": label.get("class", {}).get("color") if label.get("class") else None,
        })

    return {
        "labels": labels,
        "total": result.count or 0,
        "page": page,
        "limit": limit,
    }


@router.get("/datasets/{dataset_id}/images/{image_id}")
async def get_labels_for_image(dataset_id: str, image_id: str):
    """Get all labels for a specific image in a dataset."""
    result = supabase_service.client.table("cls_labels").select(
        "*, class:cls_classes(name, color)"
    ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    labels = []
    for label in result.data or []:
        labels.append({
            **label,
            "class_name": label.get("class", {}).get("name") if label.get("class") else None,
            "class_color": label.get("class", {}).get("color") if label.get("class") else None,
        })

    return labels


@router.post("/datasets/{dataset_id}/images/{image_id}", response_model=CLSLabelResponse)
async def set_label_for_image(dataset_id: str, image_id: str, data: CLSLabelCreate):
    """Set a label for an image.

    For single-label datasets, this replaces any existing label.
    For multi-label datasets, this adds a new label.
    """
    # Get dataset to check task_type
    dataset = supabase_service.client.table("cls_datasets").select("task_type").eq("id", dataset_id).single().execute()

    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    task_type = dataset.data.get("task_type", "single_label")

    # For single-label, delete existing labels first
    if task_type == "single_label":
        supabase_service.client.table("cls_labels").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Check if this class already assigned (for multi-label)
    if task_type == "multi_label":
        existing = supabase_service.client.table("cls_labels").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).eq("class_id", data.class_id).execute()

        if existing.data:
            raise HTTPException(status_code=409, detail="This class is already assigned to this image")

    # Create label
    label_data = {
        "dataset_id": dataset_id,
        "image_id": image_id,
        **data.model_dump(),
    }

    result = supabase_service.client.table("cls_labels").insert(label_data).execute()

    # Update dataset image status
    supabase_service.client.table("cls_dataset_images").update({
        "status": "labeled",
        "labeled_at": "now()",
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()
    supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": data.class_id}).execute()

    # Get class info for response
    class_info = supabase_service.client.table("cls_classes").select("name, color").eq("id", data.class_id).single().execute()

    return {
        **result.data[0],
        "class_name": class_info.data.get("name") if class_info.data else None,
        "class_color": class_info.data.get("color") if class_info.data else None,
    }


@router.delete("/datasets/{dataset_id}/images/{image_id}")
async def clear_labels_for_image(dataset_id: str, image_id: str):
    """Clear all labels for an image."""
    # Get class IDs before deleting (for stats update)
    labels = supabase_service.client.table("cls_labels").select("class_id").eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
    class_ids = [l["class_id"] for l in labels.data or []]

    # Delete labels
    supabase_service.client.table("cls_labels").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update dataset image status
    supabase_service.client.table("cls_dataset_images").update({
        "status": "pending",
        "labeled_at": None,
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()

    for class_id in class_ids:
        supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": class_id}).execute()

    return {"success": True, "cleared": len(class_ids)}


@router.delete("/{label_id}")
async def delete_label(label_id: str):
    """Delete a specific label."""
    # Get label info
    label = supabase_service.client.table("cls_labels").select("dataset_id, image_id, class_id").eq("id", label_id).single().execute()

    if not label.data:
        raise HTTPException(status_code=404, detail="Label not found")

    dataset_id = label.data["dataset_id"]
    image_id = label.data["image_id"]
    class_id = label.data["class_id"]

    # Delete label
    supabase_service.client.table("cls_labels").delete().eq("id", label_id).execute()

    # Check if image has other labels
    remaining = supabase_service.client.table("cls_labels").select("id", count="exact").eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    if not remaining.count or remaining.count == 0:
        # No more labels, update status
        supabase_service.client.table("cls_dataset_images").update({
            "status": "pending",
            "labeled_at": None,
        }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()
    supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": class_id}).execute()

    return {"success": True}


@router.patch("/{label_id}", response_model=CLSLabelResponse)
async def update_label(label_id: str, data: CLSLabelUpdate):
    """Update a label (e.g., mark as reviewed or change class)."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # If marking as reviewed, set reviewed_at
    if update_data.get("is_reviewed"):
        update_data["reviewed_at"] = "now()"

    result = supabase_service.client.table("cls_labels").update(update_data).eq("id", label_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Label not found")

    return result.data[0]


# ===========================================
# Bulk Operations
# ===========================================

@router.post("/datasets/{dataset_id}/bulk")
async def bulk_set_labels(dataset_id: str, data: CLSBulkLabelRequest):
    """Set the same label for multiple images."""
    created = 0
    updated = 0
    errors = []

    # Get dataset task type
    dataset = supabase_service.client.table("cls_datasets").select("task_type").eq("id", dataset_id).single().execute()
    task_type = dataset.data.get("task_type", "single_label") if dataset.data else "single_label"

    for image_id in data.image_ids:
        try:
            if task_type == "single_label":
                # Check if label exists
                existing = supabase_service.client.table("cls_labels").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

                if existing.data:
                    # Update existing
                    supabase_service.client.table("cls_labels").update({
                        "class_id": data.class_id,
                    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
                    updated += 1
                else:
                    # Create new
                    supabase_service.client.table("cls_labels").insert({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "class_id": data.class_id,
                    }).execute()
                    created += 1
            else:
                # Multi-label: add if not exists
                existing = supabase_service.client.table("cls_labels").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).eq("class_id", data.class_id).execute()

                if not existing.data:
                    supabase_service.client.table("cls_labels").insert({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "class_id": data.class_id,
                    }).execute()
                    created += 1

            # Update dataset image status
            supabase_service.client.table("cls_dataset_images").update({
                "status": "labeled",
                "labeled_at": "now()",
            }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

        except Exception as e:
            errors.append(f"{image_id}: {str(e)}")

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()
    supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": data.class_id}).execute()

    return {
        "success": True,
        "created": created,
        "updated": updated,
        "errors": errors,
    }


@router.post("/datasets/{dataset_id}/bulk-clear")
async def bulk_clear_labels(dataset_id: str, data: CLSBulkClearLabelsRequest):
    """Clear labels for multiple images."""
    # Get class IDs for stats update
    labels = supabase_service.client.table("cls_labels").select("class_id").eq("dataset_id", dataset_id).in_("image_id", data.image_ids).execute()
    class_ids = list(set(l["class_id"] for l in labels.data or []))

    # Delete labels
    supabase_service.client.table("cls_labels").delete().eq("dataset_id", dataset_id).in_("image_id", data.image_ids).execute()

    # Update dataset image statuses
    supabase_service.client.table("cls_dataset_images").update({
        "status": "pending",
        "labeled_at": None,
    }).eq("dataset_id", dataset_id).in_("image_id", data.image_ids).execute()

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()

    for class_id in class_ids:
        supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": class_id}).execute()

    return {"success": True, "cleared": len(data.image_ids)}
