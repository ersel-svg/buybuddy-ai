"""
Classification - Labeling Router

Endpoints for the labeling workflow (annotation page equivalent).
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    LabelingQueueRequest,
    LabelingQueueResponse,
    LabelingImageResponse,
    LabelingProgressResponse,
    CLSLabelCreate,
)

router = APIRouter()


@router.get("/queue/{dataset_id}", response_model=LabelingQueueResponse)
async def get_labeling_queue(
    dataset_id: str,
    mode: str = "unlabeled",
    split: Optional[str] = None,
    class_id: Optional[str] = None,
    limit: int = 100,
):
    """Get queue of images for labeling based on mode.

    Modes:
    - all: All images in order
    - unlabeled: Only images without labels
    - review: Only AI-generated labels needing review
    - random: Random unlabeled images
    - low_confidence: AI labels with low confidence
    """
    query = supabase_service.client.table("cls_dataset_images").select("image_id").eq("dataset_id", dataset_id)

    if split:
        query = query.eq("split", split)

    if mode == "unlabeled":
        query = query.eq("status", "pending")
    elif mode == "review":
        query = query.eq("status", "review")
    elif mode == "completed":
        query = query.in_("status", ["labeled", "completed"])

    # For random mode, we'd ideally use SQL random() but supabase-py doesn't support it directly
    # So we get all and shuffle in Python
    if mode == "random":
        query = query.eq("status", "pending")

    result = query.order("added_at").limit(limit).execute()

    image_ids = [item["image_id"] for item in result.data or []]

    # If filtering by class
    if class_id and mode not in ["unlabeled", "pending"]:
        labels = supabase_service.client.table("cls_labels").select("image_id").eq("dataset_id", dataset_id).eq("class_id", class_id).in_("image_id", image_ids).execute()
        image_ids = [l["image_id"] for l in labels.data or []]

    # For random mode, shuffle
    if mode == "random":
        import random
        random.shuffle(image_ids)

    # Get total count
    count_query = supabase_service.client.table("cls_dataset_images").select("id", count="exact").eq("dataset_id", dataset_id)
    if mode == "unlabeled":
        count_query = count_query.eq("status", "pending")
    elif mode == "review":
        count_query = count_query.eq("status", "review")
    count_result = count_query.execute()

    return LabelingQueueResponse(
        image_ids=image_ids,
        total=count_result.count or 0,
        mode=mode,
    )


@router.get("/image/{dataset_id}/{image_id}", response_model=LabelingImageResponse)
async def get_labeling_image(dataset_id: str, image_id: str):
    """Get a single image for labeling with context."""
    # Get image details
    image_result = supabase_service.client.table("cls_images").select("*").eq("id", image_id).single().execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get dataset image status
    dataset_image = supabase_service.client.table("cls_dataset_images").select("status, split, added_at").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()

    if not dataset_image.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Get current labels
    labels_result = supabase_service.client.table("cls_labels").select(
        "*, class:cls_classes(name, color)"
    ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    labels = []
    for label in labels_result.data or []:
        labels.append({
            **label,
            "class_name": label.get("class", {}).get("name") if label.get("class") else None,
            "class_color": label.get("class", {}).get("color") if label.get("class") else None,
        })

    # Get position in queue (for navigation)
    all_images = supabase_service.client.table("cls_dataset_images").select("image_id").eq("dataset_id", dataset_id).order("added_at").execute()

    image_ids = [item["image_id"] for item in all_images.data or []]
    position = image_ids.index(image_id) + 1 if image_id in image_ids else 0
    total = len(image_ids)

    # Get prev/next
    idx = image_ids.index(image_id) if image_id in image_ids else -1
    prev_id = image_ids[idx - 1] if idx > 0 else None
    next_id = image_ids[idx + 1] if idx < len(image_ids) - 1 else None

    return LabelingImageResponse(
        image=image_result.data,
        current_labels=labels,
        dataset_image_status=dataset_image.data.get("status", "pending"),
        position=position,
        total=total,
        prev_image_id=prev_id,
        next_image_id=next_id,
    )


@router.post("/image/{dataset_id}/{image_id}")
async def save_label(dataset_id: str, image_id: str, data: CLSLabelCreate):
    """Save label for an image during labeling workflow."""
    # Get dataset task type
    dataset = supabase_service.client.table("cls_datasets").select("task_type").eq("id", dataset_id).single().execute()
    task_type = dataset.data.get("task_type", "single_label") if dataset.data else "single_label"

    # For single-label, delete existing labels first
    if task_type == "single_label":
        supabase_service.client.table("cls_labels").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

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

    # Get next image in queue
    all_images = supabase_service.client.table("cls_dataset_images").select("image_id").eq("dataset_id", dataset_id).order("added_at").execute()
    image_ids = [item["image_id"] for item in all_images.data or []]

    idx = image_ids.index(image_id) if image_id in image_ids else -1
    next_id = image_ids[idx + 1] if idx >= 0 and idx < len(image_ids) - 1 else None

    return {
        "success": True,
        "label_id": result.data[0]["id"] if result.data else None,
        "next_image_id": next_id,
    }


@router.post("/skip/{dataset_id}/{image_id}")
async def skip_image(dataset_id: str, image_id: str):
    """Skip an image during labeling workflow."""
    supabase_service.client.table("cls_dataset_images").update({
        "status": "skipped",
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Get next image
    all_images = supabase_service.client.table("cls_dataset_images").select("image_id").eq("dataset_id", dataset_id).order("added_at").execute()
    image_ids = [item["image_id"] for item in all_images.data or []]

    idx = image_ids.index(image_id) if image_id in image_ids else -1
    next_id = image_ids[idx + 1] if idx >= 0 and idx < len(image_ids) - 1 else None

    return {
        "success": True,
        "next_image_id": next_id,
    }


@router.get("/progress/{dataset_id}", response_model=LabelingProgressResponse)
async def get_labeling_progress(dataset_id: str):
    """Get labeling progress for a dataset."""
    result = supabase_service.client.rpc("get_cls_labeling_progress", {"p_dataset_id": dataset_id}).execute()

    if result.data:
        return LabelingProgressResponse(**result.data)

    # Fallback calculation
    images = supabase_service.client.table("cls_dataset_images").select("status").eq("dataset_id", dataset_id).execute()

    counts = {"pending": 0, "labeled": 0, "review": 0, "completed": 0, "skipped": 0}
    for img in images.data or []:
        status = img.get("status", "pending")
        if status in counts:
            counts[status] += 1

    total = sum(counts.values())
    labeled = counts["labeled"] + counts["completed"]

    return LabelingProgressResponse(
        total=total,
        labeled=labeled,
        pending=counts["pending"],
        review=counts["review"],
        completed=counts["completed"],
        skipped=counts["skipped"],
        progress_pct=round((labeled / total * 100) if total > 0 else 0, 1),
    )


@router.post("/mark-reviewed/{dataset_id}/{image_id}")
async def mark_as_reviewed(dataset_id: str, image_id: str):
    """Mark all labels for an image as reviewed."""
    supabase_service.client.table("cls_labels").update({
        "is_reviewed": True,
        "reviewed_at": "now()",
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update dataset image status if it was in review
    supabase_service.client.table("cls_dataset_images").update({
        "status": "completed",
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).eq("status", "review").execute()

    return {"success": True}
