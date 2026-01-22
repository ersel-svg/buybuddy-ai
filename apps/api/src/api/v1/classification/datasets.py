"""
Classification - Datasets Router

Endpoints for managing classification datasets.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSDatasetCreate,
    CLSDatasetUpdate,
    CLSDatasetResponse,
    CLSDatasetImageResponse,
    CLSAddImagesRequest,
    CLSRemoveImagesRequest,
    CLSDatasetVersionCreate,
    CLSDatasetVersionResponse,
    AutoSplitRequest,
    ManualSplitRequest,
    SplitStatsResponse,
)

router = APIRouter()


@router.get("", response_model=list[CLSDatasetResponse])
async def list_datasets(
    search: Optional[str] = None,
):
    """List all classification datasets."""
    query = supabase_service.client.table("cls_datasets").select("*")

    if search:
        query = query.ilike("name", f"%{search}%")

    query = query.order("created_at", desc=True)

    result = query.execute()

    return result.data or []


@router.get("/{dataset_id}", response_model=CLSDatasetResponse)
async def get_dataset(dataset_id: str):
    """Get a single dataset by ID."""
    result = supabase_service.client.table("cls_datasets").select("*").eq("id", dataset_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return result.data


@router.post("", response_model=CLSDatasetResponse)
async def create_dataset(data: CLSDatasetCreate):
    """Create a new classification dataset."""
    dataset_data = data.model_dump(exclude_unset=True)

    # Set defaults
    if "split_ratios" not in dataset_data:
        dataset_data["split_ratios"] = {"train": 0.8, "val": 0.1, "test": 0.1}
    if "preprocessing" not in dataset_data:
        dataset_data["preprocessing"] = {"image_size": 224, "normalize": True}

    result = supabase_service.client.table("cls_datasets").insert(dataset_data).execute()

    return result.data[0]


@router.patch("/{dataset_id}", response_model=CLSDatasetResponse)
async def update_dataset(dataset_id: str, data: CLSDatasetUpdate):
    """Update a dataset."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("cls_datasets").update(update_data).eq("id", dataset_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return result.data[0]


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    # This will cascade delete dataset_images and labels due to FK constraints
    supabase_service.client.table("cls_datasets").delete().eq("id", dataset_id).execute()

    return {"success": True, "message": "Dataset deleted"}


# ===========================================
# Dataset Images
# ===========================================

@router.get("/{dataset_id}/images")
async def list_dataset_images(
    dataset_id: str,
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    split: Optional[str] = None,
    class_id: Optional[str] = None,
    labeled: Optional[bool] = None,
):
    """List images in a dataset with their labels."""
    offset = (page - 1) * limit

    # Get dataset images with image details
    query = supabase_service.client.table("cls_dataset_images").select(
        "*, image:cls_images(*)"
    ).eq("dataset_id", dataset_id)

    if status:
        query = query.eq("status", status)
    if split:
        query = query.eq("split", split)

    query = query.order("added_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    # Get labels for these images
    image_ids = [di["image_id"] for di in result.data or []]

    labels_result = supabase_service.client.table("cls_labels").select(
        "*, class:cls_classes(name, color)"
    ).eq("dataset_id", dataset_id).in_("image_id", image_ids).execute() if image_ids else {"data": []}

    # Group labels by image
    labels_by_image = {}
    for label in labels_result.data or []:
        img_id = label["image_id"]
        if img_id not in labels_by_image:
            labels_by_image[img_id] = []

        label_with_class = {
            **label,
            "class_name": label.get("class", {}).get("name") if label.get("class") else None,
            "class_color": label.get("class", {}).get("color") if label.get("class") else None,
        }
        labels_by_image[img_id].append(label_with_class)

    # Filter by class if specified
    if class_id:
        filtered_data = []
        for di in result.data or []:
            img_labels = labels_by_image.get(di["image_id"], [])
            if any(l["class_id"] == class_id for l in img_labels):
                filtered_data.append(di)
        result.data = filtered_data

    # Filter by labeled status
    if labeled is not None:
        filtered_data = []
        for di in result.data or []:
            has_labels = di["image_id"] in labels_by_image
            if labeled == has_labels:
                filtered_data.append(di)
        result.data = filtered_data

    # Attach labels to response
    dataset_images = []
    for di in result.data or []:
        dataset_images.append({
            **di,
            "labels": labels_by_image.get(di["image_id"], []),
        })

    # Get total count
    count_query = supabase_service.client.table("cls_dataset_images").select("id", count="exact").eq("dataset_id", dataset_id)
    if status:
        count_query = count_query.eq("status", status)
    if split:
        count_query = count_query.eq("split", split)
    count_result = count_query.execute()

    return {
        "images": dataset_images,
        "total": count_result.count or 0,
        "page": page,
        "limit": limit,
    }


@router.post("/{dataset_id}/images/add")
async def add_images_to_dataset(dataset_id: str, data: CLSAddImagesRequest):
    """Add images to a dataset."""
    added = 0
    skipped = 0

    for image_id in data.image_ids:
        # Check if already in dataset
        existing = supabase_service.client.table("cls_dataset_images").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

        if existing.data:
            skipped += 1
            continue

        supabase_service.client.table("cls_dataset_images").insert({
            "dataset_id": dataset_id,
            "image_id": image_id,
            "status": "pending",
        }).execute()
        added += 1

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()

    return {"success": True, "added": added, "skipped": skipped}


@router.post("/{dataset_id}/images/remove")
async def remove_images_from_dataset(dataset_id: str, data: CLSRemoveImagesRequest):
    """Remove images from a dataset."""
    # This also removes labels due to cascade
    for image_id in data.image_ids:
        supabase_service.client.table("cls_dataset_images").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": dataset_id}).execute()

    return {"success": True, "removed": len(data.image_ids)}


# ===========================================
# Split Management
# ===========================================

@router.post("/{dataset_id}/split/auto")
async def auto_split_dataset(dataset_id: str, data: AutoSplitRequest):
    """Automatically split dataset into train/val/test."""
    import random

    # Validate ratios
    total = data.train_ratio + data.val_ratio + data.test_ratio
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail="Split ratios must sum to 1.0")

    # Get all images in dataset
    images_result = supabase_service.client.table("cls_dataset_images").select("id, image_id").eq("dataset_id", dataset_id).execute()
    images = images_result.data or []

    if not images:
        raise HTTPException(status_code=400, detail="Dataset has no images")

    # Shuffle with optional seed
    if data.seed:
        random.seed(data.seed)
    random.shuffle(images)

    # Calculate split sizes
    total_count = len(images)
    train_count = int(total_count * data.train_ratio)
    val_count = int(total_count * data.val_ratio)

    # Assign splits
    train_ids = [img["id"] for img in images[:train_count]]
    val_ids = [img["id"] for img in images[train_count:train_count + val_count]]
    test_ids = [img["id"] for img in images[train_count + val_count:]]

    # Update database
    if train_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "train"}).in_("id", train_ids).execute()
    if val_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "val"}).in_("id", val_ids).execute()
    if test_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "test"}).in_("id", test_ids).execute()

    return {
        "success": True,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
    }


@router.post("/{dataset_id}/split/manual")
async def manual_split_dataset(dataset_id: str, data: ManualSplitRequest):
    """Manually assign splits to images."""
    # Update each split
    if data.train_image_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "train"}).eq("dataset_id", dataset_id).in_("image_id", data.train_image_ids).execute()

    if data.val_image_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "val"}).eq("dataset_id", dataset_id).in_("image_id", data.val_image_ids).execute()

    if data.test_image_ids:
        supabase_service.client.table("cls_dataset_images").update({"split": "test"}).eq("dataset_id", dataset_id).in_("image_id", data.test_image_ids).execute()

    return {
        "success": True,
        "train_count": len(data.train_image_ids),
        "val_count": len(data.val_image_ids),
        "test_count": len(data.test_image_ids),
    }


@router.get("/{dataset_id}/split/stats", response_model=SplitStatsResponse)
async def get_split_stats(dataset_id: str):
    """Get split statistics for a dataset."""
    # Get counts by split
    result = supabase_service.client.table("cls_dataset_images").select("split").eq("dataset_id", dataset_id).execute()

    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    for item in result.data or []:
        split = item.get("split")
        if split in counts:
            counts[split] += 1
        else:
            counts["unassigned"] += 1

    # Get class distribution per split
    class_dist = supabase_service.client.rpc("get_cls_class_distribution", {"p_dataset_id": dataset_id}).execute()

    return SplitStatsResponse(
        train_count=counts["train"],
        val_count=counts["val"],
        test_count=counts["test"],
        unassigned_count=counts["unassigned"],
        class_distribution=class_dist.data if class_dist.data else {},
    )


# ===========================================
# Dataset Versions
# ===========================================

@router.get("/{dataset_id}/versions", response_model=list[CLSDatasetVersionResponse])
async def list_dataset_versions(dataset_id: str):
    """List all versions of a dataset."""
    result = supabase_service.client.table("cls_dataset_versions").select("*").eq("dataset_id", dataset_id).order("version_number", desc=True).execute()

    return result.data or []


@router.post("/{dataset_id}/versions", response_model=CLSDatasetVersionResponse)
async def create_dataset_version(dataset_id: str, data: CLSDatasetVersionCreate):
    """Create a new dataset version (snapshot for training)."""
    # Get current version number
    existing = supabase_service.client.table("cls_dataset_versions").select("version_number").eq("dataset_id", dataset_id).order("version_number", desc=True).limit(1).execute()

    version_number = (existing.data[0]["version_number"] + 1) if existing.data else 1

    # Get dataset images by split
    images_result = supabase_service.client.table("cls_dataset_images").select("image_id, split").eq("dataset_id", dataset_id).execute()

    train_ids = [i["image_id"] for i in images_result.data or [] if i.get("split") == "train"]
    val_ids = [i["image_id"] for i in images_result.data or [] if i.get("split") == "val"]
    test_ids = [i["image_id"] for i in images_result.data or [] if i.get("split") == "test"]

    # Get labels and classes
    labels_result = supabase_service.client.table("cls_labels").select("class_id").eq("dataset_id", dataset_id).execute()
    class_ids = list(set(l["class_id"] for l in labels_result.data or []))

    classes_result = supabase_service.client.table("cls_classes").select("id, name").in_("id", class_ids).execute() if class_ids else {"data": []}

    class_names = [c["name"] for c in classes_result.data or []]
    class_mapping = {c["id"]: i for i, c in enumerate(classes_result.data or [])}

    # Create version
    version_data = {
        "dataset_id": dataset_id,
        "version_number": version_number,
        "name": data.name or f"v{version_number}",
        "description": data.description,
        "image_count": len(train_ids) + len(val_ids) + len(test_ids),
        "labeled_image_count": len(set(l["image_id"] for l in labels_result.data or [] if l.get("image_id"))) if labels_result.data else 0,
        "class_count": len(class_ids),
        "class_mapping": class_mapping,
        "class_names": class_names,
        "split_counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
        "test_image_ids": test_ids,
    }

    result = supabase_service.client.table("cls_dataset_versions").insert(version_data).execute()

    # Update dataset version counter
    supabase_service.client.table("cls_datasets").update({"version": version_number}).eq("id", dataset_id).execute()

    return result.data[0]


@router.get("/{dataset_id}/versions/{version_id}", response_model=CLSDatasetVersionResponse)
async def get_dataset_version(dataset_id: str, version_id: str):
    """Get a specific dataset version."""
    result = supabase_service.client.table("cls_dataset_versions").select("*").eq("id", version_id).eq("dataset_id", dataset_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Version not found")

    return result.data


# ===========================================
# Dataset Health
# ===========================================

@router.get("/{dataset_id}/health")
async def get_dataset_health(dataset_id: str):
    """Get dataset health metrics (class balance, unlabeled images, etc.)."""
    # Get class distribution
    class_dist = supabase_service.client.rpc("get_cls_class_distribution", {"p_dataset_id": dataset_id}).execute()

    classes = class_dist.data if class_dist.data else []

    # Calculate imbalance
    counts = [c.get("image_count", 0) for c in classes]
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # Get labeling progress
    progress = supabase_service.client.rpc("get_cls_labeling_progress", {"p_dataset_id": dataset_id}).execute()

    return {
        "class_count": len(classes),
        "class_distribution": classes,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "max_class_count": max_count,
        "min_class_count": min_count,
        "labeling_progress": progress.data if progress.data else {},
        "warnings": _get_health_warnings(classes, progress.data if progress.data else {}),
    }


def _get_health_warnings(classes: list, progress: dict) -> list[str]:
    """Generate health warnings for dataset."""
    warnings = []

    counts = [c.get("image_count", 0) for c in classes]
    if counts:
        max_count = max(counts)
        min_count = min(counts)

        if min_count > 0 and max_count / min_count > 10:
            warnings.append(f"High class imbalance: {max_count}:{min_count} ratio. Consider oversampling or class weights.")

        if min_count < 10:
            warnings.append(f"Some classes have very few images (min: {min_count}). Consider adding more samples.")

    labeled_pct = progress.get("progress_pct", 0)
    if labeled_pct < 100:
        unlabeled = progress.get("pending", 0)
        warnings.append(f"{unlabeled} images are unlabeled ({100 - labeled_pct:.1f}% remaining).")

    return warnings
