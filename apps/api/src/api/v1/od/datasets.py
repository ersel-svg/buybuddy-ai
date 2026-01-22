"""
Object Detection - Datasets Router

Endpoints for managing OD datasets.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException

from services.supabase import supabase_service
from schemas.od import (
    ODDatasetCreate,
    ODDatasetUpdate,
    ODDatasetResponse,
    ODDatasetWithImagesResponse,
    ODAddImagesRequest,
    ExportRequest,
    ExportJobResponse,
    DatasetVersionCreate,
    DatasetVersionResponse,
)

router = APIRouter()


@router.get("", response_model=list[ODDatasetResponse])
async def list_datasets():
    """List all datasets."""
    result = supabase_service.client.table("od_datasets").select("*").order("created_at", desc=True).execute()
    return result.data or []


@router.get("/{dataset_id}", response_model=ODDatasetResponse)
async def get_dataset(dataset_id: str):
    """Get a single dataset by ID."""
    result = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return result.data


@router.get("/{dataset_id}/images")
async def get_dataset_images(
    dataset_id: str,
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    split: Optional[str] = None,
):
    """Get images in a dataset with their annotation counts."""
    offset = (page - 1) * limit

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build query for dataset_images with joined image data
    query = supabase_service.client.table("od_dataset_images").select(
        "*, image:od_images(*)",
        count="exact"
    ).eq("dataset_id", dataset_id)

    if status:
        query = query.eq("status", status)
    if split:
        query = query.eq("split", split)

    query = query.order("added_at", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    return {
        "images": result.data or [],
        "total": result.count or 0,
        "page": page,
        "limit": limit,
    }


@router.post("", response_model=ODDatasetResponse)
async def create_dataset(data: ODDatasetCreate):
    """Create a new dataset and copy template classes to it."""
    dataset_data = data.model_dump()
    result = supabase_service.client.table("od_datasets").insert(dataset_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create dataset")

    dataset_id = result.data[0]["id"]

    # Copy all template classes (those with NULL dataset_id) to this new dataset
    templates = supabase_service.client.table("od_classes").select("*").is_("dataset_id", "null").execute()

    if templates.data:
        new_classes = []
        for template in templates.data:
            new_classes.append({
                "dataset_id": dataset_id,
                "name": template["name"],
                "display_name": template.get("display_name"),
                "description": template.get("description"),
                "color": template["color"],
                "category": template.get("category"),
                "aliases": template.get("aliases"),
                "is_active": True,
                "is_system": template.get("is_system", False),
                "annotation_count": 0,
            })

        if new_classes:
            supabase_service.client.table("od_classes").insert(new_classes).execute()

            # Update dataset class_count
            supabase_service.client.table("od_datasets").update({
                "class_count": len(new_classes)
            }).eq("id", dataset_id).execute()

            result.data[0]["class_count"] = len(new_classes)

    return result.data[0]


@router.patch("/{dataset_id}", response_model=ODDatasetResponse)
async def update_dataset(dataset_id: str, data: ODDatasetUpdate):
    """Update a dataset."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("od_datasets").update(update_data).eq("id", dataset_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return result.data[0]


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset and all its annotations."""
    # Verify exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete dataset (cascade will delete dataset_images and annotations)
    supabase_service.client.table("od_datasets").delete().eq("id", dataset_id).execute()

    return {"status": "deleted", "id": dataset_id}


@router.post("/{dataset_id}/images")
async def add_images_to_dataset(dataset_id: str, data: ODAddImagesRequest):
    """Add images to a dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify all images exist
    images = supabase_service.client.table("od_images").select("id").in_("id", data.image_ids).execute()
    if len(images.data or []) != len(data.image_ids):
        raise HTTPException(status_code=400, detail="One or more images not found")

    # Check which images are already in the dataset
    existing = supabase_service.client.table("od_dataset_images").select("image_id").eq("dataset_id", dataset_id).in_("image_id", data.image_ids).execute()
    existing_ids = set(item["image_id"] for item in existing.data or [])

    # Add new images
    new_image_ids = [img_id for img_id in data.image_ids if img_id not in existing_ids]

    if not new_image_ids:
        return {"added": 0, "skipped": len(data.image_ids)}

    # Insert dataset_images records
    records = [{"dataset_id": dataset_id, "image_id": img_id} for img_id in new_image_ids]
    supabase_service.client.table("od_dataset_images").insert(records).execute()

    # Update dataset image count
    new_count = (dataset.data.get("image_count", 0) or 0) + len(new_image_ids)
    supabase_service.client.table("od_datasets").update({"image_count": new_count}).eq("id", dataset_id).execute()

    return {"added": len(new_image_ids), "skipped": len(existing_ids)}


@router.delete("/{dataset_id}/images/{image_id}")
async def remove_image_from_dataset(dataset_id: str, image_id: str, delete_completely: bool = True):
    """
    Remove an image from a dataset.

    If delete_completely=True (default), also deletes:
    - Annotations for this image
    - The image record from od_images
    - The image file from storage

    If delete_completely=False, only removes the dataset link (image stays in system).
    """
    # Get dataset_image record
    di = supabase_service.client.table("od_dataset_images").select("*").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()

    if not di.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Get image data for storage deletion
    image_data = None
    if delete_completely:
        image_result = supabase_service.client.table("od_images").select("filename").eq("id", image_id).single().execute()
        image_data = image_result.data

    # Delete annotations for this image in this dataset
    supabase_service.client.table("od_annotations").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Delete dataset_image record
    supabase_service.client.table("od_dataset_images").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # If delete_completely, also delete the image itself
    if delete_completely:
        # Check if image is used in other datasets
        other_datasets = supabase_service.client.table("od_dataset_images").select("id").eq("image_id", image_id).limit(1).execute()

        if not other_datasets.data:
            # Not used elsewhere, safe to delete completely
            # Delete from storage
            if image_data and image_data.get("filename"):
                try:
                    supabase_service.client.storage.from_("od-images").remove([image_data["filename"]])
                except Exception:
                    pass  # Continue even if storage delete fails

            # Delete image record
            supabase_service.client.table("od_images").delete().eq("id", image_id).execute()

    # Update dataset counts
    dataset = supabase_service.client.table("od_datasets").select("image_count, annotation_count").eq("id", dataset_id).single().execute()
    if dataset.data:
        new_image_count = max(0, (dataset.data.get("image_count", 0) or 0) - 1)
        new_annotation_count = max(0, (dataset.data.get("annotation_count", 0) or 0) - (di.data.get("annotation_count", 0) or 0))
        supabase_service.client.table("od_datasets").update({
            "image_count": new_image_count,
            "annotation_count": new_annotation_count,
        }).eq("id", dataset_id).execute()

    return {"status": "removed", "dataset_id": dataset_id, "image_id": image_id, "deleted_completely": delete_completely}


@router.post("/{dataset_id}/images/bulk-remove")
async def remove_images_bulk(dataset_id: str, image_ids: list[str], delete_completely: bool = True):
    """
    Remove multiple images from a dataset.

    If delete_completely=True (default), also deletes:
    - Annotations for these images
    - Image records from od_images (if not used in other datasets)
    - Image files from storage
    """
    removed = 0
    deleted_from_storage = 0

    # Get all image filenames for storage deletion
    images_data = {}
    if delete_completely:
        images_result = supabase_service.client.table("od_images").select("id, filename").in_("id", image_ids).execute()
        images_data = {img["id"]: img["filename"] for img in (images_result.data or [])}

    for image_id in image_ids:
        try:
            # Delete annotations
            supabase_service.client.table("od_annotations").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
            # Delete dataset_image
            result = supabase_service.client.table("od_dataset_images").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
            if result.data:
                removed += 1

                # If delete_completely, also delete the image itself
                if delete_completely:
                    # Check if image is used in other datasets
                    other_datasets = supabase_service.client.table("od_dataset_images").select("id").eq("image_id", image_id).limit(1).execute()

                    if not other_datasets.data:
                        # Not used elsewhere, safe to delete completely
                        filename = images_data.get(image_id)
                        if filename:
                            try:
                                supabase_service.client.storage.from_("od-images").remove([filename])
                                deleted_from_storage += 1
                            except Exception:
                                pass

                        # Delete image record
                        supabase_service.client.table("od_images").delete().eq("id", image_id).execute()
        except Exception:
            continue

    # Update dataset count
    count_result = supabase_service.client.table("od_dataset_images").select("id", count="exact").eq("dataset_id", dataset_id).execute()
    new_count = count_result.count or 0
    supabase_service.client.table("od_datasets").update({"image_count": new_count}).eq("id", dataset_id).execute()

    return {"removed": removed, "deleted_from_storage": deleted_from_storage}


@router.get("/{dataset_id}/classes")
async def get_dataset_classes(dataset_id: str, include_templates: bool = False):
    """Get all classes for a specific dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get classes for this dataset
    query = supabase_service.client.table("od_classes").select("*")

    if include_templates:
        # Include both dataset classes and templates (for UI showing all available)
        query = query.or_(f"dataset_id.eq.{dataset_id},dataset_id.is.null")
    else:
        query = query.eq("dataset_id", dataset_id)

    query = query.eq("is_active", True).order("name")
    result = query.execute()

    return result.data or []


@router.post("/{dataset_id}/classes")
async def add_class_to_dataset(dataset_id: str, data: dict):
    """Create a new class for a specific dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if name already exists in this dataset
    existing = supabase_service.client.table("od_classes").select("id").eq("name", data["name"]).eq("dataset_id", dataset_id).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail=f"Class '{data['name']}' already exists in this dataset")

    # Create the class
    class_data = {
        "dataset_id": dataset_id,
        "name": data["name"],
        "display_name": data.get("display_name"),
        "color": data.get("color", "#3B82F6"),
        "category": data.get("category"),
        "is_active": True,
        "annotation_count": 0,
    }

    result = supabase_service.client.table("od_classes").insert(class_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create class")

    # Update dataset class_count
    count_result = supabase_service.client.table("od_classes").select("id", count="exact").eq("dataset_id", dataset_id).execute()
    supabase_service.client.table("od_datasets").update({"class_count": count_result.count or 0}).eq("id", dataset_id).execute()

    return result.data[0]


@router.get("/{dataset_id}/stats")
async def get_dataset_stats(dataset_id: str):
    """Get detailed statistics for a dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get image status counts
    status_result = supabase_service.client.table("od_dataset_images").select("status").eq("dataset_id", dataset_id).execute()
    status_counts = {"pending": 0, "annotating": 0, "completed": 0, "skipped": 0}
    for item in status_result.data or []:
        status = item.get("status", "pending")
        if status in status_counts:
            status_counts[status] += 1

    # Get annotation count by class
    class_result = supabase_service.client.rpc("get_od_dataset_class_stats", {"p_dataset_id": dataset_id}).execute()

    # Fallback if RPC doesn't exist - just count annotations
    annotation_count = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).execute()

    return {
        "dataset": dataset.data,
        "images_by_status": status_counts,
        "total_annotations": annotation_count.count or 0,
        "class_distribution": class_result.data if class_result.data else [],
    }


@router.patch("/{dataset_id}/images/{image_id}/status")
async def update_image_status(dataset_id: str, image_id: str, status: str):
    """Update the status of an image in a dataset."""
    valid_statuses = ["pending", "annotating", "completed", "skipped"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    update_data = {"status": status}
    if status == "completed":
        from datetime import datetime, timezone
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    result = supabase_service.client.table("od_dataset_images").update(update_data).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Update dataset annotated_image_count if completed
    if status == "completed":
        count = supabase_service.client.table("od_dataset_images").select("id", count="exact").eq("dataset_id", dataset_id).eq("status", "completed").execute()
        supabase_service.client.table("od_datasets").update({"annotated_image_count": count.count or 0}).eq("id", dataset_id).execute()

    return result.data[0]


@router.post("/{dataset_id}/images/bulk-status")
async def update_images_status_bulk(dataset_id: str, image_ids: list[str], status: str):
    """
    Update the status of multiple images in a dataset at once.

    Args:
        dataset_id: The dataset ID
        image_ids: List of image IDs to update
        status: New status (pending, annotating, completed, skipped)

    Returns:
        Count of updated images
    """
    from datetime import datetime, timezone

    valid_statuses = ["pending", "annotating", "completed", "skipped"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    if not image_ids:
        return {"updated": 0, "message": "No image IDs provided"}

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = {"status": status}
    if status == "completed":
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Bulk update all images in a single query
    result = supabase_service.client.table("od_dataset_images").update(
        update_data
    ).eq("dataset_id", dataset_id).in_("image_id", image_ids).execute()

    updated_count = len(result.data) if result.data else 0

    # Update dataset annotated_image_count
    count = supabase_service.client.table("od_dataset_images").select(
        "id", count="exact"
    ).eq("dataset_id", dataset_id).eq("status", "completed").execute()
    supabase_service.client.table("od_datasets").update({
        "annotated_image_count": count.count or 0
    }).eq("id", dataset_id).execute()

    return {
        "updated": updated_count,
        "status": status,
        "message": f"Updated {updated_count} images to '{status}'"
    }


@router.post("/{dataset_id}/images/bulk-status-by-filter")
async def update_images_status_by_filter(
    dataset_id: str,
    new_status: str,
    current_status: Optional[str] = None,
    has_annotations: Optional[bool] = None,
):
    """
    Update status of images matching filter criteria.

    This is useful for bulk operations like:
    - Mark all images with annotations as "completed"
    - Mark all "pending" images as "skipped"

    Args:
        dataset_id: The dataset ID
        new_status: New status to set (pending, annotating, completed, skipped)
        current_status: Only update images with this current status
        has_annotations: If True, only update images that have annotations; if False, only without

    Returns:
        Count of updated images
    """
    from datetime import datetime, timezone

    valid_statuses = ["pending", "annotating", "completed", "skipped"]
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build query to find matching images
    query = supabase_service.client.table("od_dataset_images").select("image_id, annotation_count").eq("dataset_id", dataset_id)

    if current_status:
        query = query.eq("status", current_status)

    result = query.execute()
    all_images = result.data or []

    # Filter by has_annotations if specified
    if has_annotations is not None:
        if has_annotations:
            # Images with annotation_count > 0
            all_images = [img for img in all_images if (img.get("annotation_count") or 0) > 0]
        else:
            # Images with annotation_count == 0 or None
            all_images = [img for img in all_images if (img.get("annotation_count") or 0) == 0]

    if not all_images:
        return {"updated": 0, "message": "No images match the filter criteria"}

    image_ids = [img["image_id"] for img in all_images]

    # Prepare update data
    update_data = {"status": new_status}
    if new_status == "completed":
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Bulk update
    supabase_service.client.table("od_dataset_images").update(
        update_data
    ).eq("dataset_id", dataset_id).in_("image_id", image_ids).execute()

    # Update dataset annotated_image_count
    count = supabase_service.client.table("od_dataset_images").select(
        "id", count="exact"
    ).eq("dataset_id", dataset_id).eq("status", "completed").execute()
    supabase_service.client.table("od_datasets").update({
        "annotated_image_count": count.count or 0
    }).eq("id", dataset_id).execute()

    return {
        "updated": len(image_ids),
        "status": new_status,
        "message": f"Updated {len(image_ids)} images to '{new_status}'"
    }


@router.post("/{dataset_id}/recalculate-counts")
async def recalculate_dataset_counts_endpoint(dataset_id: str):
    """
    Recalculate all denormalized counts for a dataset.

    This fixes counts that may be out of sync after imports or other operations.
    Updates: image_count, annotation_count, annotated_image_count
    """
    from services.roboflow_streaming import recalculate_dataset_counts

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id, name").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    counts = recalculate_dataset_counts(dataset_id)

    return {
        "success": True,
        "dataset_id": dataset_id,
        "dataset_name": dataset.data.get("name"),
        "counts": counts,
    }


# ===========================================
# Export Endpoints
# ===========================================

@router.post("/{dataset_id}/export", response_model=ExportJobResponse)
async def export_dataset(dataset_id: str, data: ExportRequest):
    """
    Export a dataset in the specified format.

    Supported formats:
    - yolo: YOLO format with data.yaml and label files
    - coco: COCO format with annotations JSON

    Returns a job with download URL when complete.
    """
    from services.od_export import od_export_service

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate format
    if data.format not in ["yolo", "coco"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: yolo, coco")

    # Create and process export job
    config = data.config.model_dump() if data.config else None

    job = await od_export_service.create_export_job(
        dataset_id=dataset_id,
        format=data.format,
        include_images=data.include_images,
        version_id=data.version_id,
        split=data.split,
        config=config,
    )

    return ExportJobResponse(
        job_id=job["id"],
        status=job["status"],
        download_url=job.get("download_url"),
        progress=job.get("progress", 0),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at"),
    )


@router.get("/{dataset_id}/export/{job_id}")
async def get_export_status(dataset_id: str, job_id: str):
    """Get the status of an export job."""
    # For now, exports are synchronous so this just returns not found
    # In async version, this would check job status in database
    raise HTTPException(status_code=404, detail="Export job not found or already completed")


# ===========================================
# Dataset Versioning Endpoints
# ===========================================

@router.get("/{dataset_id}/versions", response_model=list[DatasetVersionResponse])
async def list_versions(dataset_id: str):
    """List all versions of a dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = supabase_service.client.table("od_dataset_versions").select("*").eq("dataset_id", dataset_id).order("version_number", desc=True).execute()

    return result.data or []


@router.post("/{dataset_id}/versions", response_model=DatasetVersionResponse)
async def create_version(dataset_id: str, data: DatasetVersionCreate):
    """
    Create a new version (snapshot) of the dataset.

    This creates a frozen copy of the current dataset state with
    train/val/test split assignments.
    """
    import random
    from datetime import datetime, timezone
    from uuid import uuid4

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get current version number
    latest = supabase_service.client.table("od_dataset_versions").select("version_number").eq("dataset_id", dataset_id).order("version_number", desc=True).limit(1).execute()
    next_version = (latest.data[0]["version_number"] + 1) if latest.data else 1

    # Get dataset images
    images = supabase_service.client.table("od_dataset_images").select("id, image_id, status").eq("dataset_id", dataset_id).execute()
    annotated_images = [img for img in (images.data or []) if img.get("status") in ["completed", "annotating"]]

    if not annotated_images:
        raise HTTPException(status_code=400, detail="No annotated images in dataset")

    # Get classes
    classes = supabase_service.client.table("od_classes").select("id, name").eq("dataset_id", dataset_id).eq("is_active", True).execute()
    class_mapping = {cls["id"]: idx for idx, cls in enumerate(classes.data or [])}

    # Get annotation count
    ann_count = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).execute()

    # Assign splits
    random.seed(42)
    random.shuffle(annotated_images)

    total = len(annotated_images)
    train_end = int(total * data.train_split)
    val_end = train_end + int(total * data.val_split)

    train_ids = [img["image_id"] for img in annotated_images[:train_end]]
    val_ids = [img["image_id"] for img in annotated_images[train_end:val_end]]
    test_ids = [img["image_id"] for img in annotated_images[val_end:]]

    # Update split assignments in od_dataset_images
    if train_ids:
        supabase_service.client.table("od_dataset_images").update({"split": "train"}).eq("dataset_id", dataset_id).in_("image_id", train_ids).execute()
    if val_ids:
        supabase_service.client.table("od_dataset_images").update({"split": "val"}).eq("dataset_id", dataset_id).in_("image_id", val_ids).execute()
    if test_ids:
        supabase_service.client.table("od_dataset_images").update({"split": "test"}).eq("dataset_id", dataset_id).in_("image_id", test_ids).execute()

    # Create version record
    version_id = str(uuid4())
    version_name = data.name or f"v{next_version}"

    version_data = {
        "id": version_id,
        "dataset_id": dataset_id,
        "version_number": next_version,
        "name": version_name,
        "description": data.description,
        "image_count": len(annotated_images),
        "annotation_count": ann_count.count or 0,
        "class_count": len(classes.data or []),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "class_mapping": class_mapping,
        "split_config": {
            "train_split": data.train_split,
            "val_split": data.val_split,
            "test_split": data.test_split,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("od_dataset_versions").insert(version_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create version")

    # Update dataset version number
    supabase_service.client.table("od_datasets").update({"version": next_version}).eq("id", dataset_id).execute()

    return result.data[0]


@router.get("/{dataset_id}/versions/{version_id}", response_model=DatasetVersionResponse)
async def get_version(dataset_id: str, version_id: str):
    """Get a specific version of a dataset."""
    result = supabase_service.client.table("od_dataset_versions").select("*").eq("dataset_id", dataset_id).eq("id", version_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Version not found")

    return result.data


@router.delete("/{dataset_id}/versions/{version_id}")
async def delete_version(dataset_id: str, version_id: str):
    """Delete a dataset version."""
    # Verify exists
    version = supabase_service.client.table("od_dataset_versions").select("id").eq("dataset_id", dataset_id).eq("id", version_id).single().execute()
    if not version.data:
        raise HTTPException(status_code=404, detail="Version not found")

    supabase_service.client.table("od_dataset_versions").delete().eq("id", version_id).execute()

    return {"status": "deleted", "id": version_id}
