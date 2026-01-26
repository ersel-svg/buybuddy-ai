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
    merchant_ids: Optional[str] = None,
    store_ids: Optional[str] = None,
    sort_by: Optional[str] = None,  # "recently_annotated" or None (default: added_at)
):
    """Get images in a dataset with their annotation counts.

    Args:
        merchant_ids: Comma-separated merchant IDs to filter by
        store_ids: Comma-separated store IDs to filter by
        sort_by: Sort order - "recently_annotated" to sort by last annotation time
    """
    offset = (page - 1) * limit

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # If filtering by merchant/store, first get the matching image IDs
    filtered_image_ids = None
    if merchant_ids or store_ids:
        img_query = supabase_service.client.table("od_images").select("id")

        if merchant_ids:
            merchant_list = [int(m.strip()) for m in merchant_ids.split(",") if m.strip().isdigit()]
            if merchant_list:
                img_query = img_query.in_("merchant_id", merchant_list)

        if store_ids:
            store_list = [int(s.strip()) for s in store_ids.split(",") if s.strip().isdigit()]
            if store_list:
                img_query = img_query.in_("store_id", store_list)

        img_result = img_query.execute()
        filtered_image_ids = [r["id"] for r in (img_result.data or [])]

        if not filtered_image_ids:
            # No images match the filter, return empty
            return {"images": [], "total": 0, "page": page, "limit": limit}

    # Build query for dataset_images with joined image data
    query = supabase_service.client.table("od_dataset_images").select(
        "*, image:od_images(*)",
        count="exact"
    ).eq("dataset_id", dataset_id)

    if status:
        query = query.eq("status", status)
    if split:
        query = query.eq("split", split)

    # Apply image ID filter if merchant/store filter was used
    if filtered_image_ids is not None:
        # Batch the IN clause to avoid URL length limits
        BATCH_SIZE = 100
        if len(filtered_image_ids) <= BATCH_SIZE:
            query = query.in_("image_id", filtered_image_ids)
        else:
            # For large filters, we need to fetch all and filter in Python
            # This is less efficient but avoids URL limits
            all_results = []
            filtered_set = set(filtered_image_ids)

            # Fetch without image_id filter
            base_query = supabase_service.client.table("od_dataset_images").select(
                "*, image:od_images(*)"
            ).eq("dataset_id", dataset_id)
            if status:
                base_query = base_query.eq("status", status)
            if split:
                base_query = base_query.eq("split", split)

            all_data = base_query.execute()
            all_results = [r for r in (all_data.data or []) if r.get("image_id") in filtered_set]

            # Manual pagination
            total = len(all_results)
            paginated = all_results[offset:offset + limit]

            return {
                "images": paginated,
                "total": total,
                "page": page,
                "limit": limit,
            }

    # Apply sort order
    if sort_by == "recently_annotated":
        # Sort by last_annotated_at descending, with nulls last
        query = query.order("last_annotated_at", desc=True, nullsfirst=False)
    else:
        # Default: sort by added_at descending
        query = query.order("added_at", desc=True)

    query = query.range(offset, offset + limit - 1)
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


@router.post("/{dataset_id}/images/bulk-remove/async")
async def remove_images_bulk_async(
    dataset_id: str,
    image_ids: list[str] | None = None,
    statuses: str | None = None,
    has_annotations: bool | None = None,
    delete_completely: bool = True,
):
    """
    Async version: Remove multiple images from a dataset as a background job.

    Use this for large removals (>50 images). Supports:
    - Explicit image_ids list
    - Filter-based removal via statuses/has_annotations

    Args:
        dataset_id: Target dataset ID
        image_ids: Specific image IDs to remove (optional)
        statuses: Comma-separated statuses to filter (optional)
        has_annotations: Filter by annotation presence (optional)
        delete_completely: If True, delete images from system entirely (default: True)

    Returns:
        Job ID for tracking progress
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets")\
        .select("id, name")\
        .eq("id", dataset_id)\
        .single()\
        .execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Need either image_ids or filters
    if not image_ids and not statuses and has_annotations is None:
        raise HTTPException(status_code=400, detail="Provide image_ids or filter criteria")

    # Build config
    config = {
        "dataset_id": dataset_id,
        "delete_completely": delete_completely,
    }

    if image_ids:
        config["image_ids"] = image_ids
    else:
        config["filters"] = {}
        if statuses:
            config["filters"]["statuses"] = statuses
        if has_annotations is not None:
            config["filters"]["has_annotations"] = has_annotations

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_bulk_remove_from_dataset",
        "status": "pending",
        "progress": 0,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk remove job queued for dataset '{dataset.data['name']}'",
    }


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
    """Get detailed statistics for a dataset (training wizard compatible)."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = dataset.data

    # Get image status counts
    status_result = supabase_service.client.table("od_dataset_images").select("status").eq("dataset_id", dataset_id).execute()
    status_counts = {"pending": 0, "annotating": 0, "completed": 0, "skipped": 0}
    for item in status_result.data or []:
        status = item.get("status", "pending")
        if status in status_counts:
            status_counts[status] += 1

    # Get annotation count
    annotation_result = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).execute()
    total_annotations = annotation_result.count or 0

    # Get classes and build distribution
    class_names = []
    class_distribution = {}
    try:
        classes = supabase_service.client.table("od_classes").select("id, name").eq("dataset_id", dataset_id).execute()
        for cls in classes.data or []:
            class_names.append(cls["name"])
            count_result = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).eq("class_id", cls["id"]).execute()
            class_distribution[cls["name"]] = count_result.count or 0
    except Exception:
        pass

    # Calculate avg annotations per image
    annotated_count = ds.get("annotated_image_count", 0)
    avg_annotations = total_annotations / annotated_count if annotated_count > 0 else 0

    # Get image size stats (simplified - using defaults if not available)
    # In production, you'd query actual image dimensions
    min_size = {"width": 640, "height": 480}
    max_size = {"width": 1920, "height": 1080}
    avg_size = {"width": 1280, "height": 720}

    # Return training wizard compatible format
    return {
        "name": ds.get("name", "Unknown"),
        "image_count": ds.get("image_count", 0),
        "annotated_image_count": ds.get("annotated_image_count", 0),
        "annotation_count": total_annotations,
        "class_names": class_names,
        "class_distribution": class_distribution,
        "avg_annotations_per_image": round(avg_annotations, 2),
        "min_image_size": min_size,
        "max_image_size": max_size,
        "avg_image_size": avg_size,
        # Also include legacy format for backward compatibility
        "dataset": ds,
        "images_by_status": status_counts,
        "total_annotations": total_annotations,
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

    # Bulk update in batches (Supabase has limits on IN clause size)
    BATCH_SIZE = 100
    updated_count = 0
    for i in range(0, len(image_ids), BATCH_SIZE):
        batch = image_ids[i:i + BATCH_SIZE]
        result = supabase_service.client.table("od_dataset_images").update(
            update_data
        ).eq("dataset_id", dataset_id).in_("image_id", batch).execute()
        updated_count += len(result.data) if result.data else 0

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

    # Bulk update in batches (Supabase has limits on IN clause size)
    BATCH_SIZE = 100
    total_updated = 0
    for i in range(0, len(image_ids), BATCH_SIZE):
        batch = image_ids[i:i + BATCH_SIZE]
        supabase_service.client.table("od_dataset_images").update(
            update_data
        ).eq("dataset_id", dataset_id).in_("image_id", batch).execute()
        total_updated += len(batch)

    # Update dataset annotated_image_count
    count = supabase_service.client.table("od_dataset_images").select(
        "id", count="exact"
    ).eq("dataset_id", dataset_id).eq("status", "completed").execute()
    supabase_service.client.table("od_datasets").update({
        "annotated_image_count": count.count or 0
    }).eq("id", dataset_id).execute()

    return {
        "updated": total_updated,
        "status": new_status,
        "message": f"Updated {total_updated} images to '{new_status}'"
    }


@router.post("/{dataset_id}/images/bulk-status/async")
async def update_images_status_bulk_async(
    dataset_id: str,
    new_status: str,
    image_ids: list[str] | None = None,
    current_status: Optional[str] = None,
    has_annotations: Optional[bool] = None,
):
    """
    Async version: Update status of multiple images as a background job.

    Use this for large status updates (>200 images). Supports:
    - Explicit image_ids list
    - Filter-based updates via current_status/has_annotations

    Args:
        dataset_id: Target dataset ID
        new_status: Status to set (pending, annotating, completed, skipped)
        image_ids: Specific image IDs to update (optional)
        current_status: Only update images with this current status (optional)
        has_annotations: Filter by annotation presence (optional)

    Returns:
        Job ID for tracking progress
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    valid_statuses = ["pending", "annotating", "completed", "skipped"]
    if new_status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets")\
        .select("id, name")\
        .eq("id", dataset_id)\
        .single()\
        .execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Need either image_ids or filters
    if not image_ids and current_status is None and has_annotations is None:
        raise HTTPException(status_code=400, detail="Provide image_ids or filter criteria")

    # Build config
    config = {
        "dataset_id": dataset_id,
        "new_status": new_status,
    }

    if image_ids:
        config["image_ids"] = image_ids
    else:
        config["filters"] = {}
        if current_status:
            config["filters"]["current_status"] = current_status
        if has_annotations is not None:
            config["filters"]["has_annotations"] = has_annotations

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_bulk_update_status",
        "status": "pending",
        "progress": 0,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk status update job queued for dataset '{dataset.data['name']}'",
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


@router.post("/{dataset_id}/export/async")
async def export_dataset_async(dataset_id: str, data: ExportRequest):
    """
    Async version: Export dataset as a background job with progress tracking.

    Use this for large datasets where sync export might timeout.
    Monitor progress via job status endpoint.

    Args:
        dataset_id: Dataset to export
        data: Export configuration (format, include_images, etc.)

    Returns:
        Job ID for tracking progress
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets")\
        .select("id, name")\
        .eq("id", dataset_id)\
        .single()\
        .execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate format
    if data.format not in ["yolo", "coco"]:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: yolo, coco")

    # Build config
    config = {
        "dataset_id": dataset_id,
        "format": data.format,
        "include_images": data.include_images,
    }

    if data.version_id:
        config["version_id"] = data.version_id
    if data.split:
        config["split"] = data.split
    if data.config:
        config_dict = data.config.model_dump()
        if config_dict.get("train_split"):
            config["train_split"] = config_dict["train_split"]
        if config_dict.get("val_split"):
            config["val_split"] = config_dict["val_split"]
        if config_dict.get("test_split"):
            config["test_split"] = config_dict["test_split"]

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_export_dataset",
        "status": "pending",
        "progress": 0,
        "config": config,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Export job queued for dataset '{dataset.data['name']}'",
    }


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
