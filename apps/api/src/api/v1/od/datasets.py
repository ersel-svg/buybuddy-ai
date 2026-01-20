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
async def remove_image_from_dataset(dataset_id: str, image_id: str):
    """Remove an image from a dataset (also deletes its annotations in this dataset)."""
    # Get dataset_image record
    di = supabase_service.client.table("od_dataset_images").select("*").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()

    if not di.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Delete annotations for this image in this dataset
    supabase_service.client.table("od_annotations").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Delete dataset_image record
    supabase_service.client.table("od_dataset_images").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update dataset counts
    dataset = supabase_service.client.table("od_datasets").select("image_count, annotation_count").eq("id", dataset_id).single().execute()
    if dataset.data:
        new_image_count = max(0, (dataset.data.get("image_count", 0) or 0) - 1)
        new_annotation_count = max(0, (dataset.data.get("annotation_count", 0) or 0) - (di.data.get("annotation_count", 0) or 0))
        supabase_service.client.table("od_datasets").update({
            "image_count": new_image_count,
            "annotation_count": new_annotation_count,
        }).eq("id", dataset_id).execute()

    return {"status": "removed", "dataset_id": dataset_id, "image_id": image_id}


@router.post("/{dataset_id}/images/bulk-remove")
async def remove_images_bulk(dataset_id: str, image_ids: list[str]):
    """Remove multiple images from a dataset."""
    removed = 0
    for image_id in image_ids:
        try:
            # Delete annotations
            supabase_service.client.table("od_annotations").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
            # Delete dataset_image
            result = supabase_service.client.table("od_dataset_images").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
            if result.data:
                removed += 1
        except Exception:
            continue

    # Update dataset count
    count_result = supabase_service.client.table("od_dataset_images").select("id", count="exact").eq("dataset_id", dataset_id).execute()
    new_count = count_result.count or 0
    supabase_service.client.table("od_datasets").update({"image_count": new_count}).eq("id", dataset_id).execute()

    return {"removed": removed}


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
