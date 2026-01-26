"""
Object Detection - Annotations Router

Endpoints for managing bounding box annotations.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException

from services.supabase import supabase_service
from schemas.od import (
    ODAnnotationCreate,
    ODAnnotationUpdate,
    ODAnnotationResponse,
    ODBulkAnnotationsCreate,
    ODBulkAnnotationsResponse,
)

router = APIRouter()


@router.get("/datasets/{dataset_id}/images/{image_id}", response_model=list[ODAnnotationResponse])
async def get_image_annotations(dataset_id: str, image_id: str):
    """Get all annotations for an image in a dataset."""
    # Join with class info
    result = supabase_service.client.table("od_annotations").select(
        "*, class:od_classes(name, color)"
    ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Transform to include class_name and class_color
    annotations = []
    for ann in result.data or []:
        ann_data = {
            "id": ann["id"],
            "dataset_id": ann["dataset_id"],
            "image_id": ann["image_id"],
            "class_id": ann["class_id"],
            "class_name": ann["class"]["name"] if ann.get("class") else None,
            "class_color": ann["class"]["color"] if ann.get("class") else None,
            "bbox": {
                "x": ann["bbox_x"],
                "y": ann["bbox_y"],
                "width": ann["bbox_width"],
                "height": ann["bbox_height"],
            },
            "polygon": ann.get("polygon"),
            "is_ai_generated": ann.get("is_ai_generated", False),
            "confidence": ann.get("confidence"),
            "ai_model": ann.get("ai_model"),
            "is_reviewed": ann.get("is_reviewed", False),
            "attributes": ann.get("attributes"),
            "created_at": ann["created_at"],
            "updated_at": ann["updated_at"],
        }
        annotations.append(ann_data)

    return annotations


@router.post("/datasets/{dataset_id}/images/{image_id}", response_model=ODAnnotationResponse)
async def create_annotation(dataset_id: str, image_id: str, data: ODAnnotationCreate):
    """Create a new annotation for an image."""
    # Verify dataset and image exist and image is in dataset
    di = supabase_service.client.table("od_dataset_images").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()
    if not di.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Verify class exists
    cls = supabase_service.client.table("od_classes").select("id, name, color").eq("id", data.class_id).single().execute()
    if not cls.data:
        raise HTTPException(status_code=404, detail="Class not found")

    # Create annotation
    annotation_data = {
        "dataset_id": dataset_id,
        "image_id": image_id,
        "class_id": data.class_id,
        "bbox_x": data.bbox.x,
        "bbox_y": data.bbox.y,
        "bbox_width": data.bbox.width,
        "bbox_height": data.bbox.height,
        "polygon": [{"x": p.x, "y": p.y} for p in data.polygon] if data.polygon else None,
        "is_ai_generated": data.is_ai_generated,
        "confidence": data.confidence,
        "ai_model": data.ai_model,
        "attributes": data.attributes or {},
    }

    result = supabase_service.client.table("od_annotations").insert(annotation_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create annotation")

    # Update counts
    _update_annotation_counts(dataset_id, image_id, data.class_id, delta=1)

    # Return with class info
    ann = result.data[0]
    return {
        **ann,
        "class_name": cls.data["name"],
        "class_color": cls.data["color"],
        "bbox": {
            "x": ann["bbox_x"],
            "y": ann["bbox_y"],
            "width": ann["bbox_width"],
            "height": ann["bbox_height"],
        },
    }


@router.post("/datasets/{dataset_id}/images/{image_id}/bulk", response_model=ODBulkAnnotationsResponse)
async def create_annotations_bulk(dataset_id: str, image_id: str, data: ODBulkAnnotationsCreate):
    """Create multiple annotations at once."""
    # Verify dataset and image
    di = supabase_service.client.table("od_dataset_images").select("id").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()
    if not di.data:
        raise HTTPException(status_code=404, detail="Image not in dataset")

    # Prepare annotation records
    records = []
    class_counts = {}

    for ann in data.annotations:
        records.append({
            "dataset_id": dataset_id,
            "image_id": image_id,
            "class_id": ann.class_id,
            "bbox_x": ann.bbox.x,
            "bbox_y": ann.bbox.y,
            "bbox_width": ann.bbox.width,
            "bbox_height": ann.bbox.height,
            "polygon": [{"x": p.x, "y": p.y} for p in ann.polygon] if ann.polygon else None,
            "is_ai_generated": ann.is_ai_generated,
            "confidence": ann.confidence,
            "ai_model": ann.ai_model,
            "attributes": ann.attributes or {},
        })
        class_counts[ann.class_id] = class_counts.get(ann.class_id, 0) + 1

    # Insert all
    result = supabase_service.client.table("od_annotations").insert(records).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create annotations")

    # Update counts
    for class_id, count in class_counts.items():
        _update_annotation_counts(dataset_id, image_id, class_id, delta=count)

    return ODBulkAnnotationsResponse(
        created=len(result.data),
        annotation_ids=[ann["id"] for ann in result.data],
    )


@router.patch("/{annotation_id}", response_model=ODAnnotationResponse)
async def update_annotation(annotation_id: str, data: ODAnnotationUpdate):
    """Update an annotation."""
    # Get current annotation
    current = supabase_service.client.table("od_annotations").select("*").eq("id", annotation_id).single().execute()
    if not current.data:
        raise HTTPException(status_code=404, detail="Annotation not found")

    update_data = {}

    if data.class_id is not None:
        # Verify new class exists
        cls = supabase_service.client.table("od_classes").select("id").eq("id", data.class_id).single().execute()
        if not cls.data:
            raise HTTPException(status_code=404, detail="Class not found")
        update_data["class_id"] = data.class_id

        # Update class counts if changed
        if data.class_id != current.data["class_id"]:
            _update_class_count(current.data["class_id"], delta=-1)
            _update_class_count(data.class_id, delta=1)

    if data.bbox is not None:
        update_data["bbox_x"] = data.bbox.x
        update_data["bbox_y"] = data.bbox.y
        update_data["bbox_width"] = data.bbox.width
        update_data["bbox_height"] = data.bbox.height

    if data.polygon is not None:
        update_data["polygon"] = [{"x": p.x, "y": p.y} for p in data.polygon]

    if data.is_reviewed is not None:
        update_data["is_reviewed"] = data.is_reviewed
        if data.is_reviewed:
            from datetime import datetime, timezone
            update_data["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    if data.attributes is not None:
        update_data["attributes"] = data.attributes

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("od_annotations").update(update_data).eq("id", annotation_id).execute()

    # Get class info for response
    ann = result.data[0]
    cls = supabase_service.client.table("od_classes").select("name, color").eq("id", ann["class_id"]).single().execute()

    return {
        **ann,
        "class_name": cls.data["name"] if cls.data else None,
        "class_color": cls.data["color"] if cls.data else None,
        "bbox": {
            "x": ann["bbox_x"],
            "y": ann["bbox_y"],
            "width": ann["bbox_width"],
            "height": ann["bbox_height"],
        },
    }


@router.delete("/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Delete an annotation."""
    # Get annotation first
    ann = supabase_service.client.table("od_annotations").select("*").eq("id", annotation_id).single().execute()
    if not ann.data:
        raise HTTPException(status_code=404, detail="Annotation not found")

    # Delete
    supabase_service.client.table("od_annotations").delete().eq("id", annotation_id).execute()

    # Update counts
    _update_annotation_counts(ann.data["dataset_id"], ann.data["image_id"], ann.data["class_id"], delta=-1)

    return {"status": "deleted", "id": annotation_id}


@router.delete("/datasets/{dataset_id}/images/{image_id}/bulk")
async def delete_annotations_bulk(dataset_id: str, image_id: str, annotation_ids: Optional[list[str]] = None):
    """Delete multiple annotations. If annotation_ids not provided, deletes all for the image."""
    if annotation_ids:
        # Delete specific annotations
        result = supabase_service.client.table("od_annotations").delete().in_("id", annotation_ids).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
    else:
        # Delete all annotations for image
        result = supabase_service.client.table("od_annotations").delete().eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    deleted_count = len(result.data or [])

    # Recalculate counts
    _recalculate_image_annotation_count(dataset_id, image_id)
    _recalculate_dataset_annotation_count(dataset_id)

    return {"deleted": deleted_count}


# ===========================================
# Helper functions
# ===========================================

def _update_annotation_counts(dataset_id: str, image_id: str, class_id: str, delta: int):
    """Update annotation counts on dataset, dataset_image, and class."""
    # Update dataset_images annotation_count and status
    di = supabase_service.client.table("od_dataset_images").select("annotation_count, status").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()
    if di.data:
        new_count = max(0, (di.data.get("annotation_count", 0) or 0) + delta)
        current_status = di.data.get("status", "pending")
        # Update status based on annotation count (only if not already completed/skipped)
        new_status = current_status
        if current_status not in ("completed", "skipped"):
            new_status = "annotated" if new_count > 0 else "pending"
        supabase_service.client.table("od_dataset_images").update({
            "annotation_count": new_count,
            "status": new_status
        }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()

    # Update dataset annotation_count
    dataset = supabase_service.client.table("od_datasets").select("annotation_count").eq("id", dataset_id).single().execute()
    if dataset.data:
        new_count = max(0, (dataset.data.get("annotation_count", 0) or 0) + delta)
        supabase_service.client.table("od_datasets").update({"annotation_count": new_count}).eq("id", dataset_id).execute()

    # Update class annotation_count
    _update_class_count(class_id, delta)


def _update_class_count(class_id: str, delta: int):
    """Update annotation count on a class."""
    cls = supabase_service.client.table("od_classes").select("annotation_count").eq("id", class_id).single().execute()
    if cls.data:
        new_count = max(0, (cls.data.get("annotation_count", 0) or 0) + delta)
        supabase_service.client.table("od_classes").update({"annotation_count": new_count}).eq("id", class_id).execute()


def _recalculate_image_annotation_count(dataset_id: str, image_id: str):
    """Recalculate annotation count for an image in a dataset."""
    count_result = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
    new_count = count_result.count or 0

    # Get current status to preserve completed/skipped
    di = supabase_service.client.table("od_dataset_images").select("status").eq("dataset_id", dataset_id).eq("image_id", image_id).single().execute()
    current_status = di.data.get("status", "pending") if di.data else "pending"

    # Update status based on annotation count (only if not already completed/skipped)
    new_status = current_status
    if current_status not in ("completed", "skipped"):
        new_status = "annotated" if new_count > 0 else "pending"

    supabase_service.client.table("od_dataset_images").update({
        "annotation_count": new_count,
        "status": new_status
    }).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()


def _recalculate_dataset_annotation_count(dataset_id: str):
    """Recalculate total annotation count for a dataset."""
    count = supabase_service.client.table("od_annotations").select("id", count="exact").eq("dataset_id", dataset_id).execute()
    supabase_service.client.table("od_datasets").update({"annotation_count": count.count or 0}).eq("id", dataset_id).execute()
