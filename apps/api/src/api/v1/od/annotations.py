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

# Duplicate detection threshold - IoU above this is considered duplicate
DUPLICATE_IOU_THRESHOLD = 0.95


def _compute_iou(box1: dict, box2: dict) -> float:
    """Compute IoU (Intersection over Union) between two bboxes."""
    x1_1, y1_1 = box1["x"], box1["y"]
    x2_1, y2_1 = box1["x"] + box1["width"], box1["y"] + box1["height"]

    x1_2, y1_2 = box2["x"], box2["y"]
    x2_2, y2_2 = box2["x"] + box2["width"], box2["y"] + box2["height"]

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def _check_duplicate_annotation(
    dataset_id: str,
    image_id: str,
    class_id: str,
    bbox: dict,
    existing_annotations: list = None,
) -> bool:
    """
    Check if an annotation with similar coordinates already exists.

    Args:
        dataset_id: Dataset ID
        image_id: Image ID
        class_id: Class ID
        bbox: Bounding box dict with x, y, width, height
        existing_annotations: Optional pre-fetched annotations to check against

    Returns:
        True if duplicate exists, False otherwise
    """
    if existing_annotations is None:
        # Fetch existing annotations for this image and class
        result = supabase_service.client.table("od_annotations").select(
            "bbox_x, bbox_y, bbox_width, bbox_height"
        ).eq("dataset_id", dataset_id).eq("image_id", image_id).eq("class_id", class_id).execute()
        existing_annotations = result.data or []

    for existing in existing_annotations:
        existing_bbox = {
            "x": existing.get("bbox_x", existing.get("x", 0)),
            "y": existing.get("bbox_y", existing.get("y", 0)),
            "width": existing.get("bbox_width", existing.get("width", 0)),
            "height": existing.get("bbox_height", existing.get("height", 0)),
        }

        iou = _compute_iou(bbox, existing_bbox)
        if iou >= DUPLICATE_IOU_THRESHOLD:
            return True

    return False


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

    # Check for duplicate annotation (same location, same class)
    bbox_dict = {"x": data.bbox.x, "y": data.bbox.y, "width": data.bbox.width, "height": data.bbox.height}
    if _check_duplicate_annotation(dataset_id, image_id, data.class_id, bbox_dict):
        raise HTTPException(
            status_code=409,
            detail="Duplicate annotation: An annotation with similar coordinates already exists for this class"
        )

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

    # Fetch existing annotations for duplicate checking
    existing_result = supabase_service.client.table("od_annotations").select(
        "class_id, bbox_x, bbox_y, bbox_width, bbox_height"
    ).eq("dataset_id", dataset_id).eq("image_id", image_id).execute()
    existing_by_class: dict[str, list] = {}
    for existing in existing_result.data or []:
        class_id = existing["class_id"]
        if class_id not in existing_by_class:
            existing_by_class[class_id] = []
        existing_by_class[class_id].append(existing)

    # Prepare annotation records, filtering out duplicates
    records = []
    class_counts = {}
    skipped_duplicates = 0

    # Also track new annotations to avoid duplicates within the same batch
    new_by_class: dict[str, list] = {}

    for ann in data.annotations:
        bbox_dict = {"x": ann.bbox.x, "y": ann.bbox.y, "width": ann.bbox.width, "height": ann.bbox.height}

        # Check against existing annotations
        existing_for_class = existing_by_class.get(ann.class_id, [])
        if _check_duplicate_annotation(dataset_id, image_id, ann.class_id, bbox_dict, existing_for_class):
            skipped_duplicates += 1
            continue

        # Check against other annotations in this batch
        new_for_class = new_by_class.get(ann.class_id, [])
        is_duplicate_in_batch = False
        for new_ann in new_for_class:
            iou = _compute_iou(bbox_dict, new_ann)
            if iou >= DUPLICATE_IOU_THRESHOLD:
                is_duplicate_in_batch = True
                break

        if is_duplicate_in_batch:
            skipped_duplicates += 1
            continue

        # Track this annotation for intra-batch duplicate checking
        if ann.class_id not in new_by_class:
            new_by_class[ann.class_id] = []
        new_by_class[ann.class_id].append(bbox_dict)

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

    if skipped_duplicates > 0:
        print(f"[Bulk Annotation] Skipped {skipped_duplicates} duplicate annotations")

    if not records:
        return ODBulkAnnotationsResponse(created=0, annotation_ids=[])

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
    """
    Update annotation counts on dataset, dataset_image, and class.

    NOTE: Uses recalculate instead of delta to avoid race conditions (Lost Update bug).
    When multiple concurrent requests try to update counts, delta-based updates can lose data:
    - Thread A reads count=5, adds 1, writes 6
    - Thread B reads count=5, adds 1, writes 6
    - Expected: 7, Actual: 6 (lost update!)

    Recalculate is slightly slower but guarantees correctness.
    """
    # Recalculate image annotation count from actual data
    _recalculate_image_annotation_count(dataset_id, image_id)

    # Recalculate dataset annotation count
    _recalculate_dataset_annotation_count(dataset_id)

    # Recalculate class annotation count
    _recalculate_class_count(class_id)


def _update_class_count(class_id: str, delta: int):
    """Update annotation count on a class. Deprecated: use _recalculate_class_count instead."""
    _recalculate_class_count(class_id)


def _recalculate_class_count(class_id: str):
    """Recalculate annotation count for a class from actual data."""
    count_result = supabase_service.client.table("od_annotations").select("id", count="exact").eq("class_id", class_id).execute()
    new_count = count_result.count or 0
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
