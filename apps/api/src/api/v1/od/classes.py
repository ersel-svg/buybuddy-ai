"""
Object Detection - Classes Router

Endpoints for managing detection classes.
"""

from typing import Optional
from difflib import SequenceMatcher
from fastapi import APIRouter, HTTPException

from services.supabase import supabase_service
from schemas.od import (
    ODClassCreate,
    ODClassUpdate,
    ODClassResponse,
    ODClassMergeRequest,
)

router = APIRouter()


@router.get("", response_model=list[ODClassResponse])
async def list_classes(
    dataset_id: Optional[str] = None,
    category: Optional[str] = None,
    is_active: Optional[bool] = True,
    include_templates: bool = False,
):
    """List detection classes. If dataset_id is provided, returns classes for that dataset."""
    query = supabase_service.client.table("od_classes").select("*")

    if dataset_id:
        # Get classes for specific dataset
        if include_templates:
            # Include both dataset classes and templates
            query = query.or_(f"dataset_id.eq.{dataset_id},dataset_id.is.null")
        else:
            query = query.eq("dataset_id", dataset_id)
    else:
        # Get all classes (legacy behavior) or only templates
        pass  # Returns all classes

    if is_active is not None:
        query = query.eq("is_active", is_active)
    if category:
        query = query.eq("category", category)

    query = query.order("name")
    result = query.execute()

    return result.data or []


@router.get("/templates", response_model=list[ODClassResponse])
async def list_template_classes(
    category: Optional[str] = None,
):
    """List template classes (classes without a dataset, used as templates for new datasets)."""
    query = supabase_service.client.table("od_classes").select("*").is_("dataset_id", "null")

    if category:
        query = query.eq("category", category)

    query = query.order("name")
    result = query.execute()

    return result.data or []


@router.get("/duplicates")
async def detect_duplicates(dataset_id: str, threshold: float = 0.75):
    """
    Detect potentially duplicate classes based on name similarity.

    Args:
        dataset_id: The dataset to check for duplicates
        threshold: Similarity threshold (0-1), default 0.75

    Returns:
        Groups of similar classes with similarity scores
    """
    # Get all classes for the dataset
    result = supabase_service.client.table("od_classes").select("*").eq("dataset_id", dataset_id).eq("is_active", True).execute()
    classes = result.data or []

    if len(classes) < 2:
        return {"groups": [], "total_groups": 0}

    # Find similar pairs
    similar_groups = []
    processed_ids = set()

    for i, class_a in enumerate(classes):
        if class_a["id"] in processed_ids:
            continue

        group = {
            "classes": [
                {
                    "id": class_a["id"],
                    "name": class_a["name"],
                    "display_name": class_a.get("display_name"),
                    "annotation_count": class_a.get("annotation_count", 0) or 0,
                    "is_system": class_a.get("is_system", False),
                    "color": class_a.get("color"),
                }
            ],
            "max_similarity": 0.0,
        }

        for j, class_b in enumerate(classes):
            if i >= j or class_b["id"] in processed_ids:
                continue

            # Calculate similarity using SequenceMatcher
            name_a = class_a["name"].lower()
            name_b = class_b["name"].lower()
            similarity = SequenceMatcher(None, name_a, name_b).ratio()

            # Also check if one is plural of another (simple check)
            if name_a + "s" == name_b or name_b + "s" == name_a:
                similarity = max(similarity, 0.95)
            if name_a + "es" == name_b or name_b + "es" == name_a:
                similarity = max(similarity, 0.95)

            if similarity >= threshold:
                group["classes"].append({
                    "id": class_b["id"],
                    "name": class_b["name"],
                    "display_name": class_b.get("display_name"),
                    "annotation_count": class_b.get("annotation_count", 0) or 0,
                    "is_system": class_b.get("is_system", False),
                    "color": class_b.get("color"),
                    "similarity": round(similarity, 2),
                })
                group["max_similarity"] = max(group["max_similarity"], similarity)
                processed_ids.add(class_b["id"])

        if len(group["classes"]) > 1:
            # Sort by annotation count descending (suggest keeping the one with most annotations)
            group["classes"].sort(key=lambda x: x["annotation_count"], reverse=True)
            group["suggested_target"] = group["classes"][0]["id"]
            group["suggested_sources"] = [c["id"] for c in group["classes"][1:]]
            group["total_annotations"] = sum(c["annotation_count"] for c in group["classes"])
            similar_groups.append(group)
            processed_ids.add(class_a["id"])

    # Sort groups by max similarity descending
    similar_groups.sort(key=lambda x: x["max_similarity"], reverse=True)

    return {
        "groups": similar_groups,
        "total_groups": len(similar_groups),
    }


@router.get("/{class_id}", response_model=ODClassResponse)
async def get_class(class_id: str):
    """Get a single class by ID."""
    result = supabase_service.client.table("od_classes").select("*").eq("id", class_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Class not found")

    return result.data


@router.post("", response_model=ODClassResponse)
async def create_class(data: ODClassCreate):
    """Create a new detection class for a specific dataset or as a template."""
    # Check if name already exists in the same dataset (or globally for templates)
    query = supabase_service.client.table("od_classes").select("id").eq("name", data.name)
    if data.dataset_id:
        query = query.eq("dataset_id", data.dataset_id)
    else:
        query = query.is_("dataset_id", "null")

    existing = query.execute()
    if existing.data:
        scope = f"dataset" if data.dataset_id else "template classes"
        raise HTTPException(status_code=400, detail=f"Class with name '{data.name}' already exists in {scope}")

    class_data = data.model_dump()
    result = supabase_service.client.table("od_classes").insert(class_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create class")

    # Log the change
    supabase_service.client.table("od_class_changes").insert({
        "change_type": "create",
        "class_id": result.data[0]["id"],
        "new_name": data.name,
    }).execute()

    # Update dataset class_count if this class is for a dataset
    if data.dataset_id:
        count_result = supabase_service.client.table("od_classes").select("id", count="exact").eq("dataset_id", data.dataset_id).execute()
        supabase_service.client.table("od_datasets").update({"class_count": count_result.count or 0}).eq("id", data.dataset_id).execute()

    return result.data[0]


@router.patch("/{class_id}", response_model=ODClassResponse)
async def update_class(class_id: str, data: ODClassUpdate):
    """Update a class."""
    # Get current class
    current = supabase_service.client.table("od_classes").select("*").eq("id", class_id).single().execute()
    if not current.data:
        raise HTTPException(status_code=404, detail="Class not found")

    update_data = data.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # If renaming, add old name to aliases and log
    if "name" in update_data and update_data["name"] != current.data["name"]:
        old_name = current.data["name"]
        aliases = current.data.get("aliases") or []
        if old_name not in aliases:
            aliases.append(old_name)
        update_data["aliases"] = aliases

        # Log the rename
        supabase_service.client.table("od_class_changes").insert({
            "change_type": "rename",
            "class_id": class_id,
            "old_name": old_name,
            "new_name": update_data["name"],
        }).execute()

    result = supabase_service.client.table("od_classes").update(update_data).eq("id", class_id).execute()

    return result.data[0]


@router.delete("/{class_id}")
async def delete_class(class_id: str, force: bool = False):
    """Delete a class."""
    # Get current class
    current = supabase_service.client.table("od_classes").select("*").eq("id", class_id).single().execute()
    if not current.data:
        raise HTTPException(status_code=404, detail="Class not found")

    # Check for annotations using this class
    annotation_count = current.data.get("annotation_count", 0)
    if annotation_count > 0 and not force:
        raise HTTPException(
            status_code=400,
            detail=f"Class has {annotation_count} annotations. Use force=true to delete anyway."
        )

    # Log the deletion
    supabase_service.client.table("od_class_changes").insert({
        "change_type": "delete",
        "class_id": class_id,
        "old_name": current.data["name"],
        "affected_annotation_count": annotation_count,
    }).execute()

    # If force=true and annotations exist, delete them first
    if annotation_count > 0 and force:
        # Get all annotations before deleting to update counts
        annotations = supabase_service.client.table("od_annotations").select("dataset_id, image_id").eq("class_id", class_id).execute()

        # Delete annotations
        supabase_service.client.table("od_annotations").delete().eq("class_id", class_id).execute()

        # Update counts for affected images and datasets
        affected_datasets = set()
        affected_images = set()
        for ann in annotations.data or []:
            affected_datasets.add(ann["dataset_id"])
            affected_images.add((ann["dataset_id"], ann["image_id"]))

        # Recalculate counts for affected images
        from api.v1.od.annotations import _recalculate_image_annotation_count
        for dataset_id, image_id in affected_images:
            _recalculate_image_annotation_count(dataset_id, image_id)

        # Recalculate counts for affected datasets
        from api.v1.od.annotations import _recalculate_dataset_annotation_count
        for dataset_id in affected_datasets:
            _recalculate_dataset_annotation_count(dataset_id)

    # Delete the class
    try:
        supabase_service.client.table("od_classes").delete().eq("id", class_id).execute()
    except Exception as e:
        import traceback
        error_detail = f"Cannot delete class: {str(e)}"
        print(f"ERROR deleting class {class_id}: {error_detail}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=error_detail)

    return {"status": "deleted", "id": class_id}


@router.post("/merge")
async def merge_classes(data: ODClassMergeRequest):
    """
    Merge multiple classes into one target class - BACKGROUND JOB.

    Creates a background job to handle large merges that would timeout.
    Poll GET /api/v1/jobs/{job_id} to track progress.
    """
    # Validate target class exists
    target = supabase_service.client.table("od_classes").select("*").eq("id", data.target_class_id).single().execute()
    if not target.data:
        raise HTTPException(status_code=404, detail="Target class not found")

    # Validate source classes exist
    sources = supabase_service.client.table("od_classes").select("*").in_("id", data.source_class_ids).execute()
    if len(sources.data or []) != len(data.source_class_ids):
        raise HTTPException(status_code=400, detail="One or more source classes not found")

    # Calculate total annotations to merge
    total_annotations = sum(
        (s.get("annotation_count", 0) or 0)
        for s in sources.data
        if s["id"] != data.target_class_id
    )

    # For small merges (< 10K annotations), do it inline for faster response
    if total_annotations < 10000:
        return await _merge_classes_inline(data, target, sources)

    # For large merges, create a background job
    try:
        from services.local_jobs import create_local_job

        job = await create_local_job(
            job_type="local_class_merge",
            config={
                "target_class_id": data.target_class_id,
                "source_class_ids": data.source_class_ids,
            }
        )

        return {
            "status": "pending",
            "job_id": job["id"],
            "message": f"Merge job created for {total_annotations:,} annotations. Poll GET /api/v1/jobs/{job['id']} for progress.",
            "target_class_id": data.target_class_id,
            "estimated_annotations": total_annotations,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create merge job: {str(e)}"
        )


async def _merge_classes_inline(data: ODClassMergeRequest, target, sources):
    """Inline merge for small operations (< 10K annotations)."""
    total_moved = 0

    for source_id in data.source_class_ids:
        if source_id == data.target_class_id:
            continue

        source_class = next((s for s in sources.data if s["id"] == source_id), None)
        source_annotation_count = source_class.get("annotation_count", 0) or 0 if source_class else 0

        # Direct update
        moved_for_source = 0
        try:
            update_result = supabase_service.client.table("od_annotations").update({
                "class_id": data.target_class_id
            }).eq("class_id", source_id).execute()

            moved_for_source = len(update_result.data) if update_result.data else source_annotation_count
        except Exception as e:
            print(f"[Merge] Direct update failed: {e}")
            # Fallback to batch
            while True:
                batch_result = supabase_service.client.table("od_annotations") \
                    .select("id") \
                    .eq("class_id", source_id) \
                    .limit(1000) \
                    .execute()

                if not batch_result.data:
                    break

                batch_ids = [ann["id"] for ann in batch_result.data]
                supabase_service.client.table("od_annotations").update({
                    "class_id": data.target_class_id
                }).in_("id", batch_ids).execute()
                moved_for_source += len(batch_ids)

        total_moved += moved_for_source

        # Log and delete
        supabase_service.client.table("od_class_changes").insert({
            "change_type": "merge",
            "source_class_ids": [source_id],
            "target_class_id": data.target_class_id,
            "old_name": source_class["name"] if source_class else None,
            "new_name": target.data["name"],
            "affected_annotation_count": moved_for_source,
        }).execute()

        supabase_service.client.table("od_classes").delete().eq("id", source_id).execute()

    # Update target count
    new_count = (target.data.get("annotation_count", 0) or 0) + total_moved
    supabase_service.client.table("od_classes").update({
        "annotation_count": new_count
    }).eq("id", data.target_class_id).execute()

    return {
        "status": "merged",
        "target_class_id": data.target_class_id,
        "merged_count": len(data.source_class_ids),
        "annotations_moved": total_moved,
    }


@router.get("/categories/list")
async def list_categories():
    """Get unique category values."""
    result = supabase_service.client.table("od_classes").select("category").not_.is_("category", "null").execute()

    categories = list(set(item["category"] for item in result.data or [] if item["category"]))
    categories.sort()

    return {"categories": categories}
