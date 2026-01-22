"""
Classification - Classes Router

Endpoints for managing classification classes.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSClassCreate,
    CLSClassUpdate,
    CLSClassResponse,
    CLSClassMergeRequest,
    CLSClassBulkCreate,
)

router = APIRouter()


@router.get("", response_model=list[CLSClassResponse])
async def list_classes(
    include_inactive: bool = False,
    search: Optional[str] = None,
):
    """List all classification classes."""
    query = supabase_service.client.table("cls_classes").select("*")

    if not include_inactive:
        query = query.eq("is_active", True)

    if search:
        query = query.ilike("name", f"%{search}%")

    query = query.order("name")

    result = query.execute()

    return result.data or []


@router.get("/hierarchy")
async def get_class_hierarchy():
    """Get class hierarchy tree."""
    result = supabase_service.client.table("cls_classes").select("*").eq("is_active", True).execute()

    classes = result.data or []

    # Build tree
    class_map = {c["id"]: {**c, "children": []} for c in classes}

    roots = []
    for c in classes:
        parent_id = c.get("parent_class_id")
        if parent_id and parent_id in class_map:
            class_map[parent_id]["children"].append(class_map[c["id"]])
        else:
            roots.append(class_map[c["id"]])

    return roots


@router.get("/{class_id}", response_model=CLSClassResponse)
async def get_class(class_id: str):
    """Get a single class by ID."""
    result = supabase_service.client.table("cls_classes").select("*").eq("id", class_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Class not found")

    return result.data


@router.post("", response_model=CLSClassResponse)
async def create_class(data: CLSClassCreate):
    """Create a new classification class for a dataset."""
    # Check for duplicate name within the same dataset
    existing = supabase_service.client.table("cls_classes").select("id").eq("dataset_id", data.dataset_id).eq("name", data.name).execute()

    if existing.data:
        raise HTTPException(status_code=409, detail="Class with this name already exists in this dataset")

    class_data = data.model_dump()

    result = supabase_service.client.table("cls_classes").insert(class_data).execute()

    # Update dataset class count
    try:
        supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": data.dataset_id}).execute()
    except Exception as e:
        logger.warning(f"Failed to update dataset stats: {e}")

    return result.data[0]


@router.post("/bulk", response_model=list[CLSClassResponse])
async def create_classes_bulk(data: CLSClassBulkCreate):
    """Create multiple classes at once."""
    created = []
    errors = []

    for cls in data.classes:
        try:
            existing = supabase_service.client.table("cls_classes").select("id").eq("name", cls.name).execute()

            if existing.data:
                errors.append(f"Class '{cls.name}' already exists")
                continue

            result = supabase_service.client.table("cls_classes").insert(cls.model_dump()).execute()
            created.append(result.data[0])

        except Exception as e:
            errors.append(f"Failed to create '{cls.name}': {str(e)}")

    return created


@router.patch("/{class_id}", response_model=CLSClassResponse)
async def update_class(class_id: str, data: CLSClassUpdate):
    """Update a class."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Check name uniqueness if updating name
    if "name" in update_data:
        existing = supabase_service.client.table("cls_classes").select("id").eq("name", update_data["name"]).neq("id", class_id).execute()

        if existing.data:
            raise HTTPException(status_code=409, detail="Class with this name already exists")

    result = supabase_service.client.table("cls_classes").update(update_data).eq("id", class_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Class not found")

    return result.data[0]


@router.delete("/{class_id}")
async def delete_class(class_id: str):
    """Delete a class (soft delete by setting is_active=False)."""
    # Check if class has labels
    labels = supabase_service.client.table("cls_labels").select("id", count="exact").eq("class_id", class_id).execute()

    if labels.count and labels.count > 0:
        # Soft delete
        supabase_service.client.table("cls_classes").update({"is_active": False}).eq("id", class_id).execute()
        return {"success": True, "message": "Class deactivated (has existing labels)"}

    # Hard delete if no labels
    supabase_service.client.table("cls_classes").delete().eq("id", class_id).execute()

    return {"success": True, "message": "Class deleted"}


@router.post("/merge")
async def merge_classes(data: CLSClassMergeRequest):
    """Merge multiple classes into one."""
    # Update all labels to target class
    affected_labels = 0

    for source_id in data.source_class_ids:
        if source_id == data.target_class_id:
            continue

        result = supabase_service.client.table("cls_labels").update({"class_id": data.target_class_id}).eq("class_id", source_id).execute()

        affected_labels += len(result.data or [])

        # Deactivate source class
        supabase_service.client.table("cls_classes").update({"is_active": False}).eq("id", source_id).execute()

    # Update target class image count
    supabase_service.client.rpc("update_cls_class_image_count", {"p_class_id": data.target_class_id}).execute()

    return {
        "success": True,
        "affected_labels": affected_labels,
        "merged_classes": len(data.source_class_ids),
    }
