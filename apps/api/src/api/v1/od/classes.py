"""
Object Detection - Classes Router

Endpoints for managing detection classes.
"""

from typing import Optional
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
    category: Optional[str] = None,
    is_active: Optional[bool] = True,
    include_counts: bool = True,
):
    """List all detection classes."""
    query = supabase_service.client.table("od_classes").select("*")

    if is_active is not None:
        query = query.eq("is_active", is_active)
    if category:
        query = query.eq("category", category)

    query = query.order("name")
    result = query.execute()

    return result.data or []


@router.get("/{class_id}", response_model=ODClassResponse)
async def get_class(class_id: str):
    """Get a single class by ID."""
    result = supabase_service.client.table("od_classes").select("*").eq("id", class_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Class not found")

    return result.data


@router.post("", response_model=ODClassResponse)
async def create_class(data: ODClassCreate):
    """Create a new detection class."""
    # Check if name already exists
    existing = supabase_service.client.table("od_classes").select("id").eq("name", data.name).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail=f"Class with name '{data.name}' already exists")

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

    return result.data[0]


@router.patch("/{class_id}", response_model=ODClassResponse)
async def update_class(class_id: str, data: ODClassUpdate):
    """Update a class."""
    # Get current class
    current = supabase_service.client.table("od_classes").select("*").eq("id", class_id).single().execute()
    if not current.data:
        raise HTTPException(status_code=404, detail="Class not found")

    # Prevent modifying system classes' name
    if current.data["is_system"] and data.name and data.name != current.data["name"]:
        raise HTTPException(status_code=400, detail="Cannot rename system classes")

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

    if current.data["is_system"]:
        raise HTTPException(status_code=400, detail="Cannot delete system classes")

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

    # Delete the class (will fail if annotations exist due to RESTRICT)
    try:
        supabase_service.client.table("od_classes").delete().eq("id", class_id).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot delete class: {str(e)}")

    return {"status": "deleted", "id": class_id}


@router.post("/merge")
async def merge_classes(data: ODClassMergeRequest):
    """Merge multiple classes into one target class."""
    # Validate target class exists
    target = supabase_service.client.table("od_classes").select("*").eq("id", data.target_class_id).single().execute()
    if not target.data:
        raise HTTPException(status_code=404, detail="Target class not found")

    # Validate source classes exist
    sources = supabase_service.client.table("od_classes").select("*").in_("id", data.source_class_ids).execute()
    if len(sources.data or []) != len(data.source_class_ids):
        raise HTTPException(status_code=400, detail="One or more source classes not found")

    # Check no system classes in sources
    for source in sources.data:
        if source["is_system"]:
            raise HTTPException(status_code=400, detail=f"Cannot merge system class '{source['name']}'")

    total_moved = 0

    # Move annotations from source classes to target
    for source_id in data.source_class_ids:
        if source_id == data.target_class_id:
            continue

        # Update annotations
        result = supabase_service.client.table("od_annotations").update({
            "class_id": data.target_class_id
        }).eq("class_id", source_id).execute()

        moved = len(result.data or [])
        total_moved += moved

        # Get source name for logging
        source_class = next((s for s in sources.data if s["id"] == source_id), None)

        # Log the merge
        supabase_service.client.table("od_class_changes").insert({
            "change_type": "merge",
            "source_class_ids": [source_id],
            "target_class_id": data.target_class_id,
            "old_name": source_class["name"] if source_class else None,
            "new_name": target.data["name"],
            "affected_annotation_count": moved,
        }).execute()

        # Delete source class
        supabase_service.client.table("od_classes").delete().eq("id", source_id).execute()

    # Update target class annotation count
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
