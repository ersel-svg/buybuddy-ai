"""
Object Detection - Images Router

Endpoints for managing OD images (upload, list, CRUD).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from uuid import uuid4

from services.supabase import supabase_service
from schemas.od import (
    ODImageResponse,
    ODImagesResponse,
    ODImageUpdate,
)

router = APIRouter()


@router.get("", response_model=ODImagesResponse)
async def list_images(
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    source: Optional[str] = None,
    folder: Optional[str] = None,
    search: Optional[str] = None,
):
    """List OD images with pagination and filters."""
    offset = (page - 1) * limit

    # Build query
    query = supabase_service.client.table("od_images").select("*", count="exact")

    # Apply filters
    if status:
        query = query.eq("status", status)
    if source:
        query = query.eq("source", source)
    if folder:
        query = query.eq("folder", folder)
    if search:
        query = query.ilike("filename", f"%{search}%")

    # Order and paginate
    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return ODImagesResponse(
        images=result.data or [],
        total=result.count or 0,
        page=page,
        limit=limit,
    )


@router.get("/{image_id}", response_model=ODImageResponse)
async def get_image(image_id: str):
    """Get a single image by ID."""
    result = supabase_service.client.table("od_images").select("*").eq("id", image_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    return result.data


@router.post("", response_model=ODImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(None),
):
    """Upload a new image."""
    from PIL import Image
    import io

    # Read file content
    content = await file.read()

    # Get image dimensions
    img = Image.open(io.BytesIO(content))
    width, height = img.size

    # Generate unique filename
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    unique_filename = f"{uuid4()}.{ext}"

    # Upload to Supabase Storage
    storage_path = f"od-images/{unique_filename}"

    try:
        supabase_service.client.storage.from_("od-images").upload(
            storage_path,
            content,
            {"content-type": file.content_type or "image/jpeg"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to storage: {str(e)}")

    # Get public URL
    image_url = supabase_service.client.storage.from_("od-images").get_public_url(storage_path)

    # Create database record
    image_data = {
        "filename": unique_filename,
        "original_filename": file.filename,
        "image_url": image_url,
        "width": width,
        "height": height,
        "file_size_bytes": len(content),
        "mime_type": file.content_type or "image/jpeg",
        "source": "upload",
        "folder": folder,
        "status": "pending",
    }

    result = supabase_service.client.table("od_images").insert(image_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create image record")

    return result.data[0]


@router.post("/bulk", response_model=list[ODImageResponse])
async def upload_images_bulk(files: list[UploadFile] = File(...)):
    """Upload multiple images."""
    from PIL import Image
    import io

    uploaded = []

    for file in files:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        width, height = img.size

        ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid4()}.{ext}"
        storage_path = f"od-images/{unique_filename}"

        try:
            supabase_service.client.storage.from_("od-images").upload(
                storage_path,
                content,
                {"content-type": file.content_type or "image/jpeg"},
            )

            image_url = supabase_service.client.storage.from_("od-images").get_public_url(storage_path)

            image_data = {
                "filename": unique_filename,
                "original_filename": file.filename,
                "image_url": image_url,
                "width": width,
                "height": height,
                "file_size_bytes": len(content),
                "source": "upload",
                "status": "pending",
            }

            result = supabase_service.client.table("od_images").insert(image_data).execute()
            if result.data:
                uploaded.append(result.data[0])
        except Exception as e:
            print(f"Failed to upload {file.filename}: {e}")
            continue

    return uploaded


@router.patch("/{image_id}", response_model=ODImageResponse)
async def update_image(image_id: str, data: ODImageUpdate):
    """Update image metadata."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("od_images").update(update_data).eq("id", image_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    return result.data[0]


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    """Delete an image."""
    # Get image first
    image = supabase_service.client.table("od_images").select("*").eq("id", image_id).single().execute()

    if not image.data:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if image is used in any dataset
    dataset_check = supabase_service.client.table("od_dataset_images").select("id").eq("image_id", image_id).limit(1).execute()

    if dataset_check.data:
        raise HTTPException(status_code=400, detail="Image is used in a dataset. Remove from dataset first.")

    # Delete from storage
    try:
        filename = image.data["filename"]
        supabase_service.client.storage.from_("od-images").remove([f"od-images/{filename}"])
    except Exception:
        pass  # Continue even if storage delete fails

    # Delete from database
    supabase_service.client.table("od_images").delete().eq("id", image_id).execute()

    return {"status": "deleted", "id": image_id}


@router.delete("/bulk")
async def delete_images_bulk(image_ids: list[str]):
    """Delete multiple images."""
    deleted = 0
    errors = []

    for image_id in image_ids:
        try:
            # Check if used in dataset
            dataset_check = supabase_service.client.table("od_dataset_images").select("id").eq("image_id", image_id).limit(1).execute()
            if dataset_check.data:
                errors.append(f"{image_id}: Used in dataset")
                continue

            # Get image for filename
            image = supabase_service.client.table("od_images").select("filename").eq("id", image_id).single().execute()
            if image.data:
                try:
                    supabase_service.client.storage.from_("od-images").remove([f"od-images/{image.data['filename']}"])
                except Exception:
                    pass

            supabase_service.client.table("od_images").delete().eq("id", image_id).execute()
            deleted += 1
        except Exception as e:
            errors.append(f"{image_id}: {str(e)}")

    return {"deleted": deleted, "errors": errors}
