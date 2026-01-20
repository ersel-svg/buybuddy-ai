"""
Object Detection - Images Router

Endpoints for managing OD images (upload, list, CRUD).
Includes advanced import features (URL, annotated datasets, duplicate detection).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, Depends
from pydantic import BaseModel
from uuid import uuid4
import json

from services.supabase import supabase_service, SupabaseService

def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service
from services.buybuddy import buybuddy_service
from services.od_import import (
    import_from_urls,
    preview_import,
    import_annotated_dataset,
    get_duplicate_groups,
    check_duplicate_by_phash,
    check_duplicate_by_hash,
    calculate_phash,
    calculate_file_hash,
    ClassMapping,
)
from schemas.od import (
    ODImageResponse,
    ODImagesResponse,
    ODImageUpdate,
    ImportURLsRequest,
    ImportPreviewResponse,
    ImportAnnotatedRequest,
    ImportResultResponse,
    DuplicateCheckRequest,
    DuplicateCheckResponse,
    DuplicateGroupsResponse,
    BulkOperationRequest,
    BulkTagRequest,
    BulkMoveRequest,
    BulkOperationResponse,
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
    merchant_id: Optional[int] = None,
    store_id: Optional[int] = None,
    # Multi-select filters (comma-separated)
    statuses: Optional[str] = None,
    sources: Optional[str] = None,
    folders: Optional[str] = None,
    merchant_ids: Optional[str] = None,
    store_ids: Optional[str] = None,
):
    """List OD images with pagination and filters.

    Supports both single and multi-select filters:
    - status/statuses: Filter by image status (comma-separated for multi)
    - source/sources: Filter by image source (comma-separated for multi)
    - folder/folders: Filter by folder (comma-separated for multi)
    - merchant_id/merchant_ids: Filter by merchant (comma-separated for multi)
    - store_id/store_ids: Filter by store (comma-separated for multi)
    """
    offset = (page - 1) * limit

    # Build query
    query = supabase_service.client.table("od_images").select("*", count="exact")

    # Apply filters - support both single and multi-select
    # Status filter
    if statuses:
        status_list = [s.strip() for s in statuses.split(",") if s.strip()]
        if status_list:
            query = query.in_("status", status_list)
    elif status:
        query = query.eq("status", status)

    # Source filter
    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]
        if source_list:
            query = query.in_("source", source_list)
    elif source:
        query = query.eq("source", source)

    # Folder filter
    if folders:
        folder_list = [f.strip() for f in folders.split(",") if f.strip()]
        if folder_list:
            query = query.in_("folder", folder_list)
    elif folder:
        query = query.eq("folder", folder)

    # Search
    if search:
        query = query.ilike("filename", f"%{search}%")

    # Merchant filter
    if merchant_ids:
        merchant_list = [int(m.strip()) for m in merchant_ids.split(",") if m.strip().isdigit()]
        if merchant_list:
            query = query.in_("merchant_id", merchant_list)
    elif merchant_id:
        query = query.eq("merchant_id", merchant_id)

    # Store filter
    if store_ids:
        store_list = [int(s.strip()) for s in store_ids.split(",") if s.strip().isdigit()]
        if store_list:
            query = query.in_("store_id", store_list)
    elif store_id:
        query = query.eq("store_id", store_id)

    # Order and paginate
    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return ODImagesResponse(
        images=result.data or [],
        total=result.count or 0,
        page=page,
        limit=limit,
    )


@router.get("/filters/options")
async def get_filter_options():
    """Get filter options with counts for FilterDrawer component."""
    # Get all images for counting
    all_images = supabase_service.client.table("od_images").select(
        "status, source, folder, merchant_id, merchant_name, store_id, store_name"
    ).execute()

    images = all_images.data or []

    # Count by status
    status_counts = {}
    source_counts = {}
    folder_counts = {}
    merchant_counts = {}
    store_data = {}

    for img in images:
        # Status
        status = img.get("status") or "pending"
        status_counts[status] = status_counts.get(status, 0) + 1

        # Source
        source = img.get("source") or "upload"
        source_counts[source] = source_counts.get(source, 0) + 1

        # Folder
        folder = img.get("folder")
        if folder:
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        # Merchant
        merchant_id = img.get("merchant_id")
        if merchant_id:
            if merchant_id not in merchant_counts:
                merchant_counts[merchant_id] = {
                    "id": merchant_id,
                    "name": img.get("merchant_name") or f"Merchant {merchant_id}",
                    "count": 0
                }
            merchant_counts[merchant_id]["count"] += 1

        # Store
        store_id = img.get("store_id")
        if store_id:
            if store_id not in store_data:
                store_data[store_id] = {
                    "id": store_id,
                    "name": img.get("store_name") or f"Store {store_id}",
                    "count": 0
                }
            store_data[store_id]["count"] += 1

    # Format status options
    status_options = [
        {"value": status, "label": status.replace("_", " ").title(), "count": count}
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Format source options
    source_labels = {
        "upload": "Upload",
        "buybuddy_sync": "BuyBuddy Sync",
        "import": "Import",
        "url": "URL Import",
    }
    source_options = [
        {"value": source, "label": source_labels.get(source, source.title()), "count": count}
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1])
    ]

    # Format folder options
    folder_options = [
        {"value": folder, "label": folder, "count": count}
        for folder, count in sorted(folder_counts.items(), key=lambda x: x[0])
    ]

    # Format merchant options
    merchant_options = [
        {"value": str(m["id"]), "label": m["name"], "count": m["count"]}
        for m in sorted(merchant_counts.values(), key=lambda x: x["name"])
    ]

    # Format store options
    store_options = [
        {"value": str(s["id"]), "label": s["name"], "count": s["count"]}
        for s in sorted(store_data.values(), key=lambda x: x["name"])
    ]

    return {
        "status": status_options,
        "source": source_options,
        "folder": folder_options,
        "merchant": merchant_options,
        "store": store_options,
        # Also return raw data for backwards compatibility
        "merchants": [{"id": m["id"], "name": m["name"]} for m in merchant_counts.values()],
        "stores": [{"id": s["id"], "name": s["name"]} for s in store_data.values()],
    }


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
    skip_duplicates: bool = Form(False),
):
    """Upload a new image with duplicate detection."""
    from PIL import Image
    import io

    # Read file content
    content = await file.read()

    # Get image dimensions
    img = Image.open(io.BytesIO(content))
    width, height = img.size

    # Calculate hashes for duplicate detection
    file_hash = calculate_file_hash(content)
    phash = calculate_phash(content)

    # Check for duplicates if requested
    if skip_duplicates:
        existing = await check_duplicate_by_hash(file_hash)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate image exists: {existing['id']}"
            )

        if phash:
            similar = await check_duplicate_by_phash(phash, threshold=5)
            if similar:
                raise HTTPException(
                    status_code=409,
                    detail=f"Similar image exists: {similar[0]['id']} ({similar[0]['similarity']:.0%} similar)"
                )

    # Generate unique filename
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    unique_filename = f"{uuid4()}.{ext}"

    # Upload to Supabase Storage (path is relative to bucket)
    try:
        supabase_service.client.storage.from_("od-images").upload(
            unique_filename,
            content,
            {"content-type": file.content_type or "image/jpeg"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to storage: {str(e)}")

    # Get public URL
    image_url = supabase_service.client.storage.from_("od-images").get_public_url(unique_filename)

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
        "file_hash": file_hash,
        "phash": phash,
    }

    result = supabase_service.client.table("od_images").insert(image_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create image record")

    return result.data[0]


@router.post("/bulk", response_model=list[ODImageResponse])
async def upload_images_bulk(
    files: list[UploadFile] = File(...),
    skip_duplicates: bool = Form(False),
):
    """Upload multiple images with duplicate detection."""
    from PIL import Image
    import io

    uploaded = []
    skipped = 0

    for file in files:
        content = await file.read()

        try:
            img = Image.open(io.BytesIO(content))
            width, height = img.size
        except Exception:
            continue

        # Calculate hashes
        file_hash = calculate_file_hash(content)
        phash = calculate_phash(content)

        # Check for duplicates
        if skip_duplicates:
            existing = await check_duplicate_by_hash(file_hash)
            if existing:
                skipped += 1
                continue

            if phash:
                similar = await check_duplicate_by_phash(phash, threshold=5)
                if similar:
                    skipped += 1
                    continue

        ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid4()}.{ext}"

        try:
            supabase_service.client.storage.from_("od-images").upload(
                unique_filename,
                content,
                {"content-type": file.content_type or "image/jpeg"},
            )

            image_url = supabase_service.client.storage.from_("od-images").get_public_url(unique_filename)

            image_data = {
                "filename": unique_filename,
                "original_filename": file.filename,
                "image_url": image_url,
                "width": width,
                "height": height,
                "file_size_bytes": len(content),
                "source": "upload",
                "status": "pending",
                "file_hash": file_hash,
                "phash": phash,
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


# NOTE: /bulk route MUST be defined BEFORE /{image_id} to avoid route matching issues
@router.delete("/bulk")
async def delete_images_bulk(image_ids: list[str]):
    """Delete multiple images using optimized batch operations."""
    if not image_ids:
        return {"deleted": 0, "errors": []}

    errors = []

    # 1. Get all images data in ONE query
    images_result = supabase_service.client.table("od_images").select("id, filename").in_("id", image_ids).execute()
    images_map = {img["id"]: img["filename"] for img in (images_result.data or [])}

    # 2. Check which images are used in datasets in ONE query
    dataset_check = supabase_service.client.table("od_dataset_images").select("image_id").in_("image_id", image_ids).execute()
    used_in_dataset = set(item["image_id"] for item in (dataset_check.data or []))

    # Filter out images used in datasets
    deletable_ids = []
    deletable_filenames = []
    for img_id in image_ids:
        if img_id in used_in_dataset:
            errors.append(f"{img_id}: Used in dataset")
        elif img_id in images_map:
            deletable_ids.append(img_id)
            deletable_filenames.append(images_map[img_id])
        else:
            errors.append(f"{img_id}: Not found")

    if not deletable_ids:
        return {"deleted": 0, "errors": errors}

    # 3. Delete from storage in batch (Supabase supports bulk delete)
    try:
        if deletable_filenames:
            supabase_service.client.storage.from_("od-images").remove(deletable_filenames)
    except Exception as e:
        # Log but don't fail - storage cleanup is secondary
        pass

    # 4. Delete from database in ONE query
    try:
        supabase_service.client.table("od_images").delete().in_("id", deletable_ids).execute()
        deleted = len(deletable_ids)
    except Exception as e:
        errors.append(f"Database delete failed: {str(e)}")
        deleted = 0

    return {"deleted": deleted, "errors": errors}


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
        supabase_service.client.storage.from_("od-images").remove([filename])
    except Exception:
        pass  # Continue even if storage delete fails

    # Delete from database
    supabase_service.client.table("od_images").delete().eq("id", image_id).execute()

    return {"status": "deleted", "id": image_id}


# ===========================================
# Import Endpoints
# ===========================================

@router.post("/import/url", response_model=ImportResultResponse)
async def import_from_url(data: ImportURLsRequest):
    """
    Import images from a list of URLs.

    Supports up to 100 URLs at once. Images are downloaded, validated,
    and stored. Duplicate detection using perceptual hash (pHash).
    """
    result = await import_from_urls(
        urls=data.urls,
        folder=data.folder,
        skip_duplicates=data.skip_duplicates,
    )

    # If dataset_id provided, add images to dataset
    if data.dataset_id and result.images_imported > 0:
        # Get recently imported images
        recent = supabase_service.client.table("od_images").select("id").eq(
            "source", "url"
        ).order("created_at", desc=True).limit(result.images_imported).execute()

        if recent.data:
            image_ids = [img["id"] for img in recent.data]
            for img_id in image_ids:
                try:
                    supabase_service.client.table("od_dataset_images").insert({
                        "dataset_id": data.dataset_id,
                        "image_id": img_id,
                        "status": "pending"
                    }).execute()
                except Exception:
                    pass

            # Update dataset count
            count = supabase_service.client.table("od_dataset_images").select(
                "id", count="exact"
            ).eq("dataset_id", data.dataset_id).execute()

            supabase_service.client.table("od_datasets").update({
                "image_count": count.count or 0
            }).eq("id", data.dataset_id).execute()

    return ImportResultResponse(
        success=result.success,
        images_imported=result.images_imported,
        images_skipped=result.images_skipped,
        duplicates_found=result.duplicates_found,
        errors=result.errors,
    )


@router.post("/import/preview", response_model=ImportPreviewResponse)
async def preview_annotated_import(file: UploadFile = File(...)):
    """
    Preview what will be imported from an annotated dataset file.

    Supports:
    - ZIP files containing COCO, YOLO, Pascal VOC, or LabelMe annotations
    - Single COCO JSON file
    - Single LabelMe JSON file

    Returns detected format, image/annotation counts, and class list.
    """
    content = await file.read()
    preview = await preview_import(content, file.filename)

    return ImportPreviewResponse(
        format_detected=preview.format_detected,
        total_images=preview.total_images,
        total_annotations=preview.total_annotations,
        classes_found=preview.classes_found,
        sample_images=preview.sample_images,
        errors=preview.errors,
    )


@router.post("/import/annotated", response_model=ImportResultResponse)
async def import_annotated(
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    class_mapping_json: str = Form(...),
    skip_duplicates: bool = Form(True),
    merge_annotations: bool = Form(False),
):
    """
    Import an annotated dataset from a ZIP file.

    The ZIP should contain:
    - Images in a folder (images/, train/, etc.)
    - Annotations in one of supported formats (COCO, YOLO, VOC, LabelMe)

    class_mapping_json should be a JSON string array of:
    [{"source_name": "class1", "target_class_id": "uuid", "skip": false}, ...]
    """
    # Parse class mapping
    try:
        mapping_data = json.loads(class_mapping_json)
        class_mapping = [
            ClassMapping(
                source_name=m["source_name"],
                target_class_id=m.get("target_class_id"),
                create_new=m.get("create_new", False),
                skip=m.get("skip", False),
                color=m.get("color"),
            )
            for m in mapping_data
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid class_mapping_json: {e}")

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Handle class creation for "create_new" mappings
    for mapping in class_mapping:
        if mapping.create_new and not mapping.target_class_id:
            # Create new class in dataset
            new_class = supabase_service.client.table("od_classes").insert({
                "dataset_id": dataset_id,
                "name": mapping.source_name.lower().replace(" ", "_"),
                "display_name": mapping.source_name,
                "color": mapping.color or "#3B82F6",
                "is_active": True,
                "annotation_count": 0,
            }).execute()

            if new_class.data:
                mapping.target_class_id = new_class.data[0]["id"]

    content = await file.read()
    result = await import_annotated_dataset(
        zip_content=content,
        dataset_id=dataset_id,
        class_mapping=class_mapping,
        skip_duplicates=skip_duplicates,
        merge_annotations=merge_annotations,
    )

    # Update class annotation counts
    for mapping in class_mapping:
        if mapping.target_class_id:
            count = supabase_service.client.table("od_annotations").select(
                "id", count="exact"
            ).eq("class_id", mapping.target_class_id).execute()

            supabase_service.client.table("od_classes").update({
                "annotation_count": count.count or 0
            }).eq("id", mapping.target_class_id).execute()

    return ImportResultResponse(
        success=result.success,
        images_imported=result.images_imported,
        annotations_imported=result.annotations_imported,
        images_skipped=result.images_skipped,
        duplicates_found=result.duplicates_found,
        errors=result.errors,
    )


# ===========================================
# Duplicate Detection Endpoints
# ===========================================

@router.post("/check-duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate(file: UploadFile = File(...)):
    """
    Check if an image already exists in the library.

    Uses both file hash (exact match) and pHash (similar image detection).
    """
    content = await file.read()

    file_hash = calculate_file_hash(content)
    phash = calculate_phash(content)

    # Check exact match first
    exact_match = await check_duplicate_by_hash(file_hash)
    if exact_match:
        return DuplicateCheckResponse(
            is_duplicate=True,
            similar_images=[{
                "id": exact_match["id"],
                "filename": exact_match["filename"],
                "image_url": exact_match["image_url"],
                "similarity": 1.0,
            }]
        )

    # Check similar by pHash
    if phash:
        similar = await check_duplicate_by_phash(phash, threshold=10)
        if similar:
            return DuplicateCheckResponse(
                is_duplicate=True,
                similar_images=[
                    {
                        "id": img["id"],
                        "filename": img["filename"],
                        "image_url": img["image_url"],
                        "similarity": img["similarity"],
                    }
                    for img in similar[:5]
                ]
            )

    return DuplicateCheckResponse(is_duplicate=False, similar_images=[])


@router.get("/duplicates", response_model=DuplicateGroupsResponse)
async def list_duplicate_groups(threshold: int = 10):
    """
    Get all groups of duplicate/similar images.

    Args:
        threshold: Hamming distance threshold (0-64). Lower = more similar.
                   Default 10 means ~84% similar.

    Returns groups of similar images sorted by similarity.
    """
    groups = await get_duplicate_groups(threshold=threshold)

    return DuplicateGroupsResponse(
        groups=[
            {"images": g["images"], "max_similarity": g["max_similarity"]}
            for g in groups
        ],
        total_groups=len(groups),
    )


@router.post("/duplicates/resolve")
async def resolve_duplicates(
    keep_image_id: str,
    delete_image_ids: list[str],
    merge_to_datasets: bool = False,
):
    """
    Resolve a duplicate group by keeping one image and deleting others.

    Args:
        keep_image_id: The image ID to keep
        delete_image_ids: List of image IDs to delete
        merge_to_datasets: If True, transfer dataset memberships to kept image
    """
    deleted = 0
    errors = []

    for img_id in delete_image_ids:
        if img_id == keep_image_id:
            continue

        try:
            if merge_to_datasets:
                # Get datasets this image belongs to
                memberships = supabase_service.client.table("od_dataset_images").select(
                    "dataset_id"
                ).eq("image_id", img_id).execute()

                for membership in memberships.data or []:
                    # Check if kept image already in this dataset
                    existing = supabase_service.client.table("od_dataset_images").select(
                        "id"
                    ).eq("dataset_id", membership["dataset_id"]).eq(
                        "image_id", keep_image_id
                    ).execute()

                    if not existing.data:
                        # Add kept image to dataset
                        supabase_service.client.table("od_dataset_images").insert({
                            "dataset_id": membership["dataset_id"],
                            "image_id": keep_image_id,
                        }).execute()

            # Delete annotations for this image
            supabase_service.client.table("od_annotations").delete().eq("image_id", img_id).execute()

            # Delete dataset memberships
            supabase_service.client.table("od_dataset_images").delete().eq("image_id", img_id).execute()

            # Get filename for storage deletion
            image = supabase_service.client.table("od_images").select("filename").eq("id", img_id).single().execute()
            if image.data:
                try:
                    supabase_service.client.storage.from_("od-images").remove([image.data["filename"]])
                except Exception:
                    pass

            # Delete image record
            supabase_service.client.table("od_images").delete().eq("id", img_id).execute()
            deleted += 1

        except Exception as e:
            errors.append(f"{img_id}: {str(e)}")

    return {"deleted": deleted, "errors": errors}


# ===========================================
# Bulk Operations
# ===========================================

@router.post("/bulk/tags", response_model=BulkOperationResponse)
async def bulk_tag_images(data: BulkTagRequest):
    """
    Add, remove, or replace tags on multiple images.

    operation can be:
    - "add": Add tags to existing tags
    - "remove": Remove specified tags
    - "replace": Replace all tags with new ones
    """
    affected = 0
    errors = []

    for img_id in data.image_ids:
        try:
            if data.operation == "replace":
                supabase_service.client.table("od_images").update({
                    "tags": data.tags
                }).eq("id", img_id).execute()
                affected += 1
            else:
                # Get current tags
                img = supabase_service.client.table("od_images").select("tags").eq("id", img_id).single().execute()
                current_tags = img.data.get("tags") or [] if img.data else []

                if data.operation == "add":
                    new_tags = list(set(current_tags + data.tags))
                elif data.operation == "remove":
                    new_tags = [t for t in current_tags if t not in data.tags]
                else:
                    new_tags = current_tags

                supabase_service.client.table("od_images").update({
                    "tags": new_tags
                }).eq("id", img_id).execute()
                affected += 1

        except Exception as e:
            errors.append(f"{img_id}: {str(e)}")

    return BulkOperationResponse(success=True, affected_count=affected, errors=errors)


@router.post("/bulk/move", response_model=BulkOperationResponse)
async def bulk_move_images(data: BulkMoveRequest):
    """Move multiple images to a folder."""
    try:
        result = supabase_service.client.table("od_images").update({
            "folder": data.folder
        }).in_("id", data.image_ids).execute()

        return BulkOperationResponse(
            success=True,
            affected_count=len(result.data or []),
            errors=[]
        )
    except Exception as e:
        return BulkOperationResponse(
            success=False,
            affected_count=0,
            errors=[str(e)]
        )


@router.post("/bulk/add-to-dataset", response_model=BulkOperationResponse)
async def bulk_add_to_dataset(image_ids: list[str], dataset_id: str):
    """Add multiple images to a dataset."""
    added = 0
    skipped = 0
    errors = []

    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    for img_id in image_ids:
        try:
            # Check if already in dataset
            existing = supabase_service.client.table("od_dataset_images").select(
                "id"
            ).eq("dataset_id", dataset_id).eq("image_id", img_id).execute()

            if existing.data:
                skipped += 1
                continue

            # Add to dataset
            supabase_service.client.table("od_dataset_images").insert({
                "dataset_id": dataset_id,
                "image_id": img_id,
                "status": "pending"
            }).execute()
            added += 1

        except Exception as e:
            errors.append(f"{img_id}: {str(e)}")

    # Update dataset count
    count = supabase_service.client.table("od_dataset_images").select(
        "id", count="exact"
    ).eq("dataset_id", dataset_id).execute()

    supabase_service.client.table("od_datasets").update({
        "image_count": count.count or 0
    }).eq("id", dataset_id).execute()

    return BulkOperationResponse(
        success=True,
        affected_count=added,
        errors=errors + ([f"Skipped {skipped} (already in dataset)"] if skipped > 0 else [])
    )


# ===========================================
# Bulk Operations by Filters
# ===========================================

class BulkFilterRequest(BaseModel):
    """Request for bulk operations using filters instead of IDs."""
    search: Optional[str] = None
    statuses: Optional[str] = None
    sources: Optional[str] = None
    folders: Optional[str] = None
    merchant_ids: Optional[str] = None
    store_ids: Optional[str] = None


def build_filter_query(filters: BulkFilterRequest):
    """Build a query with filters applied."""
    query = supabase_service.client.table("od_images").select("id")

    if filters.statuses:
        status_list = [s.strip() for s in filters.statuses.split(",") if s.strip()]
        if status_list:
            query = query.in_("status", status_list)

    if filters.sources:
        source_list = [s.strip() for s in filters.sources.split(",") if s.strip()]
        if source_list:
            query = query.in_("source", source_list)

    if filters.folders:
        folder_list = [f.strip() for f in filters.folders.split(",") if f.strip()]
        if folder_list:
            query = query.in_("folder", folder_list)

    if filters.search:
        query = query.ilike("filename", f"%{filters.search}%")

    if filters.merchant_ids:
        merchant_list = [int(m.strip()) for m in filters.merchant_ids.split(",") if m.strip().isdigit()]
        if merchant_list:
            query = query.in_("merchant_id", merchant_list)

    if filters.store_ids:
        store_list = [int(s.strip()) for s in filters.store_ids.split(",") if s.strip().isdigit()]
        if store_list:
            query = query.in_("store_id", store_list)

    return query


@router.post("/bulk/delete-by-filters")
async def bulk_delete_by_filters(filters: BulkFilterRequest):
    """Delete all images matching the given filters using optimized batch operations."""
    # Get all matching image IDs and filenames
    query = build_filter_query(filters)
    result = query.execute()

    if not result.data:
        return {"deleted": 0, "total_matched": 0, "errors": [], "message": "No images match the filters"}

    image_ids = [img["id"] for img in result.data]
    images_map = {img["id"]: img.get("filename", "") for img in result.data}
    total_matched = len(image_ids)
    errors = []

    # Check which images are used in datasets in ONE query
    dataset_check = supabase_service.client.table("od_dataset_images").select("image_id").in_("image_id", image_ids).execute()
    used_in_dataset = set(item["image_id"] for item in (dataset_check.data or []))

    # Filter out images used in datasets
    deletable_ids = []
    deletable_filenames = []
    for img_id in image_ids:
        if img_id in used_in_dataset:
            errors.append(f"{img_id}: Used in dataset")
        else:
            deletable_ids.append(img_id)
            if images_map.get(img_id):
                deletable_filenames.append(images_map[img_id])

    if not deletable_ids:
        return {
            "deleted": 0,
            "total_matched": total_matched,
            "errors": errors[:20] if errors else [],
            "message": f"No images could be deleted (all are used in datasets)"
        }

    # Delete from storage in batches (Supabase has a limit on bulk operations)
    STORAGE_BATCH_SIZE = 100
    for i in range(0, len(deletable_filenames), STORAGE_BATCH_SIZE):
        batch = deletable_filenames[i:i + STORAGE_BATCH_SIZE]
        try:
            supabase_service.client.storage.from_("od-images").remove(batch)
        except Exception:
            pass  # Storage cleanup is secondary

    # Delete from database in batches
    DB_BATCH_SIZE = 500
    deleted = 0
    for i in range(0, len(deletable_ids), DB_BATCH_SIZE):
        batch = deletable_ids[i:i + DB_BATCH_SIZE]
        try:
            supabase_service.client.table("od_images").delete().in_("id", batch).execute()
            deleted += len(batch)
        except Exception as e:
            errors.append(f"Batch delete failed: {str(e)}")

    return {
        "deleted": deleted,
        "total_matched": total_matched,
        "errors": errors[:20] if errors else [],
        "message": f"Deleted {deleted} of {total_matched} images"
    }


@router.post("/bulk/add-to-dataset-by-filters")
async def bulk_add_to_dataset_by_filters(dataset_id: str, filters: BulkFilterRequest):
    """Add all images matching filters to a dataset."""
    # Verify dataset exists
    dataset = supabase_service.client.table("od_datasets").select("id").eq("id", dataset_id).single().execute()
    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get all matching image IDs
    query = build_filter_query(filters)
    result = query.execute()

    if not result.data:
        return {"added": 0, "skipped": 0, "errors": [], "message": "No images match the filters"}

    image_ids = [img["id"] for img in result.data]
    added = 0
    skipped = 0
    errors = []

    # Get existing images in dataset
    existing_result = supabase_service.client.table("od_dataset_images").select("image_id").eq("dataset_id", dataset_id).execute()
    existing_ids = {r["image_id"] for r in (existing_result.data or [])}

    # Add in batches
    BATCH_SIZE = 100
    for i in range(0, len(image_ids), BATCH_SIZE):
        batch = image_ids[i:i + BATCH_SIZE]
        new_links = []
        for img_id in batch:
            if img_id in existing_ids:
                skipped += 1
                continue
            new_links.append({
                "dataset_id": dataset_id,
                "image_id": img_id,
                "status": "pending"
            })

        if new_links:
            try:
                supabase_service.client.table("od_dataset_images").insert(new_links).execute()
                added += len(new_links)
            except Exception as e:
                errors.append(f"Batch {i//BATCH_SIZE + 1}: {str(e)}")

    # Update dataset count
    count = supabase_service.client.table("od_dataset_images").select(
        "id", count="exact"
    ).eq("dataset_id", dataset_id).execute()

    supabase_service.client.table("od_datasets").update({
        "image_count": count.count or 0
    }).eq("id", dataset_id).execute()

    return {
        "added": added,
        "skipped": skipped,
        "total_matched": len(image_ids),
        "errors": errors[:10] if errors else [],
        "message": f"Added {added} images, skipped {skipped} (already in dataset)"
    }


# ===========================================
# BuyBuddy Sync Endpoints
# ===========================================

@router.get("/buybuddy/status")
async def check_buybuddy_status():
    """Check if BuyBuddy API is configured and accessible."""
    configured = buybuddy_service.is_configured()

    if not configured:
        return {
            "configured": False,
            "accessible": False,
            "message": "BuyBuddy API credentials not configured"
        }

    try:
        # Try to get a small batch to verify access
        await buybuddy_service.get_evaluation_images(limit=1)
        return {
            "configured": True,
            "accessible": True,
            "message": "BuyBuddy API is accessible"
        }
    except Exception as e:
        return {
            "configured": True,
            "accessible": False,
            "message": f"Failed to access BuyBuddy API: {str(e)}"
        }


@router.get("/buybuddy/preview")
async def preview_buybuddy_sync(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    is_annotated: Optional[bool] = Query(None, description="Filter by annotation status"),
    is_approved: Optional[bool] = Query(None, description="Filter by approval status"),
    limit: int = Query(10, description="Number of preview images"),
):
    """Preview images available for sync from BuyBuddy."""
    if not buybuddy_service.is_configured():
        raise HTTPException(status_code=400, detail="BuyBuddy API not configured")

    try:
        result = await buybuddy_service.preview_od_sync(
            start_date=start_date,
            end_date=end_date,
            store_id=store_id,
            is_annotated=is_annotated,
            is_approved=is_approved,
            limit=limit,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preview BuyBuddy images: {str(e)}")


class BuyBuddySyncRequest(BaseModel):
    """Request to sync images from BuyBuddy."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    store_id: Optional[int] = None
    is_annotated: Optional[bool] = None
    is_approved: Optional[bool] = None
    max_images: Optional[int] = None
    dataset_id: Optional[str] = None
    tags: Optional[list[str]] = None


@router.post("/buybuddy/sync")
async def sync_from_buybuddy(
    request: BuyBuddySyncRequest,
    supabase: SupabaseService = Depends(get_supabase),
):
    """Sync images from BuyBuddy to OD system."""
    if not buybuddy_service.is_configured():
        raise HTTPException(status_code=400, detail="BuyBuddy API not configured")

    try:
        # Fetch images from BuyBuddy
        images = await buybuddy_service.sync_od_images(
            start_date=request.start_date,
            end_date=request.end_date,
            store_id=request.store_id,
            is_annotated=request.is_annotated,
            is_approved=request.is_approved,
            max_images=request.max_images,
        )

        if not images:
            return {
                "synced": 0,
                "skipped": 0,
                "errors": [],
                "message": "No images found matching criteria"
            }

        # Get all existing buybuddy_image_ids in one query (much faster)
        bb_ids = [img["buybuddy_image_id"] for img in images]
        existing_result = supabase.client.table("od_images").select("buybuddy_image_id").in_(
            "buybuddy_image_id", bb_ids
        ).execute()
        existing_ids = {r["buybuddy_image_id"] for r in (existing_result.data or [])}

        # Prepare batch insert data
        new_images = []
        skipped = len(existing_ids)

        for img in images:
            if img["buybuddy_image_id"] in existing_ids:
                continue

            image_url = img["image_url"]
            filename = image_url.split("/")[-1] if image_url else f"buybuddy_{img['buybuddy_image_id']}.jpg"

            image_data = {
                "filename": filename,
                "image_url": image_url,
                "buybuddy_image_id": img["buybuddy_image_id"],
                "source": "upload",
                "status": "pending",
                "width": 0,
                "height": 0,
                # Direct columns for filtering
                "merchant_id": img.get("merchant_id"),
                "merchant_name": img.get("merchant_name"),
                "store_id": img.get("store_id"),
                "store_name": img.get("store_name"),
                # Full metadata for reference
                "metadata": {
                    "source_type": "buybuddy",
                    "image_type": img.get("image_type"),
                    "basket_id": img.get("basket_id"),
                    "basket_identifier": img.get("basket_identifier"),
                    "merchant_id": img.get("merchant_id"),
                    "merchant_name": img.get("merchant_name"),
                    "store_id": img.get("store_id"),
                    "store_name": img.get("store_name"),
                    "store_code": img.get("store_code"),
                    "is_annotated": img.get("is_annotated"),
                    "is_approved": img.get("is_approved"),
                    "annotation_id": img.get("annotation_id"),
                    "inserted_at": img.get("inserted_at"),
                }
            }

            if request.tags:
                image_data["tags"] = request.tags

            new_images.append(image_data)

        # Batch insert (100 at a time)
        synced = 0
        errors = []
        BATCH_SIZE = 100

        for i in range(0, len(new_images), BATCH_SIZE):
            batch = new_images[i:i + BATCH_SIZE]
            try:
                result = supabase.client.table("od_images").insert(batch).execute()
                synced += len(result.data) if result.data else 0
            except Exception as e:
                errors.append(f"Batch {i//BATCH_SIZE + 1}: {str(e)}")

        # Add to dataset if specified
        if request.dataset_id and synced > 0:
            try:
                # Get IDs of newly inserted images
                new_ids_result = supabase.client.table("od_images").select("id").in_(
                    "buybuddy_image_id", [img["buybuddy_image_id"] for img in new_images[:synced]]
                ).execute()

                if new_ids_result.data:
                    dataset_links = [
                        {"dataset_id": request.dataset_id, "image_id": r["id"], "status": "pending"}
                        for r in new_ids_result.data
                    ]
                    # Batch insert dataset links
                    for i in range(0, len(dataset_links), BATCH_SIZE):
                        batch = dataset_links[i:i + BATCH_SIZE]
                        try:
                            supabase.client.table("od_dataset_images").insert(batch).execute()
                        except Exception:
                            pass  # Ignore duplicate key errors
            except Exception as e:
                errors.append(f"Dataset linking: {str(e)}")

        return {
            "synced": synced,
            "skipped": skipped,
            "total_found": len(images),
            "errors": errors[:10] if errors else [],
            "message": f"Synced {synced} images, skipped {skipped} duplicates"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync from BuyBuddy: {str(e)}")


# ===========================================
# Roboflow Import Endpoints
# ===========================================

from services.roboflow import roboflow_service, RoboflowService


class RoboflowValidateRequest(BaseModel):
    """Request to validate Roboflow API key."""
    api_key: str


@router.post("/roboflow/validate-key")
async def validate_roboflow_key(request: RoboflowValidateRequest):
    """Validate a Roboflow API key and return workspace info."""
    result = await roboflow_service.validate_api_key(request.api_key)
    return result


@router.get("/roboflow/projects")
async def list_roboflow_projects(
    api_key: str = Query(..., description="Roboflow API key"),
    workspace: str = Query(..., description="Workspace URL/slug"),
):
    """List projects in a Roboflow workspace."""
    try:
        projects = await roboflow_service.list_projects(api_key, workspace)
        return projects  # Return array directly
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to list projects: {str(e)}")


@router.get("/roboflow/versions")
async def list_roboflow_versions(
    api_key: str = Query(..., description="Roboflow API key"),
    workspace: str = Query(..., description="Workspace URL/slug"),
    project: str = Query(..., description="Project URL/slug"),
):
    """List versions of a Roboflow project."""
    try:
        versions = await roboflow_service.list_versions(api_key, workspace, project)
        return versions  # Return array directly
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to list versions: {str(e)}")


@router.get("/roboflow/preview")
async def preview_roboflow_import(
    api_key: str = Query(..., description="Roboflow API key"),
    workspace: str = Query(..., description="Workspace URL/slug"),
    project: str = Query(..., description="Project URL/slug"),
    version: int = Query(..., description="Version number"),
):
    """Preview what will be imported from a Roboflow dataset."""
    try:
        preview = await roboflow_service.preview_import(api_key, workspace, project, version)
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preview import: {str(e)}")


class RoboflowClassMapping(BaseModel):
    """Class mapping for Roboflow import."""
    source_name: str
    target_class_id: Optional[str] = None
    create_new: bool = False
    skip: bool = False
    color: Optional[str] = None


class RoboflowImportRequest(BaseModel):
    """Request to import from Roboflow."""
    api_key: str
    workspace: str
    project: str
    version: int
    dataset_id: str
    class_mapping: list[RoboflowClassMapping]
    format: str = "coco"
    skip_duplicates: bool = True
    merge_annotations: bool = False


async def _run_roboflow_import(job_id: str, request_data: dict):
    """Background task to run Roboflow import with disk streaming."""
    import os
    import logging
    import httpx
    from services.od_import import import_annotated_dataset_from_file, ClassMapping

    logger = logging.getLogger(__name__)

    def update_job(progress: int, stage: str, message: str, status: str = "running"):
        """Helper to update job progress."""
        try:
            supabase_service.client.table("jobs").update({
                "status": status,
                "progress": progress,
                "result": {"stage": stage, "message": message}
            }).eq("id", job_id).execute()
        except Exception:
            pass  # Don't fail if we can't update progress

    def fail_job(error: str):
        """Helper to mark job as failed."""
        logger.error(f"Roboflow import job {job_id} failed: {error}")
        try:
            supabase_service.client.table("jobs").update({
                "status": "failed",
                "error": error
            }).eq("id", job_id).execute()
        except Exception:
            pass

    temp_file_path = None

    try:
        # Update job status to downloading
        update_job(5, "downloading", "Starting download from Roboflow...")

        # Track progress for UI updates
        last_progress_mb = 0

        def progress_callback(downloaded_bytes: int, total_bytes: int | None):
            nonlocal last_progress_mb
            mb_downloaded = downloaded_bytes // (1024 * 1024)

            # Update progress every 10MB
            if mb_downloaded >= last_progress_mb + 10:
                last_progress_mb = mb_downloaded

                if total_bytes:
                    # Calculate percentage based on actual file size
                    total_mb = total_bytes // (1024 * 1024)
                    download_pct = min(45, 5 + int((downloaded_bytes / total_bytes) * 40))
                    update_job(
                        download_pct,
                        "downloading",
                        f"Downloaded {mb_downloaded} MB / {total_mb} MB..."
                    )
                else:
                    # Unknown total size, just show progress
                    download_progress = min(45, 5 + int(mb_downloaded / 100))
                    update_job(
                        download_progress,
                        "downloading",
                        f"Downloaded {mb_downloaded} MB..."
                    )

        # Download to temporary file (memory-efficient)
        try:
            temp_file_path, total_bytes = await roboflow_service.download_dataset_to_file(
                api_key=request_data["api_key"],
                workspace=request_data["workspace"],
                project=request_data["project"],
                version=request_data["version"],
                format=request_data.get("format", "coco"),
                max_retries=3,
                progress_callback=progress_callback,
            )

        except httpx.TimeoutException as e:
            fail_job(
                f"Download timed out after multiple retries. "
                f"The dataset might be too large or the connection is slow. "
                f"Error: {str(e)}"
            )
            return
        except httpx.NetworkError as e:
            fail_job(
                f"Network error during download. Please check your internet connection. "
                f"Error: {str(e)}"
            )
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                fail_job("Invalid API key or unauthorized access to this project.")
            elif e.response.status_code == 404:
                fail_job("Project or version not found. Please verify the project details.")
            elif e.response.status_code == 429:
                fail_job("Rate limited by Roboflow. Please wait a few minutes and try again.")
            else:
                fail_job(f"Roboflow API error (HTTP {e.response.status_code}): {str(e)}")
            return

        if not temp_file_path or not os.path.exists(temp_file_path):
            fail_job("Download failed - no file was created")
            return

        # Verify file size
        file_size = os.path.getsize(temp_file_path)
        if file_size == 0:
            fail_job("Download failed - empty file received from Roboflow")
            return

        total_mb = file_size // (1024 * 1024)
        update_job(50, "processing", f"Download complete ({total_mb} MB). Processing images...")

        logger.info(f"Roboflow download complete for job {job_id}: {total_mb} MB")

        # Convert class mapping
        class_mapping_list = [
            ClassMapping(
                source_name=m["source_name"],
                target_class_id=m.get("target_class_id"),
                create_new=m.get("create_new", False),
                skip=m.get("skip", False),
                color=m.get("color"),
            )
            for m in request_data.get("class_mapping", [])
        ]

        # Import using file-based import function
        result = await import_annotated_dataset_from_file(
            zip_file_path=temp_file_path,
            dataset_id=request_data["dataset_id"],
            class_mapping=class_mapping_list,
            skip_duplicates=request_data.get("skip_duplicates", True),
            merge_annotations=request_data.get("merge_annotations", False),
        )

        # Update job with final result
        supabase_service.client.table("jobs").update({
            "status": "completed" if result.success else "failed",
            "progress": 100,
            "result": {
                "stage": "completed",
                "success": result.success,
                "images_imported": result.images_imported,
                "annotations_imported": result.annotations_imported,
                "images_skipped": result.images_skipped,
                "duplicates_found": result.duplicates_found,
                "errors": result.errors[:20] if result.errors else [],
            },
            "error": result.errors[0] if result.errors and not result.success else None
        }).eq("id", job_id).execute()

        logger.info(
            f"Roboflow import job {job_id} completed: "
            f"{result.images_imported} images, {result.annotations_imported} annotations"
        )

    except MemoryError:
        fail_job(
            "Out of memory while processing images. "
            "The dataset might be too large. Try importing a smaller version."
        )
    except Exception as e:
        logger.exception(f"Unexpected error in Roboflow import job {job_id}")
        fail_job(f"Unexpected error: {str(e)}")

    finally:
        # Always clean up temp file after processing
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")


@router.post("/roboflow/import")
async def import_from_roboflow(
    request: RoboflowImportRequest,
    background_tasks: BackgroundTasks,
):
    """Start a background import from Roboflow. Returns job ID for tracking."""
    try:
        # Verify dataset exists
        dataset = supabase_service.client.table("od_datasets").select("id, name").eq("id", request.dataset_id).single().execute()
        if not dataset.data:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Create job record
        job_data = {
            "type": "roboflow_import",
            "status": "pending",
            "progress": 0,
            "config": {
                "workspace": request.workspace,
                "project": request.project,
                "version": request.version,
                "dataset_id": request.dataset_id,
                "format": request.format,
            }
        }
        job_result = supabase_service.client.table("jobs").insert(job_data).execute()
        job = job_result.data[0]

        # Prepare request data for background task
        request_data = {
            "api_key": request.api_key,
            "workspace": request.workspace,
            "project": request.project,
            "version": request.version,
            "dataset_id": request.dataset_id,
            "format": request.format,
            "skip_duplicates": request.skip_duplicates,
            "merge_annotations": request.merge_annotations,
            "class_mapping": [
                {
                    "source_name": m.source_name,
                    "target_class_id": m.target_class_id,
                    "create_new": m.create_new,
                    "skip": m.skip,
                    "color": m.color,
                }
                for m in request.class_mapping
            ]
        }

        # Start background task
        background_tasks.add_task(_run_roboflow_import, job["id"], request_data)

        return {
            "job_id": job["id"],
            "status": "started",
            "message": "Import started. Use job_id to track progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start import: {str(e)}")


@router.get("/roboflow/import/{job_id}")
async def get_roboflow_import_status(job_id: str):
    """Get the status of a Roboflow import job."""
    result = supabase_service.client.table("jobs").select("*").eq("id", job_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = result.data
    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job.get("updated_at"),
    }
