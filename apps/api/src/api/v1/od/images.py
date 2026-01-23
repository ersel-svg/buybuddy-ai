"""
Object Detection - Images Router

Endpoints for managing OD images (upload, list, CRUD).
Includes advanced import features (URL, annotated datasets, duplicate detection).
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, Depends
from pydantic import BaseModel
from uuid import uuid4
import json

logger = logging.getLogger(__name__)

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
    parallel_upload_images,
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
    """Get filter options with counts for FilterDrawer component.

    Uses SQL aggregation via RPC for optimal performance with large datasets.
    """
    try:
        # Use optimized SQL aggregation via RPC function
        result = supabase_service.client.rpc("get_od_image_filter_options").execute()

        if result.data:
            return result.data

        # Fallback to empty structure if RPC returns nothing
        return {
            "status": [],
            "source": [],
            "folder": [],
            "merchant": [],
            "store": [],
            "merchants": [],
            "stores": [],
        }
    except Exception as e:
        # If RPC fails (e.g., function doesn't exist yet), fall back to Python aggregation
        logger.warning(f"RPC get_od_image_filter_options failed, using fallback: {e}")
        return await _get_filter_options_fallback()


async def _get_filter_options_fallback():
    """
    Fallback filter options using optimized separate queries.
    
    Instead of fetching all rows (8000+) and counting in Python,
    we use separate COUNT queries with LIMIT to get aggregated results.
    This is much faster and uses less memory.
    """
    import asyncio
    
    # Helper to run count query
    def get_status_counts():
        """Get status counts using a simple approach."""
        result = supabase_service.client.table("od_images").select("status").execute()
        counts = {}
        for row in result.data or []:
            status = row.get("status") or "pending"
            counts[status] = counts.get(status, 0) + 1
        return counts
    
    def get_source_counts():
        """Get source counts."""
        result = supabase_service.client.table("od_images").select("source").execute()
        counts = {}
        for row in result.data or []:
            source = row.get("source") or "upload"
            counts[source] = counts.get(source, 0) + 1
        return counts
    
    def get_folder_counts():
        """Get folder counts - only non-null folders."""
        result = supabase_service.client.table("od_images").select("folder").not_.is_("folder", "null").execute()
        counts = {}
        for row in result.data or []:
            folder = row.get("folder")
            if folder:
                counts[folder] = counts.get(folder, 0) + 1
        return counts
    
    def get_merchant_data():
        """Get merchant data with counts."""
        result = supabase_service.client.table("od_images").select("merchant_id, merchant_name").not_.is_("merchant_id", "null").execute()
        merchants = {}
        for row in result.data or []:
            mid = row.get("merchant_id")
            if mid:
                if mid not in merchants:
                    merchants[mid] = {
                        "id": mid,
                        "name": row.get("merchant_name") or f"Merchant {mid}",
                        "count": 0
                    }
                merchants[mid]["count"] += 1
        return merchants
    
    def get_store_data():
        """Get store data with counts."""
        result = supabase_service.client.table("od_images").select("store_id, store_name").not_.is_("store_id", "null").execute()
        stores = {}
        for row in result.data or []:
            sid = row.get("store_id")
            if sid:
                if sid not in stores:
                    stores[sid] = {
                        "id": sid,
                        "name": row.get("store_name") or f"Store {sid}",
                        "count": 0
                    }
                stores[sid]["count"] += 1
        return stores

    # Run queries (these are lighter because we only select specific columns)
    status_counts = get_status_counts()
    source_counts = get_source_counts()
    folder_counts = get_folder_counts()
    merchant_data = get_merchant_data()
    store_data = get_store_data()

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
        for m in sorted(merchant_data.values(), key=lambda x: x["name"])
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
        "merchants": [{"id": m["id"], "name": m["name"]} for m in merchant_data.values()],
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
    """
    Upload multiple images with duplicate detection.

    Uses SOTA batch processing with:
    - Adaptive rate limiting
    - Parallel uploads (2 concurrent)
    - Batch processing (100 images per batch)
    - Automatic retry with exponential backoff
    """
    from PIL import Image
    import io

    # ============================================
    # PHASE 1: Prepare all images (validation + duplicate check)
    # ============================================
    upload_items = []  # (filename, content, content_type)
    upload_metadata = []  # (filename, original_filename, width, height, file_size, file_hash, phash)
    skipped = 0

    for file in files:
        content = await file.read()

        # Validate image
        try:
            img = Image.open(io.BytesIO(content))
            width, height = img.size
        except Exception:
            logger.warning(f"Invalid image: {file.filename}")
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

        # Generate unique filename
        ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid4()}.{ext}"
        content_type = file.content_type or "image/jpeg"

        upload_items.append((unique_filename, content, content_type))
        upload_metadata.append({
            "filename": unique_filename,
            "original_filename": file.filename,
            "width": width,
            "height": height,
            "file_size": len(content),
            "file_hash": file_hash,
            "phash": phash,
        })

    if not upload_items:
        logger.info(f"No images to upload (skipped: {skipped})")
        return []

    logger.info(f"Prepared {len(upload_items)} images for bulk upload (skipped: {skipped})")

    # ============================================
    # PHASE 2: Parallel upload to storage (SOTA)
    # ============================================
    successful_files, failed_uploads = await parallel_upload_images(
        upload_items,
        max_concurrent=2,
        batch_size=100,
        batch_delay=2.0,
    )

    if failed_uploads:
        for filename, error in failed_uploads:
            logger.warning(f"Failed to upload {filename}: {error}")

    if not successful_files:
        logger.error("All uploads failed")
        return []

    # ============================================
    # PHASE 3: Insert to database (batch)
    # ============================================
    successful_set = set(successful_files)
    images_to_insert = []

    for meta in upload_metadata:
        if meta["filename"] in successful_set:
            image_url = supabase_service.client.storage.from_("od-images").get_public_url(meta["filename"])
            images_to_insert.append({
                "filename": meta["filename"],
                "original_filename": meta["original_filename"],
                "image_url": image_url,
                "width": meta["width"],
                "height": meta["height"],
                "file_size_bytes": meta["file_size"],
                "source": "upload",
                "status": "pending",
                "file_hash": meta["file_hash"],
                "phash": meta["phash"],
            })

    uploaded = []
    if images_to_insert:
        try:
            # Batch insert (500 at a time)
            batch_size = 500
            for i in range(0, len(images_to_insert), batch_size):
                batch = images_to_insert[i:i + batch_size]
                result = supabase_service.client.table("od_images").insert(batch).execute()
                if result.data:
                    uploaded.extend(result.data)
            logger.info(f"Inserted {len(uploaded)} images to database")
        except Exception as e:
            logger.error(f"Database insert failed: {e}")

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


async def _run_buybuddy_sync(job_id: str, request_data: dict, resume_checkpoint=None):
    """
    Background task to run BuyBuddy sync with progress tracking.

    Args:
        job_id: The job ID
        request_data: Sync configuration
        resume_checkpoint: Optional BuyBuddySyncCheckpoint to resume from
    """
    import time
    from datetime import datetime, timezone
    from services.import_checkpoint import checkpoint_service, BuyBuddySyncCheckpoint

    start_time = time.time()

    # Initialize or use existing checkpoint
    if resume_checkpoint:
        checkpoint = resume_checkpoint
        logger.info(f"Resuming BuyBuddy sync job {job_id} from {len(checkpoint.synced_ids)} images")
    else:
        checkpoint = BuyBuddySyncCheckpoint(job_id=job_id)

    def update_job(progress: int, stage: str, message: str, status: str = "running",
                   synced: int = 0, skipped: int = 0, total: int = 0):
        """Helper to update job progress."""
        elapsed = time.time() - start_time
        speed = synced / elapsed if elapsed > 0 and synced > 0 else 0
        remaining = total - synced - skipped if total > 0 else 0
        eta = remaining / speed if speed > 0 else 0

        result_data = {
            "stage": stage,
            "message": message,
            "can_resume": True,
            "synced": synced,
            "skipped": skipped,
            "total_found": total,
            "images_per_second": round(speed, 2),
            "eta_seconds": int(eta),
            "elapsed_seconds": int(elapsed),
            "started_at": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
        }

        # Preserve checkpoint in result
        result_data["checkpoint"] = checkpoint.to_dict()

        try:
            supabase_service.client.table("jobs").update({
                "status": status,
                "progress": progress,
                "result": result_data,
            }).eq("id", job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update job progress: {e}")

    def fail_job(error: str):
        """Helper to mark job as failed with resume capability."""
        checkpoint_service.save_buybuddy_checkpoint(checkpoint)
        try:
            supabase_service.client.table("jobs").update({
                "status": "failed",
                "error": error,
                "result": {
                    "stage": "failed",
                    "error": error,
                    "can_resume": True,
                    "checkpoint": checkpoint.to_dict(),
                    "synced_count": len(checkpoint.synced_ids),
                    "total_images": checkpoint.total_images,
                }
            }).eq("id", job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update job failure status: {e}")

    try:
        update_job(5, "fetching", "Fetching images from BuyBuddy API...")

        # Step 1: Get total count and fetch images
        images = await buybuddy_service.sync_od_images(
            start_date=request_data.get("start_date"),
            end_date=request_data.get("end_date"),
            store_id=request_data.get("store_id"),
            is_annotated=request_data.get("is_annotated"),
            is_approved=request_data.get("is_approved"),
            max_images=request_data.get("max_images"),
        )

        if not images:
            supabase_service.client.table("jobs").update({
                "status": "completed",
                "progress": 100,
                "result": {
                    "stage": "completed",
                    "success": True,
                    "synced": 0,
                    "skipped": 0,
                    "total_found": 0,
                    "message": "No images found matching criteria",
                }
            }).eq("id", job_id).execute()
            return

        checkpoint.total_images = len(images)
        update_job(10, "checking", f"Found {len(images)} images, checking for duplicates...",
                  synced=0, skipped=0, total=len(images))

        # Step 2: Check existing images (batched)
        bb_ids = [img["buybuddy_image_id"] for img in images]
        existing_ids = set()
        BATCH_SIZE = 100

        for i in range(0, len(bb_ids), BATCH_SIZE):
            batch = bb_ids[i:i + BATCH_SIZE]
            try:
                existing_result = supabase_service.client.table("od_images").select(
                    "buybuddy_image_id"
                ).in_("buybuddy_image_id", batch).execute()
                existing_ids.update(r["buybuddy_image_id"] for r in (existing_result.data or []))
            except Exception as e:
                logger.warning(f"Error checking existing IDs: {e}")

        # If resuming, add already synced IDs to existing
        if resume_checkpoint:
            existing_ids.update(checkpoint.synced_ids)

        skipped = len(existing_ids)

        # Step 3: Prepare new images
        new_images = []
        tags = request_data.get("tags")

        for img in images:
            if img["buybuddy_image_id"] in existing_ids:
                continue

            image_url = img["image_url"]
            filename = image_url.split("/")[-1] if image_url else f"buybuddy_{img['buybuddy_image_id']}.jpg"

            image_data = {
                "filename": filename,
                "image_url": image_url,
                "buybuddy_image_id": img["buybuddy_image_id"],
                "source": "buybuddy_sync",  # Fixed: was "upload"
                "status": "pending",
                "width": 0,
                "height": 0,
                "merchant_id": img.get("merchant_id"),
                "merchant_name": img.get("merchant_name"),
                "store_id": img.get("store_id"),
                "store_name": img.get("store_name"),
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
                },
            }

            if tags:
                image_data["tags"] = tags

            new_images.append(image_data)

        if not new_images:
            supabase_service.client.table("jobs").update({
                "status": "completed",
                "progress": 100,
                "result": {
                    "stage": "completed",
                    "success": True,
                    "synced": 0,
                    "skipped": skipped,
                    "total_found": len(images),
                    "message": f"All {skipped} images already exist",
                }
            }).eq("id", job_id).execute()
            return

        update_job(15, "syncing", f"Syncing {len(new_images)} new images...",
                  synced=0, skipped=skipped, total=len(images))

        # Step 4: Insert in batches with progress updates
        synced = 0
        INSERT_BATCH_SIZE = 100

        for i in range(0, len(new_images), INSERT_BATCH_SIZE):
            batch = new_images[i:i + INSERT_BATCH_SIZE]

            try:
                result = supabase_service.client.table("od_images").insert(batch).execute()
                batch_synced = len(result.data) if result.data else 0
                synced += batch_synced

                # Update checkpoint
                for img in batch[:batch_synced]:
                    checkpoint.synced_ids.add(img["buybuddy_image_id"])

                # Update progress (15-85%)
                progress = 15 + int((synced / len(new_images)) * 70) if new_images else 85
                update_job(
                    progress, "syncing",
                    f"Synced {synced}/{len(new_images)} images...",
                    synced=synced, skipped=skipped, total=len(images)
                )

                # Save checkpoint periodically
                if synced % 500 == 0:
                    checkpoint_service.save_buybuddy_checkpoint(checkpoint)

            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                for img in batch:
                    checkpoint.failed_ids[img["buybuddy_image_id"]] = str(e)

        # Step 5: Link to dataset if specified
        dataset_id = request_data.get("dataset_id")
        if dataset_id and synced > 0:
            update_job(88, "linking", "Adding images to dataset...",
                      synced=synced, skipped=skipped, total=len(images))

            try:
                # Get IDs of newly inserted images
                synced_bb_ids = list(checkpoint.synced_ids)
                new_ids_data = []

                for i in range(0, len(synced_bb_ids), BATCH_SIZE):
                    batch = synced_bb_ids[i:i + BATCH_SIZE]
                    batch_result = supabase_service.client.table("od_images").select("id").in_(
                        "buybuddy_image_id", batch
                    ).execute()
                    new_ids_data.extend(batch_result.data or [])

                if new_ids_data:
                    dataset_links = [
                        {"dataset_id": dataset_id, "image_id": r["id"], "status": "pending"}
                        for r in new_ids_data
                    ]

                    for i in range(0, len(dataset_links), BATCH_SIZE):
                        batch = dataset_links[i:i + BATCH_SIZE]
                        try:
                            supabase_service.client.table("od_dataset_images").insert(batch).execute()
                        except Exception:
                            pass  # Ignore duplicate key errors

            except Exception as e:
                logger.warning(f"Dataset linking failed: {e}")

        # Step 6: Complete
        final_result = {
            "stage": "completed",
            "success": True,
            "synced": synced,
            "skipped": skipped,
            "total_found": len(images),
            "failed": len(checkpoint.failed_ids),
            "errors": list(checkpoint.failed_ids.values())[:20] if checkpoint.failed_ids else [],
            "message": f"Synced {synced} images, skipped {skipped} duplicates",
        }

        supabase_service.client.table("jobs").update({
            "status": "completed",
            "progress": 100,
            "result": final_result,
        }).eq("id", job_id).execute()

        logger.info(f"BuyBuddy sync job {job_id} completed: {synced} synced, {skipped} skipped")

    except Exception as e:
        import traceback
        logger.exception(f"BuyBuddy sync job {job_id} failed: {e}")
        fail_job(f"Unexpected error: {str(e)}")


@router.post("/buybuddy/sync")
async def sync_from_buybuddy(request: BuyBuddySyncRequest):
    """Start a background BuyBuddy sync job. Returns job ID for tracking."""
    import asyncio

    if not buybuddy_service.is_configured():
        raise HTTPException(status_code=400, detail="BuyBuddy API not configured")

    # Create job record
    job_data = {
        "type": "buybuddy_sync",
        "status": "pending",
        "progress": 0,
        "config": {
            "start_date": request.start_date,
            "end_date": request.end_date,
            "store_id": request.store_id,
            "is_annotated": request.is_annotated,
            "is_approved": request.is_approved,
            "max_images": request.max_images,
            "dataset_id": request.dataset_id,
            "tags": request.tags,
        }
    }

    try:
        job_result = supabase_service.client.table("jobs").insert(job_data).execute()
        job = job_result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

    # Prepare request data for background task
    request_data = {
        "start_date": request.start_date,
        "end_date": request.end_date,
        "store_id": request.store_id,
        "is_annotated": request.is_annotated,
        "is_approved": request.is_approved,
        "max_images": request.max_images,
        "dataset_id": request.dataset_id,
        "tags": request.tags,
    }

    # Start background task
    asyncio.create_task(_run_buybuddy_sync(job["id"], request_data))

    return {
        "job_id": job["id"],
        "status": "started",
        "message": "BuyBuddy sync started. Use job_id to track progress.",
    }


@router.get("/buybuddy/sync/{job_id}")
async def get_buybuddy_sync_status(job_id: str):
    """Get the status of a BuyBuddy sync job."""
    result = supabase_service.client.table("jobs").select("*").eq("id", job_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = result.data
    job_result = job.get("result", {})

    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "result": job_result,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "can_resume": job_result.get("can_resume", False) if job_result else False,
    }


@router.post("/buybuddy/sync/{job_id}/retry")
async def retry_buybuddy_sync(job_id: str):
    """Retry a failed BuyBuddy sync job."""
    import asyncio
    from services.import_checkpoint import checkpoint_service

    result = supabase_service.client.table("jobs").select("*").eq("id", job_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = result.data

    if job["type"] != "buybuddy_sync":
        raise HTTPException(status_code=400, detail="Job is not a BuyBuddy sync")

    if job["status"] not in ["failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Cannot retry job with status: {job['status']}")

    config = job.get("config", {})

    # Try to load checkpoint
    checkpoint = checkpoint_service.load_buybuddy_checkpoint(job_id)

    # Update job status
    resume_progress = 0
    if checkpoint and checkpoint.can_resume():
        synced_count = len(checkpoint.synced_ids)
        total = checkpoint.total_images
        resume_progress = 15 + int((synced_count / total) * 70) if total > 0 else 15

    supabase_service.client.table("jobs").update({
        "status": "running",
        "error": None,
        "progress": resume_progress,
    }).eq("id", job_id).execute()

    # Start background task
    asyncio.create_task(_run_buybuddy_sync(job_id, config, checkpoint))

    if checkpoint and checkpoint.can_resume():
        return {
            "job_id": job_id,
            "status": "resumed",
            "message": f"Resuming from {len(checkpoint.synced_ids)} synced images",
            "resumed_from_checkpoint": True,
        }
    else:
        return {
            "job_id": job_id,
            "status": "restarted",
            "message": "Starting fresh sync",
            "resumed_from_checkpoint": False,
        }


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
    use_streaming: bool = True  # NEW: Use streaming by default


async def _run_roboflow_import_streaming(job_id: str, request_data: dict, resume_checkpoint=None):
    """
    Background task to run Roboflow import using STREAMING.

    This imports images directly from Roboflow URLs without downloading
    the entire ZIP file first. Much faster and more memory-efficient.

    Args:
        job_id: The job ID
        request_data: Import configuration
        resume_checkpoint: Optional StreamingCheckpoint to resume from
    """
    import logging
    from services.roboflow_streaming import streaming_import_from_roboflow, StreamingCheckpoint
    from services.import_checkpoint import checkpoint_service

    logger = logging.getLogger(__name__)

    import time
    from datetime import datetime, timezone

    start_time = time.time()
    total_annotations = [0]  # Use list to allow modification in nested function

    def update_job(progress: int, stage: str, message: str, status: str = "running",
                   images_processed: int = 0, images_total: int = 0, annotations: int = 0):
        """Helper to update job progress with detailed stats."""
        if annotations > 0:
            total_annotations[0] = annotations

        try:
            elapsed = time.time() - start_time
            speed = images_processed / elapsed if elapsed > 0 and images_processed > 0 else 0
            remaining = images_total - images_processed if images_total > 0 else 0
            eta = remaining / speed if speed > 0 else 0

            # Preserve existing checkpoint data when updating progress
            existing_checkpoint = None
            try:
                existing_job = supabase_service.client.table("jobs").select("result").eq("id", job_id).single().execute()
                if existing_job.data and existing_job.data.get("result"):
                    existing_checkpoint = existing_job.data["result"].get("checkpoint")
            except Exception:
                pass

            result_data = {
                "stage": stage,
                "message": message,
                "can_resume": True,
                # Real-time stats
                "images_processed": images_processed,
                "images_total": images_total,
                "annotations_imported": total_annotations[0],
                "images_per_second": round(speed, 2),
                "eta_seconds": int(eta),
                "elapsed_seconds": int(elapsed),
                "started_at": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            }

            # Preserve checkpoint if it exists
            if existing_checkpoint:
                result_data["checkpoint"] = existing_checkpoint

            supabase_service.client.table("jobs").update({
                "status": status,
                "progress": progress,
                "result": result_data,
            }).eq("id", job_id).execute()
        except Exception:
            pass

    def fail_job(error: str):
        """Helper to mark job as failed with resume capability."""
        logger.error(f"Roboflow streaming import job {job_id} failed: {error}")
        try:
            # Load current checkpoint to preserve progress info
            current_checkpoint = checkpoint_service.load_streaming_checkpoint(job_id)
            checkpoint_data = current_checkpoint.to_dict() if current_checkpoint else {}

            supabase_service.client.table("jobs").update({
                "status": "failed",
                "error": error,
                "result": {
                    "stage": "failed",
                    "error": error,
                    "can_resume": True,  # Streaming can always resume
                    "checkpoint": checkpoint_data,
                    "processed_count": len(current_checkpoint.processed_ids) if current_checkpoint else 0,
                    "total_images": current_checkpoint.total_images if current_checkpoint else 0,
                }
            }).eq("id", job_id).execute()
        except Exception:
            pass

    try:
        update_job(5, "initializing", "Starting streaming import from Roboflow...")

        last_update_count = [0]

        def progress_callback(processed: int, total: int, message: str):
            # Update every 10 images for real-time feedback
            if processed >= last_update_count[0] + 10 or processed == total:
                last_update_count[0] = processed
                pct = 5 + int((processed / total) * 90) if total > 0 else 5  # 5-95%
                update_job(
                    pct, "streaming", message,
                    images_processed=processed,
                    images_total=total,
                    annotations=total_annotations[0]
                )

        # Run streaming import
        print(f"[DEBUG] Starting streaming_import_from_roboflow for job {job_id}")
        if resume_checkpoint:
            print(f"[DEBUG] Resuming from checkpoint with {len(resume_checkpoint.processed_ids)} processed images")
        result = await streaming_import_from_roboflow(
            api_key=request_data["api_key"],
            workspace=request_data["workspace"],
            project=request_data["project"],
            dataset_id=request_data["dataset_id"],
            class_mapping=request_data.get("class_mapping", []),
            concurrency=10,  # 10 parallel uploads
            progress_callback=progress_callback,
            job_id=job_id,
            checkpoint=resume_checkpoint,
        )
        print(f"[DEBUG] streaming_import_from_roboflow returned: success={result.success}, images={result.images_imported}")

        # Complete the job
        if result.success:
            supabase_service.client.table("jobs").update({
                "status": "completed",
                "progress": 100,
                "result": {
                    "stage": "completed",
                    "success": True,
                    "images_imported": result.images_imported,
                    "annotations_imported": result.annotations_imported,
                    "images_skipped": result.images_skipped,
                    "images_failed": result.images_failed,
                    "errors": result.errors[:20] if result.errors else [],
                }
            }).eq("id", job_id).execute()

            logger.info(
                f"Roboflow streaming import job {job_id} completed: "
                f"{result.images_imported} images, {result.annotations_imported} annotations"
            )
        else:
            error_msg = result.errors[0] if result.errors else "Import failed"
            fail_job(error_msg)

    except Exception as e:
        import traceback
        print(f"[DEBUG] EXCEPTION in streaming import task {job_id}: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        logger.exception(f"Unexpected error in Roboflow streaming import job {job_id}")
        fail_job(f"Unexpected error: {str(e)}")


async def _run_roboflow_import(job_id: str, request_data: dict, resume_checkpoint=None):
    """
    Background task to run Roboflow import with checkpoint support.

    Can be called fresh or with a checkpoint to resume after API restart.

    Args:
        job_id: The job ID
        request_data: Import configuration
        resume_checkpoint: Optional ImportCheckpoint to resume from
    """
    import os
    import logging
    import httpx
    from services.od_import import import_annotated_dataset_from_file, ClassMapping
    from services.import_checkpoint import checkpoint_service, ImportCheckpoint

    logger = logging.getLogger(__name__)

    # Initialize or use existing checkpoint
    if resume_checkpoint:
        checkpoint = resume_checkpoint
        logger.info(f"Resuming job {job_id} from stage: {checkpoint.stage}")
    else:
        checkpoint = ImportCheckpoint(job_id=job_id, stage="downloading")
        checkpoint_service.create_job_dir(job_id)

    zip_path = str(checkpoint_service.get_zip_path(job_id))

    def update_job(progress: int, stage: str, message: str, status: str = "running"):
        """Helper to update job progress."""
        try:
            supabase_service.client.table("jobs").update({
                "status": status,
                "progress": progress,
                "result": {
                    "stage": stage,
                    "message": message,
                    "checkpoint": checkpoint.to_dict(),
                    "can_resume": checkpoint.can_resume(),
                }
            }).eq("id", job_id).execute()
        except Exception:
            pass

    def fail_job(error: str, can_resume: bool = False):
        """Helper to mark job as failed."""
        logger.error(f"Roboflow import job {job_id} failed: {error}")
        checkpoint.error_message = error
        checkpoint_service.save_checkpoint(checkpoint)
        try:
            supabase_service.client.table("jobs").update({
                "status": "failed",
                "error": error,
                "result": {
                    "stage": checkpoint.stage,
                    "checkpoint": checkpoint.to_dict(),
                    "can_resume": can_resume and checkpoint.download_complete,
                    "error": error,
                }
            }).eq("id", job_id).execute()
        except Exception:
            pass

    try:
        # ========== STAGE 1: DOWNLOAD ==========
        if not checkpoint.download_complete:
            checkpoint.stage = "downloading"
            update_job(5, "downloading", "Starting download from Roboflow...")

            last_progress_mb = 0

            def progress_callback(downloaded_bytes: int, total_bytes: int | None):
                nonlocal last_progress_mb
                mb_downloaded = downloaded_bytes // (1024 * 1024)

                if mb_downloaded >= last_progress_mb + 10:
                    last_progress_mb = mb_downloaded

                    if total_bytes:
                        total_mb = total_bytes // (1024 * 1024)
                        download_pct = min(45, 5 + int((downloaded_bytes / total_bytes) * 40))
                        update_job(download_pct, "downloading", f"Downloaded {mb_downloaded} MB / {total_mb} MB...")
                    else:
                        download_progress = min(45, 5 + int(mb_downloaded / 100))
                        update_job(download_progress, "downloading", f"Downloaded {mb_downloaded} MB...")

            try:
                # Download to persistent path (survives restart)
                _, total_bytes, file_hash = await roboflow_service.download_dataset_to_path(
                    api_key=request_data["api_key"],
                    workspace=request_data["workspace"],
                    project=request_data["project"],
                    version=request_data["version"],
                    target_path=zip_path,
                    format=request_data.get("format", "coco"),
                    max_retries=3,
                    progress_callback=progress_callback,
                )

                # Save download checkpoint
                checkpoint.download_complete = True
                checkpoint.zip_file_path = zip_path
                checkpoint.zip_file_size = total_bytes
                checkpoint.zip_file_hash = file_hash
                checkpoint_service.save_checkpoint(checkpoint)

            except httpx.TimeoutException as e:
                fail_job(f"Download timed out: {str(e)}", can_resume=False)
                return
            except httpx.NetworkError as e:
                fail_job(f"Network error: {str(e)}", can_resume=False)
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    fail_job("Invalid API key or unauthorized access.")
                elif e.response.status_code == 404:
                    fail_job("Project or version not found.")
                elif e.response.status_code == 429:
                    fail_job("Rate limited by Roboflow. Please wait and retry.")
                else:
                    fail_job(f"HTTP error {e.response.status_code}: {str(e)}")
                return

        # Verify ZIP exists and is valid
        if not checkpoint_service.verify_zip_integrity(job_id, checkpoint.zip_file_hash):
            fail_job("ZIP file missing or corrupted. Please retry the import.", can_resume=False)
            return

        file_size = os.path.getsize(zip_path)
        total_mb = file_size // (1024 * 1024)

        # ========== STAGE 2: PROCESSING ==========
        checkpoint.stage = "processing"
        update_job(50, "processing", f"Download complete ({total_mb} MB). Processing images...")
        checkpoint_service.save_checkpoint(checkpoint)

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
            zip_file_path=zip_path,
            dataset_id=request_data["dataset_id"],
            class_mapping=class_mapping_list,
            skip_duplicates=request_data.get("skip_duplicates", True),
            merge_annotations=request_data.get("merge_annotations", False),
        )

        # ========== STAGE 3: COMPLETED ==========
        checkpoint.stage = "completed"
        checkpoint_service.save_checkpoint(checkpoint)

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

        # Clean up on successful completion
        checkpoint_service.cleanup_job(job_id)

    except MemoryError:
        fail_job("Out of memory. Try importing a smaller dataset.", can_resume=True)
    except Exception as e:
        logger.exception(f"Unexpected error in Roboflow import job {job_id}")
        fail_job(f"Unexpected error: {str(e)}", can_resume=checkpoint.download_complete)


@router.post("/roboflow/import")
async def import_from_roboflow(
    request: RoboflowImportRequest,
    background_tasks: BackgroundTasks,
):
    """Start a background import from Roboflow. Returns job ID for tracking."""
    import asyncio

    try:
        # Verify dataset exists
        dataset = supabase_service.client.table("od_datasets").select("id, name").eq("id", request.dataset_id).single().execute()
        if not dataset.data:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Create job record (store API key for resume capability)
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
                "api_key": request.api_key,  # Stored for resume
                "skip_duplicates": request.skip_duplicates,
                "merge_annotations": request.merge_annotations,
                "use_streaming": request.use_streaming,  # NEW
                "class_mapping": [
                    {
                        "source_name": m.source_name,
                        "target_class_id": m.target_class_id,
                        "create_new": m.create_new,
                        "skip": m.skip,
                        "color": m.color,
                    }
                    for m in request.class_mapping
                ],
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

        # Start background task using asyncio.create_task for async functions
        # This ensures the task runs on the current event loop properly
        if request.use_streaming:
            asyncio.create_task(_run_roboflow_import_streaming(job["id"], request_data))
            message = "Streaming import started. Use job_id to track progress."
        else:
            asyncio.create_task(_run_roboflow_import(job["id"], request_data))
            message = "Import started (ZIP mode). Use job_id to track progress."

        return {
            "job_id": job["id"],
            "status": "started",
            "message": message,
            "mode": "streaming" if request.use_streaming else "zip",
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

    # Don't expose API key in response
    job_result = job.get("result", {})
    if job_result and "checkpoint" in job_result:
        # Remove sensitive data from checkpoint in response
        checkpoint = job_result.get("checkpoint", {})
        if checkpoint:
            checkpoint.pop("api_key", None)

    return {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "result": job_result,
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job.get("updated_at"),
        "can_resume": job_result.get("can_resume", False) if job_result else False,
    }


@router.post("/roboflow/import/{job_id}/retry")
async def retry_roboflow_import(
    job_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Retry a failed Roboflow import job.

    If the download was complete, it will resume from the processing stage.
    Otherwise, it will start a fresh download.
    """
    import asyncio
    from services.import_checkpoint import checkpoint_service

    # Get the job
    result = supabase_service.client.table("jobs").select("*").eq("id", job_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = result.data

    if job["type"] != "roboflow_import":
        raise HTTPException(status_code=400, detail="Job is not a Roboflow import")

    # Check if job can be retried
    # Allow retry for: failed, cancelled, or stale running jobs (no progress for 2+ minutes)
    from datetime import datetime, timedelta

    can_retry = job["status"] in ["failed", "cancelled", "interrupted"]

    if job["status"] == "running":
        # Check if job is stale (no update for 2 minutes)
        updated_at = job.get("updated_at") or job.get("created_at")
        if updated_at:
            try:
                # Parse the timestamp
                if updated_at.endswith("Z"):
                    updated_at = updated_at[:-1] + "+00:00"
                last_update = datetime.fromisoformat(updated_at.replace("+00:00", ""))
                stale_threshold = datetime.utcnow() - timedelta(minutes=2)
                if last_update < stale_threshold:
                    can_retry = True
                    # Mark as failed since the task died
                    supabase_service.client.table("jobs").update({
                        "status": "failed",
                        "error": "Job was interrupted (server restart). Resuming..."
                    }).eq("id", job_id).execute()
            except Exception:
                pass

    if not can_retry:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status: {job['status']}. Only failed, cancelled, interrupted, or stale running jobs can be retried."
        )

    # Get config with API key
    config = job.get("config", {})
    if not config.get("api_key"):
        raise HTTPException(
            status_code=400,
            detail="Cannot retry: API key not found in job config. Please start a new import."
        )

    # Check if this was a streaming import
    use_streaming = config.get("use_streaming", False)

    # Prepare request data from stored config
    request_data = {
        "api_key": config.get("api_key"),
        "workspace": config.get("workspace"),
        "project": config.get("project"),
        "version": config.get("version"),
        "dataset_id": config.get("dataset_id"),
        "format": config.get("format", "coco"),
        "skip_duplicates": config.get("skip_duplicates", True),
        "merge_annotations": config.get("merge_annotations", False),
        "class_mapping": config.get("class_mapping", []),
    }

    if use_streaming:
        # Handle streaming import retry
        streaming_checkpoint = checkpoint_service.load_streaming_checkpoint(job_id)

        if streaming_checkpoint and streaming_checkpoint.can_resume():
            # Resume from checkpoint
            processed_count = len(streaming_checkpoint.processed_ids)
            total_images = streaming_checkpoint.total_images
            resume_progress = 5 + int((processed_count / total_images) * 90) if total_images > 0 else 5

            supabase_service.client.table("jobs").update({
                "status": "running",
                "error": None,
                "progress": resume_progress,
            }).eq("id", job_id).execute()

            asyncio.create_task(_run_roboflow_import_streaming(
                job_id, request_data, streaming_checkpoint
            ))

            return {
                "job_id": job_id,
                "status": "resumed",
                "message": f"Resuming from {processed_count}/{total_images} images",
                "resumed_from_checkpoint": True,
                "processed_count": processed_count,
                "total_images": total_images,
            }
        else:
            # Start fresh streaming import
            supabase_service.client.table("jobs").update({
                "status": "running",
                "error": None,
                "progress": 0,
            }).eq("id", job_id).execute()

            asyncio.create_task(_run_roboflow_import_streaming(job_id, request_data, None))
            return {
                "job_id": job_id,
                "status": "restarted",
                "message": "Starting fresh streaming import (no checkpoint found)",
                "resumed_from_checkpoint": False,
            }
    else:
        # Handle ZIP import retry (existing logic)
        checkpoint = checkpoint_service.load_checkpoint(job_id)

        # Check if we can resume (download complete and ZIP exists)
        can_resume = (
            checkpoint and
            checkpoint.download_complete and
            checkpoint_service.verify_zip_integrity(job_id, checkpoint.zip_file_hash)
        )

        # Update job status
        supabase_service.client.table("jobs").update({
            "status": "running",
            "error": None,
            "progress": checkpoint.download_complete * 50 if checkpoint else 0,
        }).eq("id", job_id).execute()

        # Start background task
        if can_resume:
            asyncio.create_task(_run_roboflow_import(job_id, request_data, checkpoint))
            return {
                "job_id": job_id,
                "status": "resumed",
                "message": f"Resuming from stage: {checkpoint.stage}",
                "resumed_from_checkpoint": True,
            }
        else:
            # Clean up any partial files and start fresh
            checkpoint_service.cleanup_job(job_id)
            asyncio.create_task(_run_roboflow_import(job_id, request_data, None))
            return {
                "job_id": job_id,
                "status": "restarted",
                "message": "Starting fresh download (no valid checkpoint found)",
                "resumed_from_checkpoint": False,
            }
