"""
Classification - Images Router

Endpoints for managing classification images (upload, list, CRUD).
Includes import features from various sources and duplicate detection.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from uuid import uuid4
import hashlib
import io

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSImageResponse,
    CLSImagesResponse,
    CLSImageUpdate,
    ImportURLsRequest,
    ImportFromProductsRequest,
    ImportFromCutoutsRequest,
    ImportFromODRequest,
    ImportResultResponse,
    BulkOperationRequest,
    BulkTagRequest,
    BulkMoveRequest,
    BulkAddToDatasetRequest,
    BulkOperationResponse,
    DuplicateCheckRequest,
    DuplicateCheckResponse,
)

router = APIRouter()

# Storage bucket name
CLS_IMAGES_BUCKET = "cls-images"


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def calculate_phash(content: bytes) -> str:
    """Calculate perceptual hash of image."""
    try:
        import imagehash
        from PIL import Image
        img = Image.open(io.BytesIO(content))
        return str(imagehash.phash(img))
    except Exception as e:
        logger.warning(f"Failed to calculate phash: {e}")
        return ""


@router.get("", response_model=CLSImagesResponse)
async def list_images(
    page: int = 1,
    limit: int = 50,
    status: Optional[str] = None,
    source: Optional[str] = None,
    folder: Optional[str] = None,
    search: Optional[str] = None,
    # Multi-select filters (comma-separated)
    statuses: Optional[str] = None,
    sources: Optional[str] = None,
    folders: Optional[str] = None,
):
    """List classification images with pagination and filters."""
    offset = (page - 1) * limit

    query = supabase_service.client.table("cls_images").select("*", count="exact")

    # Apply filters
    if statuses:
        status_list = [s.strip() for s in statuses.split(",") if s.strip()]
        if status_list:
            query = query.in_("status", status_list)
    elif status:
        query = query.eq("status", status)

    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]
        if source_list:
            query = query.in_("source", source_list)
    elif source:
        query = query.eq("source", source)

    if folders:
        folder_list = [f.strip() for f in folders.split(",") if f.strip()]
        if folder_list:
            query = query.in_("folder", folder_list)
    elif folder:
        query = query.eq("folder", folder)

    if search:
        query = query.ilike("filename", f"%{search}%")

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return CLSImagesResponse(
        images=result.data or [],
        total=result.count or 0,
        page=page,
        limit=limit,
    )


@router.get("/filters/options")
async def get_filter_options():
    """Get filter options for FilterDrawer component."""
    try:
        result = supabase_service.client.rpc("get_cls_image_filter_options").execute()

        if result.data:
            return result.data

        return {
            "statuses": [],
            "sources": [],
            "folders": [],
            "total_count": 0,
        }
    except Exception as e:
        logger.warning(f"RPC get_cls_image_filter_options failed: {e}")
        return await _get_filter_options_fallback()


async def _get_filter_options_fallback():
    """Fallback filter options using Python aggregation."""
    all_images = supabase_service.client.table("cls_images").select(
        "status, source, folder"
    ).execute()

    images = all_images.data or []

    status_counts = {}
    source_counts = {}
    folder_counts = {}

    for img in images:
        status = img.get("status") or "pending"
        status_counts[status] = status_counts.get(status, 0) + 1

        source = img.get("source") or "upload"
        source_counts[source] = source_counts.get(source, 0) + 1

        folder = img.get("folder")
        if folder:
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

    return {
        "statuses": [{"value": k, "label": k.replace("_", " ").title(), "count": v}
                     for k, v in sorted(status_counts.items(), key=lambda x: -x[1])],
        "sources": [{"value": k, "label": k.replace("_", " ").title(), "count": v}
                    for k, v in sorted(source_counts.items(), key=lambda x: -x[1])],
        "folders": [{"value": k, "label": k, "count": v}
                    for k, v in sorted(folder_counts.items())],
        "total_count": len(images),
    }


@router.get("/{image_id}", response_model=CLSImageResponse)
async def get_image(image_id: str):
    """Get a single image by ID."""
    result = supabase_service.client.table("cls_images").select("*").eq("id", image_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    return result.data


@router.post("", response_model=CLSImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    folder: Optional[str] = Form(None),
    skip_duplicates: bool = Form(False),
    dataset_id: Optional[str] = Form(None),
    label: Optional[str] = Form(None),
):
    """Upload a single image with duplicate detection and optional labeling."""
    from PIL import Image

    content = await file.read()

    # Get image dimensions
    img = Image.open(io.BytesIO(content))
    width, height = img.size

    # Calculate hashes for duplicate detection
    file_hash = calculate_file_hash(content)
    phash = calculate_phash(content)

    # Check for duplicates
    if skip_duplicates and file_hash:
        existing = supabase_service.client.table("cls_images").select("id").eq("file_hash", file_hash).execute()
        if existing.data:
            raise HTTPException(status_code=409, detail="Duplicate image already exists")

    # Get or create class if label provided
    class_id = None
    if label and dataset_id:
        existing_class = supabase_service.client.table("cls_dataset_classes").select("id").eq("dataset_id", dataset_id).eq("name", label).execute()
        if existing_class.data:
            class_id = existing_class.data[0]["id"]
        else:
            # Generate a new UUID for the class
            new_class_id = str(uuid4())
            
            # Insert into cls_dataset_classes
            supabase_service.client.table("cls_dataset_classes").insert({
                "id": new_class_id,
                "dataset_id": dataset_id,
                "name": label,
                "display_name": label,
            }).execute()
            
            # Also insert into cls_classes for foreign key compatibility
            try:
                supabase_service.client.table("cls_classes").insert({
                    "id": new_class_id,
                    "name": label,
                    "display_name": label,
                }).execute()
            except Exception as e:
                logger.warning(f"Could not insert into cls_classes (may already exist): {e}")
            
            class_id = new_class_id

    # Generate filename
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    filename = f"{uuid4()}.{ext}"

    # Upload to storage
    storage_path = filename
    supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).upload(
        storage_path,
        content,
        {"content-type": file.content_type or "image/jpeg"}
    )

    # Get public URL
    image_url = supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).get_public_url(storage_path)

    # Create database record
    image_data = {
        "filename": filename,
        "original_filename": file.filename,
        "image_url": image_url,
        "storage_path": storage_path,
        "width": width,
        "height": height,
        "file_size_bytes": len(content),
        "source": "upload",
        "folder": folder,
        "status": "labeled" if class_id else "pending",
        "file_hash": file_hash,
        "phash": phash,
    }

    result = supabase_service.client.table("cls_images").insert(image_data).execute()
    image_id = result.data[0]["id"]

    # Add to dataset if specified
    if dataset_id:
        supabase_service.client.table("cls_dataset_images").insert({
            "dataset_id": dataset_id,
            "image_id": image_id,
            "status": "labeled" if class_id else "pending",
        }).execute()

        # Create label if class determined
        if class_id:
            supabase_service.client.table("cls_labels").insert({
                "dataset_id": dataset_id,
                "image_id": image_id,
                "class_id": class_id,
            }).execute()

    return result.data[0]


@router.post("/bulk", response_model=list[CLSImageResponse])
async def upload_images_bulk(
    files: list[UploadFile] = File(...),
    folder: Optional[str] = Form(None),
    skip_duplicates: bool = Form(False),
):
    """Bulk upload images."""
    from PIL import Image

    uploaded_images = []
    errors = []

    for file in files:
        try:
            content = await file.read()

            img = Image.open(io.BytesIO(content))
            width, height = img.size

            file_hash = calculate_file_hash(content)
            phash = calculate_phash(content)

            if skip_duplicates and file_hash:
                existing = supabase_service.client.table("cls_images").select("id").eq("file_hash", file_hash).execute()
                if existing.data:
                    continue

            ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            filename = f"{uuid4()}.{ext}"

            storage_path = filename
            supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).upload(
                storage_path,
                content,
                {"content-type": file.content_type or "image/jpeg"}
            )

            image_url = supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).get_public_url(storage_path)

            image_data = {
                "filename": filename,
                "original_filename": file.filename,
                "image_url": image_url,
                "storage_path": storage_path,
                "width": width,
                "height": height,
                "file_size_bytes": len(content),
                "source": "upload",
                "folder": folder,
                "status": "pending",
                "file_hash": file_hash,
                "phash": phash,
            }

            result = supabase_service.client.table("cls_images").insert(image_data).execute()
            uploaded_images.append(result.data[0])

        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            errors.append(str(e))

    return uploaded_images


@router.patch("/{image_id}", response_model=CLSImageResponse)
async def update_image(image_id: str, data: CLSImageUpdate):
    """Update image metadata."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("cls_images").update(update_data).eq("id", image_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    return result.data[0]


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    """Delete an image."""
    # Get image to delete from storage
    image = supabase_service.client.table("cls_images").select("storage_path").eq("id", image_id).single().execute()

    if not image.data:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete from storage
    if image.data.get("storage_path"):
        try:
            supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).remove([image.data["storage_path"]])
        except Exception as e:
            logger.warning(f"Failed to delete from storage: {e}")

    # Delete from database
    supabase_service.client.table("cls_images").delete().eq("id", image_id).execute()

    return {"success": True, "message": "Image deleted"}


# ===========================================
# Import Endpoints
# ===========================================

@router.post("/import/url", response_model=ImportResultResponse)
async def import_from_urls(data: ImportURLsRequest, background_tasks: BackgroundTasks):
    """Import images from URLs."""
    import httpx
    from PIL import Image

    images_imported = 0
    images_skipped = 0
    duplicates_found = 0
    labels_created = 0
    classes_created = 0
    errors = []

    # Get or create class if label provided
    class_id = None
    if data.label and data.dataset_id:
        existing_class = supabase_service.client.table("cls_dataset_classes").select("id").eq("dataset_id", data.dataset_id).eq("name", data.label).execute()
        if existing_class.data:
            class_id = existing_class.data[0]["id"]
        else:
            # Generate a new UUID for the class
            import uuid
            new_class_id = str(uuid.uuid4())
            
            # Insert into cls_dataset_classes
            supabase_service.client.table("cls_dataset_classes").insert({
                "id": new_class_id,
                "dataset_id": data.dataset_id,
                "name": data.label,
                "display_name": data.label,
            }).execute()
            
            # Also insert into cls_classes for foreign key compatibility
            try:
                supabase_service.client.table("cls_classes").insert({
                    "id": new_class_id,
                    "name": data.label,
                    "display_name": data.label,
                }).execute()
            except Exception as e:
                logger.warning(f"Could not insert into cls_classes (may already exist): {e}")
            
            class_id = new_class_id
            classes_created += 1

    async with httpx.AsyncClient(timeout=120) as client:
        for url in data.urls[:100]:  # Limit to 100
            try:
                response = await client.get(url)
                response.raise_for_status()
                content = response.content

                file_hash = calculate_file_hash(content)

                if data.skip_duplicates and file_hash:
                    existing = supabase_service.client.table("cls_images").select("id").eq("file_hash", file_hash).execute()
                    if existing.data:
                        duplicates_found += 1
                        images_skipped += 1
                        continue

                img = Image.open(io.BytesIO(content))
                width, height = img.size

                ext = url.split(".")[-1].split("?")[0][:4] or "jpg"
                filename = f"{uuid4()}.{ext}"

                supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).upload(
                    filename,
                    content,
                    {"content-type": f"image/{ext}"}
                )

                image_url = supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).get_public_url(filename)

                image_data = {
                    "filename": filename,
                    "original_filename": url.split("/")[-1].split("?")[0],
                    "image_url": image_url,
                    "storage_path": filename,
                    "width": width,
                    "height": height,
                    "file_size_bytes": len(content),
                    "source": "url_import",
                    "folder": data.folder,
                    "status": "labeled" if class_id else "pending",
                    "file_hash": file_hash,
                    "phash": calculate_phash(content),
                    "metadata": {"source_url": url},
                }

                result = supabase_service.client.table("cls_images").insert(image_data).execute()

                # Add to dataset if specified
                if data.dataset_id and result.data:
                    image_id = result.data[0]["id"]
                    supabase_service.client.table("cls_dataset_images").insert({
                        "dataset_id": data.dataset_id,
                        "image_id": image_id,
                        "status": "labeled" if class_id else "pending",
                    }).execute()

                    if class_id:
                        supabase_service.client.table("cls_labels").insert({
                            "dataset_id": data.dataset_id,
                            "image_id": image_id,
                            "class_id": class_id,
                        }).execute()
                        labels_created += 1

                images_imported += 1

            except Exception as e:
                logger.error(f"Failed to import {url}: {e}")
                errors.append(f"{url}: {str(e)}")
                images_skipped += 1

    return ImportResultResponse(
        success=len(errors) == 0,
        images_imported=images_imported,
        images_skipped=images_skipped,
        duplicates_found=duplicates_found,
        labels_created=labels_created,
        classes_created=classes_created,
        errors=errors,
    )


@router.post("/import/products", response_model=ImportResultResponse)
async def import_from_products(data: ImportFromProductsRequest, background_tasks: BackgroundTasks):
    """Import images from Products module - OPTIMIZED with batch inserts."""
    images_imported = 0
    images_skipped = 0
    duplicates_found = 0
    labels_created = 0
    classes_created = 0
    errors = []

    logger.info(f"Starting product import: {len(data.product_ids or [])} products, image_types={data.image_types}, label_source={data.label_source}")

    try:
        # Process product_ids in batches of 100 to avoid Supabase limitations
        product_ids = data.product_ids or []
        batch_size = 100
        all_products = []
        
        if product_ids:
            for i in range(0, len(product_ids), batch_size):
                batch_ids = product_ids[i:i + batch_size]
                batch_result = supabase_service.client.table("products").select("*").in_("id", batch_ids).execute()
                all_products.extend(batch_result.data or [])
        else:
            query = supabase_service.client.table("products").select("*")
            if data.status:
                query = query.eq("status", data.status)
            if data.category:
                query = query.eq("category", data.category)
            if data.brand:
                query = query.eq("brand_name", data.brand)
            products_result = query.limit(1000).execute()
            all_products = products_result.data or []

        logger.info(f"Total products to process: {len(all_products)}")

        # Fetch all product_images in batches
        all_product_images = []
        product_ids_to_fetch = [p["id"] for p in all_products]
        
        for i in range(0, len(product_ids_to_fetch), batch_size):
            batch_ids = product_ids_to_fetch[i:i + batch_size]
            images_query = supabase_service.client.table("product_images").select("*").in_("product_id", batch_ids)
            if data.image_types:
                images_query = images_query.in_("image_type", data.image_types)
            images_result = images_query.limit(10000).execute()
            all_product_images.extend(images_result.data or [])

        logger.info(f"Total product images found: {len(all_product_images)}")

        # Get existing duplicates in one query if skip_duplicates is enabled
        existing_source_ids = set()
        if data.skip_duplicates and all_product_images:
            all_source_ids = [img["id"] for img in all_product_images]
            for i in range(0, len(all_source_ids), 500):
                batch_ids = all_source_ids[i:i + 500]
                existing = supabase_service.client.table("cls_images").select("source_id").eq("source_type", "product").in_("source_id", batch_ids).execute()
                existing_source_ids.update(e["source_id"] for e in (existing.data or []))
            logger.info(f"Found {len(existing_source_ids)} existing duplicates")

        # Group images by product_id
        images_by_product = {}
        for img in all_product_images:
            pid = img.get("product_id")
            if pid not in images_by_product:
                images_by_product[pid] = []
            images_by_product[pid].append(img)

        # Get or create classes - batch fetch existing classes first
        class_cache = {}
        unique_labels = set()
        
        for product in all_products:
            label_value = None
            if data.label_source == "product_id":
                label_value = product.get("id")
            elif data.label_source == "category":
                label_value = product.get("category")
            elif data.label_source == "brand":
                label_value = product.get("brand_name")
            elif data.label_source == "product_name":
                label_value = product.get("product_name") or product.get("name")
            elif data.label_source == "custom" and data.custom_label:
                label_value = data.custom_label
            if label_value:
                unique_labels.add(str(label_value))

        # Batch fetch existing classes
        if unique_labels and data.dataset_id:
            label_list = list(unique_labels)
            for i in range(0, len(label_list), 100):
                batch_labels = label_list[i:i + 100]
                existing_classes = supabase_service.client.table("cls_dataset_classes").select("id, name").eq("dataset_id", data.dataset_id).in_("name", batch_labels).execute()
                for cls in (existing_classes.data or []):
                    class_cache[cls["name"]] = cls["id"]
        
        # Create missing classes in batch
        missing_labels = unique_labels - set(class_cache.keys())
        if missing_labels and data.dataset_id:
            import uuid
            new_classes_dataset = []
            new_classes_global = []
            for label in missing_labels:
                new_id = str(uuid.uuid4())
                class_cache[label] = new_id
                new_classes_dataset.append({
                    "id": new_id,
                    "dataset_id": data.dataset_id,
                    "name": label,
                    "display_name": label,
                })
                new_classes_global.append({
                    "id": new_id,
                    "name": label,
                    "display_name": label,
                })
            
            # Batch insert classes
            if new_classes_dataset:
                supabase_service.client.table("cls_dataset_classes").insert(new_classes_dataset).execute()
                classes_created += len(new_classes_dataset)
                try:
                    supabase_service.client.table("cls_classes").insert(new_classes_global).execute()
                except Exception as e:
                    logger.warning(f"Some classes may already exist in cls_classes: {e}")

        logger.info(f"Class cache ready with {len(class_cache)} classes")

        # Prepare batch data for inserts
        images_to_insert = []
        image_metadata_map = []  # Track product_id, class_id for each image
        
        for product in all_products:
            label_value = None
            if data.label_source == "product_id":
                label_value = str(product.get("id"))
            elif data.label_source == "category":
                label_value = str(product.get("category")) if product.get("category") else None
            elif data.label_source == "brand":
                label_value = str(product.get("brand_name")) if product.get("brand_name") else None
            elif data.label_source == "product_name":
                label_value = str(product.get("product_name") or product.get("name")) if (product.get("product_name") or product.get("name")) else None
            elif data.label_source == "custom" and data.custom_label:
                label_value = data.custom_label
            
            class_id = class_cache.get(label_value) if label_value else None
            product_images = images_by_product.get(product["id"], [])[:data.max_frames_per_product]

            for pimg in product_images:
                # Skip duplicates
                if data.skip_duplicates and pimg["id"] in existing_source_ids:
                    duplicates_found += 1
                    images_skipped += 1
                    continue

                images_to_insert.append({
                    "filename": pimg.get("image_path", "").split("/")[-1] or f"{uuid4()}.jpg",
                    "image_url": pimg.get("image_url"),
                    "source": "products_import",
                    "source_type": "product",
                    "source_id": pimg["id"],
                    "status": "labeled" if class_id else "pending",
                    "metadata": {
                        "product_id": product["id"],
                        "product_name": product.get("product_name") or product.get("name"),
                        "image_type": pimg.get("image_type"),
                    },
                })
                image_metadata_map.append({"class_id": class_id})

        logger.info(f"Prepared {len(images_to_insert)} images for batch insert")

        # Build source_id to metadata mapping for correct label assignment after insert
        source_to_metadata = {}
        for i, img_data in enumerate(images_to_insert):
            source_to_metadata[img_data["source_id"]] = image_metadata_map[i]

        # Batch insert images (in chunks of 250 for reliability)
        insert_batch_size = 250
        all_inserted_images = []

        for i in range(0, len(images_to_insert), insert_batch_size):
            batch = images_to_insert[i:i + insert_batch_size]
            try:
                result = supabase_service.client.table("cls_images").insert(batch).execute()
                all_inserted_images.extend(result.data or [])
                logger.info(f"Inserted batch {i//insert_batch_size + 1}: {len(result.data or [])} images")
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                errors.append(f"Batch insert error: {str(e)}")

        images_imported = len(all_inserted_images)

        # Batch insert dataset images and labels using source_id mapping (fixes index mismatch bug)
        if data.dataset_id and all_inserted_images:
            dataset_images_to_insert = []
            labels_to_insert = []

            for img in all_inserted_images:
                # Use source_id to get correct metadata (not positional index)
                metadata = source_to_metadata.get(img.get("source_id"), {})
                class_id = metadata.get("class_id")

                dataset_images_to_insert.append({
                    "dataset_id": data.dataset_id,
                    "image_id": img["id"],
                    "status": "labeled" if class_id else "pending",
                })

                if class_id:
                    labels_to_insert.append({
                        "dataset_id": data.dataset_id,
                        "image_id": img["id"],
                        "class_id": class_id,
                    })

            # Batch insert dataset images
            for i in range(0, len(dataset_images_to_insert), insert_batch_size):
                batch = dataset_images_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_dataset_images").insert(batch).execute()
                except Exception as e:
                    logger.error(f"Dataset images batch insert failed: {e}")
                    errors.append(f"Dataset images error: {str(e)}")

            # Batch insert labels
            for i in range(0, len(labels_to_insert), insert_batch_size):
                batch = labels_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_labels").insert(batch).execute()
                    labels_created += len(batch)
                except Exception as e:
                    logger.error(f"Labels batch insert failed: {e}")
                    errors.append(f"Labels error: {str(e)}")

        # Update dataset stats
        if data.dataset_id:
            try:
                supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": data.dataset_id}).execute()
            except Exception as e:
                logger.warning(f"Failed to update dataset stats: {e}")

        logger.info(f"Product import completed: imported={images_imported}, skipped={images_skipped}, duplicates={duplicates_found}, labels={labels_created}")

    except Exception as e:
        logger.error(f"Failed to import from products: {e}")
        errors.append(str(e))

    return ImportResultResponse(
        success=len(errors) == 0,
        images_imported=images_imported,
        images_skipped=images_skipped,
        duplicates_found=duplicates_found,
        labels_created=labels_created,
        classes_created=classes_created,
        errors=errors,
    )


@router.post("/import/cutouts", response_model=ImportResultResponse)
async def import_from_cutouts(data: ImportFromCutoutsRequest, background_tasks: BackgroundTasks):
    """Import images from Cutouts module - OPTIMIZED with batch inserts."""
    images_imported = 0
    images_skipped = 0
    duplicates_found = 0
    labels_created = 0
    classes_created = 0
    errors = []

    try:
        # Fetch cutouts in batches if cutout_ids provided
        cutouts = []
        if data.cutout_ids:
            batch_size = 100
            for i in range(0, len(data.cutout_ids), batch_size):
                batch_ids = data.cutout_ids[i:i + batch_size]
                batch_result = supabase_service.client.table("cutout_images").select("*, matched_product:products(*)").in_("id", batch_ids).execute()
                cutouts.extend(batch_result.data or [])
        else:
            query = supabase_service.client.table("cutout_images").select("*, matched_product:products(*)")
            if data.only_matched:
                query = query.not_.is_("matched_product_id", "null")
            cutouts_result = query.limit(5000).execute()
            cutouts = cutouts_result.data or []

        logger.info(f"Total cutouts to process: {len(cutouts)}")

        # Get existing duplicates in one query
        existing_source_ids = set()
        if data.skip_duplicates and cutouts:
            all_cutout_ids = [c["id"] for c in cutouts]
            for i in range(0, len(all_cutout_ids), 500):
                batch_ids = all_cutout_ids[i:i + 500]
                existing = supabase_service.client.table("cls_images").select("source_id").eq("source_type", "cutout").in_("source_id", batch_ids).execute()
                existing_source_ids.update(e["source_id"] for e in (existing.data or []))
            logger.info(f"Found {len(existing_source_ids)} existing duplicates")

        # Collect unique labels
        unique_labels = set()
        for cutout in cutouts:
            matched_product = cutout.get("matched_product")
            label_value = None
            if data.label_source == "matched_product_id" and matched_product:
                label_value = str(matched_product.get("id"))
            elif data.label_source == "custom" and data.custom_label:
                label_value = data.custom_label
            if label_value:
                unique_labels.add(label_value)

        # Batch fetch/create classes
        class_cache = {}
        if unique_labels and data.dataset_id:
            label_list = list(unique_labels)
            for i in range(0, len(label_list), 100):
                batch_labels = label_list[i:i + 100]
                existing_classes = supabase_service.client.table("cls_dataset_classes").select("id, name").eq("dataset_id", data.dataset_id).in_("name", batch_labels).execute()
                for cls in (existing_classes.data or []):
                    class_cache[cls["name"]] = cls["id"]

            # Create missing classes in batch
            missing_labels = unique_labels - set(class_cache.keys())
            if missing_labels:
                import uuid
                new_classes_dataset = []
                new_classes_global = []
                for label in missing_labels:
                    new_id = str(uuid.uuid4())
                    class_cache[label] = new_id
                    new_classes_dataset.append({
                        "id": new_id,
                        "dataset_id": data.dataset_id,
                        "name": label,
                        "display_name": label,
                    })
                    new_classes_global.append({
                        "id": new_id,
                        "name": label,
                        "display_name": label,
                    })
                if new_classes_dataset:
                    supabase_service.client.table("cls_dataset_classes").insert(new_classes_dataset).execute()
                    classes_created += len(new_classes_dataset)
                    try:
                        supabase_service.client.table("cls_classes").insert(new_classes_global).execute()
                    except Exception as e:
                        logger.warning(f"Some classes may already exist in cls_classes: {e}")

        # Prepare batch data
        images_to_insert = []
        image_metadata_map = []

        for cutout in cutouts:
            if data.skip_duplicates and cutout["id"] in existing_source_ids:
                duplicates_found += 1
                images_skipped += 1
                continue

            matched_product = cutout.get("matched_product")
            label_value = None
            if data.label_source == "matched_product_id" and matched_product:
                label_value = str(matched_product.get("id"))
            elif data.label_source == "custom" and data.custom_label:
                label_value = data.custom_label

            class_id = class_cache.get(label_value) if label_value else None

            images_to_insert.append({
                "filename": f"cutout_{cutout['id']}.jpg",
                "image_url": cutout.get("image_url"),
                "source": "cutouts_import",
                "source_type": "cutout",
                "source_id": cutout["id"],
                "status": "labeled" if class_id else "pending",
                "metadata": {
                    "external_id": cutout.get("external_id"),
                    "matched_product_id": cutout.get("matched_product_id"),
                },
            })
            image_metadata_map.append({"class_id": class_id})

        logger.info(f"Prepared {len(images_to_insert)} cutouts for batch insert")

        # Build source_id to metadata mapping for correct label assignment after insert
        source_to_metadata = {}
        for i, img_data in enumerate(images_to_insert):
            source_to_metadata[img_data["source_id"]] = image_metadata_map[i]

        # Batch insert images (in chunks of 250 for reliability)
        insert_batch_size = 250
        all_inserted_images = []

        for i in range(0, len(images_to_insert), insert_batch_size):
            batch = images_to_insert[i:i + insert_batch_size]
            try:
                result = supabase_service.client.table("cls_images").insert(batch).execute()
                all_inserted_images.extend(result.data or [])
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                errors.append(f"Batch insert error: {str(e)}")

        images_imported = len(all_inserted_images)

        # Batch insert dataset images and labels using source_id mapping (fixes index mismatch bug)
        if data.dataset_id and all_inserted_images:
            dataset_images_to_insert = []
            labels_to_insert = []

            for img in all_inserted_images:
                # Use source_id to get correct metadata (not positional index)
                metadata = source_to_metadata.get(img.get("source_id"), {})
                class_id = metadata.get("class_id")

                dataset_images_to_insert.append({
                    "dataset_id": data.dataset_id,
                    "image_id": img["id"],
                    "status": "labeled" if class_id else "pending",
                })
                if class_id:
                    labels_to_insert.append({
                        "dataset_id": data.dataset_id,
                        "image_id": img["id"],
                        "class_id": class_id,
                    })

            for i in range(0, len(dataset_images_to_insert), insert_batch_size):
                batch = dataset_images_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_dataset_images").insert(batch).execute()
                except Exception as e:
                    logger.error(f"Dataset images batch insert failed: {e}")

            for i in range(0, len(labels_to_insert), insert_batch_size):
                batch = labels_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_labels").insert(batch).execute()
                    labels_created += len(batch)
                except Exception as e:
                    logger.error(f"Labels batch insert failed: {e}")

        # Update dataset stats
        if data.dataset_id:
            try:
                supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": data.dataset_id}).execute()
            except Exception as e:
                logger.warning(f"Failed to update dataset stats: {e}")

        logger.info(f"Cutouts import completed: imported={images_imported}, duplicates={duplicates_found}")

    except Exception as e:
        logger.error(f"Failed to import from cutouts: {e}")
        errors.append(str(e))

    return ImportResultResponse(
        success=len(errors) == 0,
        images_imported=images_imported,
        images_skipped=images_skipped,
        duplicates_found=duplicates_found,
        labels_created=labels_created,
        classes_created=classes_created,
        errors=errors,
    )


@router.post("/import/od-images", response_model=ImportResultResponse)
async def import_from_od(data: ImportFromODRequest, background_tasks: BackgroundTasks):
    """Import images from Object Detection module - OPTIMIZED with batch inserts."""
    images_imported = 0
    images_skipped = 0
    duplicates_found = 0
    labels_created = 0
    classes_created = 0
    errors = []

    # Get or create class if label provided
    class_id = None
    if data.label and data.dataset_id:
        existing_class = supabase_service.client.table("cls_dataset_classes").select("id").eq("dataset_id", data.dataset_id).eq("name", data.label).execute()
        if existing_class.data:
            class_id = existing_class.data[0]["id"]
        else:
            import uuid
            new_class_id = str(uuid.uuid4())
            supabase_service.client.table("cls_dataset_classes").insert({
                "id": new_class_id,
                "dataset_id": data.dataset_id,
                "name": data.label,
                "display_name": data.label,
            }).execute()
            try:
                supabase_service.client.table("cls_classes").insert({
                    "id": new_class_id,
                    "name": data.label,
                    "display_name": data.label,
                }).execute()
            except Exception as e:
                logger.warning(f"Could not insert into cls_classes (may already exist): {e}")
            class_id = new_class_id
            classes_created += 1

    try:
        # Fetch OD images in batches
        all_od_images = []
        if data.od_image_ids:
            batch_size = 100
            for i in range(0, len(data.od_image_ids), batch_size):
                batch_ids = data.od_image_ids[i:i + batch_size]
                batch_result = supabase_service.client.table("od_images").select("*").in_("id", batch_ids).execute()
                all_od_images.extend(batch_result.data or [])
        else:
            od_images = supabase_service.client.table("od_images").select("*").limit(5000).execute()
            all_od_images = od_images.data or []

        logger.info(f"Total OD images to process: {len(all_od_images)}")

        # Get existing duplicates in one query
        existing_source_ids = set()
        if data.skip_duplicates and all_od_images:
            all_source_ids = [img["id"] for img in all_od_images]
            for i in range(0, len(all_source_ids), 500):
                batch_ids = all_source_ids[i:i + 500]
                existing = supabase_service.client.table("cls_images").select("source_id").eq("source_type", "od_image").in_("source_id", batch_ids).execute()
                existing_source_ids.update(e["source_id"] for e in (existing.data or []))
            logger.info(f"Found {len(existing_source_ids)} existing duplicates")

        # Prepare batch data
        images_to_insert = []
        for od_img in all_od_images:
            if data.skip_duplicates and od_img["id"] in existing_source_ids:
                duplicates_found += 1
                images_skipped += 1
                continue

            images_to_insert.append({
                "filename": od_img.get("filename"),
                "original_filename": od_img.get("original_filename"),
                "image_url": od_img.get("image_url"),
                "width": od_img.get("width"),
                "height": od_img.get("height"),
                "file_size_bytes": od_img.get("file_size_bytes"),
                "source": "od_import",
                "source_type": "od_image",
                "source_id": od_img["id"],
                "folder": od_img.get("folder"),
                "status": "labeled" if class_id else "pending",
            })

        logger.info(f"Prepared {len(images_to_insert)} OD images for batch insert")

        # Batch insert images (in chunks of 250 for reliability)
        insert_batch_size = 250
        all_inserted_images = []

        for i in range(0, len(images_to_insert), insert_batch_size):
            batch = images_to_insert[i:i + insert_batch_size]
            try:
                result = supabase_service.client.table("cls_images").insert(batch).execute()
                all_inserted_images.extend(result.data or [])
            except Exception as e:
                logger.error(f"Batch insert failed: {e}")
                errors.append(f"Batch insert error: {str(e)}")

        images_imported = len(all_inserted_images)

        # Batch insert dataset images and labels
        if data.dataset_id and all_inserted_images:
            dataset_images_to_insert = []
            labels_to_insert = []

            for img in all_inserted_images:
                dataset_images_to_insert.append({
                    "dataset_id": data.dataset_id,
                    "image_id": img["id"],
                    "status": "labeled" if class_id else "pending",
                })
                if class_id:
                    labels_to_insert.append({
                        "dataset_id": data.dataset_id,
                        "image_id": img["id"],
                        "class_id": class_id,
                    })

            for i in range(0, len(dataset_images_to_insert), insert_batch_size):
                batch = dataset_images_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_dataset_images").insert(batch).execute()
                except Exception as e:
                    logger.error(f"Dataset images batch insert failed: {e}")

            for i in range(0, len(labels_to_insert), insert_batch_size):
                batch = labels_to_insert[i:i + insert_batch_size]
                try:
                    supabase_service.client.table("cls_labels").insert(batch).execute()
                    labels_created += len(batch)
                except Exception as e:
                    logger.error(f"Labels batch insert failed: {e}")

        # Update dataset stats
        if data.dataset_id:
            try:
                supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": data.dataset_id}).execute()
            except Exception as e:
                logger.warning(f"Failed to update dataset stats: {e}")

        logger.info(f"OD import completed: imported={images_imported}, duplicates={duplicates_found}")

    except Exception as e:
        logger.error(f"Failed to import from OD: {e}")
        errors.append(str(e))

    return ImportResultResponse(
        success=len(errors) == 0,
        images_imported=images_imported,
        images_skipped=images_skipped,
        duplicates_found=duplicates_found,
        labels_created=labels_created,
        classes_created=classes_created,
        errors=errors,
    )


# ===========================================
# Bulk IDs Endpoints (for Select All Filtered)
# ===========================================

@router.get("/bulk-ids/products")
async def get_product_bulk_ids(
    search: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    has_image: bool = True,
    limit: int = 10000,
):
    """Get all product IDs matching filters (for Select All Filtered feature)."""
    query = supabase_service.client.table("products").select("id")

    if has_image:
        query = query.not_.is_("primary_image_url", "null")
    if search:
        query = query.or_(f"product_name.ilike.%{search}%,barcode.ilike.%{search}%")
    if category and category != "all":
        query = query.eq("category", category)
    if brand and brand != "all":
        query = query.eq("brand_name", brand)

    result = query.limit(min(limit, 10000)).execute()
    ids = [p["id"] for p in result.data or []]

    return {
        "ids": ids,
        "total": len(ids),
        "filters_applied": {
            "search": search,
            "category": category,
            "brand": brand,
            "has_image": has_image,
        }
    }


@router.get("/bulk-ids/cutouts")
async def get_cutout_bulk_ids(
    search: Optional[str] = None,
    is_matched: Optional[bool] = None,
    limit: int = 10000,
):
    """Get all cutout IDs matching filters (for Select All Filtered feature)."""
    query = supabase_service.client.table("cutout_images").select("id")

    if search:
        query = query.or_(f"predicted_upc.ilike.%{search}%,external_id.ilike.%{search}%")
    if is_matched is True:
        query = query.not_.is_("matched_product_id", "null")
    elif is_matched is False:
        query = query.is_("matched_product_id", "null")

    result = query.limit(min(limit, 10000)).execute()
    ids = [c["id"] for c in result.data or []]

    return {
        "ids": ids,
        "total": len(ids),
        "filters_applied": {
            "search": search,
            "is_matched": is_matched,
        }
    }


@router.get("/bulk-ids/od-images")
async def get_od_bulk_ids(
    search: Optional[str] = None,
    folder: Optional[str] = None,
    limit: int = 10000,
):
    """Get all OD image IDs matching filters (for Select All Filtered feature)."""
    query = supabase_service.client.table("od_images").select("id")

    if search:
        query = query.ilike("filename", f"%{search}%")
    if folder:
        query = query.eq("folder", folder)

    result = query.limit(min(limit, 10000)).execute()
    ids = [img["id"] for img in result.data or []]

    return {
        "ids": ids,
        "total": len(ids),
        "filters_applied": {
            "search": search,
            "folder": folder,
        }
    }


# ===========================================
# Bulk Operations
# ===========================================

@router.post("/bulk/tags", response_model=BulkOperationResponse)
async def bulk_update_tags(data: BulkTagRequest):
    """Add, remove, or replace tags for multiple images."""
    affected = 0
    errors = []

    for image_id in data.image_ids:
        try:
            image = supabase_service.client.table("cls_images").select("tags").eq("id", image_id).single().execute()

            if not image.data:
                continue

            current_tags = image.data.get("tags") or []

            if data.action == "add":
                new_tags = list(set(current_tags + data.tags))
            elif data.action == "remove":
                new_tags = [t for t in current_tags if t not in data.tags]
            else:  # replace
                new_tags = data.tags

            supabase_service.client.table("cls_images").update({"tags": new_tags}).eq("id", image_id).execute()
            affected += 1

        except Exception as e:
            errors.append(f"{image_id}: {str(e)}")

    return BulkOperationResponse(success=True, affected_count=affected, errors=errors)


@router.post("/bulk/move", response_model=BulkOperationResponse)
async def bulk_move_to_folder(data: BulkMoveRequest):
    """Move multiple images to a folder."""
    result = supabase_service.client.table("cls_images").update({"folder": data.folder}).in_("id", data.image_ids).execute()

    return BulkOperationResponse(success=True, affected_count=len(result.data or []))


@router.post("/bulk/add-to-dataset", response_model=BulkOperationResponse)
async def bulk_add_to_dataset(data: BulkAddToDatasetRequest):
    """Add multiple images to a dataset."""
    affected = 0
    errors = []

    for image_id in data.image_ids:
        try:
            # Check if already in dataset
            existing = supabase_service.client.table("cls_dataset_images").select("id").eq("dataset_id", data.dataset_id).eq("image_id", image_id).execute()

            if existing.data:
                continue

            supabase_service.client.table("cls_dataset_images").insert({
                "dataset_id": data.dataset_id,
                "image_id": image_id,
                "status": "pending",
            }).execute()
            affected += 1

        except Exception as e:
            errors.append(f"{image_id}: {str(e)}")

    # Update dataset stats
    supabase_service.client.rpc("update_cls_dataset_stats", {"p_dataset_id": data.dataset_id}).execute()

    return BulkOperationResponse(success=True, affected_count=affected, errors=errors)


@router.post("/bulk/delete", response_model=BulkOperationResponse)
async def bulk_delete(data: BulkOperationRequest):
    """Delete multiple images."""
    affected = 0
    errors = []

    for image_id in data.image_ids:
        try:
            image = supabase_service.client.table("cls_images").select("storage_path").eq("id", image_id).single().execute()

            if image.data and image.data.get("storage_path"):
                try:
                    supabase_service.client.storage.from_(CLS_IMAGES_BUCKET).remove([image.data["storage_path"]])
                except:
                    pass

            supabase_service.client.table("cls_images").delete().eq("id", image_id).execute()
            affected += 1

        except Exception as e:
            errors.append(f"{image_id}: {str(e)}")

    return BulkOperationResponse(success=True, affected_count=affected, errors=errors)


# ===========================================
# Async Bulk Operations (Background Jobs)
# ===========================================

ASYNC_DELETE_THRESHOLD = 100
ASYNC_ADD_THRESHOLD = 100
ASYNC_TAG_THRESHOLD = 200


@router.post("/bulk/delete/async")
async def bulk_delete_async(data: BulkOperationRequest):
    """
    Async version: Delete multiple images as a background job.

    Use this for large deletions (>100 images).
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    if len(data.image_ids) < ASYNC_DELETE_THRESHOLD:
        # Use sync version for small batches
        return await bulk_delete(data)

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_cls_bulk_delete_images",
        "status": "pending",
        "progress": 0,
        "config": {
            "image_ids": data.image_ids,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk delete job queued for {len(data.image_ids)} images",
    }


@router.post("/bulk/add-to-dataset/async")
async def bulk_add_to_dataset_async(data: BulkAddToDatasetRequest):
    """
    Async version: Add multiple images to a dataset as a background job.

    Use this for large additions (>100 images).
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    if len(data.image_ids) < ASYNC_ADD_THRESHOLD:
        # Use sync version for small batches
        return await bulk_add_to_dataset(data)

    # Verify dataset exists
    dataset = supabase_service.client.table("cls_datasets")\
        .select("id, name")\
        .eq("id", data.dataset_id)\
        .single()\
        .execute()

    if not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_cls_bulk_add_to_dataset",
        "status": "pending",
        "progress": 0,
        "config": {
            "dataset_id": data.dataset_id,
            "image_ids": data.image_ids,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk add job queued for {len(data.image_ids)} images to '{dataset.data['name']}'",
    }


@router.post("/bulk/tags/async")
async def bulk_update_tags_async(data: BulkTagRequest):
    """
    Async version: Update tags for multiple images as a background job.

    Use this for large tag updates (>200 images).
    """
    from uuid import uuid4
    from datetime import datetime, timezone

    if len(data.image_ids) < ASYNC_TAG_THRESHOLD:
        # Use sync version for small batches
        return await bulk_update_tags(data)

    # Create job record
    job_id = str(uuid4())
    job_data = {
        "id": job_id,
        "type": "local_cls_bulk_update_tags",
        "status": "pending",
        "progress": 0,
        "config": {
            "image_ids": data.image_ids,
            "action": data.action,
            "tags": data.tags,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    result = supabase_service.client.table("jobs").insert(job_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create job")

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Bulk tag update job queued for {len(data.image_ids)} images",
    }


# ===========================================
# Duplicate Detection
# ===========================================

@router.post("/duplicates/check", response_model=DuplicateCheckResponse)
async def check_duplicate(data: DuplicateCheckRequest):
    """Check if an image is a duplicate."""
    exact_match = None
    similar_matches = []

    if data.file_hash:
        result = supabase_service.client.table("cls_images").select("*").eq("file_hash", data.file_hash).execute()
        if result.data:
            exact_match = result.data[0]

    # Perceptual hash similarity check would require custom implementation
    # For now, just check exact hash

    return DuplicateCheckResponse(
        is_duplicate=exact_match is not None,
        exact_match=exact_match,
        similar_matches=similar_matches,
    )


@router.get("/duplicates")
async def get_duplicate_groups():
    """Get groups of duplicate images."""
    # Get all images with their hashes
    result = supabase_service.client.table("cls_images").select("id, filename, image_url, file_hash").not_.is_("file_hash", "null").execute()

    # Group by hash
    hash_groups = {}
    for img in result.data or []:
        h = img.get("file_hash")
        if h:
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(img)

    # Filter to only groups with duplicates
    duplicate_groups = [
        {"group_id": h, "images": imgs, "count": len(imgs)}
        for h, imgs in hash_groups.items()
        if len(imgs) > 1
    ]

    return {
        "groups": duplicate_groups,
        "total_duplicates": sum(len(g["images"]) - 1 for g in duplicate_groups),
    }
