"""
Supabase Data Fetching Utilities for URL-Based OD Training.

Handles pagination for large datasets (722k+ annotations).
Fetches dataset images, annotations, and classes directly from Supabase.
"""

import time
from typing import List, Dict, Any, Optional


def fetch_with_pagination(
    client,
    table: str,
    select: str,
    filters: Dict[str, Any],
    page_size: int = 1000,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Fetch data from Supabase with automatic pagination.

    Args:
        client: Supabase client
        table: Table name
        select: Select clause
        filters: Dict of column -> value for eq() filters
        page_size: Number of rows per page
        max_retries: Retry attempts per page

    Returns:
        List of all rows
    """
    all_data = []
    offset = 0

    while True:
        for attempt in range(max_retries):
            try:
                query = client.table(table).select(select)

                for col, val in filters.items():
                    if isinstance(val, list):
                        query = query.in_(col, val)
                    elif val is True or val is False:
                        query = query.eq(col, val)
                    else:
                        query = query.eq(col, val)

                query = query.range(offset, offset + page_size - 1)
                result = query.execute()

                batch = result.data or []
                all_data.extend(batch)

                # Check if we've reached the end
                if len(batch) < page_size:
                    return all_data

                offset += page_size
                break  # Success, continue to next page

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"[WARNING] Fetch failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to fetch from {table} after {max_retries} attempts: {e}")

    return all_data


def fetch_classes(client, dataset_id: str) -> List[Dict]:
    """
    Fetch all classes for a dataset.

    Args:
        client: Supabase client
        dataset_id: Dataset UUID

    Returns:
        List of class dicts with id, name, display_name, color
    """
    print("  Fetching classes...")

    # OD classes are global (not per-dataset), so we need to get classes
    # that have annotations in this dataset
    # First get unique class_ids from annotations
    try:
        # Try to get classes that are used in this dataset's annotations
        ann_result = client.table("od_annotations").select("class_id").eq("dataset_id", dataset_id).limit(10000).execute()

        if ann_result.data:
            unique_class_ids = list(set(ann["class_id"] for ann in ann_result.data if ann.get("class_id")))

            if unique_class_ids:
                classes_result = client.table("od_classes").select("id, name, display_name, color").in_("id", unique_class_ids).execute()
                classes = classes_result.data or []
                print(f"  Found {len(classes)} classes used in dataset")
                return classes
    except Exception as e:
        print(f"  [WARNING] Could not fetch classes from annotations: {e}")

    # Fallback: get all active classes
    classes_result = client.table("od_classes").select("id, name, display_name, color").eq("is_active", True).execute()
    classes = classes_result.data or []
    print(f"  Found {len(classes)} active classes (fallback)")
    return classes


def fetch_dataset_images(
    client,
    dataset_id: str,
    version_id: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch all images in a dataset with their metadata.
    Joins od_dataset_images with od_images.

    Args:
        client: Supabase client
        dataset_id: Dataset UUID
        version_id: Optional version UUID (not used yet, for future)

    Returns:
        List of image dicts with image_id, split, status, and nested image info
    """
    print("  Fetching dataset images...")

    all_images = []
    offset = 0
    page_size = 1000

    while True:
        try:
            query = client.table("od_dataset_images").select(
                "image_id, split, status, image:od_images(id, image_url, width, height, original_filename)"
            ).eq("dataset_id", dataset_id)

            # Only get completed or annotating images (skip pending/skipped)
            query = query.in_("status", ["completed", "annotating"])

            query = query.range(offset, offset + page_size - 1)
            result = query.execute()

            batch = result.data or []
            all_images.extend(batch)

            if len(batch) < page_size:
                break
            offset += page_size

        except Exception as e:
            print(f"  [ERROR] Failed to fetch images at offset {offset}: {e}")
            raise

    print(f"  Found {len(all_images)} images")
    return all_images


def fetch_annotations_chunked(
    client,
    dataset_id: str,
    image_ids: List[str],
    chunk_size: int = 500,
    page_size: int = 1000,
) -> Dict[str, List[Dict]]:
    """
    Fetch all annotations for given images.

    Handles 722k+ annotations by:
    1. Chunking image_ids (Supabase IN queries limited to ~500 items)
    2. Paginating results within each chunk

    Args:
        client: Supabase client
        dataset_id: Dataset UUID
        image_ids: List of image UUIDs to fetch annotations for
        chunk_size: Number of image_ids per IN query (max ~500)
        page_size: Number of rows per page

    Returns:
        Dict mapping image_id -> list of annotations
    """
    print("  Fetching annotations...")

    annotations_by_image: Dict[str, List[Dict]] = {}
    total_fetched = 0

    # Process image_ids in chunks
    for i in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[i:i + chunk_size]
        offset = 0

        while True:
            try:
                result = (
                    client.table("od_annotations")
                    .select("image_id, class_id, bbox_x, bbox_y, bbox_width, bbox_height, confidence, is_ai_generated")
                    .eq("dataset_id", dataset_id)
                    .in_("image_id", chunk_ids)
                    .range(offset, offset + page_size - 1)
                    .execute()
                )

                batch = result.data or []

                for ann in batch:
                    img_id = ann["image_id"]
                    if img_id not in annotations_by_image:
                        annotations_by_image[img_id] = []

                    annotations_by_image[img_id].append({
                        "class_id": ann["class_id"],
                        "bbox": {
                            "x": ann["bbox_x"],
                            "y": ann["bbox_y"],
                            "width": ann["bbox_width"],
                            "height": ann["bbox_height"],
                        },
                        "confidence": ann.get("confidence"),
                        "is_ai_generated": ann.get("is_ai_generated", False),
                    })

                total_fetched += len(batch)

                if len(batch) < page_size:
                    break
                offset += page_size

            except Exception as e:
                print(f"  [ERROR] Failed to fetch annotations at chunk {i}, offset {offset}: {e}")
                # Continue with next chunk instead of failing completely
                break

        # Progress indicator for large datasets
        if len(image_ids) > 1000 and (i + chunk_size) % 5000 == 0:
            print(f"    Progress: {min(i + chunk_size, len(image_ids))}/{len(image_ids)} images processed, {total_fetched} annotations fetched")

    print(f"  Found {total_fetched} annotations for {len(annotations_by_image)} images")
    return annotations_by_image


def build_url_dataset_data(
    supabase_url: str,
    supabase_key: str,
    dataset_id: str,
    version_id: Optional[str] = None,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main function to fetch all dataset data for URL-based training.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key
        dataset_id: Dataset UUID
        version_id: Optional version UUID
        train_split: Train/val split ratio (default 0.8)
        seed: Random seed for split

    Returns:
        {
            "train_images": [...],
            "val_images": [...],
            "class_mapping": {class_id: index},
            "class_names": [...],
            "num_classes": int,
        }
    """
    from supabase import create_client

    print(f"[Supabase] Fetching dataset {dataset_id}...")
    client = create_client(supabase_url, supabase_key)

    # 1. Fetch classes
    classes = fetch_classes(client, dataset_id)
    class_mapping = {cls["id"]: idx for idx, cls in enumerate(classes)}
    class_names = [cls["name"] for cls in classes]
    print(f"  Classes: {class_names}")

    # 2. Fetch dataset images
    dataset_images = fetch_dataset_images(client, dataset_id, version_id)

    if not dataset_images:
        raise ValueError(f"No images found in dataset {dataset_id}")

    # 3. Fetch annotations
    image_ids = [img["image_id"] for img in dataset_images]
    annotations_by_image = fetch_annotations_chunked(client, dataset_id, image_ids)

    # 4. Build image data with annotations
    train_images = []
    val_images = []

    for img_data in dataset_images:
        image_info = img_data.get("image")
        if not image_info:
            continue

        image_url = image_info.get("image_url")
        if not image_url:
            continue

        item = {
            "image_id": img_data["image_id"],
            "image_url": image_url,
            "width": image_info.get("width") or 640,
            "height": image_info.get("height") or 640,
            "original_filename": image_info.get("original_filename"),
            "annotations": annotations_by_image.get(img_data["image_id"], []),
        }

        # Use split from database if available
        split = img_data.get("split")
        if split == "val" or split == "validation":
            val_images.append(item)
        elif split == "test":
            # Skip test images for training
            continue
        else:
            train_images.append(item)

    # Auto-split if no val data or all in train
    if not val_images and train_images:
        import random
        random.seed(seed)
        random.shuffle(train_images)
        split_idx = int(len(train_images) * train_split)
        val_images = train_images[split_idx:]
        train_images = train_images[:split_idx]
        print(f"  Auto-split applied: {len(train_images)} train, {len(val_images)} val")

    # Count total annotations
    train_ann_count = sum(len(img["annotations"]) for img in train_images)
    val_ann_count = sum(len(img["annotations"]) for img in val_images)

    print(f"[Supabase] Dataset ready:")
    print(f"  Train: {len(train_images)} images, {train_ann_count} annotations")
    print(f"  Val: {len(val_images)} images, {val_ann_count} annotations")
    print(f"  Classes: {len(class_names)}")

    return {
        "train_images": train_images,
        "val_images": val_images,
        "class_mapping": class_mapping,
        "class_names": class_names,
        "num_classes": len(classes),
    }
