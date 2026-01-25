"""
Supabase Data Fetching Utilities for SOTA CLS (Classification) Training.

Fetches training data from Supabase based on dataset_id.
Handles pagination for large datasets (50K+ labeled images).

SOTA Pattern:
- API sends dataset_id + credentials
- Worker fetches cls_labels and cls_images from DB
- No massive payload from API

Example job_input:
{
    "training_run_id": "uuid",
    "dataset_id": "uuid",
    "config": {...},
    "supabase_url": "...",
    "supabase_key": "..."
}
"""

import time
import random
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
    Fetch data from Supabase with automatic pagination and retry.

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
                    elif val is None:
                        query = query.is_(col, "null")
                    else:
                        query = query.eq(col, val)

                query = query.range(offset, offset + page_size - 1)
                result = query.execute()

                batch = result.data or []
                all_data.extend(batch)

                if len(batch) < page_size:
                    return all_data

                offset += page_size
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[WARNING] Fetch failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to fetch from {table} after {max_retries} attempts: {e}")

    return all_data


def fetch_labels_paginated(
    client,
    dataset_id: str,
    page_size: int = 1000,
) -> List[Dict]:
    """
    Fetch all labels with joined image URLs for a dataset.

    Args:
        client: Supabase client
        dataset_id: Dataset UUID
        page_size: Number of rows per page

    Returns:
        List of label dicts with nested cls_images
    """
    all_labels = []
    offset = 0

    while True:
        try:
            result = client.table("cls_labels").select(
                "image_id, class_id, cls_images!inner(id, image_url)"
            ).eq("dataset_id", dataset_id).range(offset, offset + page_size - 1).execute()

            batch = result.data or []
            all_labels.extend(batch)

            if len(batch) < page_size:
                break

            offset += page_size

            # Progress indicator for large datasets
            if offset % 10000 == 0:
                print(f"    Fetched {offset} labels...")

        except Exception as e:
            print(f"[ERROR] Failed to fetch labels at offset {offset}: {e}")
            raise

    return all_labels


def fetch_classes(
    client,
    dataset_id: str,
) -> List[Dict]:
    """
    Fetch all classes for a dataset, sorted by name.

    Args:
        client: Supabase client
        dataset_id: Dataset UUID

    Returns:
        List of class dicts with id and name, sorted by name
    """
    result = client.table("cls_classes").select(
        "id, name"
    ).eq("dataset_id", dataset_id).order("name").execute()

    return result.data or []


def build_url_dataset_data(
    supabase_url: str,
    supabase_key: str,
    dataset_id: str,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main function to fetch CLS dataset data from Supabase.

    SOTA Pattern: Worker fetches data based on dataset_id from API.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key
        dataset_id: Dataset UUID
        train_split: Train/val split ratio (default 0.8)
        seed: Random seed for reproducible split

    Returns:
        {
            "train_urls": [{"url": "...", "label": 0}, ...],
            "val_urls": [{"url": "...", "label": 0}, ...],
            "class_names": ["class1", "class2", ...],
            "num_classes": 10,
        }
    """
    from supabase import create_client

    print(f"[Supabase Fetcher CLS] Fetching dataset {dataset_id}...")

    # Create Supabase client
    url_with_slash = supabase_url if supabase_url.endswith("/") else supabase_url + "/"
    client = create_client(url_with_slash, supabase_key)

    # 1. Fetch classes (sorted by name for consistent ordering)
    print("  Fetching classes...")
    classes = fetch_classes(client, dataset_id)

    if not classes:
        raise ValueError(f"No classes found for dataset {dataset_id}")

    # Create class_id -> index mapping (sorted by name)
    class_id_to_idx = {cls["id"]: idx for idx, cls in enumerate(classes)}
    class_names = [cls["name"] for cls in classes]
    print(f"  Found {len(classes)} classes: {class_names[:5]}{'...' if len(classes) > 5 else ''}")

    # 2. Fetch all labels with image URLs
    print("  Fetching labeled images...")
    labels = fetch_labels_paginated(client, dataset_id)
    print(f"  Found {len(labels)} labeled images")

    if not labels:
        raise ValueError(f"No labeled images found for dataset {dataset_id}")

    # 3. Build URL list with labels
    all_urls = []
    skipped = 0

    for label in labels:
        image_data = label.get("cls_images")
        if not image_data:
            skipped += 1
            continue

        image_url = image_data.get("image_url")
        if not image_url:
            skipped += 1
            continue

        class_id = label.get("class_id")
        class_idx = class_id_to_idx.get(class_id)
        if class_idx is None:
            skipped += 1
            continue

        all_urls.append({
            "url": image_url,
            "label": class_idx,
        })

    if skipped > 0:
        print(f"  [WARNING] Skipped {skipped} labels with missing data")

    if not all_urls:
        raise ValueError("No valid labeled images after processing")

    # 4. Shuffle and split into train/val
    print("  Splitting into train/val...")
    random.seed(seed)
    random.shuffle(all_urls)

    split_idx = int(len(all_urls) * train_split)
    train_urls = all_urls[:split_idx]
    val_urls = all_urls[split_idx:]

    print(f"\n[Supabase Fetcher CLS] Dataset ready:")
    print(f"  Train: {len(train_urls)} images")
    print(f"  Val: {len(val_urls)} images")
    print(f"  Classes: {len(class_names)}")

    return {
        "train_urls": train_urls,
        "val_urls": val_urls,
        "class_names": class_names,
        "num_classes": len(class_names),
    }
