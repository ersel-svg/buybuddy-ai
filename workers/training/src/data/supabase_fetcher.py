"""
Supabase Data Fetching Utilities for SOTA Embedding Training.

Fetches training data from Supabase based on source_config.
Handles pagination for large datasets (10K+ products).

SOTA Pattern:
- API sends source_config with product_ids + image_config
- Worker fetches products and builds training_images
- No massive payload from API

Example source_config:
{
    "product_ids": ["uuid1", "uuid2", ...],
    "train_product_ids": ["uuid1", ...],
    "val_product_ids": ["uuid2", ...],
    "test_product_ids": [...],
    "image_config": {
        "image_types": ["synthetic", "real"],
        "frame_selection": "first",
        "max_frames_per_type": 10,
        "include_matched_cutouts": true,
    },
    "label_config": {
        "label_field": "brand_name",
    }
}
"""

import time
from typing import List, Dict, Any, Optional, Tuple


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


def fetch_products_paginated(
    client,
    product_ids: List[str],
    chunk_size: int = 500,
) -> List[Dict]:
    """
    Fetch products by IDs with chunked IN queries.

    Args:
        client: Supabase client
        product_ids: List of product UUIDs
        chunk_size: Max IDs per IN query (Supabase limit ~500)

    Returns:
        List of product dictionaries
    """
    all_products = []

    for i in range(0, len(product_ids), chunk_size):
        chunk_ids = product_ids[i:i + chunk_size]

        result = client.table("products").select(
            "id, barcode, brand_name, category, product_name, frames_path, frame_count, identifiers"
        ).in_("id", chunk_ids).execute()

        all_products.extend(result.data or [])

        if len(product_ids) > 1000 and (i + chunk_size) % 5000 == 0:
            print(f"    Products fetched: {min(i + chunk_size, len(product_ids))}/{len(product_ids)}")

    return all_products


def fetch_matched_cutouts(
    client,
    product_ids: List[str],
    chunk_size: int = 500,
) -> List[Dict]:
    """
    Fetch matched cutouts for products.

    Args:
        client: Supabase client
        product_ids: List of product UUIDs
        chunk_size: Max IDs per IN query

    Returns:
        List of cutout dictionaries with image_url and matched_product_id
    """
    all_cutouts = []

    for i in range(0, len(product_ids), chunk_size):
        chunk_ids = product_ids[i:i + chunk_size]

        result = client.table("cutout_images").select(
            "id, image_url, matched_product_id"
        ).in_("matched_product_id", chunk_ids).not_.is_("image_url", "null").execute()

        all_cutouts.extend(result.data or [])

    return all_cutouts


def generate_frame_urls(
    frames_path: str,
    frame_count: int,
    frame_selection: str = "first",
    max_frames: int = 10,
    frame_interval: int = 5,
) -> List[Dict]:
    """
    Generate frame URLs from frames_path.

    Args:
        frames_path: Base path for frames (e.g., "https://storage.../products/uuid/frames")
        frame_count: Total number of frames available
        frame_selection: Selection strategy (first, all, key_frames, interval)
        max_frames: Maximum frames to select
        frame_interval: Interval for "interval" selection

    Returns:
        List of frame dicts with url, image_type, frame_index, domain
    """
    if not frames_path or frame_count <= 0:
        return []

    # Determine which frame indices to use
    if frame_selection == "first":
        indices = [0]
    elif frame_selection == "all":
        indices = list(range(frame_count))
    elif frame_selection == "key_frames":
        # Pick 4 frames at 0°, 90°, 180°, 270° (roughly)
        step = max(1, frame_count // 4)
        indices = [0] + [i * step for i in range(1, 4) if i * step < frame_count]
        indices = indices[:max_frames]
    elif frame_selection == "interval":
        indices = list(range(0, frame_count, frame_interval))[:max_frames]
    else:
        indices = [0]

    # Generate URLs
    base_url = frames_path.rstrip("/")
    frames = []
    for idx in indices:
        frames.append({
            "url": f"{base_url}/frame_{idx:04d}.png",
            "image_type": "synthetic",
            "frame_index": idx,
            "domain": "synthetic",
        })

    return frames


def build_training_data(
    supabase_url: str,
    supabase_key: str,
    source_config: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Main function to fetch training data from Supabase.

    SOTA Pattern: Worker fetches data based on source_config from API.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key
        source_config: Configuration dict with product_ids, image_config, etc.

    Returns:
        Tuple of (train_data, val_data, test_data, train_images, val_images, test_images)
        - *_data: List of product metadata dicts
        - *_images: Dict mapping product_id -> list of image dicts with URLs
    """
    from supabase import create_client

    print("[Supabase Fetcher] Building training data from source_config...")

    # Parse source_config
    product_ids = source_config.get("product_ids", [])
    train_ids = set(source_config.get("train_product_ids", []))
    val_ids = set(source_config.get("val_product_ids", []))
    test_ids = set(source_config.get("test_product_ids", []))
    image_config = source_config.get("image_config", {})
    label_config = source_config.get("label_config", {})

    if not product_ids:
        raise ValueError("source_config must contain product_ids")

    print(f"  Product IDs: {len(product_ids)}")
    print(f"  Train/Val/Test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    # Create Supabase client
    url_with_slash = supabase_url if supabase_url.endswith("/") else supabase_url + "/"
    client = create_client(url_with_slash, supabase_key)

    # 1. Fetch products with pagination
    print("  Fetching products...")
    products = fetch_products_paginated(client, product_ids)
    print(f"  Fetched {len(products)} products")

    if not products:
        raise ValueError(f"No products found for given IDs")

    # Create lookup dict
    products_by_id = {p["id"]: p for p in products}

    # 2. Build training_images dict
    print("  Building training images...")
    training_images: Dict[str, List[Dict]] = {}
    image_stats = {"synthetic": 0, "real": 0, "augmented": 0, "cutout": 0}

    image_types = image_config.get("image_types", ["synthetic"])
    frame_selection = image_config.get("frame_selection", "first")
    max_frames = image_config.get("max_frames_per_type", 10)
    frame_interval = image_config.get("frame_interval", 5)

    # Synthetic frames
    if "synthetic" in image_types:
        for product_id, product in products_by_id.items():
            frames_path = product.get("frames_path")
            frame_count = product.get("frame_count", 0)

            if frames_path and frame_count > 0:
                frames = generate_frame_urls(
                    frames_path=frames_path,
                    frame_count=frame_count,
                    frame_selection=frame_selection,
                    max_frames=max_frames,
                    frame_interval=frame_interval,
                )
                if frames:
                    training_images[product_id] = frames
                    image_stats["synthetic"] += len(frames)

    # Matched cutouts (real domain)
    if image_config.get("include_matched_cutouts", False):
        print("  Fetching matched cutouts...")
        cutouts = fetch_matched_cutouts(client, product_ids)
        print(f"  Found {len(cutouts)} cutouts")

        for cutout in cutouts:
            product_id = cutout.get("matched_product_id")
            if product_id and product_id in products_by_id:
                if product_id not in training_images:
                    training_images[product_id] = []
                training_images[product_id].append({
                    "url": cutout["image_url"],
                    "image_type": "cutout",
                    "frame_index": 0,
                    "domain": "real",
                    "cutout_id": cutout["id"],
                })
                image_stats["cutout"] += 1

    print(f"  Image stats: {image_stats}")

    # 3. Split into train/val/test
    train_images = {pid: imgs for pid, imgs in training_images.items() if pid in train_ids}
    val_images = {pid: imgs for pid, imgs in training_images.items() if pid in val_ids}
    test_images = {pid: imgs for pid, imgs in training_images.items() if pid in test_ids}

    train_data = [products_by_id[pid] for pid in train_ids if pid in products_by_id]
    val_data = [products_by_id[pid] for pid in val_ids if pid in products_by_id]
    test_data = [products_by_id[pid] for pid in test_ids if pid in products_by_id]

    # 4. Summary
    total_train_images = sum(len(imgs) for imgs in train_images.values())
    total_val_images = sum(len(imgs) for imgs in val_images.values())
    total_test_images = sum(len(imgs) for imgs in test_images.values())

    print(f"\n[Supabase Fetcher] Training data ready:")
    print(f"  Train: {len(train_data)} products, {total_train_images} images")
    print(f"  Val: {len(val_data)} products, {total_val_images} images")
    print(f"  Test: {len(test_data)} products, {total_test_images} images")

    return train_data, val_data, test_data, train_images, val_images, test_images
