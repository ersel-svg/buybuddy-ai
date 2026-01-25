"""
Supabase Data Fetching Utilities for Embedding Extraction.

SOTA Pattern: Worker fetches images from DB instead of receiving them in payload.
This allows handling 60K+ images without hitting RunPod's ~10MB payload limit.

Based on OD Training supabase_fetcher.py pattern.
"""

import time
from typing import List, Dict, Any, Optional


def fetch_with_pagination(
    client,
    table: str,
    select: str,
    filters: Optional[Dict[str, Any]] = None,
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
    filters = filters or {}

    while True:
        for attempt in range(max_retries):
            try:
                query = client.table(table).select(select)

                for col, val in filters.items():
                    if val is None:
                        continue
                    elif col.endswith("_is_null"):
                        actual_col = col.replace("_is_null", "")
                        if val:
                            query = query.is_(actual_col, "null")
                        else:
                            query = query.not_.is_(actual_col, "null")
                    elif col.endswith("_not_null"):
                        actual_col = col.replace("_not_null", "")
                        if val:
                            query = query.not_.is_(actual_col, "null")
                    elif col.endswith("_gt"):
                        actual_col = col.replace("_gt", "")
                        query = query.gt(actual_col, val)
                    elif col.endswith("_gte"):
                        actual_col = col.replace("_gte", "")
                        query = query.gte(actual_col, val)
                    elif col.endswith("_lt"):
                        actual_col = col.replace("_lt", "")
                        query = query.lt(actual_col, val)
                    elif col.endswith("_lte"):
                        actual_col = col.replace("_lte", "")
                        query = query.lte(actual_col, val)
                    elif isinstance(val, list):
                        query = query.in_(col, val)
                    elif isinstance(val, bool):
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


def fetch_cutout_images(
    client,
    filters: Optional[Dict[str, Any]] = None,
    page_size: int = 1000,
) -> List[Dict]:
    """
    Fetch cutout images from Supabase.

    Args:
        client: Supabase client
        filters: Optional filters like {"has_embedding": False, "predicted_upc_not_null": True}
        page_size: Page size for pagination

    Returns:
        List of cutout image dicts with id, image_url, predicted_upc
    """
    print("  Fetching cutout images...")

    # Build filter dict
    db_filters = {}
    if filters:
        if "has_embedding" in filters:
            db_filters["has_embedding"] = filters["has_embedding"]
        if filters.get("cutout_filter_has_upc"):
            db_filters["predicted_upc_not_null"] = True

    cutouts = fetch_with_pagination(
        client=client,
        table="cutout_images",
        select="id, image_url, predicted_upc",
        filters=db_filters,
        page_size=page_size,
    )

    # Filter out cutouts without image_url
    valid_cutouts = [c for c in cutouts if c.get("image_url")]
    print(f"  Found {len(valid_cutouts)} valid cutout images")

    return valid_cutouts


def fetch_product_images(
    client,
    source: str = "all",
    product_ids: Optional[List[str]] = None,
    product_dataset_id: Optional[str] = None,
    product_filter: Optional[Dict[str, Any]] = None,
    frame_selection: str = "first",
    max_frames: int = 10,
    frame_interval: int = 5,
    page_size: int = 1000,
) -> List[Dict]:
    """
    Fetch product images from Supabase.

    Args:
        client: Supabase client
        source: Product source filter - "all", "selected", "dataset", "filter", "matched", "new"
        product_ids: Specific product IDs to fetch (for "selected" source)
        product_dataset_id: Dataset ID to filter by (for "dataset" source)
        product_filter: Custom filters dict (for "filter" source)
        frame_selection: "first", "all", "key_frames", "interval"
        max_frames: Maximum frames per product
        frame_interval: Interval between frames (for "interval" selection)
        page_size: Page size for pagination

    Returns:
        List of product image dicts
    """
    print(f"  Fetching product images (source={source}, frame_selection={frame_selection})...")

    # Build filters based on source
    filters = {"frame_count_gt": 0}  # Always require frames

    if source == "selected" and product_ids:
        # Specific product IDs
        filters["id"] = product_ids
        print(f"    Filtering by {len(product_ids)} specific product IDs")
    elif source == "dataset" and product_dataset_id:
        # Products from specific dataset
        filters["dataset_id"] = product_dataset_id
        print(f"    Filtering by dataset ID: {product_dataset_id}")
    elif source == "filter" and product_filter:
        # Custom filters
        filters.update(product_filter)
        print(f"    Applying custom filters: {product_filter}")
    elif source == "matched":
        # Products with matched cutouts - handled below
        pass
    elif source == "new":
        # Products without embeddings - handled below
        pass

    # Fetch products with frames
    products = fetch_with_pagination(
        client=client,
        table="products",
        select="id, barcode, brand_name, product_name, frames_path, frame_count, dataset_id",
        filters=filters,
        page_size=page_size,
    )

    # For "matched" source, filter by products with matched cutouts
    if source == "matched":
        matched_ids_result = client.table("cutout_images").select("matched_product_id").not_.is_("matched_product_id", "null").execute()
        matched_product_ids = set(c["matched_product_id"] for c in (matched_ids_result.data or []))
        products = [p for p in products if p["id"] in matched_product_ids]

    # For "new" source, filter by products without embeddings
    if source == "new":
        # This would require checking product_images table
        # For now, just return all products (can be enhanced later)
        pass

    print(f"  Found {len(products)} products with frames")

    # Generate frame URLs for each product
    images = []
    for p in products:
        product_id = p["id"]
        frames_path = p.get("frames_path", "")
        frame_count = p.get("frame_count", 0)

        if not frames_path or frame_count <= 0:
            continue

        # Determine which frame indices to use
        if frame_selection == "first":
            frame_indices = [0]
        elif frame_selection == "all":
            frame_indices = list(range(min(frame_count, max_frames)))
        elif frame_selection == "key_frames":
            # Sample evenly spaced frames
            if frame_count <= max_frames:
                frame_indices = list(range(frame_count))
            else:
                step = frame_count / max_frames
                frame_indices = [int(i * step) for i in range(max_frames)]
        elif frame_selection == "interval":
            frame_indices = list(range(0, frame_count, frame_interval))[:max_frames]
        else:
            frame_indices = [0]

        # Generate URLs
        for frame_idx in frame_indices:
            frame_url = f"{frames_path.rstrip('/')}/frame_{frame_idx:04d}.png"
            images.append({
                "id": f"{product_id}_synthetic_{frame_idx}",
                "url": frame_url,
                "type": "product",
                "metadata": {
                    "source": "product",
                    "product_id": product_id,
                    "image_type": "synthetic",
                    "frame_index": frame_idx,
                    "domain": "synthetic",
                    "barcode": p.get("barcode"),
                    "brand_name": p.get("brand_name"),
                },
            })

    print(f"  Generated {len(images)} product frame URLs")
    return images


def build_extraction_data(
    supabase_url: str,
    supabase_key: str,
    source_config: Dict[str, Any],
) -> List[Dict]:
    """
    Main function to fetch all images for embedding extraction.

    This is the SOTA pattern: instead of receiving 60K images in payload,
    worker fetches them from DB using this function.

    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service role key
        source_config: Extraction configuration
            {
                "type": "cutouts" | "products" | "both",
                "filters": {
                    "has_embedding": false,
                    "cutout_filter_has_upc": true,
                    "product_source": "all" | "selected" | "dataset" | "filter" | "matched" | "new",
                    "product_ids": ["uuid1", "uuid2"],  # For "selected" source
                    "product_dataset_id": "dataset_uuid",  # For "dataset" source
                    "product_filter": {"brand": "Nike"}  # For "filter" source
                },
                "frame_selection": "first" | "all" | "key_frames" | "interval",
                "max_frames": 10,
                "frame_interval": 5,
                "product_collection": "products_dinov2_base",
                "cutout_collection": "cutouts_dinov2_base"
            }

    Returns:
        List of image dicts ready for extraction:
        [{
            "id": "cutout_123" or "product_uuid_synthetic_0",
            "url": "https://...",
            "type": "cutout" | "product",
            "collection": "collection_name",
            "metadata": {...}
        }]
    """
    from supabase import create_client

    print(f"\n[Supabase Fetcher] Building extraction data...")
    print(f"  Source config: {source_config}")

    client = create_client(supabase_url, supabase_key)

    source_type = source_config.get("type", "both")
    filters = source_config.get("filters", {})
    frame_selection = source_config.get("frame_selection", "first")
    max_frames = source_config.get("max_frames", 10)
    frame_interval = source_config.get("frame_interval", 5)
    product_collection = source_config.get("product_collection", "products")
    cutout_collection = source_config.get("cutout_collection", "cutouts")

    all_images = []

    # Fetch cutouts if requested
    if source_type in ["cutouts", "both"]:
        cutouts = fetch_cutout_images(
            client=client,
            filters={
                "has_embedding": filters.get("has_embedding"),
                "cutout_filter_has_upc": filters.get("cutout_filter_has_upc"),
            },
        )

        for c in cutouts:
            all_images.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "collection": cutout_collection,
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "domain": "real",
                    "predicted_upc": c.get("predicted_upc"),
                },
            })

    # Fetch products if requested
    if source_type in ["products", "both"]:
        product_source = filters.get("product_source", "all")
        product_images = fetch_product_images(
            client=client,
            source=product_source,
            product_ids=filters.get("product_ids"),
            product_dataset_id=filters.get("product_dataset_id"),
            product_filter=filters.get("product_filter"),
            frame_selection=frame_selection,
            max_frames=max_frames,
            frame_interval=frame_interval,
        )

        for img in product_images:
            img["collection"] = product_collection
            all_images.append(img)

    print(f"\n[Supabase Fetcher] Total images: {len(all_images)}")
    print(f"  Cutouts: {sum(1 for i in all_images if i['type'] == 'cutout')}")
    print(f"  Products: {sum(1 for i in all_images if i['type'] == 'product')}")

    return all_images
