# 🎯 SOTA Integration Fix Plan

**Created:** 2026-01-26
**Status:** Ready for Implementation
**Priority:** Critical

---

## 📊 Executive Summary

This document outlines a comprehensive, state-of-the-art (SOTA) plan to fix all integration issues across the BuyBuddy AI training and extraction systems. The plan ensures 100% frontend-API-worker integration with backward compatibility.

### Systems Analyzed
1. ✅ **Embedding Trainer** - 100% integrated, no issues
2. ✅ **OD Trainer** - 90% integrated, production-ready
3. ⚠️ **CLS Trainer** - 98% integrated, 1 minor fix needed
4. ❌ **Embedding Extraction (Matching Mode)** - Multiple issues require fixes

### Issues Identified
- **Critical (P1):** Embedding Extraction - Multiple collection support missing
- **Important (P2):** Embedding Extraction - product_ids and dataset_id filters missing
- **Important (P3):** CLS Trainer - Config key mismatch

---

## 🔍 Pre-Analysis: Current State

### ✅ Working Systems

#### Embedding Trainer
- **Status:** 100% integrated
- **Frontend:** Sends full SOTA config (combined loss, P-K sampling, curriculum, etc.)
- **API:** Accepts and forwards all config
- **Worker:** Reads `sota_config` dict and applies all features
- **Verdict:** No changes needed

#### OD Trainer
- **Status:** 90% integrated, production-ready
- **Frontend:** 7-step wizard with 50+ config parameters
- **API:** Uses `extra: "allow"` schema, accepts all configs
- **Worker:** Processes 90% of configs correctly
- **Minor Gaps:** Test split not used, momentum parameter missing (non-critical)
- **Verdict:** Production-ready as-is

#### Embedding Extraction (Production/Training/Evaluation)
- **Status:** Correct collection strategy
- **Production:** Single collection (correct)
- **Training:** Single collection (correct)
- **Evaluation:** Single collection (correct)
- **Verdict:** No changes needed

### ❌ Systems Requiring Fixes

#### Embedding Extraction (Matching Mode)
- **Issue 1:** Multiple collections not supported in worker
- **Issue 2:** product_ids filter not passed to worker
- **Issue 3:** product_dataset_id filter not passed to worker
- **Impact:** Matching mode cannot separate products/cutouts into different collections

#### CLS Trainer
- **Issue:** Worker expects `preload_config` but API sends `data_loading`
- **Impact:** Frontend data loading configuration ignored

---

## 📋 PHASE 1: Embedding Extraction - Matching Mode Fix

### Background: Why Multiple Collections?

**Use Case:** Cross-domain product matching
- **Product Collection:** Synthetic 360° product renders
- **Cutout Collection:** Real shelf images (cutouts)
- **Query Pattern:** Search cutout → Find matching product

**Why Separate Collections?**
```python
# ❌ Single collection problem:
query_vector = embed(cutout_image)
results = qdrant.search(
    collection_name="all_images",  # Returns other cutouts too!
    query_vector=query_vector
)

# ✅ Separate collections solution:
query_vector = embed(cutout_image)
results = qdrant.search(
    collection_name="products_only",  # Returns only products!
    query_vector=query_vector
)
```

---

### Problem 1.1: Multiple Collections Support Missing (WORKER)

**Affected File:** `/Users/erselgokmen/Ai-pipeline/buybuddy-ai/workers/embedding-extraction/src/handler.py`

**Current Implementation (Lines 400-470):**
```python
# All images go to the same collection
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    # ... embedding extraction ...

    qdrant.upsert(
        collection_name=collection_name,  # ❌ Single collection!
        points=points,
        wait=True,
    )
```

**Root Cause:**
- API sends `product_collection` and `cutout_collection` in `source_config`
- `supabase_fetcher.py` adds `img["collection"]` field to each image
- Handler ignores this field and uses only primary `collection_name`

**SOTA Solution:**

Add new functions to `handler.py`:

```python
def process_images_by_collection(
    images: List[Dict],
    qdrant: QdrantClient,
    model,
    device,
    batch_size: int,
    job_id: str,
    supabase: Optional[Client] = None,
):
    """
    Process images grouped by their target collection.

    This enables Matching mode to separate products and cutouts
    into different collections for cross-domain search.

    Args:
        images: List of image dicts with 'collection' field
        qdrant: Qdrant client
        model: Embedding model
        device: torch device
        batch_size: Batch size for processing
        job_id: Job ID for progress tracking
        supabase: Supabase client for status updates

    Returns:
        tuple[int, int]: (processed_count, failed_count)
    """
    from collections import defaultdict

    # Group images by collection
    images_by_collection = defaultdict(list)
    for img in images:
        collection_name = img.get("collection", "default")
        images_by_collection[collection_name].append(img)

    print(f"\n[Collection Grouping] Found {len(images_by_collection)} collections:")
    for coll_name, coll_images in images_by_collection.items():
        print(f"  - {coll_name}: {len(coll_images)} images")

    total_processed = 0
    total_failed = 0

    # Process each collection separately
    for collection_name, collection_images in images_by_collection.items():
        print(f"\n{'='*60}")
        print(f"Processing Collection: {collection_name}")
        print(f"{'='*60}")
        print(f"Total images: {len(collection_images)}")

        # Ensure collection exists with correct dimension
        embedding_dim = model.embedding_dim if hasattr(model, 'embedding_dim') else 768
        ensure_collection_exists(qdrant, collection_name, embedding_dim)

        processed, failed = process_collection_batch(
            images=collection_images,
            collection_name=collection_name,
            qdrant=qdrant,
            model=model,
            device=device,
            batch_size=batch_size,
            job_id=job_id,
            supabase=supabase,
        )

        total_processed += processed
        total_failed += failed

        print(f"\n[{collection_name}] Complete:")
        print(f"  - Processed: {processed}")
        print(f"  - Failed: {failed}")

    return total_processed, total_failed


def process_collection_batch(
    images: List[Dict],
    collection_name: str,
    qdrant: QdrantClient,
    model,
    device,
    batch_size: int,
    job_id: str,
    supabase: Optional[Client] = None,
) -> tuple[int, int]:
    """
    Process a single collection's images in batches.

    Args:
        images: Images for this collection
        collection_name: Target collection name
        qdrant: Qdrant client
        model: Embedding model
        device: torch device
        batch_size: Batch size
        job_id: Job ID for progress
        supabase: Supabase client

    Returns:
        tuple[int, int]: (processed, failed)
    """
    import traceback
    from qdrant_client.models import PointStruct

    processed_count = 0
    failed_count = 0

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        try:
            # Extract embeddings
            urls = [img["url"] for img in batch]
            embeddings = extract_embeddings_batch(urls, model, device)

            # Build Qdrant points
            points = []
            url_to_embedding = dict(zip(urls, embeddings))

            for img in batch:
                if img["url"] in url_to_embedding:
                    embedding = url_to_embedding[img["url"]]
                    if embedding is not None:
                        point_id = parse_point_id(img["id"])
                        payload = {
                            "id": img["id"],
                            "type": img.get("type", "unknown"),
                            "url": img["url"],
                            **(img.get("metadata", {})),
                        }

                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding.tolist(),
                            payload=payload,
                        ))
                    else:
                        failed_count += 1
                else:
                    failed_count += 1

            # Upsert to THIS collection
            if points:
                qdrant.upsert(
                    collection_name=collection_name,  # ✅ Correct collection!
                    points=points,
                    wait=True,
                )
                processed_count += len(points)
                print(f"  [{collection_name}] Batch {i//batch_size + 1}: Upserted {len(points)} points")

            # Update Supabase for cutouts
            if supabase:
                for img in batch:
                    if img.get("type") == "cutout" and img["url"] in url_to_embedding:
                        try:
                            supabase.table("cutout_images").update({
                                "has_embedding": True,
                                "qdrant_point_id": img["id"],
                            }).eq("id", img["id"]).execute()
                        except Exception as e:
                            print(f"  [WARNING] Failed to update cutout {img['id']}: {e}")

        except Exception as e:
            print(f"  Batch error: {e}")
            failed_count += len(batch)
            traceback.print_exc()

        # Update progress
        if supabase:
            update_embedding_job(
                supabase,
                job_id,
                processed_images=processed_count + failed_count,
            )

    return processed_count, failed_count


def ensure_collection_exists(
    qdrant: QdrantClient,
    collection_name: str,
    embedding_dim: int,
):
    """
    Ensure Qdrant collection exists with correct configuration.

    Args:
        qdrant: Qdrant client
        collection_name: Collection to check/create
        embedding_dim: Vector dimension
    """
    from qdrant_client.models import Distance, VectorParams

    try:
        # Check if collection exists
        qdrant.get_collection(collection_name)
        print(f"  [Qdrant] Collection '{collection_name}' exists")
    except Exception:
        # Create collection if it doesn't exist
        print(f"  [Qdrant] Creating collection '{collection_name}' (dim={embedding_dim})")
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
```

**Integration Point (Replace existing loop around line 400):**
```python
# BEFORE:
# for i in range(0, len(images), batch_size):
#     batch = images[i:i + batch_size]
#     ... single collection processing ...

# AFTER:
processed_count, failed_count = process_images_by_collection(
    images=images,
    qdrant=qdrant,
    model=model,
    device=device,
    batch_size=batch_size,
    job_id=job_id,
    supabase=supabase,
)
```

**Benefits:**
- ✅ Supports multiple collections per job
- ✅ Backward compatible (single collection still works)
- ✅ Clean separation of concerns
- ✅ Better logging and debugging

---

### Problem 1.2: product_ids and dataset_id Missing (API)

**Affected File:** `/Users/erselgokmen/Ai-pipeline/buybuddy-ai/apps/api/src/api/v1/embeddings.py`

**Current Implementation (Lines 2093-2105):**
```python
"source_config": {
    "type": "both" if request.include_cutouts else "products",
    "filters": {
        "has_embedding": False if request.collection_mode == "create" else None,
        "product_source": request.product_source,
        "cutout_filter_has_upc": request.cutout_filter_has_upc,
        # ❌ product_ids missing!
        # ❌ product_dataset_id missing!
    },
    "frame_selection": request.frame_selection,
    "frame_interval": request.frame_interval,
    "max_frames": request.max_frames,
    "product_collection": product_collection,
    "cutout_collection": cutout_collection if request.include_cutouts else None,
}
```

**Root Cause:**
- Frontend sends `product_ids` (for "selected" source)
- Frontend sends `product_dataset_id` (for "dataset" source)
- API accepts these in request schema but doesn't pass to worker

**SOTA Solution:**

**Update source_config (Line 2093-2105):**
```python
"source_config": {
    "type": "both" if request.include_cutouts else "products",
    "filters": {
        "has_embedding": False if request.collection_mode == "create" else None,
        "product_source": request.product_source,
        "cutout_filter_has_upc": request.cutout_filter_has_upc,
        # ✅ ADD: product_ids for "selected" source
        "product_ids": request.product_ids if request.product_source == "selected" else None,
        # ✅ ADD: dataset_id for "dataset" source
        "product_dataset_id": request.product_dataset_id if request.product_source == "dataset" else None,
    },
    "frame_selection": request.frame_selection,
    "frame_interval": request.frame_interval,
    "max_frames": request.max_frames,
    "product_collection": product_collection,
    "cutout_collection": cutout_collection if request.include_cutouts else None,
}
```

**Affected Lines:**
- Line 2098: Add product_ids
- Line 2099: Add product_dataset_id

---

### Problem 1.3: Worker Fetcher - product_ids & dataset_id Support

**Affected File:** `/Users/erselgokmen/Ai-pipeline/buybuddy-ai/workers/embedding-extraction/src/data/supabase_fetcher.py`

**Update fetch_product_images() function (Line ~142):**

```python
def fetch_product_images(
    client,
    source: str = "all",
    frame_selection: str = "first",
    max_frames: int = 10,
    frame_interval: int = 5,
    # ✅ ADD: new parameters
    product_ids: Optional[List[str]] = None,
    product_dataset_id: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch product images based on source configuration.

    Args:
        client: Supabase client
        source: Source type ("all", "matched", "new", "selected", "dataset")
        frame_selection: Frame selection strategy
        max_frames: Maximum frames per product
        frame_interval: Interval between frames
        product_ids: Specific product IDs to fetch (for "selected" source)
        product_dataset_id: Dataset ID to fetch products from (for "dataset" source)

    Returns:
        List of image dicts with URLs and metadata
    """
    print(f"\n[Fetching Products] Source: {source}")

    # Build filters based on source
    filters = {}

    # ✅ SOTA: Handle specific product IDs
    if product_ids:
        print(f"  Filter: Specific {len(product_ids)} products")
        filters["id"] = product_ids  # Will use .in_() in fetch_with_pagination

    # ✅ SOTA: Handle dataset source
    elif product_dataset_id:
        print(f"  Filter: Products from dataset {product_dataset_id}")
        # Fetch product IDs from dataset_products table
        dataset_products = fetch_with_pagination(
            client=client,
            table="dataset_products",
            select="product_id",
            filters={"dataset_id": product_dataset_id},
        )
        dataset_product_ids = [p["product_id"] for p in dataset_products]
        print(f"  Found {len(dataset_product_ids)} products in dataset")

        if not dataset_product_ids:
            print("  [WARNING] No products found in dataset")
            return []

        filters["id"] = dataset_product_ids

    # Existing source filters
    elif source == "matched":
        filters["frame_count_gt"] = 0
        filters["matched_cutout_count_gt"] = 0
    elif source == "new":
        filters["frame_count_gt"] = 0
        filters["matched_cutout_count_is_null"] = True
    elif source == "all":
        filters["frame_count_gt"] = 0

    # Fetch products
    products = fetch_with_pagination(
        client=client,
        table="products",
        select="id, barcode, brand_name, frames_path, frame_count",
        filters=filters,
    )

    print(f"  Retrieved {len(products)} products")

    # Generate frame URLs based on frame_selection
    images = []
    for p in products:
        product_id = p["id"]
        frames_path = p.get("frames_path")
        frame_count = p.get("frame_count", 0)

        if not frames_path or frame_count == 0:
            continue

        # Select frames based on strategy
        frame_indices = select_frames(
            frame_count=frame_count,
            selection=frame_selection,
            max_frames=max_frames,
            interval=frame_interval,
        )

        # Build frame URLs
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
```

**Update build_extraction_data() to pass new params (Line ~290):**
```python
def build_extraction_data(
    supabase_url: str,
    supabase_key: str,
    source_config: Dict[str, Any],
) -> List[Dict]:
    """
    Main function to fetch all images for embedding extraction.

    SOTA pattern: worker fetches images from DB instead of receiving them in payload.
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

    # ✅ Extract new filter parameters
    product_ids = filters.get("product_ids")
    product_dataset_id = filters.get("product_dataset_id")

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
            frame_selection=frame_selection,
            max_frames=max_frames,
            frame_interval=frame_interval,
            product_ids=product_ids,  # ✅ Pass through
            product_dataset_id=product_dataset_id,  # ✅ Pass through
        )

        for img in product_images:
            img["collection"] = product_collection
            all_images.append(img)

    print(f"\n[Supabase Fetcher] Total images: {len(all_images)}")
    print(f"  Cutouts: {sum(1 for i in all_images if i['type'] == 'cutout')}")
    print(f"  Products: {sum(1 for i in all_images if i['type'] == 'product')}")

    return all_images
```

---

## 📋 PHASE 2: CLS Trainer - Config Key Fix

### Problem 2.1: Data Loading Config Key Mismatch (WORKER)

**Affected File:** `/Users/erselgokmen/Ai-pipeline/buybuddy-ai/workers/cls-training/handler.py`

**Current Implementation (Line 738):**
```python
preload_config = full_config.get("preload_config", {})
```

**Root Cause:**
- Frontend sends `data_loading.preload` config
- API forwards as `data_loading` key
- Worker expects `preload_config` key (old format)
- Result: Configuration ignored

**SOTA Solution (Backward Compatible):**

```python
# Line 736-745: Replace preload_config extraction

# SOTA PATTERN: Support both old and new key formats
# Priority: data_loading.preload > preload_config (backward compat)
data_loading = full_config.get("data_loading", {})
if data_loading and isinstance(data_loading, dict):
    # New format: data_loading.preload
    preload_config = data_loading.get("preload", {})
    print(f"\n[Data Loading Config] Using 'data_loading.preload' format")
else:
    # Fallback to old key for backward compatibility
    preload_config = full_config.get("preload_config", {})
    print(f"\n[Data Loading Config] Using legacy 'preload_config' format")

# Log configuration for debugging
print(f"[Data Loading Config]")
print(f"  Preload enabled: {preload_config.get('enabled', True)}")
print(f"  Batched: {preload_config.get('batched', True)}")
print(f"  Batch size: {preload_config.get('batch_size', 500)}")
print(f"  Max workers: {preload_config.get('max_workers', 8)}")
print(f"  HTTP timeout: {preload_config.get('http_timeout', 30)}s")
```

**Benefits:**
- ✅ Supports new format (data_loading.preload)
- ✅ Backward compatible with old format (preload_config)
- ✅ Better logging for debugging
- ✅ No breaking changes for existing jobs

---

## 🧪 Testing Plan

### Test Suite 1: Embedding Extraction - Matching Mode

#### Test 1.1: Two Separate Collections (Core Functionality)
**Objective:** Verify products and cutouts go to separate collections

**Steps:**
1. Open Frontend: Embeddings page → Matching tab
2. Configure:
   - Model: dinov2-base
   - Product source: "matched"
   - Include cutouts: ✅ Yes
   - Collection names:
     - Products: `test_products_v1`
     - Cutouts: `test_cutouts_v1`
3. Start extraction
4. Monitor worker logs for:
   ```
   [Collection Grouping] Found 2 collections:
     - test_products_v1: 1500 images
     - test_cutouts_v1: 500 images

   ============================================================
   Processing Collection: test_products_v1
   ============================================================
   ...

   ============================================================
   Processing Collection: test_cutouts_v1
   ============================================================
   ...
   ```

**Verification:**
- ✅ Two collections created in Qdrant
- ✅ Products collection contains product embeddings
- ✅ Cutouts collection contains cutout embeddings
- ✅ No mixing between collections

**Cross-Domain Search Test:**
```python
# Query: Select a random cutout
cutout_vector = qdrant.retrieve(
    collection_name="test_cutouts_v1",
    ids=[cutout_id]
)[0].vector

# Search in products collection
results = qdrant.search(
    collection_name="test_products_v1",
    query_vector=cutout_vector,
    limit=5
)

# Verify: Top result should be the matching product
assert results[0].payload["product_id"] == expected_product_id
```

---

#### Test 1.2: Selected Products Filter
**Objective:** Verify product_ids filter works correctly

**Steps:**
1. Open Frontend: Matching tab
2. Configure:
   - Product source: "selected"
   - Select 5 specific products manually
   - Include cutouts: ❌ No
3. Start extraction

**Verification:**
- ✅ Worker logs show: `Filter: Specific 5 products`
- ✅ Exactly 5 products processed (check frame count)
- ✅ No other products included

---

#### Test 1.3: Dataset Source Filter
**Objective:** Verify product_dataset_id filter works correctly

**Steps:**
1. Create test dataset with 10 products
2. Open Frontend: Matching tab
3. Configure:
   - Product source: "dataset"
   - Select: "Test Dataset A"
   - Include cutouts: ❌ No
4. Start extraction

**Verification:**
- ✅ Worker logs show: `Filter: Products from dataset {dataset_id}`
- ✅ Worker logs show: `Found 10 products in dataset`
- ✅ Exactly 10 products processed
- ✅ Products match dataset contents

---

### Test Suite 2: CLS Trainer - Data Loading Config

#### Test 2.1: Data Loading Configuration Applied
**Objective:** Verify frontend config reaches worker

**Steps:**
1. Open Frontend: Classification Training page
2. Open "Data Loading (Advanced)" section
3. Configure:
   - Batched Preload: ✅ Yes
   - Batch Size: 1000
   - Max Workers: 16
   - HTTP Timeout: 60s
4. Start training

**Verification:**
- ✅ Worker logs show:
   ```
   [Data Loading Config] Using 'data_loading.preload' format
   [Data Loading Config]
     Preload enabled: True
     Batched: True
     Batch size: 1000
     Max workers: 16
     HTTP timeout: 60s
   ```
- ✅ Training uses batched preloading
- ✅ Memory usage is efficient (batched gc.collect())

---

#### Test 2.2: Backward Compatibility
**Objective:** Verify old training jobs still work

**Steps:**
1. Find existing training run from before this fix
2. Resume/retry the training

**Verification:**
- ✅ Worker logs show: `Using legacy 'preload_config' format`
- ✅ Training completes successfully
- ✅ No errors or warnings

---

### Test Suite 3: Regression Testing

#### Test 3.1: Production Mode (No Change Expected)
**Objective:** Verify Production extraction still works with single collection

**Steps:**
1. Open Frontend: Embeddings → Production tab
2. Configure:
   - Model: dinov2-base
   - Product source: "all"
   - Collection name: `prod_v1`
3. Start extraction

**Verification:**
- ✅ Single collection created
- ✅ Both products and cutouts in same collection
- ✅ No breaking changes

---

#### Test 3.2: Training Mode (No Change Expected)
**Objective:** Verify Training extraction still works

**Steps:**
1. Open Frontend: Embeddings → Training tab
2. Configure and start extraction

**Verification:**
- ✅ Single collection created
- ✅ Matched products processed correctly
- ✅ No breaking changes

---

#### Test 3.3: Evaluation Mode (No Change Expected)
**Objective:** Verify Evaluation extraction still works

**Steps:**
1. Open Frontend: Embeddings → Evaluation tab
2. Configure and start extraction

**Verification:**
- ✅ Single collection created
- ✅ Dataset products processed correctly
- ✅ No breaking changes

---

## 📈 Implementation Checklist

### Phase 1: Embedding Extraction - Matching Mode

#### Step 1.1: Worker - Multiple Collections Support
- [ ] File: `workers/embedding-extraction/src/handler.py`
- [ ] Add function: `process_images_by_collection()` (~80 lines)
- [ ] Add function: `process_collection_batch()` (~80 lines)
- [ ] Add function: `ensure_collection_exists()` (~20 lines)
- [ ] Update main handler: Replace batch loop around line 400
- [ ] Add imports: `from collections import defaultdict`
- [ ] Test locally with mock data
- [ ] Commit: `feat(worker): add multiple collection support for matching mode`

#### Step 1.2: API - product_ids & dataset_id
- [ ] File: `apps/api/src/api/v1/embeddings.py`
- [ ] Update line ~2098: Add `product_ids` to filters
- [ ] Update line ~2099: Add `product_dataset_id` to filters
- [ ] Test with Postman/curl
- [ ] Commit: `feat(api): pass product_ids and dataset_id to worker`

#### Step 1.3: Worker Fetcher - product_ids & dataset_id Support
- [ ] File: `workers/embedding-extraction/src/data/supabase_fetcher.py`
- [ ] Update `fetch_product_images()`: Add product_ids parameter
- [ ] Update `fetch_product_images()`: Add product_dataset_id parameter
- [ ] Update `fetch_product_images()`: Implement filtering logic
- [ ] Update `build_extraction_data()`: Extract and pass new params
- [ ] Test with sample data
- [ ] Commit: `feat(worker): support product_ids and dataset_id filters`

---

### Phase 2: CLS Trainer - Config Key Fix

#### Step 2.1: Worker - Data Loading Config Key
- [ ] File: `workers/cls-training/handler.py`
- [ ] Update lines 736-745: New config extraction logic
- [ ] Add logging for debugging
- [ ] Test with frontend config
- [ ] Commit: `fix(worker): support data_loading.preload config format`

---

### Phase 3: Testing

#### Integration Testing
- [ ] Test 1.1: Two separate collections
- [ ] Test 1.2: Selected products filter
- [ ] Test 1.3: Dataset source filter
- [ ] Test 2.1: Data loading config
- [ ] Test 2.2: Backward compatibility
- [ ] Test 3.1: Production mode regression
- [ ] Test 3.2: Training mode regression
- [ ] Test 3.3: Evaluation mode regression

#### Performance Testing
- [ ] Benchmark: Single collection vs multiple collections
- [ ] Monitor: Memory usage during extraction
- [ ] Monitor: Qdrant upsert performance
- [ ] Verify: No significant performance degradation

---

## 🚀 Rollout Strategy

### Stage 1: Development & Testing (1 day)
1. Implement all fixes locally
2. Run unit tests
3. Run integration tests with test data
4. Code review

### Stage 2: Staging Deployment (1 day)
1. Deploy worker changes to staging
2. Deploy API changes to staging
3. Run full test suite on staging
4. Verify logs and metrics

### Stage 3: Production Deployment (1 day)
1. Deploy during low-traffic window
2. Monitor first few jobs closely
3. Check Qdrant collection creation
4. Verify search quality
5. Monitor error rates

### Stage 4: Post-Deployment Monitoring (1 week)
1. Monitor Matching extraction jobs
2. Monitor CLS training jobs
3. Check for any unexpected errors
4. Gather user feedback

---

## 🎯 Success Criteria

### Embedding Extraction - Matching
- ✅ Two separate collections created (products + cutouts)
- ✅ Cross-domain search works correctly (cutout → product)
- ✅ Selected products mode filters correctly
- ✅ Dataset source mode filters correctly
- ✅ No performance degradation (< 5% slower acceptable)
- ✅ Zero errors in production for 7 days

### CLS Trainer
- ✅ Frontend data_loading config reaches worker
- ✅ Preload settings applied correctly
- ✅ Training remains memory-efficient
- ✅ Backward compatibility maintained
- ✅ Zero errors in production for 7 days

### Backward Compatibility
- ✅ Old training jobs work without changes
- ✅ Production/Training/Evaluation modes unaffected
- ✅ Single collection modes work correctly
- ✅ No breaking changes to existing APIs

---

## 📝 Documentation Updates

Post-implementation documentation tasks:

### API Documentation
- [ ] Update OpenAPI spec: Add product_ids parameter
- [ ] Update OpenAPI spec: Add product_dataset_id parameter
- [ ] Add examples for "selected" source
- [ ] Add examples for "dataset" source

### Worker Documentation
- [ ] Update README: Multiple collection support
- [ ] Add architecture diagram: Cross-domain search pattern
- [ ] Document collection strategy per mode
- [ ] Add troubleshooting guide

### User Documentation
- [ ] Update user guide: Matching mode workflow
- [ ] Add tutorial: Cross-domain product matching
- [ ] Document best practices: Collection naming
- [ ] Add FAQ: When to use multiple collections

---

## 🔧 Rollback Plan

If critical issues arise:

### Rollback Worker Changes
```bash
# Revert to previous worker version
cd workers/embedding-extraction
git revert <commit-hash>
git push

cd workers/cls-training
git revert <commit-hash>
git push

# Redeploy to RunPod
./deploy.sh
```

### Rollback API Changes
```bash
# Revert API changes
cd apps/api
git revert <commit-hash>
git push

# Redeploy
fly deploy
```

### Fallback Strategy
- Old jobs will continue working (backward compatible)
- New Matching jobs will fall back to single collection (degraded but functional)
- Monitor for 24 hours before re-attempting fix

---

## 📊 Metrics to Monitor

### Pre-Deployment Metrics
- Embedding extraction success rate: Target > 95%
- Average extraction time: Baseline measurement
- Memory usage: Baseline measurement
- Error rate: Current rate

### Post-Deployment Metrics
- Embedding extraction success rate: Should remain > 95%
- Average extraction time: Should be within 10% of baseline
- Memory usage: Should be within 20% of baseline
- Error rate: Should not increase by > 5%
- Collection creation success: Should be 100%
- Cross-domain search quality: Manual testing and user feedback

---

## 🎓 Lessons Learned (Post-Implementation)

_To be filled after implementation_

### What Went Well
-

### What Could Be Improved
-

### Unexpected Issues
-

### Future Improvements
-

---

## 📞 Contact & Support

**Implementation Team:**
- Backend: [Your Team]
- Frontend: [Your Team]
- DevOps: [Your Team]

**Escalation Path:**
1. Check worker logs
2. Check API logs
3. Contact backend team
4. Escalate to architecture team

---

**Document Version:** 1.0
**Last Updated:** 2026-01-26
**Status:** Ready for Implementation
