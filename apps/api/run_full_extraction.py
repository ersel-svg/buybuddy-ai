#!/usr/bin/env python3
"""
Full extraction test:
1. Delete all existing collections
2. Get USA_beverage dataset products
3. Get ALL cutouts
4. Extract embeddings for both
5. Verify everything works
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()


async def main():
    from services.supabase import supabase_service as db
    from services.qdrant import qdrant_service
    from services.runpod import runpod_service, EndpointType

    print("=" * 70)
    print("FULL EMBEDDING EXTRACTION - USA_BEVERAGE + ALL CUTOUTS")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # =========================================
    # STEP 1: Delete all existing collections
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 1: DELETING ALL EXISTING COLLECTIONS")
    print("=" * 70)

    collections = await qdrant_service.list_collections()
    print(f"Found {len(collections)} collections to delete: {collections}")

    for coll in collections:
        print(f"   Deleting {coll}...")
        await qdrant_service.delete_collection(coll)

    # Verify
    remaining = await qdrant_service.list_collections()
    print(f"   Remaining collections: {remaining}")
    assert len(remaining) == 0, "Failed to delete all collections!"
    print("   ✅ All collections deleted!")

    # =========================================
    # STEP 2: Get model info
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 2: GETTING MODEL INFO")
    print("=" * 70)

    models_result = db.client.table("embedding_models").select("*").limit(1).execute()
    if not models_result.data:
        print("❌ No embedding model found!")
        return

    model = models_result.data[0]
    model_id = model["id"]
    model_type = model.get("model_type", "dinov2-base")
    embedding_dim = model.get("embedding_dim", 768)
    model_name = model.get("name", model_type).lower().replace(" ", "_").replace("-", "_")

    print(f"   Model: {model.get('name')}")
    print(f"   Type: {model_type}")
    print(f"   Dimension: {embedding_dim}")

    # Collection names
    product_collection = f"products_{model_name}"
    cutout_collection = f"cutouts_{model_name}"

    print(f"\n   Target collections:")
    print(f"   - Products: {product_collection}")
    print(f"   - Cutouts: {cutout_collection}")

    # =========================================
    # STEP 3: Get USA_beverage dataset products
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 3: GETTING USA_BEVERAGE DATASET PRODUCTS")
    print("=" * 70)

    # Find the dataset
    datasets_result = db.client.table("datasets").select("*").ilike("name", "%usa%beverage%").execute()
    if not datasets_result.data:
        # Try alternate search
        datasets_result = db.client.table("datasets").select("*").execute()
        print(f"   All datasets: {[d.get('name') for d in datasets_result.data]}")

        # Find one with beverage or USA
        dataset = None
        for d in datasets_result.data:
            name = d.get('name', '').lower()
            if 'beverage' in name or 'usa' in name:
                dataset = d
                break

        if not dataset and datasets_result.data:
            dataset = datasets_result.data[0]  # Use first dataset
            print(f"   Using first available dataset: {dataset.get('name')}")
    else:
        dataset = datasets_result.data[0]

    if not dataset:
        print("❌ No dataset found!")
        return

    dataset_id = dataset["id"]
    dataset_name = dataset.get("name", "Unknown")
    print(f"   Found dataset: {dataset_name} (ID: {dataset_id})")

    # Get products from dataset
    products_result = db.client.table("dataset_products").select(
        "product_id, products(id, barcode, brand_name, product_name, frames_path, frame_count)"
    ).eq("dataset_id", dataset_id).execute()

    products = []
    for dp in products_result.data or []:
        p = dp.get("products")
        if p and p.get("frames_path") and p.get("frame_count", 0) > 0:
            products.append(p)

    print(f"   Products in dataset: {len(products_result.data or [])}")
    print(f"   Products with frames: {len(products)}")

    if not products:
        print("⚠️  No products with frames in this dataset!")
        # Fallback: get some products directly
        print("   Falling back to first 100 products with frames...")
        fallback_result = db.client.table("products").select(
            "id, barcode, brand_name, product_name, frames_path, frame_count"
        ).gt("frame_count", 0).limit(100).execute()
        products = fallback_result.data or []
        print(f"   Got {len(products)} products")

    # =========================================
    # STEP 4: Get ALL cutouts
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 4: GETTING ALL CUTOUTS")
    print("=" * 70)

    # Get total count first
    count_result = db.client.table("cutout_images").select("id", count="exact").execute()
    total_cutouts = count_result.count
    print(f"   Total cutouts in database: {total_cutouts}")

    # Get all cutouts (paginated)
    cutouts = []
    page_size = 1000
    offset = 0

    while True:
        cutouts_result = db.client.table("cutout_images").select(
            "id, image_url, predicted_upc"
        ).range(offset, offset + page_size - 1).execute()

        batch = cutouts_result.data or []
        if not batch:
            break

        cutouts.extend(batch)
        offset += page_size
        print(f"   Loaded {len(cutouts)}/{total_cutouts} cutouts...")

        if len(batch) < page_size:
            break

    print(f"   Total cutouts loaded: {len(cutouts)}")

    # =========================================
    # STEP 5: Build images list
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 5: BUILDING IMAGES LIST")
    print("=" * 70)

    images_to_process = []

    # Add products (first frame only)
    for p in products:
        frames_path = p.get("frames_path", "")
        if not frames_path:
            continue

        frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
        images_to_process.append({
            "id": f"{p['id']}_0",
            "url": frame_url,
            "type": "product",
            "collection": product_collection,
            "metadata": {
                "source": "product",
                "product_id": p["id"],
                "frame_index": 0,
                "is_primary": True,
                "barcode": p.get("barcode"),
                "brand_name": p.get("brand_name"),
                "product_name": p.get("product_name"),
            },
        })

    product_count = len([i for i in images_to_process if i["type"] == "product"])
    print(f"   Products to process: {product_count}")

    # Add cutouts
    for c in cutouts:
        images_to_process.append({
            "id": c["id"],
            "url": c["image_url"],
            "type": "cutout",
            "collection": cutout_collection,
            "metadata": {
                "source": "cutout",
                "cutout_id": c["id"],
                "predicted_upc": c.get("predicted_upc"),
            },
        })

    cutout_count = len([i for i in images_to_process if i["type"] == "cutout"])
    print(f"   Cutouts to process: {cutout_count}")
    print(f"   TOTAL IMAGES: {len(images_to_process)}")

    # =========================================
    # STEP 6: Create collections
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 6: CREATING COLLECTIONS")
    print("=" * 70)

    for coll in [product_collection, cutout_collection]:
        print(f"   Creating {coll}...")
        await qdrant_service.create_collection(
            collection_name=coll,
            vector_size=embedding_dim,
            distance="Cosine",
        )

    # =========================================
    # STEP 7: Process in batches (RUNPOD QUEUE)
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 7: EXTRACTING EMBEDDINGS (RUNPOD QUEUE)")
    print("=" * 70)

    batch_size = 50
    total = len(images_to_process)
    processed_count = 0
    failed_count = 0

    start_time = datetime.now()

    # Split into batches
    batches = []
    for i in range(0, total, batch_size):
        batch = images_to_process[i:i + batch_size]
        batches.append((i // batch_size, batch))

    total_batches = len(batches)
    print(f"   Total batches: {total_batches} (batch_size={batch_size})")
    print(f"   RunPod will manage parallelism based on available workers")

    # PHASE 1: Submit ALL jobs to RunPod queue
    print(f"\n   📤 PHASE 1: Submitting all {total_batches} jobs to RunPod queue...")

    pending_jobs = []  # (job_id, batch_idx, batch)

    for batch_idx, batch in batches:
        worker_input = {
            "images": [
                {"id": img["id"], "url": img["url"], "type": img["type"]}
                for img in batch
            ],
            "model_type": model_type,
            "batch_size": min(16, len(batch)),
        }

        try:
            result = await runpod_service.submit_job(
                endpoint_type=EndpointType.EMBEDDING,
                input_data=worker_input,
            )
            job_id = result.get("id")
            pending_jobs.append((job_id, batch_idx, batch))

            if (batch_idx + 1) % 50 == 0:
                print(f"      Submitted {batch_idx + 1}/{total_batches} jobs...")

        except Exception as e:
            print(f"      ❌ Failed to submit batch {batch_idx + 1}: {e}")
            failed_count += len(batch)

    print(f"   ✅ Submitted {len(pending_jobs)} jobs to RunPod queue")

    # PHASE 2: Poll for results
    print(f"\n   📥 PHASE 2: Collecting results...")

    completed_jobs = set()
    poll_interval = 2  # seconds

    while len(completed_jobs) < len(pending_jobs):
        await asyncio.sleep(poll_interval)

        for job_id, batch_idx, batch in pending_jobs:
            if job_id in completed_jobs:
                continue

            try:
                status = await runpod_service.get_job_status(
                    endpoint_type=EndpointType.EMBEDDING,
                    job_id=job_id,
                )

                job_status = status.get("status")

                if job_status == "COMPLETED":
                    completed_jobs.add(job_id)

                    output = status.get("output", {})
                    if output.get("status") != "success":
                        print(f"      Batch {batch_idx + 1}: ❌ Worker error")
                        failed_count += len(batch)
                        continue

                    embeddings = output.get("embeddings", [])
                    embedding_map = {e["id"]: e for e in embeddings}

                    # Group by collection
                    collection_points = {}
                    batch_processed = 0
                    batch_failed = 0

                    for img in batch:
                        emb_data = embedding_map.get(img["id"])
                        if emb_data and emb_data.get("vector"):
                            coll = img["collection"]
                            if coll not in collection_points:
                                collection_points[coll] = []
                            collection_points[coll].append({
                                "id": img["id"],
                                "vector": emb_data["vector"],
                                "payload": img["metadata"],
                            })
                            batch_processed += 1
                        else:
                            batch_failed += 1

                    # Upsert to collections
                    for coll_name, points in collection_points.items():
                        if points:
                            await qdrant_service.upsert_points(coll_name, points)

                    # Update cutout records
                    for img in batch:
                        if img["type"] == "cutout" and img["id"] in embedding_map:
                            try:
                                db.client.table("cutout_images").update({
                                    "has_embedding": True,
                                    "embedding_model_id": model_id,
                                }).eq("id", img["id"]).execute()
                            except Exception:
                                pass

                    processed_count += batch_processed
                    failed_count += batch_failed

                elif job_status == "FAILED":
                    completed_jobs.add(job_id)
                    print(f"      Batch {batch_idx + 1}: ❌ Job failed")
                    failed_count += len(batch)

            except Exception as e:
                # Transient error, will retry
                pass

        # Progress update
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed_count / elapsed if elapsed > 0 else 0
        pct = len(completed_jobs) / len(pending_jobs) * 100 if pending_jobs else 0
        print(f"   📊 {len(completed_jobs)}/{len(pending_jobs)} jobs done ({pct:.0f}%), {processed_count} embeddings, {rate:.1f}/sec")

    # =========================================
    # STEP 8: Verify results
    # =========================================
    print("\n" + "=" * 70)
    print("STEP 8: VERIFYING RESULTS")
    print("=" * 70)

    total_time = (datetime.now() - start_time).total_seconds()

    for coll in [product_collection, cutout_collection]:
        try:
            info = qdrant_service.client.get_collection(collection_name=coll)
            print(f"\n   {coll}:")
            print(f"      Points: {info.points_count}")
        except Exception as e:
            print(f"\n   {coll}: Error - {e}")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Products processed: {product_count}")
    print(f"   Cutouts processed: {cutout_count}")
    print(f"   Total processed: {processed_count}")
    print(f"   Total failed: {failed_count}")
    print(f"   Success rate: {processed_count / total * 100:.1f}%")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Rate: {processed_count / total_time:.1f} images/second")

    print("\n" + "=" * 70)
    if failed_count == 0:
        print("🎉 EXTRACTION COMPLETE - ALL SUCCESSFUL!")
    else:
        print(f"⚠️  EXTRACTION COMPLETE - {failed_count} FAILURES")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
