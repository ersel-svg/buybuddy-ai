#!/usr/bin/env python3
"""
Test script for embedding flow WITHOUT the worker.
Tests all API-side components: Supabase, Qdrant, data flow.

Run from apps/api directory:
    python test_embedding_flow.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

print("\n" + "=" * 70)
print("EMBEDDING FLOW TEST (Without Worker)")
print("=" * 70)
print(f"Time: {datetime.now().isoformat()}")


def test_environment():
    """Test environment variables."""
    print("\n" + "=" * 70)
    print("1. ENVIRONMENT CHECK")
    print("=" * 70)

    required = {
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        "QDRANT_URL": os.environ.get("QDRANT_URL", ""),
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "RUNPOD_ENDPOINT_EMBEDDING": os.environ.get("RUNPOD_ENDPOINT_EMBEDDING", ""),
    }

    all_ok = True
    for key, value in required.items():
        status = "OK" if value else "MISSING"
        masked = value[:30] + "..." if len(value) > 30 else value
        print(f"  {status}: {key} = {masked or '(not set)'}")
        if not value:
            all_ok = False

    return all_ok


def test_supabase_connection():
    """Test Supabase connection and data fetching."""
    print("\n" + "=" * 70)
    print("2. SUPABASE CONNECTION")
    print("=" * 70)

    try:
        from services.supabase import supabase_service

        # Test products with frames
        print("\n  Testing products table...")
        products = supabase_service.client.table("products").select(
            "id, barcode, frames_path, frame_count"
        ).gt("frame_count", 0).limit(3).execute()

        print(f"  OK Found {len(products.data)} products with frames")
        for p in products.data[:3]:
            print(f"      - {p['barcode']}: {p['frame_count']} frames")
            print(f"        frames_path: {p['frames_path'][:60]}...")

        # Test cutouts
        print("\n  Testing cutout_images table...")
        cutouts = supabase_service.client.table("cutout_images").select(
            "id, image_url, predicted_upc"
        ).limit(3).execute()

        print(f"  OK Found {len(cutouts.data)} cutouts")
        for c in cutouts.data[:3]:
            print(f"      - {c.get('predicted_upc', 'N/A')}: {c['image_url'][:50]}...")

        # Test embedding models
        print("\n  Testing embedding_models table...")
        models = supabase_service.client.table("embedding_models").select("*").execute()
        print(f"  OK Found {len(models.data)} embedding models")
        for m in models.data:
            active = " (ACTIVE)" if m.get("is_active") else ""
            print(f"      - {m['name']}: {m['model_type']}, dim={m['embedding_dim']}{active}")

        return products.data, cutouts.data, models.data

    except Exception as e:
        print(f"  FAIL Supabase error: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []


def test_qdrant_connection():
    """Test Qdrant connection."""
    print("\n" + "=" * 70)
    print("3. QDRANT CONNECTION")
    print("=" * 70)

    try:
        from services.qdrant import qdrant_service

        if not qdrant_service.is_configured():
            print("  SKIP Qdrant not configured")
            return None

        # List collections
        collections = qdrant_service.client.get_collections().collections
        print(f"  OK Connected! Collections: {[c.name for c in collections]}")

        return qdrant_service

    except Exception as e:
        print(f"  FAIL Qdrant error: {e}")
        print("  Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return None


def test_qdrant_operations(qdrant_service):
    """Test Qdrant operations (create collection, upsert, search)."""
    print("\n" + "=" * 70)
    print("4. QDRANT OPERATIONS")
    print("=" * 70)

    if not qdrant_service:
        print("  SKIP Qdrant not available")
        return False

    test_collection = "test_embedding_flow"

    try:
        # Create collection
        print(f"\n  Creating test collection '{test_collection}'...")

        # Delete if exists
        try:
            qdrant_service.client.delete_collection(test_collection)
            print("    (deleted existing)")
        except:
            pass

        qdrant_service.client.create_collection(
            collection_name=test_collection,
            vectors_config={
                "size": 768,
                "distance": "Cosine",
            },
        )
        print(f"  OK Collection created")

        # Simulate worker response - fake embeddings
        print("\n  Simulating worker response...")
        fake_embeddings = [
            {
                "id": "test-product-1",
                "type": "product",
                "vector": [0.1] * 768,  # Fake 768-dim vector
            },
            {
                "id": "test-product-2",
                "type": "product",
                "vector": [0.2] * 768,
            },
            {
                "id": "test-cutout-1",
                "type": "cutout",
                "vector": [0.15] * 768,  # Similar to product-1
            },
        ]
        print(f"  OK Simulated {len(fake_embeddings)} embeddings")

        # Upsert to Qdrant
        print("\n  Upserting to Qdrant...")
        from qdrant_client.http import models as qmodels

        points = [
            qmodels.PointStruct(
                id=emb["id"],
                vector=emb["vector"],
                payload={"type": emb["type"], "source": emb["type"]},
            )
            for emb in fake_embeddings
        ]

        qdrant_service.client.upsert(
            collection_name=test_collection,
            points=points,
        )
        print(f"  OK Upserted {len(points)} points")

        # Verify
        info = qdrant_service.client.get_collection(test_collection)
        print(f"  OK Collection has {info.points_count} points")

        # Test similarity search
        print("\n  Testing similarity search...")
        query_vector = [0.15] * 768  # Should match cutout-1 best

        results = qdrant_service.client.search(
            collection_name=test_collection,
            query_vector=query_vector,
            limit=3,
        )

        print(f"  OK Found {len(results)} results:")
        for r in results:
            print(f"      - {r.id}: score={r.score:.4f}, type={r.payload.get('type')}")

        # Cleanup
        print("\n  Cleaning up test collection...")
        qdrant_service.client.delete_collection(test_collection)
        print("  OK Deleted test collection")

        return True

    except Exception as e:
        print(f"  FAIL Qdrant operations error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_url_building(products):
    """Test building frame URLs from products."""
    print("\n" + "=" * 70)
    print("5. IMAGE URL BUILDING")
    print("=" * 70)

    if not products:
        print("  SKIP No products to test")
        return []

    import httpx

    urls_to_test = []

    for p in products[:3]:
        frames_path = p.get("frames_path", "")
        if not frames_path:
            continue

        frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
        urls_to_test.append({
            "product_id": p["id"],
            "barcode": p["barcode"],
            "url": frame_url,
        })

    print(f"\n  Testing {len(urls_to_test)} URLs...")

    valid_urls = []
    for item in urls_to_test:
        try:
            response = httpx.head(item["url"], timeout=10, follow_redirects=True)
            status = response.status_code

            if status == 200:
                print(f"  OK {item['barcode']}: {item['url'][:50]}...")
                valid_urls.append(item)
            else:
                print(f"  FAIL {item['barcode']}: HTTP {status}")

        except Exception as e:
            print(f"  FAIL {item['barcode']}: {e}")

    print(f"\n  Valid URLs: {len(valid_urls)}/{len(urls_to_test)}")
    return valid_urls


def test_full_flow_simulation(products, cutouts, models, qdrant_service):
    """Simulate the full embedding sync flow."""
    print("\n" + "=" * 70)
    print("6. FULL FLOW SIMULATION")
    print("=" * 70)

    if not qdrant_service:
        print("  SKIP Qdrant not available")
        return False

    if not models:
        print("  SKIP No embedding models found")
        return False

    # Get active model (or first one)
    model = next((m for m in models if m.get("is_active")), models[0] if models else None)
    if not model:
        print("  SKIP No model available")
        return False

    print(f"\n  Using model: {model['name']} ({model['model_type']})")

    collection_name = model.get("qdrant_collection", "test_flow_collection")
    embedding_dim = model.get("embedding_dim", 768)

    print(f"  Collection: {collection_name}")
    print(f"  Embedding dim: {embedding_dim}")

    try:
        # Step 1: Collect images (like API does)
        print("\n  Step 1: Collecting images...")
        images_to_process = []

        # Products
        for p in products[:2]:
            frames_path = p.get("frames_path", "")
            if not frames_path:
                continue
            frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
            images_to_process.append({
                "id": p["id"],
                "url": frame_url,
                "type": "product",
                "metadata": {
                    "source": "product",
                    "product_id": p["id"],
                    "barcode": p.get("barcode"),
                },
            })

        # Cutouts
        for c in cutouts[:2]:
            images_to_process.append({
                "id": c["id"],
                "url": c["image_url"],
                "type": "cutout",
                "metadata": {
                    "source": "cutout",
                    "cutout_id": c["id"],
                    "predicted_upc": c.get("predicted_upc"),
                },
            })

        print(f"  OK Collected {len(images_to_process)} images")

        # Step 2: Prepare worker input (like API does)
        print("\n  Step 2: Preparing worker input...")
        worker_input = {
            "images": [
                {"id": img["id"], "url": img["url"], "type": img["type"]}
                for img in images_to_process
            ],
            "model_type": model.get("model_type", "dinov2-base"),
            "batch_size": 16,
        }
        print(f"  OK Worker input prepared:")
        print(f"      {json.dumps(worker_input, indent=6)[:500]}...")

        # Step 3: SIMULATE worker response
        print("\n  Step 3: Simulating worker response...")
        import random
        simulated_response = {
            "status": "success",
            "embeddings": [
                {
                    "id": img["id"],
                    "type": img["type"],
                    "vector": [random.uniform(-0.1, 0.1) for _ in range(embedding_dim)],
                }
                for img in worker_input["images"]
            ],
            "processed_count": len(worker_input["images"]),
            "failed_count": 0,
            "embedding_dim": embedding_dim,
        }
        print(f"  OK Simulated response: {simulated_response['processed_count']} embeddings")

        # Step 4: Create Qdrant collection (like API does)
        print("\n  Step 4: Creating Qdrant collection...")
        from qdrant_client.http import models as qmodels

        # Delete if exists
        try:
            qdrant_service.client.delete_collection(collection_name)
        except:
            pass

        qdrant_service.client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"  OK Collection '{collection_name}' created")

        # Step 5: Upsert embeddings (like API does)
        print("\n  Step 5: Upserting embeddings to Qdrant...")
        embedding_map = {e["id"]: e for e in simulated_response["embeddings"]}

        qdrant_points = []
        for img in images_to_process:
            emb_data = embedding_map.get(img["id"])
            if emb_data and emb_data.get("vector"):
                qdrant_points.append(
                    qmodels.PointStruct(
                        id=img["id"],
                        vector=emb_data["vector"],
                        payload=img["metadata"],
                    )
                )

        qdrant_service.client.upsert(
            collection_name=collection_name,
            points=qdrant_points,
        )
        print(f"  OK Upserted {len(qdrant_points)} points")

        # Step 6: Verify and test search
        print("\n  Step 6: Verifying and testing search...")
        info = qdrant_service.client.get_collection(collection_name)
        print(f"  OK Collection has {info.points_count} points")

        # Search using first embedding
        if simulated_response["embeddings"]:
            query = simulated_response["embeddings"][0]["vector"]
            results = qdrant_service.client.search(
                collection_name=collection_name,
                query_vector=query,
                limit=5,
            )
            print(f"\n  Search results:")
            for r in results:
                print(f"      - {r.id}: score={r.score:.4f}, source={r.payload.get('source')}")

        # Cleanup
        print("\n  Cleaning up...")
        qdrant_service.client.delete_collection(collection_name)
        print("  OK Test collection deleted")

        return True

    except Exception as e:
        print(f"  FAIL Flow simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runpod_health():
    """Test RunPod endpoint health (without running a job)."""
    print("\n" + "=" * 70)
    print("7. RUNPOD ENDPOINT CHECK")
    print("=" * 70)

    try:
        import httpx

        api_key = os.environ.get("RUNPOD_API_KEY", "")
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_EMBEDDING", "")

        if not api_key or not endpoint_id:
            print("  SKIP RunPod not configured")
            return False

        print(f"  Endpoint ID: {endpoint_id}")

        # Check health
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = httpx.get(url, headers=headers, timeout=10)
        data = response.json()

        print(f"  OK Health check response:")
        print(f"      Workers: {data.get('workers', {})}")

        return True

    except Exception as e:
        print(f"  FAIL RunPod check error: {e}")
        return False


def main():
    """Run all tests."""
    results = {}

    # 1. Environment
    results["environment"] = test_environment()
    if not results["environment"]:
        print("\n FAIL Environment check failed!")
        return

    # 2. Supabase
    products, cutouts, models = test_supabase_connection()
    results["supabase"] = bool(products or cutouts or models)

    # 3. Qdrant
    qdrant = test_qdrant_connection()
    results["qdrant_connection"] = qdrant is not None

    # 4. Qdrant operations
    results["qdrant_operations"] = test_qdrant_operations(qdrant)

    # 5. Image URL building
    valid_urls = test_image_url_building(products)
    results["image_urls"] = len(valid_urls) > 0

    # 6. Full flow simulation
    results["full_flow"] = test_full_flow_simulation(products, cutouts, models, qdrant)

    # 7. RunPod health
    results["runpod"] = test_runpod_health()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Ready to test with real worker.")
    else:
        print("SOME TESTS FAILED - Fix issues before deploying.")
    print("=" * 70)


if __name__ == "__main__":
    main()
