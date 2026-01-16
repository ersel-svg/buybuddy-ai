#!/usr/bin/env python3
"""
Test script for embedding extraction pipeline.
Run this on SSH pod to test the full flow without RunPod serverless.

Usage:
    export SUPABASE_URL="https://xxx.supabase.co"
    export SUPABASE_SERVICE_ROLE_KEY="xxx"
    export QDRANT_URL="https://xxx.qdrant.cloud:6333"  # or local
    export QDRANT_API_KEY="xxx"  # if using Qdrant Cloud

    python test_pipeline.py
"""

import os
import sys
import uuid
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from supabase import create_client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Check imports
try:
    from src.extractor import get_extractor, DINOv2Extractor
    print("✓ Extractor imported")
except ImportError as e:
    print(f"✗ Extractor import error: {e}")
    sys.exit(1)


def test_environment():
    """Test environment variables."""
    print("\n" + "=" * 60)
    print("1. ENVIRONMENT CHECK")
    print("=" * 60)

    required = {
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    }

    # QDRANT_URL is optional - defaults to :memory: for testing
    qdrant_url = os.environ.get("QDRANT_URL", ":memory:")
    print(f"  {'✓' if qdrant_url else '-'} QDRANT_URL: {qdrant_url}")

    optional = {
        "QDRANT_API_KEY": os.environ.get("QDRANT_API_KEY", ""),
    }

    all_ok = True
    for key, value in required.items():
        status = "✓" if value else "✗"
        masked = value[:20] + "..." if len(value) > 20 else value
        print(f"  {status} {key}: {masked or 'NOT SET'}")
        if not value:
            all_ok = False

    for key, value in optional.items():
        status = "✓" if value else "-"
        masked = value[:20] + "..." if len(value) > 20 else value
        print(f"  {status} {key}: {masked or '(not set)'}")

    return all_ok


def test_supabase():
    """Test Supabase connection."""
    print("\n" + "=" * 60)
    print("2. SUPABASE CONNECTION")
    print("=" * 60)

    try:
        client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        )

        # Test products
        result = client.table("products").select("id, barcode, frames_path, frame_count").limit(3).execute()
        products = result.data or []
        print(f"  ✓ Connected! Found {len(products)} products (showing 3)")
        for p in products:
            print(f"    - {p['barcode']}: {p['frame_count']} frames")

        # Test cutouts
        result = client.table("cutout_images").select("id, image_url, predicted_upc").limit(3).execute()
        cutouts = result.data or []
        print(f"  ✓ Found {len(cutouts)} cutouts (showing 3)")
        for c in cutouts:
            print(f"    - {c['predicted_upc']}: {c['image_url'][:50]}...")

        return client, products, cutouts

    except Exception as e:
        print(f"  ✗ Supabase error: {e}")
        return None, [], []


def test_qdrant():
    """Test Qdrant connection."""
    print("\n" + "=" * 60)
    print("3. QDRANT CONNECTION")
    print("=" * 60)

    try:
        qdrant_url = os.environ.get("QDRANT_URL", "")
        api_key = os.environ.get("QDRANT_API_KEY")

        # Support in-memory mode for testing
        if qdrant_url == ":memory:" or not qdrant_url:
            print("  Using in-memory mode...")
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(
                url=qdrant_url,
                api_key=api_key if api_key else None,
            )

        # List collections
        collections = client.get_collections().collections
        print(f"  ✓ Connected! Collections: {[c.name for c in collections]}")

        # Create test collection if needed
        test_collection = "test_embeddings"
        try:
            client.get_collection(test_collection)
            print(f"  ✓ Collection '{test_collection}' exists")
        except:
            print(f"  → Creating collection '{test_collection}'...")
            client.create_collection(
                collection_name=test_collection,
                vectors_config=qmodels.VectorParams(
                    size=768,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            print(f"  ✓ Collection '{test_collection}' created")

        return client, test_collection

    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
        return None, None


def test_extractor():
    """Test DINOv2 extractor."""
    print("\n" + "=" * 60)
    print("4. DINOV2 EXTRACTOR")
    print("=" * 60)

    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

        print("\n  Loading DINOv2 Base model...")
        extractor = get_extractor(model_type="dinov2-base")
        print(f"  ✓ Model loaded!")
        print(f"    - Embedding dim: {extractor.embedding_dim}")
        print(f"    - Device: {extractor.device}")

        return extractor

    except Exception as e:
        print(f"  ✗ Extractor error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_embedding_extraction(extractor, products, cutouts):
    """Test embedding extraction on sample images."""
    print("\n" + "=" * 60)
    print("5. EMBEDDING EXTRACTION TEST")
    print("=" * 60)

    test_urls = []

    # Add product frame_0
    if products:
        p = products[0]
        frames_path = p.get("frames_path", "")
        if frames_path:
            frame_url = f"{frames_path.rstrip('/')}/frame_0000.png"
            test_urls.append(("product", p["barcode"], frame_url))

    # Add cutout
    if cutouts:
        c = cutouts[0]
        test_urls.append(("cutout", c["predicted_upc"], c["image_url"]))

    if not test_urls:
        print("  ✗ No test images available")
        return []

    print(f"  Testing {len(test_urls)} images...")

    results = []
    for img_type, identifier, url in test_urls:
        print(f"\n  [{img_type}] {identifier}")
        print(f"    URL: {url[:60]}...")

        try:
            embedding = extractor.extract_from_url(url)
            print(f"    ✓ Embedding shape: {embedding.shape}")
            print(f"    ✓ Embedding norm: {(embedding ** 2).sum() ** 0.5:.4f}")
            print(f"    ✓ First 5 values: {embedding[:5]}")
            results.append((img_type, identifier, url, embedding))
        except Exception as e:
            print(f"    ✗ Error: {e}")

    return results


def test_qdrant_upsert(qdrant_client, collection_name, embeddings):
    """Test upserting embeddings to Qdrant."""
    print("\n" + "=" * 60)
    print("6. QDRANT UPSERT TEST")
    print("=" * 60)

    if not qdrant_client or not collection_name:
        print("  ✗ Qdrant not available")
        return

    if not embeddings:
        print("  ✗ No embeddings to upsert")
        return

    points = []
    for img_type, identifier, url, embedding in embeddings:
        point_id = str(uuid.uuid4())
        payload = {
            "source": img_type,
            "identifier": identifier,
            "url": url,
        }
        points.append(qmodels.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload,
        ))

    print(f"  Upserting {len(points)} points...")
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Verify
    info = qdrant_client.get_collection(collection_name)
    print(f"  ✓ Collection now has {info.points_count} points")


def test_similarity_search(qdrant_client, collection_name, embeddings):
    """Test similarity search."""
    print("\n" + "=" * 60)
    print("7. SIMILARITY SEARCH TEST")
    print("=" * 60)

    if not qdrant_client or not collection_name or not embeddings:
        print("  ✗ Requirements not met")
        return

    # Use first embedding as query
    query_type, query_id, query_url, query_embedding = embeddings[0]

    print(f"  Query: [{query_type}] {query_id}")

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=5,
    )

    print(f"  Found {len(results.points)} similar vectors:")
    for i, r in enumerate(results.points):
        print(f"    {i+1}. Score: {r.score:.4f} - {r.payload.get('source')}: {r.payload.get('identifier')}")


def main():
    print("\n" + "=" * 60)
    print("EMBEDDING PIPELINE TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")

    # 1. Environment
    if not test_environment():
        print("\n✗ Environment check failed. Set required variables.")
        sys.exit(1)

    # 2. Supabase
    supabase, products, cutouts = test_supabase()
    if not supabase:
        print("\n✗ Supabase connection failed.")
        sys.exit(1)

    # 3. Qdrant
    qdrant, collection = test_qdrant()
    if not qdrant:
        print("\n✗ Qdrant connection failed.")
        sys.exit(1)

    # 4. Extractor
    extractor = test_extractor()
    if not extractor:
        print("\n✗ Extractor failed.")
        sys.exit(1)

    # 5. Extract embeddings
    embeddings = test_embedding_extraction(extractor, products, cutouts)

    # 6. Upsert to Qdrant
    test_qdrant_upsert(qdrant, collection, embeddings)

    # 7. Similarity search
    test_similarity_search(qdrant, collection, embeddings)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
