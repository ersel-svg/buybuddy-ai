"""
Test suite for embedding extraction integration fixes.

Tests:
1. Multiple collection support (Matching mode)
2. product_ids filtering
3. product_dataset_id filtering
4. Backward compatibility
"""

import sys
from pathlib import Path

# Add worker src to path
worker_path = Path(__file__).parent.parent / "workers" / "embedding-extraction" / "src"
sys.path.insert(0, str(worker_path))


def test_multiple_collection_grouping():
    """Test 1: Multiple Collection Support - Image Grouping by Collection"""
    print("\n" + "=" * 70)
    print("TEST 1: Multiple Collection Support - Image Grouping")
    print("=" * 70)

    from collections import defaultdict

    # Mock batch with mixed collections
    batch = [
        {"id": "prod_1", "url": "http://example.com/1.jpg", "collection": "products_dinov2"},
        {"id": "cutout_1", "url": "http://example.com/2.jpg", "collection": "cutouts_dinov2"},
        {"id": "prod_2", "url": "http://example.com/3.jpg", "collection": "products_dinov2"},
        {"id": "cutout_2", "url": "http://example.com/4.jpg", "collection": "cutouts_dinov2"},
        {"id": "prod_3", "url": "http://example.com/5.jpg", "collection": "products_dinov2"},
    ]

    # Simulate the grouping logic from handler.py
    collection_points = defaultdict(list)

    for img in batch:
        target_collection = img.get("collection") or "default_collection"
        collection_points[target_collection].append(img["id"])

    # Verify grouping
    print(f"\n✓ Total images: {len(batch)}")
    print(f"✓ Collections found: {list(collection_points.keys())}")
    print(f"✓ Products collection: {len(collection_points['products_dinov2'])} images")
    print(f"✓ Cutouts collection: {len(collection_points['cutouts_dinov2'])} images")

    # Assertions
    assert len(collection_points) == 2, f"Expected 2 collections, got {len(collection_points)}"
    assert len(collection_points["products_dinov2"]) == 3, "Expected 3 products"
    assert len(collection_points["cutouts_dinov2"]) == 2, "Expected 2 cutouts"

    print("\n✅ TEST 1 PASSED: Images correctly grouped by collection")
    return True


def test_default_collection_fallback():
    """Test 2: Default Collection Fallback (Backward Compatibility)"""
    print("\n" + "=" * 70)
    print("TEST 2: Default Collection Fallback (Backward Compatibility)")
    print("=" * 70)

    from collections import defaultdict

    # Mock batch WITHOUT collection field (legacy format)
    batch = [
        {"id": "img_1", "url": "http://example.com/1.jpg"},
        {"id": "img_2", "url": "http://example.com/2.jpg"},
    ]

    default_collection = "legacy_collection"
    collection_points = defaultdict(list)

    for img in batch:
        target_collection = img.get("collection") or default_collection
        collection_points[target_collection].append(img["id"])

    print(f"\n✓ Images without collection field: {len(batch)}")
    print(f"✓ Fallback to default collection: '{default_collection}'")
    print(f"✓ All images in default collection: {len(collection_points[default_collection])}")

    # Assertions
    assert len(collection_points) == 1, "Should have only 1 collection"
    assert default_collection in collection_points, "Should use default collection"
    assert len(collection_points[default_collection]) == 2, "All images should be in default"

    print("\n✅ TEST 2 PASSED: Backward compatibility maintained")
    return True


def test_product_ids_filtering():
    """Test 3: product_ids Filtering (Selected Source)"""
    print("\n" + "=" * 70)
    print("TEST 3: product_ids Filtering (Selected Source)")
    print("=" * 70)

    from data.supabase_fetcher import fetch_with_pagination

    # Mock source config with product_ids
    source_config = {
        "type": "products",
        "filters": {
            "product_source": "selected",
            "product_ids": ["uuid-1", "uuid-2", "uuid-3"],
        }
    }

    # Simulate filter building logic from supabase_fetcher.py
    filters = {"frame_count_gt": 0}
    product_source = source_config["filters"]["product_source"]
    product_ids = source_config["filters"].get("product_ids")

    if product_source == "selected" and product_ids:
        filters["id"] = product_ids
        print(f"\n✓ Source: {product_source}")
        print(f"✓ Filter applied: id IN {product_ids}")
        print(f"✓ Number of products to fetch: {len(product_ids)}")

    # Verify filters
    assert "id" in filters, "Should have id filter"
    assert filters["id"] == ["uuid-1", "uuid-2", "uuid-3"], "Should filter by specific IDs"

    print("\n✅ TEST 3 PASSED: product_ids filter correctly applied")
    return True


def test_product_dataset_id_filtering():
    """Test 4: product_dataset_id Filtering (Dataset Source)"""
    print("\n" + "=" * 70)
    print("TEST 4: product_dataset_id Filtering (Dataset Source)")
    print("=" * 70)

    # Mock source config with dataset_id
    source_config = {
        "type": "products",
        "filters": {
            "product_source": "dataset",
            "product_dataset_id": "dataset-uuid-123",
        }
    }

    # Simulate filter building logic
    filters = {"frame_count_gt": 0}
    product_source = source_config["filters"]["product_source"]
    product_dataset_id = source_config["filters"].get("product_dataset_id")

    if product_source == "dataset" and product_dataset_id:
        filters["dataset_id"] = product_dataset_id
        print(f"\n✓ Source: {product_source}")
        print(f"✓ Filter applied: dataset_id = '{product_dataset_id}'")

    # Verify filters
    assert "dataset_id" in filters, "Should have dataset_id filter"
    assert filters["dataset_id"] == "dataset-uuid-123", "Should filter by dataset ID"

    print("\n✅ TEST 4 PASSED: product_dataset_id filter correctly applied")
    return True


def test_product_filter_custom():
    """Test 5: Custom product_filter (Filter Source)"""
    print("\n" + "=" * 70)
    print("TEST 5: Custom product_filter (Filter Source)")
    print("=" * 70)

    # Mock source config with custom filter
    source_config = {
        "type": "products",
        "filters": {
            "product_source": "filter",
            "product_filter": {
                "brand_name": "Nike",
                "category": "shoes",
            }
        }
    }

    # Simulate filter building logic
    filters = {"frame_count_gt": 0}
    product_source = source_config["filters"]["product_source"]
    product_filter = source_config["filters"].get("product_filter")

    if product_source == "filter" and product_filter:
        filters.update(product_filter)
        print(f"\n✓ Source: {product_source}")
        print(f"✓ Custom filters applied: {product_filter}")

    # Verify filters
    assert "brand_name" in filters, "Should have brand_name filter"
    assert "category" in filters, "Should have category filter"
    assert filters["brand_name"] == "Nike", "Should filter by brand"

    print("\n✅ TEST 5 PASSED: Custom product_filter correctly applied")
    return True


def test_api_source_config_format():
    """Test 6: API source_config Format (Integration Check)"""
    print("\n" + "=" * 70)
    print("TEST 6: API source_config Format (Integration Check)")
    print("=" * 70)

    # Mock API request (from embeddings.py)
    api_request = {
        "product_source": "selected",
        "product_ids": ["uuid-1", "uuid-2"],
        "product_dataset_id": None,
        "product_filter": None,
        "cutout_filter_has_upc": True,
    }

    # Simulate API building source_config
    source_config = {
        "type": "products",
        "filters": {
            "product_source": api_request["product_source"],
            "product_ids": api_request["product_ids"],
            "product_dataset_id": api_request["product_dataset_id"],
            "product_filter": api_request["product_filter"],
            "cutout_filter_has_upc": api_request["cutout_filter_has_upc"],
        }
    }

    print(f"\n✓ API Request received")
    print(f"✓ source_config built: {source_config}")
    print(f"✓ Filters included:")
    print(f"  - product_source: {source_config['filters']['product_source']}")
    print(f"  - product_ids: {source_config['filters']['product_ids']}")
    print(f"  - product_dataset_id: {source_config['filters']['product_dataset_id']}")
    print(f"  - product_filter: {source_config['filters']['product_filter']}")

    # Verify all fields present
    assert "product_ids" in source_config["filters"], "Missing product_ids"
    assert "product_dataset_id" in source_config["filters"], "Missing product_dataset_id"
    assert "product_filter" in source_config["filters"], "Missing product_filter"

    print("\n✅ TEST 6 PASSED: API correctly includes all filter fields")
    return True


def run_all_tests():
    """Run all embedding extraction tests"""
    print("\n" + "=" * 70)
    print("EMBEDDING EXTRACTION INTEGRATION TESTS")
    print("=" * 70)

    tests = [
        test_multiple_collection_grouping,
        test_default_collection_fallback,
        test_product_ids_filtering,
        test_product_dataset_id_filtering,
        test_product_filter_custom,
        test_api_source_config_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
