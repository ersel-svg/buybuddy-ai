"""
Bulk Update Feature - Safe Integration Test

This test:
1. Creates temporary test products with unique barcodes
2. Tests the bulk update preview and execute endpoints
3. Verifies the updates worked correctly
4. Cleans up all test data

SAFE: Uses TEST_BARCODE_PREFIX to isolate from real data
"""

import asyncio
import uuid
import os
from datetime import datetime

# Test configuration
TEST_BARCODE_PREFIX = "TEST_BULK_"  # All test barcodes start with this
NUM_TEST_PRODUCTS = 5

# Auth credentials from environment
AUTH_USERNAME = os.getenv("BUYBUDDY_USERNAME", "ersel")
AUTH_PASSWORD = os.getenv("BUYBUDDY_PASSWORD", "1234")


async def get_auth_token(client, api_base: str) -> str:
    """Login and get auth token."""
    response = await client.post(
        f"{api_base}/auth/login",
        json={"username": AUTH_USERNAME, "password": AUTH_PASSWORD}
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    return response.json()["token"]


async def run_bulk_update_test():
    """Run the complete bulk update test."""
    import httpx

    API_BASE = "http://localhost:8000/api/v1"

    print("=" * 60)
    print("BULK UPDATE INTEGRATION TEST")
    print("=" * 60)
    print(f"\nUsing test barcode prefix: {TEST_BARCODE_PREFIX}")
    print(f"Creating {NUM_TEST_PRODUCTS} test products...\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        created_product_ids = []
        test_barcodes = []

        try:
            # ============================================
            # STEP 0: Authenticate
            # ============================================
            print("STEP 0: Authenticating...")
            token = await get_auth_token(client, API_BASE)
            headers = {"Authorization": f"Bearer {token}"}
            print(f"  Authenticated as: {AUTH_USERNAME}\n")

            # ============================================
            # STEP 1: Create test products
            # ============================================
            print("STEP 1: Creating test products...")

            for i in range(NUM_TEST_PRODUCTS):
                barcode = f"{TEST_BARCODE_PREFIX}{uuid.uuid4().hex[:8]}"
                test_barcodes.append(barcode)

                product_data = {
                    "barcode": barcode,
                    "product_name": f"Test Product {i+1}",
                    "brand_name": "Original Brand",
                    "category": "Original Category",
                    "status": "pending",
                }

                response = await client.post(
                    f"{API_BASE}/products",
                    json=product_data,
                    headers=headers
                )

                if response.status_code == 200:
                    product = response.json()
                    created_product_ids.append(product["id"])
                    print(f"  Created: {barcode} -> {product['id'][:8]}...")
                else:
                    print(f"  FAILED to create product: {response.text}")
                    return False

            print(f"\n  Total created: {len(created_product_ids)} products\n")

            # ============================================
            # STEP 2: Test system fields endpoint
            # ============================================
            print("STEP 2: Testing /system-fields endpoint...")

            response = await client.get(f"{API_BASE}/products/bulk-update/system-fields", headers=headers)
            if response.status_code == 200:
                fields = response.json()
                print(f"  Found {len(fields)} system fields")
                editable = [f for f in fields if f["editable"]]
                print(f"  Editable fields: {len(editable)}")
            else:
                print(f"  FAILED: {response.text}")
                return False

            # ============================================
            # STEP 3: Test preview endpoint
            # ============================================
            print("\nSTEP 3: Testing /preview endpoint...")

            # Simulate Excel rows - some products get updates, some fields empty
            test_rows = []
            for i, barcode in enumerate(test_barcodes):
                row = {"barcode": barcode}

                # Product 0: Update brand only
                if i == 0:
                    row["brand"] = "Updated Brand A"

                # Product 1: Update category only
                elif i == 1:
                    row["category"] = "Updated Category B"

                # Product 2: Update both brand and category
                elif i == 2:
                    row["brand"] = "Updated Brand C"
                    row["category"] = "Updated Category C"

                # Product 3: Update with SKU (identifier)
                elif i == 3:
                    row["brand"] = "Updated Brand D"
                    row["sku_code"] = "SKU-TEST-001"

                # Product 4: Empty row (should not update anything)
                # row stays as just {"barcode": barcode}

                test_rows.append(row)

            # Add a row with non-existent barcode
            test_rows.append({
                "barcode": "NONEXISTENT_12345",
                "brand": "Should Not Update"
            })

            preview_request = {
                "rows": test_rows,
                "identifier_column": "barcode",
                "field_mappings": [
                    {"source_column": "brand", "target_field": "brand_name"},
                    {"source_column": "category", "target_field": "category"},
                    {"source_column": "sku_code", "target_field": "sku"},
                ]
            }

            response = await client.post(
                f"{API_BASE}/products/bulk-update/preview",
                json=preview_request,
                headers=headers
            )

            if response.status_code == 200:
                preview = response.json()
                summary = preview["summary"]
                print(f"  Total rows: {summary['total_rows']}")
                print(f"  Matched: {summary['matched']}")
                print(f"  Will update: {summary['will_update']}")
                print(f"  Not found: {summary['not_found']}")
                print(f"  Validation errors: {summary['validation_errors']}")

                # Verify expectations
                assert summary["matched"] == NUM_TEST_PRODUCTS, f"Expected {NUM_TEST_PRODUCTS} matches"
                assert summary["not_found"] == 1, "Expected 1 not found (NONEXISTENT_12345)"
                assert summary["will_update"] == 4, "Expected 4 products to update (product 4 has no changes)"

                print("\n  Preview changes:")
                for match in preview["matches"][:5]:
                    if match["product_field_changes"] or match["identifier_field_changes"]:
                        print(f"    {match['barcode']}: {match['product_field_changes']} {match['identifier_field_changes']}")

                print("\n  Preview test PASSED!")
            else:
                print(f"  FAILED: {response.text}")
                return False

            # ============================================
            # STEP 4: Test execute endpoint
            # ============================================
            print("\nSTEP 4: Testing /execute endpoint...")

            # Build updates from preview matches
            updates = []
            for match in preview["matches"]:
                if match["product_field_changes"] or match["identifier_field_changes"]:
                    updates.append({
                        "product_id": match["product_id"],
                        "fields": match["new_values"]
                    })

            execute_request = {
                "updates": updates,
                "mode": "lenient"
            }

            response = await client.post(
                f"{API_BASE}/products/bulk-update/execute",
                json=execute_request,
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                print(f"  Success: {result['success']}")
                print(f"  Updated: {result['updated_count']}")
                print(f"  Failed: {len(result['failed'])}")
                print(f"  Execution time: {result['execution_time_ms']}ms")

                assert result["updated_count"] == 4, f"Expected 4 updates, got {result['updated_count']}"

                print("\n  Execute test PASSED!")
            else:
                print(f"  FAILED: {response.text}")
                return False

            # ============================================
            # STEP 5: Verify updates in database
            # ============================================
            print("\nSTEP 5: Verifying updates in database...")

            for i, barcode in enumerate(test_barcodes):
                response = await client.get(
                    f"{API_BASE}/products",
                    params={"search": barcode, "limit": 1},
                    headers=headers
                )

                if response.status_code == 200:
                    products = response.json()["items"]
                    if products:
                        p = products[0]

                        # Check expected values
                        if i == 0:
                            assert p["brand_name"] == "Updated Brand A", f"Product 0 brand should be Updated Brand A, got {p['brand_name']}"
                            assert p["category"] == "Original Category", "Product 0 category should be unchanged"
                        elif i == 1:
                            assert p["brand_name"] == "Original Brand", "Product 1 brand should be unchanged"
                            assert p["category"] == "Updated Category B", f"Product 1 category should be Updated Category B"
                        elif i == 2:
                            assert p["brand_name"] == "Updated Brand C"
                            assert p["category"] == "Updated Category C"
                        elif i == 3:
                            assert p["brand_name"] == "Updated Brand D"
                        elif i == 4:
                            assert p["brand_name"] == "Original Brand", "Product 4 should be unchanged (empty row)"
                            assert p["category"] == "Original Category", "Product 4 should be unchanged"

                        print(f"  {barcode}: brand={p['brand_name']}, category={p['category']} ✓")

            print("\n  Verification PASSED!")

            # ============================================
            # STEP 6: Test empty value behavior
            # ============================================
            print("\nSTEP 6: Testing empty value behavior...")

            # Update product 0 with empty brand (should NOT change)
            empty_test_rows = [{
                "barcode": test_barcodes[0],
                "brand": "",  # Empty - should not update
                "category": "New Category Test"  # Non-empty - should update
            }]

            preview_request = {
                "rows": empty_test_rows,
                "identifier_column": "barcode",
                "field_mappings": [
                    {"source_column": "brand", "target_field": "brand_name"},
                    {"source_column": "category", "target_field": "category"},
                ]
            }

            response = await client.post(
                f"{API_BASE}/products/bulk-update/preview",
                json=preview_request,
                headers=headers
            )

            if response.status_code == 200:
                preview = response.json()
                if preview["matches"]:
                    changes = preview["matches"][0]["product_field_changes"]
                    # Should only have category change, not brand
                    assert "brand_name" not in changes, "Empty brand should NOT trigger update"
                    assert "category" in changes, "Non-empty category should trigger update"
                    print(f"  Changes detected: {changes}")
                    print("  Empty value behavior PASSED! (empty fields are skipped)")

            print("\n" + "=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
            return True

        finally:
            # ============================================
            # CLEANUP: Delete all test products
            # ============================================
            print("\nCLEANUP: Deleting test products...")

            for product_id in created_product_ids:
                try:
                    response = await client.delete(f"{API_BASE}/products/{product_id}", headers=headers)
                    if response.status_code == 200:
                        print(f"  Deleted: {product_id[:8]}...")
                    else:
                        print(f"  Failed to delete {product_id}: {response.status_code}")
                except Exception as e:
                    print(f"  Error deleting {product_id}: {e}")

            print(f"\n  Cleaned up {len(created_product_ids)} test products")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING BULK UPDATE SAFE TEST")
    print("This test creates temporary products with TEST_BULK_ prefix")
    print("Real products will NOT be affected")
    print("=" * 60 + "\n")

    success = asyncio.run(run_bulk_update_test())

    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        exit(1)
