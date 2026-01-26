#!/usr/bin/env python3
"""
Test script for all SOTA improvements to bulk operations.
Tests both sync and async paths for all operations.
"""

import requests
import time
import json
from typing import Dict, Any

API_BASE = "http://localhost:8000"
AUTH_TOKEN = None  # Will be set after login

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_test_header(test_name: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}TEST: {test_name}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(message: str):
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def print_info(message: str):
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")

def print_warning(message: str):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def get_auth_headers() -> Dict[str, str]:
    """Get headers with authentication token."""
    if AUTH_TOKEN:
        return {"Authorization": f"Bearer {AUTH_TOKEN}"}
    return {}

def login() -> bool:
    """Login to get authentication token."""
    global AUTH_TOKEN

    print_info("Logging in to get authentication token...")

    response = requests.post(
        f"{API_BASE}/api/v1/auth/login",
        json={
            "username": "ersel",
            "password": "1234"
        }
    )

    if response.status_code == 200:
        result = response.json()
        AUTH_TOKEN = result.get("token")
        print_success(f"Logged in as {result.get('username')}")
        return True
    else:
        print_error(f"Login failed: {response.status_code} - {response.text}")
        return False

def wait_for_job(job_id: str, timeout: int = 180) -> Dict[str, Any]:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}", headers=get_auth_headers())
        if response.status_code != 200:
            print_error(f"Failed to get job status: {response.status_code}")
            return None

        job = response.json()

        # Show progress updates
        if job.get("progress", 0) != last_progress:
            last_progress = job["progress"]
            current_step = job.get("current_step", "Processing...")
            print_info(f"Progress: {last_progress}% - {current_step}")

        if job["status"] in ["completed", "failed", "cancelled"]:
            return job

        time.sleep(1)

    print_warning(f"Job timeout after {timeout}s")
    return None

def get_product_ids(limit: int = 100) -> list[str]:
    """Fetch product IDs from the API."""
    response = requests.get(f"{API_BASE}/api/v1/products?limit={limit}", headers=get_auth_headers())
    if response.status_code != 200:
        print_error(f"Failed to get products: {response.status_code}")
        return []

    data = response.json()
    items = data.get("items", [])
    return [item["id"] for item in items]

def get_dataset_id() -> str:
    """Get or create a test dataset."""
    # Try to get existing datasets
    response = requests.get(f"{API_BASE}/api/v1/datasets", headers=get_auth_headers())
    if response.status_code == 200:
        datasets = response.json()
        if datasets:
            return datasets[0]["id"]

    # Create new dataset if none exist
    response = requests.post(
        f"{API_BASE}/api/v1/datasets",
        headers=get_auth_headers(),
        json={
            "name": "Test Dataset for SOTA",
            "description": "Created by test script",
            "type": "object_detection"
        }
    )
    if response.status_code in [200, 201]:
        return response.json()["id"]

    print_error("Failed to get or create dataset")
    return None

# =============================================================================
# TEST 1: Bulk Delete - Sync Path (<50 products)
# =============================================================================

def test_bulk_delete_sync():
    print_test_header("Bulk Delete - Sync Path (<50 products)")

    # Get 30 product IDs
    product_ids = get_product_ids(30)
    if len(product_ids) < 30:
        print_warning(f"Only found {len(product_ids)} products, need at least 30")
        # Adjust to available count
        product_ids = product_ids[:min(30, len(product_ids))]

    print_info(f"Deleting {len(product_ids)} products (should be synchronous)")

    response = requests.post(
        f"{API_BASE}/api/v1/products/bulk-delete",
        headers=get_auth_headers(),
        json={"product_ids": product_ids}
    )

    if response.status_code == 200:
        result = response.json()
        if "job_id" in result:
            print_error("Expected sync response but got job_id (should be <50 threshold)")
        else:
            print_success(f"Sync delete successful: {result.get('message', 'OK')}")
            print_info(f"Deleted: {result.get('deleted', 0)}, Failed: {result.get('failed', 0)}")
    else:
        print_error(f"Delete failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 2: Bulk Delete - Async Path (50+ products)
# =============================================================================

def test_bulk_delete_async():
    print_test_header("Bulk Delete - Async Path (50+ products)")

    # Get 60 product IDs
    product_ids = get_product_ids(100)
    if len(product_ids) < 50:
        print_warning(f"Only found {len(product_ids)} products, need at least 50")
        print_warning("Skipping async delete test")
        return

    product_ids = product_ids[:60]  # Use exactly 60

    print_info(f"Deleting {len(product_ids)} products (should be async)")

    response = requests.post(
        f"{API_BASE}/api/v1/products/bulk-delete",
        headers=get_auth_headers(),
        json={"product_ids": product_ids}
    )

    if response.status_code == 200:
        result = response.json()
        if "job_id" not in result:
            print_error("Expected job_id but got sync response (should be 50+ threshold)")
        else:
            job_id = result["job_id"]
            print_success(f"Background job created: {job_id}")
            print_info("Waiting for job to complete...")

            job = wait_for_job(job_id)
            if job:
                if job["status"] == "completed":
                    print_success(f"Job completed: {job.get('result', {}).get('message', 'OK')}")
                    result = job.get("result", {})
                    print_info(f"Deleted: {result.get('deleted', 0)}, Failed: {result.get('failed', 0)}")
                    print_info(f"Frames deleted: {result.get('frames_deleted', 0)}")
                    print_info(f"Storage deleted: {result.get('storage_deleted', 0)}")
                else:
                    print_error(f"Job failed: {job.get('error', 'Unknown error')}")
    else:
        print_error(f"Delete failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 3: Add to Dataset - Sync Path (<50 products)
# =============================================================================

def test_add_to_dataset_sync():
    print_test_header("Add to Dataset - Sync Path (<50 products)")

    dataset_id = get_dataset_id()
    if not dataset_id:
        print_error("Cannot test without dataset")
        return

    # Get 30 product IDs
    product_ids = get_product_ids(30)
    if len(product_ids) < 30:
        print_warning(f"Only found {len(product_ids)} products")
        product_ids = product_ids[:min(30, len(product_ids))]

    print_info(f"Adding {len(product_ids)} products to dataset (should be sync)")

    response = requests.post(
        f"{API_BASE}/api/v1/datasets/{dataset_id}/products",
        headers=get_auth_headers(),
        json={"product_ids": product_ids}
    )

    if response.status_code == 200:
        result = response.json()
        if "job_id" in result:
            print_error("Expected sync response but got job_id")
        else:
            print_success(f"Sync add successful: {result.get('added_count', 0)} products added")
    else:
        print_error(f"Add failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 4: Add to Dataset - Async Path (50+ products)
# =============================================================================

def test_add_to_dataset_async():
    print_test_header("Add to Dataset - Async Path (50+ products)")

    dataset_id = get_dataset_id()
    if not dataset_id:
        print_error("Cannot test without dataset")
        return

    # Get 60 product IDs
    product_ids = get_product_ids(100)
    if len(product_ids) < 50:
        print_warning(f"Only found {len(product_ids)} products, need at least 50")
        print_warning("Skipping async add test")
        return

    product_ids = product_ids[:60]

    print_info(f"Adding {len(product_ids)} products to dataset (should be async)")

    response = requests.post(
        f"{API_BASE}/api/v1/datasets/{dataset_id}/products",
        headers=get_auth_headers(),
        json={"product_ids": product_ids}
    )

    if response.status_code == 200:
        result = response.json()
        if "job_id" not in result:
            print_error("Expected job_id but got sync response")
        else:
            job_id = result["job_id"]
            print_success(f"Background job created: {job_id}")
            print_info("Waiting for job to complete...")

            job = wait_for_job(job_id)
            if job:
                if job["status"] == "completed":
                    print_success(f"Job completed: {job.get('result', {}).get('message', 'OK')}")
                    result = job.get("result", {})
                    print_info(f"Added: {result.get('added', 0)}, Skipped: {result.get('skipped', 0)}")
                else:
                    print_error(f"Job failed: {job.get('error', 'Unknown error')}")
    else:
        print_error(f"Add failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 5: Product Matcher - Sync Path (<1000 rows)
# =============================================================================

def test_product_matcher_sync():
    print_test_header("Product Matcher - Sync Path (<1000 rows)")

    # Create small test data (100 rows)
    test_rows = []
    for i in range(100):
        test_rows.append({
            "barcode": f"test-barcode-{i}",
            "product_name": f"Test Product {i}",
            "brand_name": "Test Brand"
        })

    match_rules = [
        {"source_column": "barcode", "target_field": "barcode", "priority": 1},
        {"source_column": "product_name", "target_field": "product_name", "priority": 2}
    ]

    print_info(f"Matching {len(test_rows)} rows (should be sync)")

    response = requests.post(
        f"{API_BASE}/api/v1/product-matcher/match",
        headers=get_auth_headers(),
        json={
            "rows": test_rows,
            "mapping_config": {"match_rules": match_rules}
        }
    )

    if response.status_code == 200:
        result = response.json()
        print_success(f"Sync match successful")
        print_info(f"Matched: {result.get('matched_count', 0)}/{result.get('total', 0)}")
        print_info(f"Match rate: {result.get('match_rate', 0)}%")
    else:
        print_error(f"Match failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 6: Product Matcher - Async Path (1000+ rows)
# =============================================================================

def test_product_matcher_async():
    print_test_header("Product Matcher - Async Path (1000+ rows)")

    # Create large test data (1500 rows)
    test_rows = []
    for i in range(1500):
        test_rows.append({
            "barcode": f"test-barcode-{i}",
            "product_name": f"Test Product {i}",
            "brand_name": "Test Brand"
        })

    match_rules = [
        {"source_column": "barcode", "target_field": "barcode", "priority": 1},
        {"source_column": "product_name", "target_field": "product_name", "priority": 2}
    ]

    print_info(f"Matching {len(test_rows)} rows (should be async)")

    # First, verify sync endpoint rejects this
    print_info("Testing sync endpoint rejects large batch...")
    response = requests.post(
        f"{API_BASE}/api/v1/product-matcher/match",
        headers=get_auth_headers(),
        json={
            "rows": test_rows,
            "mapping_config": {"match_rules": match_rules}
        }
    )

    if response.status_code == 400:
        print_success("Sync endpoint correctly rejected large batch")
    else:
        print_warning(f"Sync endpoint returned unexpected status: {response.status_code}")

    # Now test async endpoint
    print_info("Testing async endpoint...")
    response = requests.post(
        f"{API_BASE}/api/v1/product-matcher/match/async",
        headers=get_auth_headers(),
        json={
            "rows": test_rows,
            "mapping_config": {"match_rules": match_rules}
        }
    )

    if response.status_code == 200:
        result = response.json()
        job_id = result.get("job_id")
        if job_id:
            print_success(f"Background job created: {job_id}")
            print_info("Waiting for job to complete...")

            job = wait_for_job(job_id, timeout=180)  # Longer timeout for matching
            if job:
                if job["status"] == "completed":
                    print_success(f"Job completed: {job.get('result', {}).get('message', 'OK')}")
                    result = job.get("result", {})
                    print_info(f"Matched: {result.get('matched', 0)}/{result.get('total', 0)}")
                    print_info(f"Match rate: {result.get('match_rate', 0)}%")
                else:
                    print_error(f"Job failed: {job.get('error', 'Unknown error')}")
        else:
            print_error("No job_id in response")
    else:
        print_error(f"Async match failed: {response.status_code} - {response.text}")

# =============================================================================
# TEST 7: CSV Export - Streaming
# =============================================================================

def test_csv_export_streaming():
    print_test_header("CSV Export - Streaming (Large Dataset)")

    print_info("Testing CSV export with streaming...")

    response = requests.post(
        f"{API_BASE}/api/v1/products/export/csv",
        headers=get_auth_headers(),
        json={
            "filters": {},
            "include_fields": ["id", "barcode", "product_name", "brand_name", "status"]
        },
        stream=True  # Important: use streaming
    )

    if response.status_code == 200:
        # Check if response is streamed
        if response.headers.get("content-disposition"):
            print_success("Export started successfully (streaming response)")

            # Download first few chunks to verify streaming
            chunk_count = 0
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                chunk_count += 1
                total_size += len(chunk)
                if chunk_count <= 3:
                    print_info(f"Received chunk {chunk_count}: {len(chunk)} bytes")
                if chunk_count >= 10:  # Don't download everything
                    print_info(f"Stopping after {chunk_count} chunks ({total_size} bytes)")
                    break

            print_success(f"Streaming verified: received {chunk_count} chunks")
        else:
            print_warning("Response doesn't appear to be a file download")
    else:
        print_error(f"Export failed: {response.status_code} - {response.text}")

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                   SOTA IMPROVEMENTS - COMPREHENSIVE TEST SUITE                ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")

    print_info(f"API Base URL: {API_BASE}")
    print_info("Starting tests...\n")

    # Login first
    if not login():
        print_error("Failed to authenticate. Aborting tests.")
        return

    print()  # Empty line after login

    try:
        # Test 1: Bulk Delete - Sync
        test_bulk_delete_sync()
        time.sleep(2)

        # Test 2: Bulk Delete - Async
        test_bulk_delete_async()
        time.sleep(2)

        # Test 3: Add to Dataset - Sync
        test_add_to_dataset_sync()
        time.sleep(2)

        # Test 4: Add to Dataset - Async
        test_add_to_dataset_async()
        time.sleep(2)

        # Test 5: Product Matcher - Sync
        test_product_matcher_sync()
        time.sleep(2)

        # Test 6: Product Matcher - Async
        test_product_matcher_async()
        time.sleep(2)

        # Test 7: CSV Export - Streaming
        test_csv_export_streaming()

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}")
        print("╔═══════════════════════════════════════════════════════════════════════════════╗")
        print("║                           ALL TESTS COMPLETED                                 ║")
        print("╚═══════════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}\n")

    except KeyboardInterrupt:
        print_warning("\n\nTests interrupted by user")
    except Exception as e:
        print_error(f"\n\nTest suite error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
