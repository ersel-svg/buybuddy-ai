#!/usr/bin/env python3
"""
End-to-End API Tests for OD AI Annotation

Run with: python tests/test_api.py --api-url http://localhost:8000

Prerequisites:
- API server running
- RunPod endpoint configured (for AI tests)
- Test image uploaded to Supabase
"""

import argparse
import json
import time
import sys
from typing import Optional
import httpx

# Test configuration
DEFAULT_API_URL = "http://localhost:8000/api/v1/od"
TEST_TIMEOUT = 120  # seconds


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s) {self.message}"


class ODAPITester:
    def __init__(self, api_url: str, verbose: bool = False):
        self.api_url = api_url.rstrip("/")
        self.verbose = verbose
        self.client = httpx.Client(timeout=TEST_TIMEOUT)
        self.results: list[TestResult] = []

        # Test data storage
        self.test_dataset_id: Optional[str] = None
        self.test_image_id: Optional[str] = None
        self.test_class_id: Optional[str] = None
        self.test_annotation_id: Optional[str] = None

    def log(self, message: str):
        if self.verbose:
            print(f"  → {message}")

    def run_test(self, name: str, func) -> TestResult:
        """Run a test function and record the result."""
        start = time.time()
        try:
            func()
            duration = time.time() - start
            result = TestResult(name, True, duration=duration)
        except AssertionError as e:
            duration = time.time() - start
            result = TestResult(name, False, str(e), duration)
        except Exception as e:
            duration = time.time() - start
            result = TestResult(name, False, f"Exception: {e}", duration)

        self.results.append(result)
        print(result)
        return result

    # ==========================================
    # Health & Stats Tests
    # ==========================================

    def test_health(self):
        """Test OD health endpoint."""
        response = self.client.get(f"{self.api_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "healthy"
        self.log(f"Health: {data}")

    def test_stats(self):
        """Test OD stats endpoint."""
        response = self.client.get(f"{self.api_url}/stats")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "total_images" in data
        assert "total_datasets" in data
        self.log(f"Stats: images={data['total_images']}, datasets={data['total_datasets']}")

    # ==========================================
    # Dataset Tests
    # ==========================================

    def test_list_datasets(self):
        """Test listing datasets."""
        response = self.client.get(f"{self.api_url}/datasets")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert isinstance(data, list)
        self.log(f"Found {len(data)} datasets")

    def test_create_dataset(self):
        """Test creating a dataset."""
        payload = {
            "name": f"Test Dataset {int(time.time())}",
            "description": "Created by E2E test",
            "annotation_type": "bbox"
        }
        response = self.client.post(f"{self.api_url}/datasets", json=payload)
        assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}: {response.text}"
        data = response.json()
        assert "id" in data
        self.test_dataset_id = data["id"]
        self.log(f"Created dataset: {self.test_dataset_id}")

    def test_get_dataset(self):
        """Test getting a dataset by ID."""
        if not self.test_dataset_id:
            raise AssertionError("No test dataset ID")

        response = self.client.get(f"{self.api_url}/datasets/{self.test_dataset_id}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["id"] == self.test_dataset_id

    # ==========================================
    # Class Tests
    # ==========================================

    def test_list_classes(self):
        """Test listing classes."""
        response = self.client.get(f"{self.api_url}/classes")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert isinstance(data, list)
        self.log(f"Found {len(data)} classes")

    def test_create_class(self):
        """Test creating a class."""
        payload = {
            "name": f"test_class_{int(time.time())}",
            "display_name": "Test Class",
            "color": "#FF5733",
            "category": "test"
        }
        response = self.client.post(f"{self.api_url}/classes", json=payload)
        assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}: {response.text}"
        data = response.json()
        assert "id" in data
        self.test_class_id = data["id"]
        self.log(f"Created class: {self.test_class_id}")

    # ==========================================
    # Image Tests
    # ==========================================

    def test_list_images(self):
        """Test listing images."""
        response = self.client.get(f"{self.api_url}/images")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "images" in data
        if data["images"]:
            self.test_image_id = data["images"][0]["id"]
            self.log(f"Found {len(data['images'])} images, using: {self.test_image_id}")
        else:
            self.log("No images found - some tests will be skipped")

    # ==========================================
    # AI Annotation Tests
    # ==========================================

    def test_ai_models_list(self):
        """Test listing available AI models."""
        response = self.client.get(f"{self.api_url}/ai/models")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "detection_models" in data
        assert "segmentation_models" in data
        self.log(f"Detection models: {[m['id'] for m in data['detection_models']]}")
        self.log(f"Segmentation models: {[m['id'] for m in data['segmentation_models']]}")

    def test_ai_predict(self):
        """Test AI prediction endpoint (requires RunPod)."""
        if not self.test_image_id:
            self.log("Skipping - no test image")
            return

        payload = {
            "image_id": self.test_image_id,
            "model": "grounding_dino",
            "text_prompt": "shelf . product",
            "box_threshold": 0.3,
            "text_threshold": 0.25
        }
        response = self.client.post(f"{self.api_url}/ai/predict", json=payload)

        # Accept 503 if endpoint not configured
        if response.status_code == 503:
            self.log("RunPod endpoint not configured - skipping")
            return

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "predictions" in data
        self.log(f"Got {len(data['predictions'])} predictions")

    def test_ai_segment(self):
        """Test AI segmentation endpoint (requires RunPod)."""
        if not self.test_image_id:
            self.log("Skipping - no test image")
            return

        payload = {
            "image_id": self.test_image_id,
            "model": "sam2",
            "prompt_type": "point",
            "point": [0.5, 0.5],
            "label": 1
        }
        response = self.client.post(f"{self.api_url}/ai/segment", json=payload)

        if response.status_code == 503:
            self.log("RunPod endpoint not configured - skipping")
            return

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "bbox" in data
        self.log(f"Got segmentation: bbox={data['bbox']}")

    # ==========================================
    # Annotation Tests
    # ==========================================

    def test_create_annotation(self):
        """Test creating an annotation."""
        if not self.test_dataset_id or not self.test_image_id or not self.test_class_id:
            self.log("Skipping - missing test data")
            return

        # First add image to dataset
        add_response = self.client.post(
            f"{self.api_url}/datasets/{self.test_dataset_id}/images",
            json={"image_ids": [self.test_image_id]}
        )
        self.log(f"Add image response: {add_response.status_code}")

        payload = {
            "class_id": self.test_class_id,
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
        }
        response = self.client.post(
            f"{self.api_url}/datasets/{self.test_dataset_id}/images/{self.test_image_id}/annotations",
            json=payload
        )
        assert response.status_code in [200, 201], f"Expected 200/201, got {response.status_code}: {response.text}"
        data = response.json()
        self.test_annotation_id = data.get("id")
        self.log(f"Created annotation: {self.test_annotation_id}")

    def test_list_annotations(self):
        """Test listing annotations."""
        if not self.test_dataset_id or not self.test_image_id:
            self.log("Skipping - missing test data")
            return

        response = self.client.get(
            f"{self.api_url}/datasets/{self.test_dataset_id}/images/{self.test_image_id}/annotations"
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        self.log(f"Found {len(data)} annotations")

    # ==========================================
    # Cleanup Tests
    # ==========================================

    def test_delete_annotation(self):
        """Test deleting an annotation."""
        if not self.test_annotation_id:
            self.log("Skipping - no test annotation")
            return

        response = self.client.delete(f"{self.api_url}/annotations/{self.test_annotation_id}")
        assert response.status_code in [200, 204], f"Expected 200/204, got {response.status_code}"
        self.log("Deleted annotation")

    def test_delete_class(self):
        """Test deleting a class."""
        if not self.test_class_id:
            self.log("Skipping - no test class")
            return

        response = self.client.delete(f"{self.api_url}/classes/{self.test_class_id}")
        assert response.status_code in [200, 204], f"Expected 200/204, got {response.status_code}"
        self.log("Deleted class")

    def test_delete_dataset(self):
        """Test deleting a dataset."""
        if not self.test_dataset_id:
            self.log("Skipping - no test dataset")
            return

        response = self.client.delete(f"{self.api_url}/datasets/{self.test_dataset_id}")
        assert response.status_code in [200, 204], f"Expected 200/204, got {response.status_code}"
        self.log("Deleted dataset")

    # ==========================================
    # Run All Tests
    # ==========================================

    def run_all(self):
        """Run all tests in order."""
        print(f"\n🧪 OD AI Annotation API Tests")
        print(f"   API URL: {self.api_url}")
        print("-" * 50)

        # Health & Stats
        self.run_test("Health Check", self.test_health)
        self.run_test("Stats Endpoint", self.test_stats)

        # Datasets
        self.run_test("List Datasets", self.test_list_datasets)
        self.run_test("Create Dataset", self.test_create_dataset)
        self.run_test("Get Dataset", self.test_get_dataset)

        # Classes
        self.run_test("List Classes", self.test_list_classes)
        self.run_test("Create Class", self.test_create_class)

        # Images
        self.run_test("List Images", self.test_list_images)

        # AI Annotation
        self.run_test("AI Models List", self.test_ai_models_list)
        self.run_test("AI Predict", self.test_ai_predict)
        self.run_test("AI Segment", self.test_ai_segment)

        # Annotations
        self.run_test("Create Annotation", self.test_create_annotation)
        self.run_test("List Annotations", self.test_list_annotations)

        # Cleanup
        self.run_test("Delete Annotation", self.test_delete_annotation)
        self.run_test("Delete Class", self.test_delete_class)
        self.run_test("Delete Dataset", self.test_delete_dataset)

        # Summary
        print("-" * 50)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)

        print(f"\n📊 Results: {passed}/{len(self.results)} passed ({total_time:.2f}s)")

        if failed > 0:
            print(f"\n❌ Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"   - {r.name}: {r.message}")
            return 1
        else:
            print("\n✅ All tests passed!")
            return 0


def main():
    parser = argparse.ArgumentParser(description="OD AI Annotation API Tests")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    tester = ODAPITester(args.api_url, verbose=args.verbose)
    sys.exit(tester.run_all())


if __name__ == "__main__":
    main()
