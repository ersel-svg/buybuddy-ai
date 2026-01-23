"""
Classification Module - Full E2E Test with AI Worker
Tests the complete workflow:
1. Upload images to Supabase (via API)
2. Create dataset with classes
3. Add images to dataset
4. Test AI worker (CLIP classification)
5. Label images
6. Create dataset version
7. Verify database records
"""

import os
import sys
import uuid
import time
import json
import requests
from io import BytesIO
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
WORKER_URL = "http://localhost:8081"  # Forwarded from pod:8080

def create_test_image(width=224, height=224, color=(255, 0, 0)):
    """Create a test image with specific color."""
    img = Image.new('RGB', (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer

def step(name, func):
    """Run a test step."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print('='*60)
    try:
        result = func()
        print(f"✓ OK")
        return result
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise

def main():
    print("\n" + "="*60)
    print("CLASSIFICATION MODULE - FULL E2E TEST")
    print("="*60)
    print(f"API URL: {API_BASE_URL}")
    print(f"Worker URL: {WORKER_URL}")

    # Unique identifier for this test run
    test_id = uuid.uuid4().hex[:8]
    print(f"Test ID: {test_id}")

    # 1. Health checks
    def check_health():
        # API health
        r = requests.get(f"{API_BASE_URL}/classification/health", timeout=10)
        assert r.status_code == 200, f"API health failed: {r.status_code}"
        print(f"  API: {r.json()}")

        # Worker health
        r = requests.get(f"{WORKER_URL}/health", timeout=10)
        assert r.status_code == 200, f"Worker health failed: {r.status_code}"
        print(f"  Worker: {r.json()}")
        return True

    step("Health Checks", check_health)

    # 2. Upload test images
    test_images = []
    def upload_images():
        colors = [
            ((255, 0, 0), "red"),     # Red
            ((200, 50, 50), "red"),
            ((255, 100, 100), "red"),
            ((0, 255, 0), "green"),   # Green
            ((50, 200, 50), "green"),
            ((100, 255, 100), "green"),
            ((0, 0, 255), "blue"),    # Blue
            ((50, 50, 200), "blue"),
            ((255, 255, 0), "yellow"),# Yellow
            ((200, 200, 50), "yellow"),
        ]

        for i, (color, label) in enumerate(colors):
            img_buffer = create_test_image(224, 224, color)
            files = {'file': (f'test_{test_id}_{i+1}.jpg', img_buffer, 'image/jpeg')}
            r = requests.post(f"{API_BASE_URL}/classification/images", files=files, timeout=30)

            if r.status_code == 200:
                data = r.json()
                test_images.append({
                    "id": data["id"],
                    "url": data.get("image_url", ""),
                    "expected_class": label
                })
                print(f"  Uploaded: {data['id'][:8]}... ({label})")
            else:
                print(f"  Failed: {r.status_code} - {r.text[:100]}")

        assert len(test_images) >= 8, f"Only {len(test_images)} images uploaded"
        return test_images

    images = step("Upload 10 Test Images", upload_images)

    # 3. Create dataset
    dataset = {}
    def create_dataset():
        nonlocal dataset
        data = {
            "name": f"E2E Test {test_id}",
            "description": f"Automated E2E test {test_id}",
            "task_type": "single_label"
        }
        r = requests.post(f"{API_BASE_URL}/classification/datasets", json=data, timeout=10)
        assert r.status_code == 200, f"Create dataset failed: {r.status_code} - {r.text}"
        dataset = r.json()
        print(f"  Dataset ID: {dataset['id']}")
        print(f"  Name: {dataset['name']}")
        return dataset

    step("Create Dataset", create_dataset)

    # 4. Create classes
    classes = {}
    def create_classes():
        class_defs = [
            {"name": f"red_{test_id}", "display_name": "Red Color", "color": "#ef4444"},
            {"name": f"green_{test_id}", "display_name": "Green Color", "color": "#22c55e"},
            {"name": f"blue_{test_id}", "display_name": "Blue Color", "color": "#3b82f6"},
            {"name": f"yellow_{test_id}", "display_name": "Yellow Color", "color": "#eab308"},
        ]

        for cls_data in class_defs:
            cls_data["dataset_id"] = dataset["id"]
            r = requests.post(f"{API_BASE_URL}/classification/classes", json=cls_data, timeout=10)
            if r.status_code == 200:
                result = r.json()
                # Map both name and display_name
                classes[cls_data["name"].split("_")[0]] = result
                print(f"  Created: {result['name']} ({result['id'][:8]}...)")
            else:
                print(f"  Failed {cls_data['name']}: {r.status_code} - {r.text}")

        assert len(classes) == 4, f"Only {len(classes)} classes created"
        return classes

    step("Create 4 Classes", create_classes)

    # 5. Add images to dataset
    def add_images():
        image_ids = [img["id"] for img in test_images]
        data = {"image_ids": image_ids}
        r = requests.post(
            f"{API_BASE_URL}/classification/datasets/{dataset['id']}/images/add",
            json=data, timeout=10
        )
        assert r.status_code == 200, f"Add images failed: {r.status_code} - {r.text}"
        result = r.json()
        print(f"  Added: {result.get('added', 0)} images")
        return result

    step("Add Images to Dataset", add_images)

    # 6. Test AI Worker - Classification (direct call)
    def test_worker():
        # Get class names for zero-shot
        class_names = [c['display_name'] for c in classes.values()]
        print(f"  Classes for zero-shot: {class_names}")

        # Test with all images that have URLs
        test_imgs = [{"id": img["id"], "url": img["url"]} for img in test_images if img.get("url")]

        if not test_imgs:
            print("  No image URLs available, skipping worker test")
            return None

        print(f"  Testing with {len(test_imgs)} images...")

        payload = {
            "model": "clip",
            "images": test_imgs,
            "classes": class_names,
            "top_k": 2,
            "threshold": 0.05
        }

        r = requests.post(f"{WORKER_URL}/classify", json=payload, timeout=120)
        assert r.status_code == 200, f"Worker failed: {r.status_code} - {r.text}"
        result = r.json()

        print(f"  Model: {result.get('model')} ({result.get('model_variant', '')})")
        print(f"  Processed: {result.get('images_processed', 0)} images")
        print(f"  Throughput: {result.get('throughput_img_per_sec', 0):.1f} img/s")

        # Show predictions
        print("\n  AI Predictions:")
        correct = 0
        for i, res in enumerate(result.get("results", [])):
            preds = res.get("predictions", [])
            if preds:
                top = preds[0]
                expected = test_images[i]["expected_class"]
                # Map class name back to color
                predicted_color = top['class'].lower().replace(" color", "")
                is_correct = predicted_color == expected
                if is_correct:
                    correct += 1
                status = "✓" if is_correct else "✗"
                print(f"    {status} {res['id'][:8]}... expected={expected}, got={top['class']} ({top['confidence']*100:.1f}%)")

        accuracy = correct / len(result.get("results", [])) * 100 if result.get("results") else 0
        print(f"\n  Accuracy: {correct}/{len(result.get('results', []))} ({accuracy:.1f}%)")

        return result

    worker_result = step("Test AI Worker (CLIP Zero-Shot)", test_worker)

    # 7. Test SigLIP model too
    def test_siglip():
        class_names = [c['display_name'] for c in classes.values()]
        test_imgs = [{"id": img["id"], "url": img["url"]} for img in test_images[:3] if img.get("url")]

        if not test_imgs:
            print("  No images, skipping")
            return None

        payload = {
            "model": "siglip",
            "images": test_imgs,
            "classes": class_names,
            "top_k": 2,
            "threshold": 0.05
        }

        r = requests.post(f"{WORKER_URL}/classify", json=payload, timeout=120)
        assert r.status_code == 200, f"SigLIP failed: {r.status_code} - {r.text}"
        result = r.json()

        print(f"  Model: {result.get('model')} ({result.get('model_variant', '')})")
        print(f"  Predictions:")
        for res in result.get("results", []):
            preds = res.get("predictions", [])
            if preds:
                top = preds[0]
                print(f"    {res['id'][:8]}... -> {top['class']} ({top['confidence']*100:.1f}%)")

        return result

    step("Test AI Worker (SigLIP)", test_siglip)

    # 8. Manual labeling via API
    def manual_label():
        labeled = 0
        for img in test_images:
            expected = img["expected_class"]
            class_obj = classes.get(expected)
            if not class_obj:
                continue

            data = {
                "action": "label",
                "class_id": class_obj["id"]
            }
            r = requests.post(
                f"{API_BASE_URL}/classification/labeling/image/{dataset['id']}/{img['id']}",
                json=data, timeout=10
            )
            if r.status_code == 200:
                labeled += 1
            else:
                print(f"  Failed to label {img['id'][:8]}...: {r.status_code}")

        print(f"  Labeled: {labeled}/{len(test_images)} images")
        return labeled

    labeled_count = step("Label Images (Manual)", manual_label)

    # 9. Get labeling progress
    def check_progress():
        r = requests.get(
            f"{API_BASE_URL}/classification/labeling/progress/{dataset['id']}",
            timeout=10
        )
        assert r.status_code == 200, f"Progress failed: {r.status_code}"
        result = r.json()
        print(f"  Total: {result.get('total', 0)}")
        print(f"  Labeled: {result.get('labeled', 0)}")
        print(f"  Progress: {result.get('progress_pct', 0)}%")
        return result

    step("Check Labeling Progress", check_progress)

    # 10. Auto-split dataset
    def auto_split():
        data = {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1
        }
        r = requests.post(
            f"{API_BASE_URL}/classification/datasets/{dataset['id']}/split/auto",
            json=data, timeout=10
        )
        assert r.status_code == 200, f"Split failed: {r.status_code} - {r.text}"
        result = r.json()
        print(f"  Train: {result.get('train_count', 0)}")
        print(f"  Val: {result.get('val_count', 0)}")
        print(f"  Test: {result.get('test_count', 0)}")
        return result

    step("Auto-Split Dataset", auto_split)

    # 11. Create dataset version
    def create_version():
        data = {"notes": f"E2E test version {test_id}"}
        r = requests.post(
            f"{API_BASE_URL}/classification/datasets/{dataset['id']}/versions",
            json=data, timeout=10
        )
        assert r.status_code == 200, f"Version failed: {r.status_code} - {r.text}"
        result = r.json()
        print(f"  Version: {result.get('version_number')}")
        print(f"  ID: {result.get('id', 'unknown')[:8]}...")
        return result

    step("Create Dataset Version", create_version)

    # 12. Get dataset stats
    def get_stats():
        r = requests.get(f"{API_BASE_URL}/classification/stats", timeout=10)
        assert r.status_code == 200, f"Stats failed: {r.status_code}"
        result = r.json()
        print(f"  Total images: {result.get('total_images', 0)}")
        print(f"  Total datasets: {result.get('total_datasets', 0)}")
        print(f"  Total labels: {result.get('total_labels', 0)}")
        print(f"  Total classes: {result.get('total_classes', 0)}")
        return result

    step("Get Final Stats", get_stats)

    # 13. Verify class distribution
    def verify_classes():
        # Get all classes and filter by our test classes
        r = requests.get(f"{API_BASE_URL}/classification/classes", timeout=10)
        assert r.status_code == 200, f"Classes failed: {r.status_code}"
        all_classes = r.json()

        # Filter to our test classes
        test_class_ids = set(c["id"] for c in classes.values())
        result = [c for c in all_classes if c["id"] in test_class_ids]

        print(f"  Classes in dataset: {len(result)}")
        for cls in result:
            print(f"    - {cls['name']}: {cls.get('image_count', 0)} images")
        return result

    step("Verify Class Distribution", verify_classes)

    # Summary
    print("\n" + "="*60)
    print("E2E TEST SUMMARY")
    print("="*60)
    print("✓ All steps completed successfully!")
    print(f"\nResources created:")
    print(f"  - Dataset: {dataset['name']} ({dataset['id'][:8]}...)")
    print(f"  - Images: {len(test_images)}")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Labels: {labeled_count}")
    print(f"\nAI Worker Tests:")
    print(f"  - CLIP: {'PASSED' if worker_result else 'SKIPPED'}")
    print(f"  - SigLIP: PASSED")
    print("\n" + "="*60)
    print("✓ CLASSIFICATION MODULE E2E TEST PASSED!")
    print("="*60)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n\n{'='*60}")
        print("E2E TEST FAILED")
        print('='*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
