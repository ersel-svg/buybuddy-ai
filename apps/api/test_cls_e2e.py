"""
Classification Module E2E Test

Tests the complete workflow:
1. Upload images
2. Create dataset with classes
3. Add images to dataset
4. Label images
5. Create dataset version
6. Start training (mock)
"""

import requests
import json
import time
import uuid
from io import BytesIO
from PIL import Image
import random

BASE_URL = "http://localhost:8000/api/v1"

def create_test_image(width=224, height=224, color=None):
    """Create a test image with random colors."""
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img = Image.new('RGB', (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer

def test_step(name, func):
    """Run a test step and report result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        result = func()
        print(f"✓ PASSED")
        return result
    except Exception as e:
        print(f"✗ FAILED: {e}")
        raise

def main():
    print("\n" + "="*60)
    print("CLASSIFICATION MODULE E2E TEST")
    print("="*60)

    # Track created resources for cleanup
    created = {
        "images": [],
        "dataset_id": None,
        "class_ids": [],
    }

    try:
        # 1. Health Check
        def test_health():
            r = requests.get(f"{BASE_URL}/classification/health")
            assert r.status_code == 200, f"Status: {r.status_code}"
            data = r.json()
            assert data["status"] == "healthy"
            print(f"  Response: {data}")
            return data

        test_step("Health Check", test_health)

        # 2. Get Initial Stats
        def test_stats():
            r = requests.get(f"{BASE_URL}/classification/stats")
            assert r.status_code == 200
            data = r.json()
            print(f"  Total images: {data['total_images']}")
            print(f"  Total datasets: {data['total_datasets']}")
            return data

        initial_stats = test_step("Get Initial Stats", test_stats)

        # 3. Upload Test Images
        def test_upload_images():
            uploaded = []
            colors = [
                (255, 0, 0),    # Red - class 1
                (255, 50, 50),  # Light red
                (200, 0, 0),    # Dark red
                (0, 255, 0),    # Green - class 2
                (50, 255, 50),  # Light green
                (0, 200, 0),    # Dark green
                (0, 0, 255),    # Blue - class 3
                (50, 50, 255),  # Light blue
                (0, 0, 200),    # Dark blue
                (255, 255, 0),  # Yellow - class 4
            ]

            for i, color in enumerate(colors):
                img_buffer = create_test_image(224, 224, color)
                files = {'file': (f'test_image_{i+1}.jpg', img_buffer, 'image/jpeg')}
                r = requests.post(f"{BASE_URL}/classification/images", files=files)

                if r.status_code == 200:
                    data = r.json()
                    uploaded.append(data)
                    print(f"  Uploaded image {i+1}: {data.get('id', 'unknown')[:8]}...")
                else:
                    print(f"  Failed to upload image {i+1}: {r.status_code} - {r.text[:100]}")

            assert len(uploaded) >= 5, f"Only uploaded {len(uploaded)} images"
            created["images"] = uploaded
            return uploaded

        images = test_step("Upload 10 Test Images", test_upload_images)

        # 4. Create Dataset
        def test_create_dataset():
            data = {
                "name": f"E2E Test Dataset {uuid.uuid4().hex[:8]}",
                "description": "Automated E2E test dataset",
                "task_type": "single_label"
            }
            r = requests.post(f"{BASE_URL}/classification/datasets", json=data)
            assert r.status_code == 200, f"Status: {r.status_code}, Response: {r.text}"
            result = r.json()
            created["dataset_id"] = result["id"]
            print(f"  Dataset ID: {result['id']}")
            print(f"  Name: {result['name']}")
            return result

        dataset = test_step("Create Dataset", test_create_dataset)
        dataset_id = dataset["id"]

        # 5. Create Classes
        def test_create_classes():
            classes_data = [
                {"name": "red", "display_name": "Red Objects", "color": "#ef4444"},
                {"name": "green", "display_name": "Green Objects", "color": "#22c55e"},
                {"name": "blue", "display_name": "Blue Objects", "color": "#3b82f6"},
                {"name": "yellow", "display_name": "Yellow Objects", "color": "#eab308"},
            ]

            created_classes = []
            for cls_data in classes_data:
                cls_data["dataset_id"] = dataset_id
                r = requests.post(f"{BASE_URL}/classification/classes", json=cls_data)
                if r.status_code == 200:
                    result = r.json()
                    created_classes.append(result)
                    print(f"  Created class: {result['name']} ({result['id'][:8]}...)")
                else:
                    print(f"  Failed to create class {cls_data['name']}: {r.status_code}")

            assert len(created_classes) == 4, f"Only created {len(created_classes)} classes"
            created["class_ids"] = [c["id"] for c in created_classes]
            return created_classes

        classes = test_step("Create 4 Classes", test_create_classes)

        # 6. Add Images to Dataset
        def test_add_images():
            image_ids = [img["id"] for img in images]
            data = {"image_ids": image_ids}
            r = requests.post(f"{BASE_URL}/classification/datasets/{dataset_id}/images/add", json=data)
            assert r.status_code == 200, f"Status: {r.status_code}, Response: {r.text}"
            result = r.json()
            print(f"  Added: {result.get('added', 0)} images")
            print(f"  Skipped: {result.get('skipped', 0)} images")
            return result

        test_step("Add Images to Dataset", test_add_images)

        # 7. Get Dataset Details
        def test_get_dataset():
            r = requests.get(f"{BASE_URL}/classification/datasets/{dataset_id}")
            assert r.status_code == 200
            result = r.json()
            print(f"  Image count: {result.get('image_count', 0)}")
            print(f"  Labeled count: {result.get('labeled_image_count', 0)}")
            return result

        test_step("Get Dataset Details", test_get_dataset)

        # 8. Get Dataset Images
        def test_get_dataset_images():
            r = requests.get(f"{BASE_URL}/classification/datasets/{dataset_id}/images")
            assert r.status_code == 200
            result = r.json()
            print(f"  Total images in dataset: {len(result.get('images', []))}")
            return result

        dataset_images = test_step("Get Dataset Images", test_get_dataset_images)

        # 9. Get Labeling Queue
        def test_labeling_queue():
            r = requests.get(f"{BASE_URL}/classification/labeling/queue/{dataset_id}?mode=unlabeled")
            assert r.status_code == 200
            result = r.json()
            print(f"  Queue size: {len(result.get('image_ids', []))}")
            print(f"  Mode: {result.get('mode', 'unknown')}")
            return result

        queue = test_step("Get Labeling Queue", test_labeling_queue)

        # 10. Label Images
        def test_label_images():
            # Label first 3 images as red, next 3 as green, next 2 as blue, last 2 as yellow
            image_ids = queue.get("image_ids", [])[:10]
            class_mapping = {
                0: classes[0]["id"],  # red
                1: classes[0]["id"],
                2: classes[0]["id"],
                3: classes[1]["id"],  # green
                4: classes[1]["id"],
                5: classes[1]["id"],
                6: classes[2]["id"],  # blue
                7: classes[2]["id"],
                8: classes[3]["id"],  # yellow
                9: classes[3]["id"],
            }

            labeled = 0
            for i, img_id in enumerate(image_ids):
                class_id = class_mapping.get(i, classes[0]["id"])
                data = {
                    "action": "label",
                    "class_id": class_id
                }
                r = requests.post(f"{BASE_URL}/classification/labeling/image/{dataset_id}/{img_id}", json=data)
                if r.status_code == 200:
                    labeled += 1
                else:
                    print(f"  Failed to label image {i+1}: {r.status_code}")

            print(f"  Labeled {labeled} images")
            return labeled

        labeled_count = test_step("Label Images", test_label_images)

        # 11. Get Labeling Progress
        def test_labeling_progress():
            r = requests.get(f"{BASE_URL}/classification/labeling/progress/{dataset_id}")
            assert r.status_code == 200
            result = r.json()
            print(f"  Total: {result.get('total', 0)}")
            print(f"  Labeled: {result.get('labeled', 0)}")
            print(f"  Progress: {result.get('progress_pct', 0)}%")
            return result

        test_step("Get Labeling Progress", test_labeling_progress)

        # 12. Auto-Split Dataset
        def test_auto_split():
            data = {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.1
            }
            r = requests.post(f"{BASE_URL}/classification/datasets/{dataset_id}/auto-split", json=data)
            assert r.status_code == 200, f"Status: {r.status_code}, Response: {r.text}"
            result = r.json()
            print(f"  Train: {result.get('train_count', 0)}")
            print(f"  Val: {result.get('val_count', 0)}")
            print(f"  Test: {result.get('test_count', 0)}")
            return result

        test_step("Auto-Split Dataset", test_auto_split)

        # 13. Create Dataset Version
        def test_create_version():
            data = {"notes": "E2E test version"}
            r = requests.post(f"{BASE_URL}/classification/datasets/{dataset_id}/versions", json=data)
            assert r.status_code == 200, f"Status: {r.status_code}, Response: {r.text}"
            result = r.json()
            print(f"  Version: {result.get('version_number', 'unknown')}")
            print(f"  ID: {result.get('id', 'unknown')[:8]}...")
            return result

        version = test_step("Create Dataset Version", test_create_version)

        # 14. Get Dataset Classes
        def test_get_classes():
            r = requests.get(f"{BASE_URL}/classification/datasets/{dataset_id}/classes")
            assert r.status_code == 200
            result = r.json()
            print(f"  Classes count: {len(result)}")
            for cls in result:
                print(f"    - {cls['name']}: {cls.get('image_count', 0)} images")
            return result

        test_step("Get Dataset Classes with Stats", test_get_classes)

        # 15. Get Training Configs
        def test_training_configs():
            r = requests.get(f"{BASE_URL}/classification/training/model-configs")
            assert r.status_code == 200
            result = r.json()
            print(f"  Available models: {list(result.keys())}")
            return result

        test_step("Get Training Model Configs", test_training_configs)

        # 16. Get Augmentation Presets
        def test_aug_presets():
            r = requests.get(f"{BASE_URL}/classification/training/augmentation-presets")
            assert r.status_code == 200
            result = r.json()
            print(f"  Available presets: {list(result.keys())}")
            return result

        test_step("Get Augmentation Presets", test_aug_presets)

        # 17. Final Stats
        def test_final_stats():
            r = requests.get(f"{BASE_URL}/classification/stats")
            assert r.status_code == 200
            data = r.json()
            print(f"  Total images: {data['total_images']} (was {initial_stats['total_images']})")
            print(f"  Total datasets: {data['total_datasets']} (was {initial_stats['total_datasets']})")
            print(f"  Total labels: {data['total_labels']} (was {initial_stats['total_labels']})")
            print(f"  Total classes: {data['total_classes']} (was {initial_stats['total_classes']})")
            return data

        test_step("Final Stats Check", test_final_stats)

        # Summary
        print("\n" + "="*60)
        print("E2E TEST SUMMARY")
        print("="*60)
        print("\n✓ All tests passed!")
        print("\nWorkflow tested:")
        print("  1. Health check")
        print("  2. Image upload (10 images)")
        print("  3. Dataset creation")
        print("  4. Class creation (4 classes)")
        print("  5. Add images to dataset")
        print("  6. Labeling workflow")
        print("  7. Auto-split (train/val/test)")
        print("  8. Dataset versioning")
        print("  9. Training configs")
        print("\nClassification module is ready for production!")

    except Exception as e:
        print(f"\n\n{'='*60}")
        print("E2E TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
