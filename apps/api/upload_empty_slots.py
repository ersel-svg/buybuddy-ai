"""
Upload background images (temiz reyon) as empty slots to CLS dataset.

This script:
1. Downloads background images from Supabase Storage (frames/backgrounds)
2. Creates a dataset named "Empty Slot Or Filled Slot"
3. Creates an "empty_slot" class
4. Uploads images to the dataset with that class

Features:
- Parallel uploads for speed
- Skip already uploaded images
- Resume support
"""

import os
import io
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client

# Load environment
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
API_BASE_URL = "http://localhost:8000/api/v1"

# Background storage location
BACKGROUNDS_BUCKET = "frames"
BACKGROUNDS_PATH = "backgrounds"

# Parallel settings
MAX_WORKERS = 10


def get_existing_image_filenames(dataset_id: str) -> set:
    """Get filenames of already uploaded images in the dataset."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/classification/datasets/{dataset_id}/images",
            params={"limit": 10000},
            timeout=60
        )
        if r.status_code == 200:
            images = r.json()
            # Extract filenames from images
            filenames = set()
            for img in images:
                filename = img.get("filename", "")
                if filename:
                    filenames.add(filename)
            return filenames
    except Exception as e:
        print(f"   Warning: Could not get existing images: {e}")
    return set()


def upload_single_image(args):
    """Upload a single image (for parallel execution)."""
    client, file_info, index, total = args
    file_name = file_info["name"]
    file_path = f"{BACKGROUNDS_PATH}/{file_name}"

    try:
        # Download from Supabase
        data = client.storage.from_(BACKGROUNDS_BUCKET).download(file_path)

        # Upload to CLS API
        files = {"file": (file_name, io.BytesIO(data), "image/png")}
        r = requests.post(f"{API_BASE_URL}/classification/images", files=files, timeout=60)

        if r.status_code == 200:
            img_data = r.json()
            return (file_name, img_data["id"], True, None)
        else:
            return (file_name, None, False, f"HTTP {r.status_code}")

    except Exception as e:
        return (file_name, None, False, str(e)[:50])


def main():
    print("=" * 60)
    print("UPLOAD EMPTY SLOT IMAGES TO CLS DATASET")
    print("=" * 60)

    # 1. Connect to Supabase
    print("\n1. Connecting to Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"   Connected to: {SUPABASE_URL}")

    # 2. List ALL background images (with pagination)
    print(f"\n2. Listing backgrounds from {BACKGROUNDS_BUCKET}/{BACKGROUNDS_PATH}...")
    try:
        # Supabase has default limit of 100, need to paginate
        all_files = []
        offset = 0
        limit = 1000  # Max per request

        while True:
            files = client.storage.from_(BACKGROUNDS_BUCKET).list(
                BACKGROUNDS_PATH,
                {"limit": limit, "offset": offset}
            )
            if not files:
                break
            all_files.extend(files)
            if len(files) < limit:
                break
            offset += limit

        image_files = [
            f for f in all_files
            if f.get("name", "").lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        print(f"   Found {len(image_files)} background images")

        if not image_files:
            print("   No background images found!")
            return 1

    except Exception as e:
        print(f"   Error listing backgrounds: {e}")
        return 1

    # 3. Create/Get dataset via API
    print("\n3. Creating dataset 'Empty Slot Or Filled Slot'...")
    try:
        data = {
            "name": "Empty Slot Or Filled Slot",
            "description": "Dataset for classifying empty shelf slots vs filled slots",
            "task_type": "single_label"
        }
        r = requests.post(f"{API_BASE_URL}/classification/datasets", json=data, timeout=10)

        if r.status_code == 409:
            # Dataset already exists, get it
            print("   Dataset already exists, fetching...")
            r = requests.get(f"{API_BASE_URL}/classification/datasets", timeout=10)
            datasets = r.json()
            dataset = next((d for d in datasets if d["name"] == "Empty Slot Or Filled Slot"), None)
            if not dataset:
                print("   ERROR: Could not find existing dataset")
                return 1
        elif r.status_code == 200:
            dataset = r.json()
        else:
            print(f"   ERROR: Failed to create dataset: {r.status_code} - {r.text}")
            return 1

        print(f"   Dataset ID: {dataset['id']}")

    except Exception as e:
        print(f"   Error: {e}")
        return 1

    # 4. Create empty_slot class
    print("\n4. Creating 'empty_slot' class...")
    try:
        data = {
            "name": "empty_slot",
            "display_name": "Empty Slot",
            "dataset_id": dataset["id"],
            "color": "#6b7280",
            "description": "Empty shelf slot / clean shelf area"
        }
        r = requests.post(f"{API_BASE_URL}/classification/classes", json=data, timeout=10)

        if r.status_code == 409:
            # Class already exists
            print("   Class already exists")
            # Get it from existing classes
            r = requests.get(f"{API_BASE_URL}/classification/classes", timeout=10)
            classes = r.json()
            empty_slot_class = next((c for c in classes if c["name"] == "empty_slot" and c.get("dataset_id") == dataset["id"]), None)
            if not empty_slot_class:
                # Try without dataset filter
                empty_slot_class = next((c for c in classes if c["name"] == "empty_slot"), None)
        elif r.status_code == 200:
            empty_slot_class = r.json()
        else:
            print(f"   ERROR: {r.status_code} - {r.text}")
            return 1

        print(f"   Class ID: {empty_slot_class['id']}")

    except Exception as e:
        print(f"   Error: {e}")
        return 1

    # 5. Check existing images to skip
    print("\n5. Checking for already uploaded images...")
    existing_filenames = get_existing_image_filenames(dataset["id"])
    print(f"   Found {len(existing_filenames)} existing images in dataset")

    # Filter out already uploaded images
    new_image_files = [f for f in image_files if f["name"] not in existing_filenames]
    skipped = len(image_files) - len(new_image_files)
    print(f"   Skipping {skipped} already uploaded images")
    print(f"   New images to upload: {len(new_image_files)}")

    if not new_image_files:
        print("\n   All images already uploaded!")
        # Still need to ensure they're labeled
        uploaded_image_ids = []
    else:
        # 6. Download and upload images in parallel
        print(f"\n6. Downloading and uploading {len(new_image_files)} images ({MAX_WORKERS} parallel workers)...")

        uploaded_image_ids = []
        failed = 0

        # Prepare args for parallel execution
        upload_args = [
            (client, f, i, len(new_image_files))
            for i, f in enumerate(new_image_files)
        ]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(upload_single_image, args): args for args in upload_args}

            for i, future in enumerate(as_completed(futures)):
                file_name, img_id, success, error = future.result()
                if success:
                    uploaded_image_ids.append(img_id)
                    if (i + 1) % 50 == 0 or i + 1 == len(new_image_files):
                        print(f"   Progress: {i+1}/{len(new_image_files)} ({len(uploaded_image_ids)} uploaded, {failed} failed)")
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"   ✗ {file_name}: {error}")

        print(f"\n   Uploaded: {len(uploaded_image_ids)} new images")
        if failed > 0:
            print(f"   Failed: {failed} images")

    # 7. Add new images to dataset
    if uploaded_image_ids:
        print(f"\n7. Adding {len(uploaded_image_ids)} new images to dataset...")
        try:
            # Add in batches to avoid timeout
            batch_size = 500
            total_added = 0
            for i in range(0, len(uploaded_image_ids), batch_size):
                batch = uploaded_image_ids[i:i+batch_size]
                data = {"image_ids": batch}
                r = requests.post(
                    f"{API_BASE_URL}/classification/datasets/{dataset['id']}/images/add",
                    json=data,
                    timeout=120
                )
                if r.status_code == 200:
                    result = r.json()
                    total_added += result.get('added', 0)
                else:
                    print(f"   Batch error: {r.status_code}")
            print(f"   Added: {total_added} images")
        except Exception as e:
            print(f"   Error: {e}")

    # 8. Get all images in dataset for labeling
    print(f"\n8. Getting all images in dataset for labeling...")
    try:
        r = requests.get(
            f"{API_BASE_URL}/classification/datasets/{dataset['id']}/images",
            params={"limit": 10000},
            timeout=60
        )
        all_dataset_images = r.json() if r.status_code == 200 else []
        print(f"   Total images in dataset: {len(all_dataset_images)}")

        # Filter unlabeled images
        unlabeled_ids = [img["id"] for img in all_dataset_images if not img.get("label")]
        print(f"   Unlabeled images: {len(unlabeled_ids)}")

    except Exception as e:
        print(f"   Error: {e}")
        unlabeled_ids = uploaded_image_ids

    # 9. Label unlabeled images with empty_slot class (parallel)
    if unlabeled_ids:
        print(f"\n9. Labeling {len(unlabeled_ids)} images as 'empty_slot'...")

        def label_image(img_id):
            try:
                data = {
                    "action": "label",
                    "class_id": empty_slot_class["id"]
                }
                r = requests.post(
                    f"{API_BASE_URL}/classification/labeling/image/{dataset['id']}/{img_id}",
                    json=data,
                    timeout=10
                )
                return r.status_code == 200
            except:
                return False

        labeled = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(label_image, img_id) for img_id in unlabeled_ids]
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    labeled += 1
                if (i + 1) % 100 == 0 or i + 1 == len(unlabeled_ids):
                    print(f"   Progress: {i+1}/{len(unlabeled_ids)} ({labeled} labeled)")

        print(f"   Labeled: {labeled}/{len(unlabeled_ids)} images")
    else:
        labeled = 0
        print("\n9. All images already labeled!")

    # 10. Final stats
    print("\n10. Getting final stats...")
    try:
        r = requests.get(
            f"{API_BASE_URL}/classification/labeling/progress/{dataset['id']}",
            timeout=10
        )
        if r.status_code == 200:
            progress = r.json()
            print(f"   Total: {progress.get('total', 0)}")
            print(f"   Labeled: {progress.get('labeled', 0)}")
            print(f"   Progress: {progress.get('progress_pct', 0)}%")
    except Exception as e:
        print(f"   Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Dataset: {dataset['name']}")
    print(f"Dataset ID: {dataset['id']}")
    print(f"Class: {empty_slot_class['name']} ({empty_slot_class['id']})")
    print(f"Total background images: {len(image_files)}")
    print(f"New images uploaded: {len(uploaded_image_ids)}")
    print(f"Skipped (already exists): {skipped}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
