"""Supabase client for worker - downloads dataset and uploads results."""

import os
import io
import json
from pathlib import Path
from supabase import create_client, Client
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from PIL import Image

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")

# Background images settings
BACKGROUNDS_BUCKET = "frames"
BACKGROUNDS_PATH = "backgrounds"

# Parallel download/upload settings
MAX_WORKERS = 10  # Concurrent connections


def get_supabase() -> Client:
    """Get Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class DatasetDownloader:
    """Downloads dataset from Supabase Storage for processing."""

    def __init__(self, local_base: Path = Path("/tmp/datasets")):
        self.client = get_supabase()
        self.local_base = local_base
        self.local_base.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, dataset_id: str) -> Path:
        """
        Download all product images for a dataset.

        Structure created:
        /tmp/datasets/{dataset_id}/
          train/
            {product_id_1}/
              frame_0000.png
              frame_0001.png
              real/
                real_image_1.jpg
            {product_id_2}/
              ...
        """
        print(f"\n📥 Downloading dataset: {dataset_id}")

        # 1. Get dataset products from DB
        response = self.client.table("dataset_products").select(
            "product_id, products(id, barcode, frames_path)"
        ).eq("dataset_id", dataset_id).execute()

        products = response.data
        if not products:
            raise ValueError(f"Dataset {dataset_id} has no products")

        print(f"   Found {len(products)} products")

        # 2. Create local directory structure
        dataset_dir = self.local_base / dataset_id / "train"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 3. Download each product's frames
        for item in products:
            product = item.get("products", {})
            product_id = product.get("id")
            barcode = product.get("barcode", product_id)
            frames_path = product.get("frames_path")

            if not frames_path:
                print(f"   ⚠️ No frames for product {barcode}")
                continue

            # Create product directory (use barcode as folder name for compatibility)
            product_dir = dataset_dir / barcode
            product_dir.mkdir(parents=True, exist_ok=True)

            # Download frames from storage
            self._download_product_frames(frames_path, product_dir)

            # Download real images if exist
            real_dir = product_dir / "real"
            real_dir.mkdir(exist_ok=True)
            self._download_real_images(product_id, real_dir)

        print(f"   ✅ Dataset downloaded to: {dataset_dir.parent}")
        return dataset_dir.parent

    def _download_product_frames(self, frames_path: str, target_dir: Path):
        """Download frames from Supabase Storage with parallel downloads."""
        try:
            # frames_path format: "{supabase_url}/storage/v1/object/public/frames/{barcode}/"
            # or just "{barcode}/" if relative
            bucket = "frames"

            # Extract folder from path
            if "/" in frames_path:
                folder = frames_path.rstrip("/").split("/")[-1]
            else:
                folder = frames_path.rstrip("/")

            # List files in bucket
            files = self.client.storage.from_(bucket).list(folder)
            image_files = [
                f for f in files
                if f.get("name", "").endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ]

            if not image_files:
                print(f"   ⚠️ No image files found in {folder}")
                return

            print(f"   📥 Downloading {len(image_files)} frames (parallel)...")

            def download_single(file_info: dict) -> Tuple[str, bool]:
                """Download a single file."""
                file_name = file_info['name']
                file_path = f"{folder}/{file_name}"
                try:
                    data = self.client.storage.from_(bucket).download(file_path)
                    local_path = target_dir / file_name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    return file_name, True
                except Exception as e:
                    return file_name, False

            # Parallel download
            success_count = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(download_single, f): f for f in image_files}
                for future in as_completed(futures):
                    name, success = future.result()
                    if success:
                        success_count += 1

            print(f"   ✅ Downloaded {success_count}/{len(image_files)} frames")

        except Exception as e:
            print(f"   ⚠️ Frame download error: {e}")

    def _download_real_images(self, product_id: str, target_dir: Path):
        """Download matched real images for a product from product_images table."""
        try:
            # Get real images from unified product_images table
            response = self.client.table("product_images").select("*").eq(
                "product_id", product_id
            ).eq("image_type", "real").execute()

            downloaded = 0
            for idx, img in enumerate(response.data):
                image_url = img.get("image_url")
                image_path = img.get("image_path")

                # Try to download from URL first (for matching images)
                if image_url and image_url.startswith("http"):
                    try:
                        resp = httpx.get(image_url, timeout=30)
                        if resp.status_code == 200:
                            # Generate filename from URL or use index
                            ext = Path(image_url).suffix or ".jpg"
                            local_path = target_dir / f"real_{idx:04d}{ext}"
                            with open(local_path, 'wb') as fp:
                                fp.write(resp.content)
                            downloaded += 1
                            continue
                    except Exception:
                        pass

                # Fallback: try to download from frames bucket storage
                if image_path:
                    try:
                        data = self.client.storage.from_("frames").download(image_path)
                        local_path = target_dir / Path(image_path).name
                        with open(local_path, 'wb') as fp:
                            fp.write(data)
                        downloaded += 1
                    except Exception:
                        pass

            if downloaded > 0:
                print(f"      Downloaded {downloaded} real images")

        except Exception as e:
            print(f"   ⚠️ Real images download error: {e}")


class ResultUploader:
    """Uploads augmented images back to Supabase Storage."""

    def __init__(self):
        self.client = get_supabase()

    def _get_barcode_to_product_map(self, dataset_id: str) -> dict:
        """Get mapping of barcode -> product_id for dataset products."""
        try:
            response = self.client.table("dataset_products").select(
                "product_id, products(id, barcode)"
            ).eq("dataset_id", dataset_id).execute()

            mapping = {}
            for item in response.data:
                product = item.get("products", {})
                barcode = product.get("barcode")
                product_id = product.get("id")
                if barcode and product_id:
                    mapping[barcode] = product_id
            return mapping
        except Exception as e:
            print(f"   ⚠️ Failed to get product mapping: {e}")
            return {}

    def upload_augmented(self, dataset_dir, dataset_id: str) -> dict:
        """
        Upload augmented images to Supabase Storage with parallel uploads.
        Images are now stored per-product in the frames bucket.

        Returns upload statistics.
        """
        print(f"\n📤 Uploading augmented images (parallel)...")

        # Use frames bucket with product-based paths
        bucket = "frames"

        # Get barcode -> product_id mapping
        barcode_map = self._get_barcode_to_product_map(dataset_id)
        if not barcode_map:
            print("   ⚠️ No product mapping found")
            return {"syn_uploaded": 0, "real_uploaded": 0, "db_inserted": 0}

        # Ensure Path object
        dataset_dir = Path(dataset_dir) if isinstance(dataset_dir, str) else dataset_dir

        # Walk through dataset directory
        train_dir = dataset_dir / "train"
        if not train_dir.exists():
            train_dir = dataset_dir

        # Collect all files to upload: (local_path, storage_path, img_type, product_id, source)
        upload_tasks: List[Tuple[Path, str, str, str, str]] = []

        for product_dir in train_dir.iterdir():
            if not product_dir.is_dir():
                continue

            barcode = product_dir.name
            product_id = barcode_map.get(barcode)

            if not product_id:
                print(f"   ⚠️ No product_id found for barcode {barcode}")
                continue

            # Synthetic augmented images (syn_*.jpg)
            for img in product_dir.glob("syn_*.jpg"):
                # New path: frames/{product_id}/augmented/syn_light_001.jpg
                storage_path = f"{product_id}/augmented/{img.name}"
                # Determine source from filename (syn_light_, syn_heavy_, etc.)
                source = "aug_syn_light" if "light" in img.name else "aug_syn_heavy" if "heavy" in img.name else "aug_synthetic"
                upload_tasks.append((img, storage_path, "augmented", product_id, source))

            # Real augmented images (real/*_aug_*.jpg)
            real_dir = product_dir / "real"
            if real_dir.exists():
                for img in real_dir.glob("*_aug_*.jpg"):
                    # New path: frames/{product_id}/augmented/real_aug_001.jpg
                    storage_path = f"{product_id}/augmented/{img.name}"
                    upload_tasks.append((img, storage_path, "augmented", product_id, "aug_real"))

        if not upload_tasks:
            print("   ⚠️ No augmented images to upload")
            return {"syn_uploaded": 0, "real_uploaded": 0, "db_inserted": 0}

        print(f"   📤 Uploading {len(upload_tasks)} images...")

        def upload_single(task: Tuple[Path, str, str, str, str]) -> Tuple[str, str, str, str, bool]:
            """Upload a single file. Returns (storage_path, img_type, product_id, source, success)"""
            local_path, storage_path, img_type, product_id, source = task
            try:
                with open(local_path, 'rb') as f:
                    data = f.read()
                    try:
                        self.client.storage.from_(bucket).upload(storage_path, data)
                    except Exception:
                        # File might exist, try update
                        self.client.storage.from_(bucket).update(storage_path, data)
                return storage_path, img_type, product_id, source, True
            except Exception as e:
                return storage_path, img_type, product_id, source, False

        # Parallel upload and collect successful uploads for DB insertion
        stats = {"syn_uploaded": 0, "real_uploaded": 0, "db_inserted": 0}
        successful_uploads: List[Tuple[str, str, str, str]] = []  # (storage_path, img_type, product_id, source)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(upload_single, task) for task in upload_tasks]
            for future in as_completed(futures):
                storage_path, img_type, product_id, source, success = future.result()
                if success:
                    successful_uploads.append((storage_path, img_type, product_id, source))
                    if "real" in source:
                        stats["real_uploaded"] += 1
                    else:
                        stats["syn_uploaded"] += 1

        print(f"   ✅ Uploaded: {stats['syn_uploaded']} syn + {stats['real_uploaded']} real")

        # Insert DB records for uploaded images
        if successful_uploads:
            db_count = self._insert_augmented_records(successful_uploads, bucket)
            stats["db_inserted"] = db_count

        return stats

    def _insert_augmented_records(
        self,
        uploads: List[Tuple[str, str, str, str]],
        bucket: str,
    ) -> int:
        """Insert augmented image records into product_images table."""
        try:
            # Group by product_id for efficient deletion of old records
            product_ids = set(u[2] for u in uploads)

            # Delete existing augmented frames for these products
            for product_id in product_ids:
                self.client.table("product_images").delete().eq(
                    "product_id", product_id
                ).eq("image_type", "augmented").execute()

            # Prepare records
            records = []
            for storage_path, img_type, product_id, source in uploads:
                image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"
                records.append({
                    "product_id": product_id,
                    "image_type": "augmented",
                    "source": source,
                    "image_path": storage_path,
                    "image_url": image_url,
                })

            # Batch insert (in chunks to avoid timeout)
            chunk_size = 500
            inserted = 0
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                self.client.table("product_images").insert(chunk).execute()
                inserted += len(chunk)

            print(f"   📝 Inserted {inserted} augmented records into database")
            return inserted

        except Exception as e:
            print(f"   ⚠️ Failed to insert augmented records: {e}")
            return 0

    def update_job_progress(self, job_id: str, progress: int, current_step: str):
        """Update job progress in database."""
        try:
            self.client.table("jobs").update({
                "progress": progress,
                "current_step": current_step,
            }).eq("runpod_job_id", job_id).execute()
        except Exception as e:
            print(f"   ⚠️ Progress update error: {e}")

    def update_job_status(self, job_id: str, status: str, result: dict = None, error: str = None):
        """Update job status in database (completed/failed)."""
        try:
            update_data = {"status": status}
            if status == "completed":
                update_data["progress"] = 100
            if result:
                update_data["result"] = result
            if error:
                update_data["error"] = error

            self.client.table("jobs").update(update_data).eq("runpod_job_id", job_id).execute()
            print(f"   ✅ Job status updated to: {status}")
        except Exception as e:
            print(f"   ⚠️ Status update error: {e}")

    def send_callback(self, result: dict):
        """Send completion callback to backend."""
        if not CALLBACK_URL:
            return
        try:
            httpx.post(
                f"{CALLBACK_URL}/api/v1/webhooks/runpod",
                json=result,
                timeout=30,
            )
        except Exception as e:
            print(f"   ⚠️ Callback error: {e}")


class BackgroundDownloader:
    """Downloads background images from Supabase Storage for augmentation."""

    def __init__(self):
        self.client = get_supabase()
        self._cached_backgrounds: Optional[List[Image.Image]] = None

    def download_backgrounds(self, max_backgrounds: int = 0) -> List[Image.Image]:
        """
        Download background images from Supabase Storage.
        Returns list of PIL Image objects for use with augmentor.load_backgrounds_from_pil()

        Args:
            max_backgrounds: Maximum number of backgrounds to download (0 = all)

        Returns:
            List of PIL Image objects
        """
        # Return cached if available
        if self._cached_backgrounds is not None:
            if max_backgrounds > 0:
                return self._cached_backgrounds[:max_backgrounds]
            return self._cached_backgrounds

        print(f"\n🖼️ Downloading background images from Supabase...")

        try:
            # List files in backgrounds folder
            files = self.client.storage.from_(BACKGROUNDS_BUCKET).list(BACKGROUNDS_PATH)
            image_files = [
                f for f in files
                if f.get("name", "").endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ]

            if not image_files:
                print("   ⚠️ No background images found")
                return []

            # Limit if specified
            if max_backgrounds > 0:
                image_files = image_files[:max_backgrounds]

            print(f"   📥 Downloading {len(image_files)} backgrounds (parallel)...")

            def download_single(file_info: dict) -> Tuple[str, Optional[Image.Image]]:
                """Download a single background image."""
                file_name = file_info['name']
                file_path = f"{BACKGROUNDS_PATH}/{file_name}"
                try:
                    data = self.client.storage.from_(BACKGROUNDS_BUCKET).download(file_path)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    return file_name, img
                except Exception as e:
                    return file_name, None

            # Parallel download
            backgrounds = []
            success_count = 0

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(download_single, f): f for f in image_files}
                for future in as_completed(futures):
                    name, img = future.result()
                    if img is not None:
                        backgrounds.append(img)
                        success_count += 1

            print(f"   ✅ Downloaded {success_count}/{len(image_files)} backgrounds")

            # Cache for reuse
            self._cached_backgrounds = backgrounds
            return backgrounds

        except Exception as e:
            print(f"   ⚠️ Background download error: {e}")
            return []

    def clear_cache(self):
        """Clear cached backgrounds to free memory."""
        self._cached_backgrounds = None


class NeighborProductDownloader:
    """Downloads neighbor product images for shelf composition."""

    def __init__(self, local_base: Path = Path("/tmp/neighbors")):
        self.client = get_supabase()
        self.local_base = local_base
        self.local_base.mkdir(parents=True, exist_ok=True)

    def download_neighbor_images(
        self, dataset_id: str, max_neighbors: int = 50
    ) -> List[str]:
        """
        Download neighbor product images from the same dataset.
        Returns list of local file paths for use with augmentor.set_neighbor_paths()

        Args:
            dataset_id: Dataset ID to get products from
            max_neighbors: Maximum number of neighbor images to download

        Returns:
            List of local file paths
        """
        print(f"\n👥 Downloading neighbor product images...")

        try:
            # Get products from dataset
            response = self.client.table("dataset_products").select(
                "product_id, products(id, barcode)"
            ).eq("dataset_id", dataset_id).execute()

            if not response.data:
                print("   ⚠️ No products found in dataset")
                return []

            product_ids = [
                item.get("products", {}).get("id")
                for item in response.data
                if item.get("products", {}).get("id")
            ]

            # Get synthetic frames for these products
            images_response = self.client.table("product_images").select(
                "product_id, image_path"
            ).in_("product_id", product_ids).eq(
                "image_type", "synthetic"
            ).limit(max_neighbors).execute()

            if not images_response.data:
                print("   ⚠️ No synthetic images found")
                return []

            print(f"   📥 Downloading {len(images_response.data)} neighbor images...")

            def download_single(img_data: dict) -> Tuple[str, Optional[str]]:
                """Download a single neighbor image."""
                image_path = img_data.get("image_path")
                if not image_path:
                    return "", None
                try:
                    data = self.client.storage.from_(BACKGROUNDS_BUCKET).download(image_path)
                    local_path = self.local_base / Path(image_path).name
                    with open(local_path, 'wb') as f:
                        f.write(data)
                    return str(local_path), str(local_path)
                except Exception:
                    return image_path, None

            # Parallel download
            local_paths = []
            success_count = 0

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(download_single, img)
                    for img in images_response.data
                ]
                for future in as_completed(futures):
                    original, local = future.result()
                    if local:
                        local_paths.append(local)
                        success_count += 1

            print(f"   ✅ Downloaded {success_count} neighbor images")
            return local_paths

        except Exception as e:
            print(f"   ⚠️ Neighbor download error: {e}")
            return []
