"""Supabase client for worker - downloads dataset and uploads results."""

import os
import io
import json
import time
import random
from pathlib import Path
from supabase import create_client, Client
from typing import Optional, List, Tuple, Callable, TypeVar
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

# Retry settings
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = [1, 2, 4]  # Exponential backoff base times
JITTER_PERCENT = 0.3  # 30% jitter

T = TypeVar('T')


def retry_with_backoff(
    operation: Callable[[], T],
    operation_name: str = "operation",
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Retry an operation with exponential backoff and jitter.

    Args:
        operation: Callable to execute
        operation_name: Name for logging
        max_retries: Maximum retry attempts

    Returns:
        Result of the operation

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            last_exception = e

            if attempt < max_retries - 1:
                # Calculate backoff with jitter
                base_wait = BASE_BACKOFF_SECONDS[min(attempt, len(BASE_BACKOFF_SECONDS) - 1)]
                jitter = base_wait * JITTER_PERCENT * random.random()
                wait_time = base_wait + jitter

                print(f"   ⚠️ {operation_name} failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                print(f"   ⏳ Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ {operation_name} failed after {max_retries} attempts: {str(e)[:100]}")

    raise last_exception


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

        # 3. Batch fetch all real images for all products (avoid N+1 queries)
        product_ids = [
            item.get("products", {}).get("id")
            for item in products
            if item.get("products", {}).get("id")
        ]

        real_images_by_product = {}
        if product_ids:
            print(f"   📥 Batch fetching real images for {len(product_ids)} products...")
            try:
                real_response = self.client.table("product_images").select("*").in_(
                    "product_id", product_ids
                ).eq("image_type", "real").execute()

                # Group by product_id
                for img in real_response.data:
                    pid = img.get("product_id")
                    if pid not in real_images_by_product:
                        real_images_by_product[pid] = []
                    real_images_by_product[pid].append(img)

                print(f"   ✅ Found {len(real_response.data)} real images across all products")
            except Exception as e:
                print(f"   ⚠️ Batch real images fetch error: {e}")

        # 4. Download each product's frames
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

            # Download real images if exist (using pre-fetched data)
            real_dir = product_dir / "real"
            real_dir.mkdir(exist_ok=True)
            product_real_images = real_images_by_product.get(product_id, [])
            if product_real_images:
                self._download_real_images_batch(product_real_images, real_dir)

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
                """Download a single file with retry."""
                file_name = file_info['name']
                file_path = f"{folder}/{file_name}"

                def do_download():
                    data = self.client.storage.from_(bucket).download(file_path)
                    local_path = target_dir / file_name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    return True

                try:
                    retry_with_backoff(do_download, f"Download {file_name}")
                    return file_name, True
                except Exception:
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

    def _download_real_images_batch(self, images: List[dict], target_dir: Path):
        """Download real images from pre-fetched data (no DB query needed)."""
        if not images:
            return

        downloaded = 0
        for idx, img in enumerate(images):
            image_url = img.get("image_url")
            image_path = img.get("image_path")

            # Try to download from URL first (for matching images)
            if image_url and image_url.startswith("http"):
                def download_from_url():
                    resp = httpx.get(image_url, timeout=30)
                    if resp.status_code == 200:
                        ext = Path(image_url).suffix or ".jpg"
                        local_path = target_dir / f"real_{idx:04d}{ext}"
                        with open(local_path, 'wb') as fp:
                            fp.write(resp.content)
                        return True
                    raise Exception(f"HTTP {resp.status_code}")

                try:
                    retry_with_backoff(download_from_url, f"Download real image {idx}")
                    downloaded += 1
                    continue
                except Exception:
                    pass

            # Fallback: try to download from frames bucket storage
            if image_path:
                def download_from_storage():
                    data = self.client.storage.from_("frames").download(image_path)
                    local_path = target_dir / Path(image_path).name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                    return True

                try:
                    retry_with_backoff(download_from_storage, f"Download real image {idx} from storage")
                    downloaded += 1
                except Exception:
                    pass

        if downloaded > 0:
            print(f"      Downloaded {downloaded} real images")

    def _download_real_images(self, product_id: str, target_dir: Path):
        """Download matched real images for a product from product_images table (legacy method)."""
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
                    def download_from_url():
                        resp = httpx.get(image_url, timeout=30)
                        if resp.status_code == 200:
                            ext = Path(image_url).suffix or ".jpg"
                            local_path = target_dir / f"real_{idx:04d}{ext}"
                            with open(local_path, 'wb') as fp:
                                fp.write(resp.content)
                            return True
                        raise Exception(f"HTTP {resp.status_code}")

                    try:
                        retry_with_backoff(download_from_url, f"Download real image {idx}")
                        downloaded += 1
                        continue
                    except Exception:
                        pass

                # Fallback: try to download from frames bucket storage
                if image_path:
                    def download_from_storage():
                        data = self.client.storage.from_("frames").download(image_path)
                        local_path = target_dir / Path(image_path).name
                        with open(local_path, 'wb') as fp:
                            fp.write(data)
                        return True

                    try:
                        retry_with_backoff(download_from_storage, f"Download real image {idx} from storage")
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
            """Upload a single file with retry. Returns (storage_path, img_type, product_id, source, success)"""
            local_path, storage_path, img_type, product_id, source = task

            def do_upload():
                with open(local_path, 'rb') as f:
                    data = f.read()
                    try:
                        self.client.storage.from_(bucket).upload(storage_path, data)
                    except Exception:
                        # File might exist, try update
                        self.client.storage.from_(bucket).update(storage_path, data)
                return True

            try:
                retry_with_backoff(do_upload, f"Upload {Path(storage_path).name}")
                return storage_path, img_type, product_id, source, True
            except Exception:
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
            db_count, failed_paths = self._insert_augmented_records(successful_uploads, bucket)
            stats["db_inserted"] = db_count

            # Cleanup: Delete orphaned files from storage (uploaded but not in DB)
            if failed_paths:
                print(f"   🧹 Cleaning up {len(failed_paths)} orphaned files from storage...")
                cleaned = self._cleanup_orphaned_files(bucket, failed_paths)
                stats["orphaned_cleaned"] = cleaned
                print(f"   ✅ Cleaned {cleaned}/{len(failed_paths)} orphaned files")

        return stats

    def _cleanup_orphaned_files(self, bucket: str, paths: List[str]) -> int:
        """Delete orphaned files from storage that weren't registered in DB."""
        cleaned = 0
        for path in paths:
            try:
                self.client.storage.from_(bucket).remove([path])
                cleaned += 1
            except Exception as e:
                print(f"   ⚠️ Failed to delete {path}: {str(e)[:50]}")
        return cleaned

    def _insert_augmented_records(
        self,
        uploads: List[Tuple[str, str, str, str]],
        bucket: str,
    ) -> Tuple[int, List[str]]:
        """Insert augmented image records into product_images table.

        Uses batch delete + batch insert pattern with error handling.

        Returns:
            Tuple of (inserted_count, list of failed storage_paths)
        """
        failed_paths: List[str] = []

        try:
            # Group by product_id for efficient deletion of old records
            product_ids = list(set(u[2] for u in uploads))

            # Batch delete existing augmented frames (single query instead of N queries)
            print(f"   🗑️ Deleting existing augmented images for {len(product_ids)} products...")
            try:
                # Use IN clause for batch delete - more efficient
                self.client.table("product_images").delete().in_(
                    "product_id", product_ids
                ).eq("image_type", "augmented").execute()
                print(f"   ✅ Old augmented images deleted")
            except Exception as e:
                print(f"   ⚠️ Delete failed (continuing with insert): {e}")

            # Prepare records with storage_path tracking
            records = []
            path_to_record_idx = {}  # Map storage_path to record index
            for idx, (storage_path, img_type, product_id, source) in enumerate(uploads):
                image_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"
                records.append({
                    "product_id": product_id,
                    "image_type": "augmented",
                    "source": source,
                    "image_path": storage_path,
                    "image_url": image_url,
                })
                path_to_record_idx[storage_path] = idx

            # Batch insert with retry (in chunks to avoid timeout)
            chunk_size = 500
            inserted = 0
            failed_chunk_indices = []

            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                chunk_num = i // chunk_size + 1
                total_chunks = (len(records) + chunk_size - 1) // chunk_size

                def do_insert():
                    self.client.table("product_images").insert(chunk).execute()
                    return True

                try:
                    retry_with_backoff(do_insert, f"Insert chunk {chunk_num}/{total_chunks}")
                    inserted += len(chunk)
                except Exception as e:
                    print(f"   ⚠️ Chunk {chunk_num} insert failed: {str(e)[:100]}")
                    failed_chunk_indices.append((i, i + len(chunk)))
                    # Collect failed paths from this chunk
                    for record in chunk:
                        failed_paths.append(record["image_path"])

            if failed_chunk_indices:
                print(f"   ⚠️ {len(failed_chunk_indices)} chunks failed to insert")

            print(f"   📝 Inserted {inserted}/{len(records)} augmented records into database")
            return inserted, failed_paths

        except Exception as e:
            print(f"   ⚠️ Failed to insert augmented records: {e}")
            # All uploads failed - return all paths as failed
            all_paths = [u[0] for u in uploads]
            return 0, all_paths

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
                """Download a single background image with retry."""
                file_name = file_info['name']
                file_path = f"{BACKGROUNDS_PATH}/{file_name}"

                def do_download():
                    data = self.client.storage.from_(BACKGROUNDS_BUCKET).download(file_path)
                    return Image.open(io.BytesIO(data)).convert("RGB")

                try:
                    img = retry_with_backoff(do_download, f"Download background {file_name}")
                    return file_name, img
                except Exception:
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
                """Download a single neighbor image with retry."""
                image_path = img_data.get("image_path")
                if not image_path:
                    return "", None

                def do_download():
                    data = self.client.storage.from_(BACKGROUNDS_BUCKET).download(image_path)
                    local_path = self.local_base / Path(image_path).name
                    with open(local_path, 'wb') as f:
                        f.write(data)
                    return str(local_path)

                try:
                    local = retry_with_backoff(do_download, f"Download neighbor {Path(image_path).name}")
                    return local, local
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
