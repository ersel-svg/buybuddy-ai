"""Supabase client for worker - downloads dataset and uploads results."""

import os
import json
from pathlib import Path
from supabase import create_client, Client
from typing import Optional
import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")


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
        """Download frames from Supabase Storage."""
        try:
            # frames_path format: "{supabase_url}/storage/v1/object/public/frames/{barcode}/"
            # or just "{barcode}/" if relative
            bucket = "frames"

            # Extract barcode from path
            if "/" in frames_path:
                barcode = frames_path.rstrip("/").split("/")[-1]
            else:
                barcode = frames_path.rstrip("/")

            # List files in bucket
            files = self.client.storage.from_(bucket).list(barcode)

            for f in files:
                if f.get("name", "").endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    file_path = f"{barcode}/{f['name']}"
                    data = self.client.storage.from_(bucket).download(file_path)

                    local_path = target_dir / f['name']
                    with open(local_path, 'wb') as fp:
                        fp.write(data)

        except Exception as e:
            print(f"   ⚠️ Frame download error: {e}")

    def _download_real_images(self, product_id: str, target_dir: Path):
        """Download matched real images for a product."""
        try:
            # Get real images from product_images table
            response = self.client.table("product_images").select("*").eq(
                "product_id", product_id
            ).eq("image_type", "real").execute()

            for img in response.data:
                image_path = img.get("image_path")
                if not image_path:
                    continue

                # Download from storage
                bucket = "real-images"
                try:
                    data = self.client.storage.from_(bucket).download(image_path)
                    local_path = target_dir / Path(image_path).name
                    with open(local_path, 'wb') as fp:
                        fp.write(data)
                except Exception:
                    pass

        except Exception as e:
            print(f"   ⚠️ Real images download error: {e}")


class ResultUploader:
    """Uploads augmented images back to Supabase Storage."""

    def __init__(self):
        self.client = get_supabase()

    def upload_augmented(self, dataset_dir, dataset_id: str) -> dict:
        """
        Upload augmented images to Supabase Storage.

        Returns upload statistics.
        """
        print(f"\n📤 Uploading augmented images...")

        bucket = "augmented-images"
        stats = {"syn_uploaded": 0, "real_uploaded": 0}

        # Ensure Path object
        dataset_dir = Path(dataset_dir) if isinstance(dataset_dir, str) else dataset_dir

        # Walk through dataset directory
        train_dir = dataset_dir / "train"
        if not train_dir.exists():
            train_dir = dataset_dir

        for product_dir in train_dir.iterdir():
            if not product_dir.is_dir():
                continue

            barcode = product_dir.name

            # Upload synthetic augmented images (syn_*.jpg)
            for img in product_dir.glob("syn_*.jpg"):
                storage_path = f"{dataset_id}/{barcode}/{img.name}"
                try:
                    with open(img, 'rb') as f:
                        self.client.storage.from_(bucket).upload(storage_path, f.read())
                    stats["syn_uploaded"] += 1
                except Exception:
                    pass  # May already exist

            # Upload real augmented images (real/*_aug_*.jpg)
            real_dir = product_dir / "real"
            if real_dir.exists():
                for img in real_dir.glob("*_aug_*.jpg"):
                    storage_path = f"{dataset_id}/{barcode}/real/{img.name}"
                    try:
                        with open(img, 'rb') as f:
                            self.client.storage.from_(bucket).upload(storage_path, f.read())
                        stats["real_uploaded"] += 1
                    except Exception:
                        pass

        print(f"   ✅ Uploaded: {stats['syn_uploaded']} syn + {stats['real_uploaded']} real")
        return stats

    def update_job_progress(self, job_id: str, progress: int, current_step: str):
        """Update job progress in database."""
        try:
            self.client.table("jobs").update({
                "progress": progress,
                "current_step": current_step,
            }).eq("runpod_job_id", job_id).execute()
        except Exception as e:
            print(f"   ⚠️ Progress update error: {e}")

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
