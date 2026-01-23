"""
URL-Based Object Detection Dataset.

Loads images from Supabase Storage URLs and annotations from database.
Supports:
- Parallel image preloading with ThreadPoolExecutor
- File-level caching for faster subsequent epochs
- Retry logic for failed downloads
- Integration with augmentation pipeline (Mosaic, MixUp, etc.)

Usage:
    from data.url_dataset import URLODDataset
    from data.supabase_fetcher import build_url_dataset_data

    # Fetch data from Supabase
    data = build_url_dataset_data(supabase_url, supabase_key, dataset_id)

    # Create dataset
    train_dataset = URLODDataset(
        image_data=data["train_images"],
        class_mapping=data["class_mapping"],
        transform=train_pipeline,
        preload=True,
    )
"""

import os
import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    import requests
    HTTP_CLIENT = "requests"


def load_image_from_url(
    url: str,
    timeout: int = 30,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load image from URL with retry logic.

    Args:
        url: Image URL
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        Tuple of (image_array, error_message)
        If successful, error_message is None
        If failed, image_array is None
    """
    from PIL import Image

    for attempt in range(retry_attempts):
        try:
            if HTTP_CLIENT == "httpx":
                response = httpx.get(url, timeout=timeout, follow_redirects=True)
                response.raise_for_status()
                content = response.content
            else:
                response = requests.get(url, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                content = response.content

            img = Image.open(BytesIO(content)).convert("RGB")
            return np.array(img), None

        except Exception as e:
            error = str(e)
            if attempt < retry_attempts - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                return None, error

    return None, "Max retries exceeded"


class URLODDataset(Dataset):
    """
    Object Detection Dataset that loads images from URLs.

    Features:
    - Parallel image preloading with ThreadPoolExecutor
    - File-level caching (images saved to disk after download)
    - Memory caching for current epoch
    - Retry logic for failed downloads
    - Integration with augmentation pipeline

    Args:
        image_data: List of dicts with keys:
            - "image_id": str
            - "image_url": str
            - "width": int
            - "height": int
            - "annotations": List of annotation dicts
        class_mapping: Dict mapping class_id (UUID) to class_index (int)
        transform: Augmentation pipeline (AugmentationPipeline)
        img_size: Target image size
        cache_dir: Directory for caching downloaded images
        max_workers: Number of parallel download workers
        preload: Whether to preload all images on init
        retry_attempts: Number of download retry attempts
        retry_delay: Base delay between retries
    """

    def __init__(
        self,
        image_data: List[Dict[str, Any]],
        class_mapping: Dict[str, int],
        transform: Optional[Callable] = None,
        img_size: int = 640,
        cache_dir: Optional[str] = None,
        max_workers: int = 8,
        preload: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        use_memory_cache: bool = False,  # Disabled by default - prevents OOM with DataLoader workers
    ):
        self.image_data = image_data
        self.class_mapping = class_mapping
        self.transform = transform
        self.img_size = img_size
        self.cache_dir = cache_dir or "/tmp/od_image_cache"
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.use_memory_cache = use_memory_cache

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory cache (only used if use_memory_cache=True)
        # WARNING: With DataLoader workers, each worker has its own cache copy
        # This can cause OOM with large datasets. File cache is recommended.
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._failed_urls: set = set()

        # Preload images if requested
        if preload and len(image_data) > 0:
            self.preload_images()

        # Set sample function for multi-image augmentations (Mosaic, MixUp)
        if transform is not None and hasattr(transform, 'set_sample_fn'):
            transform.set_sample_fn(self._sample_random)

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, target_dict)
            - image_tensor: [C, H, W] normalized tensor
            - target_dict: {'boxes': [N, 4], 'labels': [N], 'image_id': str}
        """
        item = self.image_data[idx]
        image_url = item["image_url"]
        image_id = item["image_id"]

        # Load image (from cache or URL)
        image = self._load_image(image_url, image_id)

        # Get image dimensions for bbox denormalization
        img_width = item.get("width") or image.shape[1]
        img_height = item.get("height") or image.shape[0]

        # Build annotations in expected format
        boxes = []
        labels = []
        for ann in item.get("annotations", []):
            class_id = ann.get("class_id")
            if class_id not in self.class_mapping:
                continue

            class_idx = self.class_mapping[class_id]
            bbox = ann.get("bbox", {})

            # Bbox is stored as normalized (0-1) coordinates
            # Convert to pixel coordinates for augmentation
            x = bbox.get("x", 0) * img_width
            y = bbox.get("y", 0) * img_height
            w = bbox.get("width", 0) * img_width
            h = bbox.get("height", 0) * img_height

            # Convert to xyxy format
            x2 = x + w
            y2 = y + h

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x2, y2])
            labels.append(class_idx)

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
        }

        # Apply transform (augmentation pipeline)
        if self.transform is not None:
            image, target = self.transform(image, target, idx=idx)
        else:
            # Basic transform when no pipeline provided
            image = self._basic_transform(image)
            target["boxes"] = torch.from_numpy(target["boxes"])
            target["labels"] = torch.from_numpy(target["labels"])

        return image, target

    def _load_image(self, url: str, image_id: str) -> np.ndarray:
        """
        Load image from URL with caching and retry logic.

        Priority (when use_memory_cache=True):
        1. Memory cache (fastest)
        2. File cache (fast)
        3. Download from URL (slow, with retry)
        4. Gray placeholder (fallback)

        Priority (when use_memory_cache=False - default):
        1. File cache (fast, disk-based)
        2. Download from URL (slow, with retry)
        3. Gray placeholder (fallback)
        """
        # Check memory cache first (only if enabled)
        if self.use_memory_cache and url in self._memory_cache:
            return self._memory_cache[url].copy()

        # Check file cache
        cache_path = self._get_cache_path(url)
        if os.path.exists(cache_path):
            try:
                from PIL import Image
                img = Image.open(cache_path).convert("RGB")
                img_array = np.array(img)
                # Only cache in memory if enabled
                if self.use_memory_cache:
                    self._memory_cache[url] = img_array
                    return img_array.copy()
                return img_array
            except Exception:
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass

        # Download from URL
        img_array, error = load_image_from_url(
            url,
            timeout=30,
            retry_attempts=self.retry_attempts,
            retry_delay=self.retry_delay,
        )

        if img_array is not None:
            # Save to file cache
            try:
                from PIL import Image
                img = Image.fromarray(img_array)
                img.save(cache_path, "JPEG", quality=95)
            except Exception:
                pass

            # Only cache in memory if enabled
            if self.use_memory_cache:
                self._memory_cache[url] = img_array
                return img_array.copy()
            return img_array
        else:
            # Failed - track and return placeholder
            self._failed_urls.add(url)
            print(f"[WARNING] Failed to load image {image_id}: {error}")

            # Return gray placeholder
            return np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)

    def _get_cache_path(self, url: str) -> str:
        """Get file cache path for URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{url_hash}.jpg")

    def _basic_transform(self, image: np.ndarray) -> torch.Tensor:
        """Basic transform when no augmentation pipeline provided."""
        from PIL import Image
        import torchvision.transforms as T

        # Resize
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.img_size, self.img_size))
        image = np.array(pil_img)

        # Normalize and convert to tensor
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transform(image)

    def _sample_random(self, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Sample a random image and target for multi-image augmentations.

        Used by Mosaic, MixUp, CopyPaste, etc.

        Args:
            exclude_idx: Index to exclude from sampling

        Returns:
            Tuple of (image_array, target_dict) without augmentation
        """
        if exclude_idx is not None:
            valid_indices = [i for i in range(len(self.image_data)) if i != exclude_idx]
            idx = random.choice(valid_indices) if valid_indices else 0
        else:
            idx = random.randint(0, len(self.image_data) - 1)

        item = self.image_data[idx]
        image = self._load_image(item["image_url"], item["image_id"])

        img_width = item.get("width") or image.shape[1]
        img_height = item.get("height") or image.shape[0]

        boxes = []
        labels = []
        for ann in item.get("annotations", []):
            class_id = ann.get("class_id")
            if class_id in self.class_mapping:
                bbox = ann.get("bbox", {})

                x = bbox.get("x", 0) * img_width
                y = bbox.get("y", 0) * img_height
                w = bbox.get("width", 0) * img_width
                h = bbox.get("height", 0) * img_height

                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self.class_mapping[class_id])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64)

        return image, {"boxes": boxes, "labels": labels}

    def preload_images(self, progress_callback=None):
        """
        Pre-download all images in parallel.

        This significantly speeds up the first epoch by downloading
        all images before training starts.
        """
        print(f"[URLODDataset] Pre-loading {len(self.image_data)} images...")

        def download_one(item):
            url = item["image_url"]
            image_id = item["image_id"]
            try:
                self._load_image(url, image_id)
                return url not in self._failed_urls
            except:
                return False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(download_one, self.image_data))

        loaded = sum(results)
        failed = len(self.image_data) - loaded

        if failed > 0:
            print(f"[URLODDataset] WARNING: {failed} images failed to load")
        print(f"[URLODDataset] Pre-loaded {loaded}/{len(self.image_data)} images")

        return loaded

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(set(self.class_mapping.values()))

    def get_class_names(self) -> List[str]:
        """Get class names (empty - names come from database)."""
        return []

    def get_failed_urls(self) -> set:
        """Get set of URLs that failed to load."""
        return self._failed_urls.copy()

    def clear_memory_cache(self):
        """Clear memory cache to free RAM."""
        self._memory_cache.clear()


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Custom collate function for object detection.

    Handles variable number of boxes per image by keeping targets as list.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (batched_images, list_of_targets)
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)

        # Ensure tensors
        if isinstance(target['boxes'], np.ndarray):
            target['boxes'] = torch.from_numpy(target['boxes'])
        if isinstance(target['labels'], np.ndarray):
            target['labels'] = torch.from_numpy(target['labels'])

        targets.append(target)

    # Stack images into batch
    images = torch.stack(images, dim=0)

    return images, targets


def create_url_dataloader(
    dataset: URLODDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    """
    Create DataLoader for URLODDataset.

    Args:
        dataset: URLODDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        Configured DataLoader
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
