"""
Product Dataset for Training.

Features:
- Multi-frame support per product
- Multi-image-type support (synthetic, real, augmented)
- Domain-aware sampling
- Augmentation integration
- Efficient image loading from URLs
"""

import io
import random
from typing import Optional, Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import httpx

# bb-models imports
from bb_models import get_model_config
from bb_models.utils.preprocessing import (
    get_augmentation_transform,
    get_eval_transform,
)


class ProductDataset(Dataset):
    """
    Dataset for product images with product_id-based class labels.

    Supports two data formats:
    1. NEW FORMAT (recommended): training_images dict with URLs for each product
       training_images = {
           "product_id_1": [
               {"url": "https://...", "image_type": "synthetic", "domain": "synthetic", "frame_index": 0},
               {"url": "https://...", "image_type": "real", "domain": "real", "frame_index": 0},
           ],
           ...
       }

    2. LEGACY FORMAT: products list with frames_path
       products = [
           {"id": "...", "frames_path": "https://.../.../frames", "frame_count": 36, ...},
           ...
       ]

    NOTE: We use product_id as the class label (not UPC/barcode) because:
    - A product can have multiple identifiers (barcode, short_code, UPC, EAN)
    - product_id is the unique canonical identifier
    - After inference, product_id maps to all associated identifiers
    """

    def __init__(
        self,
        products: list[dict],
        model_type: str = "dinov2-base",
        augmentation_strength: str = "medium",
        is_training: bool = True,
        frames_per_product: int = 1,
        http_timeout: float = 30.0,
        training_images: Optional[dict[str, list[dict]]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            products: List of product dicts with keys:
                - id: Product ID (unique, used as class label)
                - frames_path: Base path to frames (legacy format)
                - frame_count: Number of frames (legacy format)
                - barcode, short_code, upc: Optional identifiers (for metadata only)
                - brand_name: Optional metadata
            model_type: Model identifier for preprocessing
            augmentation_strength: Augmentation level
            is_training: Enable training augmentations
            frames_per_product: Frames to sample per product (legacy format)
            http_timeout: HTTP request timeout
            training_images: NEW FORMAT - Dict mapping product_id to list of images with URLs
        """
        self.products = products
        self.model_type = model_type
        self.augmentation_strength = augmentation_strength
        self.is_training = is_training
        self.frames_per_product = frames_per_product
        self.http_timeout = http_timeout
        self.training_images = training_images

        # Determine which format we're using
        self.use_new_format = training_images is not None and len(training_images) > 0

        # Get model config for preprocessing
        self.model_config = get_model_config(model_type)

        # Build product_id to class index mapping
        # Each unique product_id becomes a class
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self.product_by_id = {}

        if self.use_new_format:
            # New format: use keys from training_images
            unique_product_ids = sorted(training_images.keys())
        else:
            # Legacy format: use product IDs from products list
            unique_product_ids = sorted(set(p["id"] for p in products if p.get("id")))

        for idx, product_id in enumerate(unique_product_ids):
            self.product_id_to_idx[product_id] = idx
            self.idx_to_product_id[idx] = product_id

        # Build product lookup
        for p in products:
            if p.get("id"):
                self.product_by_id[p["id"]] = p

        self.num_classes = len(unique_product_ids)

        # Build samples list
        self._build_samples()

        # Setup transforms
        self._setup_transforms()

        # HTTP client for downloading images
        self.http_client = httpx.Client(timeout=http_timeout)

        # Count image types
        image_type_counts = {"synthetic": 0, "real": 0, "augmented": 0, "cutout": 0, "unknown": 0}
        for sample in self.samples:
            if self.use_new_format:
                img_type = sample.get("image_type", "unknown")
            else:
                img_type = "synthetic"  # Legacy format is all synthetic
            image_type_counts[img_type] = image_type_counts.get(img_type, 0) + 1

        print(f"ProductDataset initialized:")
        print(f"  Format: {'NEW (URLs)' if self.use_new_format else 'LEGACY (frames_path)'}")
        print(f"  Products: {len(unique_product_ids)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Classes (unique product_ids): {self.num_classes}")
        print(f"  Model: {model_type}")
        print(f"  Augmentation: {augmentation_strength}")
        print(f"  Image types: {image_type_counts}")

    def _build_samples(self):
        """Build the samples list based on format."""
        self.samples = []
        self.product_to_samples = defaultdict(list)

        if self.use_new_format:
            # NEW FORMAT: Build samples from training_images dict
            for product_id, images in self.training_images.items():
                if product_id not in self.product_id_to_idx:
                    continue

                for img in images:
                    if not img.get("url"):
                        continue

                    sample = {
                        "product_id": product_id,
                        "url": img["url"],
                        "image_type": img.get("image_type", "synthetic"),
                        "domain": img.get("domain", img.get("image_type", "synthetic")),
                        "frame_index": img.get("frame_index", 0),
                    }

                    sample_idx = len(self.samples)
                    self.samples.append(sample)
                    self.product_to_samples[product_id].append(sample_idx)
        else:
            # LEGACY FORMAT: Build samples from products with frames_path
            for prod_idx, product in enumerate(self.products):
                product_id = product.get("id")
                if not product_id or product_id not in self.product_id_to_idx:
                    continue

                frame_count = product.get("frame_count", 1)
                frames_path = product.get("frames_path", "")

                if self.is_training:
                    # For training, include all frames as separate samples
                    for frame_idx in range(frame_count):
                        sample = {
                            "product_id": product_id,
                            "prod_idx": prod_idx,
                            "frame_idx": frame_idx,
                            "url": f"{frames_path}/frame_{frame_idx:04d}.png",
                            "image_type": "synthetic",
                            "domain": "synthetic",
                            "frame_index": frame_idx,
                        }
                        sample_idx = len(self.samples)
                        self.samples.append(sample)
                        self.product_to_samples[product_id].append(sample_idx)
                else:
                    # For eval, use first frame only
                    sample = {
                        "product_id": product_id,
                        "prod_idx": prod_idx,
                        "frame_idx": 0,
                        "url": f"{frames_path}/frame_0000.png",
                        "image_type": "synthetic",
                        "domain": "synthetic",
                        "frame_index": 0,
                    }
                    self.samples.append(sample)

    def _setup_transforms(self):
        """Setup image transforms with model-specific normalization."""
        image_size = self.model_config.input_size
        image_mean = self.model_config.image_mean
        image_std = self.model_config.image_std

        # Store preprocessing config for saving with training results
        self.preprocessing_config = {
            "image_size": image_size,
            "image_mean": image_mean,
            "image_std": image_std,
            "augmentation_strength": self.augmentation_strength if self.is_training else "none",
        }

        print(f"  Preprocessing: size={image_size}, mean={image_mean}, std={image_std}")

        if self.is_training and self.augmentation_strength != "none":
            self.transform = get_augmentation_transform(
                image_size=image_size,
                image_mean=image_mean,
                image_std=image_std,
                strength=self.augmentation_strength,
            )
        else:
            self.transform = get_eval_transform(
                image_size=image_size,
                image_mean=image_mean,
                image_std=image_std,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]
        product_id = sample["product_id"]
        url = sample["url"]

        # Download and process image
        try:
            image = self._load_image(url)
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        image_tensor = self.transform(image)

        # Get label from product_id
        label = self.product_id_to_idx.get(product_id, 0)

        # Get product metadata
        product = self.product_by_id.get(product_id, {})

        return {
            "image": image_tensor,
            "label": label,
            "product_id": product_id,
            "frame_idx": sample.get("frame_index", 0),
            # Domain and category for evaluation
            "domain": sample.get("domain", "synthetic"),
            "image_type": sample.get("image_type", "synthetic"),
            "category": product.get("category", "unknown"),
            # Include identifiers as metadata (not used for training)
            "barcode": product.get("barcode"),
            "short_code": product.get("short_code"),
            "upc": product.get("upc"),
        }

    def _load_image(self, url: str) -> Image.Image:
        """Load image from URL."""
        response = self.http_client.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image

    def get_item_info(self, idx: int) -> dict[str, Any]:
        """
        Get sample info without loading the image.

        Useful for samplers that need label/domain info.

        Args:
            idx: Sample index

        Returns:
            Dictionary with label, domain, and product_id
        """
        sample = self.samples[idx]
        product_id = sample["product_id"]
        label = self.product_id_to_idx.get(product_id, 0)
        product = self.product_by_id.get(product_id, {})

        return {
            "label": label,
            "domain": sample.get("domain", "synthetic"),
            "image_type": sample.get("image_type", "synthetic"),
            "product_id": product_id,
            "frame_idx": sample.get("frame_index", 0),
            "category": product.get("category", "unknown"),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Returns:
            Tensor of weights per class
        """
        class_counts = np.zeros(self.num_classes)

        for sample in self.samples:
            product_id = sample["product_id"]
            if product_id in self.product_id_to_idx:
                class_counts[self.product_id_to_idx[product_id]] += 1

        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes

        return torch.FloatTensor(weights)

    def resample_frames(self):
        """
        Resample frames for each product (call between epochs).

        This enables different frame views across epochs.
        Only applicable for legacy format.
        """
        if not self.is_training or self.use_new_format:
            # New format already has all images, no need to resample
            return

        # Legacy format resampling
        new_samples = []
        for prod_idx, product in enumerate(self.products):
            product_id = product.get("id")
            if not product_id or product_id not in self.product_id_to_idx:
                continue

            frame_count = product.get("frame_count", 1)
            frames_path = product.get("frames_path", "")

            # Sample random frames
            if frame_count > self.frames_per_product:
                frame_indices = random.sample(
                    range(frame_count),
                    self.frames_per_product,
                )
            else:
                frame_indices = list(range(frame_count))

            for frame_idx in frame_indices:
                sample = {
                    "product_id": product_id,
                    "prod_idx": prod_idx,
                    "frame_idx": frame_idx,
                    "url": f"{frames_path}/frame_{frame_idx:04d}.png",
                    "image_type": "synthetic",
                    "domain": "synthetic",
                    "frame_index": frame_idx,
                }
                new_samples.append(sample)

        self.samples = new_samples

    def get_domain_distribution(self) -> dict[str, int]:
        """Get count of samples per domain."""
        distribution = defaultdict(int)
        for sample in self.samples:
            domain = sample.get("domain", "unknown")
            distribution[domain] += 1
        return dict(distribution)

    def get_image_type_distribution(self) -> dict[str, int]:
        """Get count of samples per image type."""
        distribution = defaultdict(int)
        for sample in self.samples:
            img_type = sample.get("image_type", "unknown")
            distribution[img_type] += 1
        return dict(distribution)


class BalancedBatchSampler:
    """
    Sampler that ensures balanced classes within each batch.

    Each batch contains samples from multiple classes,
    with similar representation per class.
    """

    def __init__(
        self,
        dataset: ProductDataset,
        batch_size: int,
        classes_per_batch: int = 8,
        samples_per_class: int = 4,
        drop_last: bool = True,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Total batch size
            classes_per_batch: Number of classes per batch
            samples_per_class: Samples per class per batch
            drop_last: Drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.drop_last = drop_last

        # Build class to sample indices mapping (using product_id as class)
        self.class_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            product_id = sample["product_id"]
            if product_id in dataset.product_id_to_idx:
                class_idx = dataset.product_id_to_idx[product_id]
                self.class_to_indices[class_idx].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.num_samples = len(dataset)

    def __iter__(self):
        """Generate batches."""
        # Shuffle class sample lists
        class_indices = {
            c: random.sample(indices, len(indices))
            for c, indices in self.class_to_indices.items()
        }
        class_pointers = {c: 0 for c in self.classes}

        batches = []
        available_classes = set(self.classes)

        while len(available_classes) >= self.classes_per_batch:
            batch = []

            # Sample classes for this batch
            batch_classes = random.sample(
                list(available_classes),
                self.classes_per_batch,
            )

            for class_idx in batch_classes:
                indices = class_indices[class_idx]
                ptr = class_pointers[class_idx]

                # Get samples for this class
                for _ in range(self.samples_per_class):
                    if ptr >= len(indices):
                        # Remove exhausted class
                        available_classes.discard(class_idx)
                        break
                    batch.append(indices[ptr])
                    ptr += 1

                class_pointers[class_idx] = ptr

            if len(batch) >= self.batch_size:
                batches.append(batch[: self.batch_size])
            elif not self.drop_last:
                batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        """Approximate number of batches."""
        return self.num_samples // self.batch_size


class DomainBalancedSampler:
    """
    Sampler that balances samples across domains (image types).

    Ensures each batch contains images from multiple domains
    (synthetic, real, augmented), preventing the model from
    overfitting to domain-specific features.
    """

    def __init__(
        self,
        dataset: ProductDataset,
        batch_size: int,
        domains_per_batch: int = 3,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Total batch size
            domains_per_batch: Number of different domains per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domains_per_batch = domains_per_batch

        # Build domain to sample indices mapping
        self.domain_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            domain = sample.get("domain", "synthetic")
            self.domain_to_indices[domain].append(idx)

        self.domains = list(self.domain_to_indices.keys())
        self.num_samples = len(dataset)

        print(f"DomainBalancedSampler: {len(self.domains)} domains - {dict(self.domain_to_indices.keys())}")
        for domain, indices in self.domain_to_indices.items():
            print(f"  {domain}: {len(indices)} samples")

    def __iter__(self):
        """Generate batches."""
        # Shuffle domain sample lists
        domain_indices = {
            d: random.sample(indices, len(indices))
            for d, indices in self.domain_to_indices.items()
        }
        domain_pointers = {d: 0 for d in self.domains}

        available_domains = set(self.domains)
        samples_per_domain = max(1, self.batch_size // min(self.domains_per_batch, len(self.domains)))

        while len(available_domains) >= 1:
            batch = []

            # Sample domains for this batch
            num_domains = min(self.domains_per_batch, len(available_domains))
            batch_domains = random.sample(
                list(available_domains),
                num_domains,
            )

            for domain in batch_domains:
                indices = domain_indices[domain]
                ptr = domain_pointers[domain]

                # Get samples from this domain
                for _ in range(samples_per_domain):
                    if ptr >= len(indices):
                        available_domains.discard(domain)
                        break
                    batch.append(indices[ptr])
                    ptr += 1

                domain_pointers[domain] = ptr

            if len(batch) >= self.batch_size:
                random.shuffle(batch)
                yield batch[: self.batch_size]
            elif batch:
                # Yield partial batch at the end
                random.shuffle(batch)
                yield batch

    def __len__(self):
        """Approximate number of batches."""
        return self.num_samples // self.batch_size
