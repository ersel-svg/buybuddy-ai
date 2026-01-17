"""
Product Dataset for Training.

Features:
- Multi-frame support per product
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

    Each product can have multiple frames. During training,
    randomly samples one frame per product per epoch.

    NOTE: We use product_id as the class label (not UPC/barcode) because:
    - A product can have multiple identifiers (barcode, short_code, UPC, EAN)
    - product_id is the unique canonical identifier
    - After inference, product_id maps to all associated identifiers

    Attributes:
        products: List of product dictionaries with frames
        model_type: Model type for preprocessing config
        augmentation_strength: 'none', 'light', 'medium', 'heavy'
        is_training: Whether to apply augmentations
    """

    def __init__(
        self,
        products: list[dict],
        model_type: str = "dinov2-base",
        augmentation_strength: str = "medium",
        is_training: bool = True,
        frames_per_product: int = 1,
        http_timeout: float = 30.0,
    ):
        """
        Initialize the dataset.

        Args:
            products: List of product dicts with keys:
                - id: Product ID (unique, used as class label)
                - frames_path: Base path to frames
                - frame_count: Number of frames
                - barcode, short_code, upc: Optional identifiers (for metadata only)
                - brand_name: Optional metadata
            model_type: Model identifier for preprocessing
            augmentation_strength: Augmentation level
            is_training: Enable training augmentations
            frames_per_product: Frames to sample per product
            http_timeout: HTTP request timeout
        """
        self.products = products
        self.model_type = model_type
        self.augmentation_strength = augmentation_strength
        self.is_training = is_training
        self.frames_per_product = frames_per_product
        self.http_timeout = http_timeout

        # Get model config for preprocessing
        self.model_config = get_model_config(model_type)

        # Build product_id to class index mapping
        # Each unique product_id becomes a class
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        unique_product_ids = sorted(set(p["id"] for p in products if p.get("id")))

        for idx, product_id in enumerate(unique_product_ids):
            self.product_id_to_idx[product_id] = idx
            self.idx_to_product_id[idx] = product_id

        self.num_classes = len(unique_product_ids)

        # Build samples list: (product_idx, frame_idx)
        self._build_samples()

        # Setup transforms
        self._setup_transforms()

        # HTTP client for downloading images
        self.http_client = httpx.Client(timeout=http_timeout)

        print(f"ProductDataset initialized:")
        print(f"  Products: {len(products)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Classes (unique product_ids): {self.num_classes}")
        print(f"  Model: {model_type}")
        print(f"  Augmentation: {augmentation_strength}")

    def _build_samples(self):
        """Build the samples list."""
        self.samples = []
        self.product_to_samples = defaultdict(list)

        for prod_idx, product in enumerate(self.products):
            product_id = product.get("id")
            if not product_id or product_id not in self.product_id_to_idx:
                continue

            frame_count = product.get("frame_count", 1)

            if self.is_training:
                # For training, include all frames as separate samples
                for frame_idx in range(frame_count):
                    sample_idx = len(self.samples)
                    self.samples.append((prod_idx, frame_idx))
                    self.product_to_samples[prod_idx].append(sample_idx)
            else:
                # For eval, use first frame only
                self.samples.append((prod_idx, 0))

    def _setup_transforms(self):
        """Setup image transforms."""
        image_size = self.model_config.input_size

        if self.is_training and self.augmentation_strength != "none":
            self.transform = get_augmentation_transform(
                image_size=image_size,
                strength=self.augmentation_strength,
            )
        else:
            self.transform = get_eval_transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample."""
        prod_idx, frame_idx = self.samples[idx]
        product = self.products[prod_idx]

        # Get frame URL
        frames_path = product.get("frames_path", "")
        frame_url = f"{frames_path}/frame_{frame_idx:04d}.png"

        # Download and process image
        try:
            image = self._load_image(frame_url)
        except Exception as e:
            print(f"Failed to load {frame_url}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Apply transforms
        image_tensor = self.transform(image)

        # Get label from product_id (not UPC)
        product_id = product.get("id")
        label = self.product_id_to_idx.get(product_id, 0)

        # Determine domain from frame path or product metadata
        domain = product.get("domain", "unknown")
        if domain == "unknown":
            # Infer from frame path if possible
            if "real" in frames_path.lower() or "_real_" in frames_path.lower():
                domain = "real"
            elif "augmented" in frames_path.lower() or "_aug_" in frames_path.lower():
                domain = "augmented"
            else:
                domain = "synthetic"

        return {
            "image": image_tensor,
            "label": label,
            "product_id": product_id,
            "frame_idx": frame_idx,
            # Domain and category for evaluation
            "domain": domain,
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
        prod_idx, frame_idx = self.samples[idx]
        product = self.products[prod_idx]

        # Get label from product_id
        product_id = product.get("id")
        label = self.product_id_to_idx.get(product_id, 0)

        # Determine domain
        domain = product.get("domain", "unknown")
        if domain == "unknown":
            frames_path = product.get("frames_path", "")
            if "real" in frames_path.lower() or "_real_" in frames_path.lower():
                domain = "real"
            elif "augmented" in frames_path.lower() or "_aug_" in frames_path.lower():
                domain = "augmented"
            else:
                domain = "synthetic"

        return {
            "label": label,
            "domain": domain,
            "product_id": product_id,
            "frame_idx": frame_idx,
            "category": product.get("category", "unknown"),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced data.

        Returns:
            Tensor of weights per class
        """
        class_counts = np.zeros(self.num_classes)

        for prod_idx, _ in self.samples:
            product = self.products[prod_idx]
            product_id = product.get("id")
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
        """
        if not self.is_training:
            return

        new_samples = []
        for prod_idx, product in enumerate(self.products):
            product_id = product.get("id")
            if not product_id or product_id not in self.product_id_to_idx:
                continue

            frame_count = product.get("frame_count", 1)

            # Sample random frames
            if frame_count > self.frames_per_product:
                frame_indices = random.sample(
                    range(frame_count),
                    self.frames_per_product,
                )
            else:
                frame_indices = list(range(frame_count))

            for frame_idx in frame_indices:
                new_samples.append((prod_idx, frame_idx))

        self.samples = new_samples


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
        for idx, (prod_idx, _) in enumerate(dataset.samples):
            product = dataset.products[prod_idx]
            product_id = product.get("id")
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
    Sampler that balances samples across domains (brands).

    Ensures each batch contains products from multiple brands,
    preventing the model from overfitting to brand-specific features.
    """

    def __init__(
        self,
        dataset: ProductDataset,
        batch_size: int,
        domains_per_batch: int = 4,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size: Total batch size
            domains_per_batch: Number of different brands per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domains_per_batch = domains_per_batch

        # Build domain (brand) to sample indices mapping
        self.domain_to_indices = defaultdict(list)
        for idx, (prod_idx, _) in enumerate(dataset.samples):
            product = dataset.products[prod_idx]
            brand = product.get("brand_name", "unknown")
            self.domain_to_indices[brand].append(idx)

        self.domains = list(self.domain_to_indices.keys())
        self.num_samples = len(dataset)

        print(f"DomainBalancedSampler: {len(self.domains)} domains")

    def __iter__(self):
        """Generate batches."""
        # Shuffle domain sample lists
        domain_indices = {
            d: random.sample(indices, len(indices))
            for d, indices in self.domain_to_indices.items()
        }
        domain_pointers = {d: 0 for d in self.domains}

        available_domains = set(self.domains)
        samples_per_domain = self.batch_size // self.domains_per_batch

        while len(available_domains) >= self.domains_per_batch:
            batch = []

            # Sample domains for this batch
            batch_domains = random.sample(
                list(available_domains),
                min(self.domains_per_batch, len(available_domains)),
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

            if batch:
                random.shuffle(batch)
                yield batch[: self.batch_size]

    def __len__(self):
        """Approximate number of batches."""
        return self.num_samples // self.batch_size
