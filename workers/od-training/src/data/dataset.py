"""
Object Detection Dataset for Training.

Supports:
- COCO format annotations
- Integration with augmentation pipeline
- Multi-image augmentation sampling
- Efficient data loading

Usage:
    from data import ODDataset, create_dataloader
    from augmentations import AugmentationPipeline

    pipeline = AugmentationPipeline.from_preset("sota", img_size=640)

    dataset = ODDataset(
        img_dir="path/to/images",
        ann_file="path/to/annotations.json",
        transform=pipeline,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True,
    )
"""

import os
import json
import random
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_image(path: str) -> np.ndarray:
    """Load image from path using cv2 or PIL."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        img = Image.open(path).convert('RGB')
        return np.array(img)


class ODDataset(Dataset):
    """
    Object Detection Dataset with augmentation support.

    Loads images and annotations in COCO format, applies augmentations,
    and returns tensors ready for training.

    Args:
        img_dir: Directory containing images
        ann_file: Path to COCO format annotation file
        transform: Augmentation pipeline (AugmentationPipeline)
        img_size: Target image size (used if no transform provided)
        class_names: Optional list of class names (for logging)
        max_samples: Optional limit on number of samples (for debugging)
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        img_size: int = 640,
        class_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.img_dir = Path(img_dir)
        self.ann_file = ann_file
        self.transform = transform
        self.img_size = img_size
        self.class_names = class_names
        self.max_samples = max_samples

        # Load COCO annotations
        self._load_annotations()

        # Set up sample function for multi-image augmentations
        if transform is not None and hasattr(transform, 'set_sample_fn'):
            transform.set_sample_fn(self._sample_random)

    def _load_annotations(self):
        """Load COCO format annotations."""
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build image id to info mapping
        self.images = {}
        for img_info in coco_data.get('images', []):
            self.images[img_info['id']] = {
                'file_name': img_info['file_name'],
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
                'annotations': [],
            }

        # Build category id to index mapping
        self.categories = {}
        self.cat_id_to_idx = {}
        for idx, cat in enumerate(coco_data.get('categories', [])):
            self.categories[cat['id']] = cat['name']
            self.cat_id_to_idx[cat['id']] = idx

        # If class_names not provided, build from categories
        if self.class_names is None:
            self.class_names = [self.categories[cat_id] for cat_id in sorted(self.categories.keys())]

        # Add annotations to images
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id in self.images:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                # Convert to xyxy format
                box = [x, y, x + w, y + h]
                # Map category id to class index
                class_idx = self.cat_id_to_idx.get(ann['category_id'], 0)

                self.images[img_id]['annotations'].append({
                    'bbox': box,
                    'class_id': class_idx,
                    'area': ann.get('area', w * h),
                    'iscrowd': ann.get('iscrowd', 0),
                })

        # Create ordered list of image ids
        self.img_ids = list(self.images.keys())

        # Apply max_samples limit
        if self.max_samples is not None:
            self.img_ids = self.img_ids[:self.max_samples]

        print(f"Loaded {len(self.img_ids)} images with "
              f"{sum(len(self.images[i]['annotations']) for i in self.img_ids)} annotations")

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, target_dict)
            - image_tensor: [C, H, W] normalized tensor
            - target_dict: {'boxes': [N, 4], 'labels': [N], 'image_id': int}
        """
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = load_image(str(img_path))

        # Get annotations
        boxes = []
        labels = []
        for ann in img_info['annotations']:
            if ann['iscrowd'] == 0:  # Skip crowd annotations
                boxes.append(ann['bbox'])
                labels.append(ann['class_id'])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
        }

        # Apply transform
        if self.transform is not None:
            image, target = self.transform(image, target, idx=idx)
        else:
            # Basic resize and normalize
            image = self._basic_transform(image)
            target['boxes'] = torch.from_numpy(target['boxes'])
            target['labels'] = torch.from_numpy(target['labels'])

        return image, target

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
        Sample a random image and target.

        Used by multi-image augmentations (Mosaic, MixUp, etc.)

        Args:
            exclude_idx: Index to exclude from sampling

        Returns:
            Tuple of (image, target) without augmentation
        """
        # Choose random index
        if exclude_idx is not None:
            valid_indices = [i for i in range(len(self.img_ids)) if i != exclude_idx]
            idx = random.choice(valid_indices)
        else:
            idx = random.randint(0, len(self.img_ids) - 1)

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = load_image(str(img_path))

        # Get annotations
        boxes = []
        labels = []
        for ann in img_info['annotations']:
            if ann['iscrowd'] == 0:
                boxes.append(ann['bbox'])
                labels.append(ann['class_id'])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        return image, target

    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names) if self.class_names else len(self.categories)

    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.class_names or list(self.categories.values())


class COCODataset(ODDataset):
    """
    Alias for ODDataset with COCO format.

    This is the same as ODDataset but with a more explicit name.
    """
    pass


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


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create DataLoader with proper collate function.

    Args:
        dataset: ODDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


class InMemoryDataset(Dataset):
    """
    In-memory dataset for small datasets or debugging.

    Loads all images into memory for faster access.
    Only suitable for small datasets.

    Args:
        img_dir: Image directory
        ann_file: Annotation file
        transform: Transform pipeline
        img_size: Image size
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transform: Optional[Callable] = None,
        img_size: int = 640,
    ):
        self.transform = transform
        self.img_size = img_size

        # Load annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build mappings
        images = {}
        for img_info in coco_data.get('images', []):
            images[img_info['id']] = {
                'file_name': img_info['file_name'],
                'annotations': [],
            }

        cat_id_to_idx = {}
        for idx, cat in enumerate(coco_data.get('categories', [])):
            cat_id_to_idx[cat['id']] = idx

        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id in images:
                x, y, w, h = ann['bbox']
                images[img_id]['annotations'].append({
                    'bbox': [x, y, x + w, y + h],
                    'class_id': cat_id_to_idx.get(ann['category_id'], 0),
                })

        # Load all images into memory
        self.data = []
        print("Loading images into memory...")
        for img_id, img_info in images.items():
            img_path = Path(img_dir) / img_info['file_name']
            if img_path.exists():
                image = load_image(str(img_path))

                boxes = [ann['bbox'] for ann in img_info['annotations']]
                labels = [ann['class_id'] for ann in img_info['annotations']]

                self.data.append({
                    'image': image,
                    'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                    'labels': np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64),
                    'image_id': img_id,
                })

        print(f"Loaded {len(self.data)} images into memory")

        # Set sample function
        if transform is not None and hasattr(transform, 'set_sample_fn'):
            transform.set_sample_fn(self._sample_random)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        sample = self.data[idx]

        image = sample['image'].copy()
        target = {
            'boxes': sample['boxes'].copy(),
            'labels': sample['labels'].copy(),
            'image_id': sample['image_id'],
        }

        if self.transform is not None:
            image, target = self.transform(image, target, idx=idx)

        return image, target

    def _sample_random(self, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sample random image for multi-image augmentations."""
        if exclude_idx is not None:
            valid_indices = [i for i in range(len(self.data)) if i != exclude_idx]
            idx = random.choice(valid_indices)
        else:
            idx = random.randint(0, len(self.data) - 1)

        sample = self.data[idx]
        return sample['image'].copy(), {
            'boxes': sample['boxes'].copy(),
            'labels': sample['labels'].copy(),
        }
