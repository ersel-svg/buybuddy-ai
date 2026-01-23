"""
Dataset module for Object Detection Training.

Provides:
- ODDataset: Main dataset class for training/validation (file-based)
- URLODDataset: URL-based dataset class (loads from Supabase URLs)
- COCODataset: COCO format dataset loader
- DataLoader utilities with collate functions
- YOLO to COCO format conversion
- Dummy dataset creation for testing
- Supabase data fetching utilities
"""

from .dataset import ODDataset, COCODataset, create_dataloader
from .yolo_to_coco import convert_yolo_to_coco, get_yolo_class_names
from .dummy_dataset import create_dummy_coco_dataset
from .url_dataset import URLODDataset, create_url_dataloader
from .supabase_fetcher import build_url_dataset_data

__all__ = [
    "ODDataset",
    "COCODataset",
    "URLODDataset",
    "create_dataloader",
    "create_url_dataloader",
    "convert_yolo_to_coco",
    "get_yolo_class_names",
    "create_dummy_coco_dataset",
    "build_url_dataset_data",
]
