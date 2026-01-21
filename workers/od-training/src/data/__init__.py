"""
Dataset module for Object Detection Training.

Provides:
- ODDataset: Main dataset class for training/validation
- COCODataset: COCO format dataset loader
- DataLoader utilities with collate functions
- YOLO to COCO format conversion
- Dummy dataset creation for testing
"""

from .dataset import ODDataset, COCODataset, create_dataloader
from .yolo_to_coco import convert_yolo_to_coco, get_yolo_class_names
from .dummy_dataset import create_dummy_coco_dataset

__all__ = [
    "ODDataset",
    "COCODataset",
    "create_dataloader",
    "convert_yolo_to_coco",
    "get_yolo_class_names",
    "create_dummy_coco_dataset",
]
