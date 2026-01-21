"""
YOLO to COCO Format Converter.

Converts YOLO format annotations to COCO format for training.

YOLO format:
- labels/train/*.txt (class_id x_center y_center width height)
- images/train/*.jpg
- data.yaml (names, nc, train, val paths)

COCO format:
- annotations/train.json
- annotations/val.json
- images/train/
- images/val/
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from datetime import datetime


def convert_yolo_to_coco(
    yolo_dataset_path: str,
    output_path: str,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Convert YOLO format dataset to COCO format.

    Args:
        yolo_dataset_path: Path to YOLO dataset root
        output_path: Output path for COCO format dataset
        class_names: Optional class names (will try to read from data.yaml)

    Returns:
        Path to converted COCO dataset
    """
    yolo_path = Path(yolo_dataset_path)
    out_path = Path(output_path)

    # Try to read data.yaml for class names
    if class_names is None:
        data_yaml = yolo_path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, "r") as f:
                data = yaml.safe_load(f)
                class_names = data.get("names", [])
        else:
            raise ValueError("class_names not provided and data.yaml not found")

    print(f"Converting YOLO dataset with {len(class_names)} classes")

    # Create output directories
    (out_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_path / "annotations").mkdir(parents=True, exist_ok=True)

    # Convert train and val splits
    for split in ["train", "val"]:
        _convert_split(
            yolo_path=yolo_path,
            out_path=out_path,
            split=split,
            class_names=class_names,
        )

    print(f"COCO dataset saved to {out_path}")
    return str(out_path)


def _convert_split(
    yolo_path: Path,
    out_path: Path,
    split: str,
    class_names: List[str],
):
    """Convert a single split (train/val)."""
    # Find images and labels
    images_dir = _find_images_dir(yolo_path, split)
    labels_dir = _find_labels_dir(yolo_path, split)

    if images_dir is None:
        print(f"Warning: No images found for {split} split")
        return

    print(f"Converting {split} split from {images_dir}")

    # COCO structure
    coco = {
        "info": {
            "description": f"Converted from YOLO format",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    image_id = 1

    # Process each image
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    for img_file in sorted(image_files):
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            continue

        # Add image info
        coco["images"].append({
            "id": image_id,
            "file_name": img_file.name,
            "width": width,
            "height": height,
        })

        # Find corresponding label file
        if labels_dir:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                annotations = _parse_yolo_labels(
                    label_file, width, height, image_id, annotation_id
                )
                coco["annotations"].extend(annotations)
                annotation_id += len(annotations)

        # Copy/symlink image to output
        out_img = out_path / "images" / split / img_file.name
        if not out_img.exists():
            import shutil
            shutil.copy2(img_file, out_img)

        image_id += 1

    # Save COCO JSON
    ann_file = out_path / "annotations" / f"{split}.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def _find_images_dir(yolo_path: Path, split: str) -> Optional[Path]:
    """Find images directory for a split."""
    possible_paths = [
        yolo_path / split / "images",
        yolo_path / "images" / split,
        yolo_path / split,
    ]
    for p in possible_paths:
        if p.exists() and p.is_dir():
            return p
    return None


def _find_labels_dir(yolo_path: Path, split: str) -> Optional[Path]:
    """Find labels directory for a split."""
    possible_paths = [
        yolo_path / split / "labels",
        yolo_path / "labels" / split,
    ]
    for p in possible_paths:
        if p.exists() and p.is_dir():
            return p
    return None


def _parse_yolo_labels(
    label_file: Path,
    img_width: int,
    img_height: int,
    image_id: int,
    start_ann_id: int,
) -> List[Dict]:
    """Parse YOLO label file and convert to COCO annotations."""
    annotations = []
    ann_id = start_ann_id

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert from YOLO (normalized xywh center) to COCO (absolute xywh top-left)
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            if w <= 0 or h <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return annotations


def get_yolo_class_names(yolo_dataset_path: str) -> List[str]:
    """Get class names from YOLO data.yaml."""
    data_yaml = Path(yolo_dataset_path) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {yolo_dataset_path}")

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    return data.get("names", [])
