"""
Create dummy COCO dataset for testing.
"""

import os
import json
import numpy as np
from PIL import Image
from datetime import datetime


def create_dummy_coco_dataset(
    output_path: str,
    num_images: int = 20,
    num_classes: int = 3,
    img_size: int = 640,
):
    """
    Create a dummy COCO format dataset for testing.

    Args:
        output_path: Output directory
        num_images: Number of images to create
        num_classes: Number of classes
        img_size: Image size
    """
    # Create directories
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/annotations", exist_ok=True)

    # Split train/val
    num_train = int(num_images * 0.8)
    num_val = num_images - num_train

    # Create train set
    train_coco = _create_split(
        output_path=output_path,
        split="train",
        num_images=num_train,
        num_classes=num_classes,
        img_size=img_size,
    )

    # Create val set
    val_coco = _create_split(
        output_path=output_path,
        split="val",
        num_images=num_val,
        num_classes=num_classes,
        img_size=img_size,
        start_id=num_train + 1,
    )

    # Save annotations
    with open(f"{output_path}/annotations/train.json", "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(f"{output_path}/annotations/val.json", "w") as f:
        json.dump(val_coco, f, indent=2)

    print(f"Created dummy dataset: {num_train} train, {num_val} val images")
    return output_path


def _create_split(
    output_path: str,
    split: str,
    num_images: int,
    num_classes: int,
    img_size: int,
    start_id: int = 1,
):
    """Create a single split."""
    coco = {
        "info": {
            "description": "Dummy COCO dataset for testing",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": [
            {"id": i, "name": f"class_{i}", "supercategory": "object"}
            for i in range(num_classes)
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1

    for i in range(num_images):
        img_id = start_id + i
        filename = f"img_{img_id:05d}.jpg"

        # Create random image
        img_array = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f"{output_path}/images/{split}/{filename}")

        # Add image info
        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_size,
            "height": img_size,
        })

        # Add random annotations (1-5 objects per image)
        num_objects = np.random.randint(1, 6)
        for _ in range(num_objects):
            # Random box
            x = np.random.randint(0, img_size - 50)
            y = np.random.randint(0, img_size - 50)
            w = np.random.randint(30, min(100, img_size - x))
            h = np.random.randint(30, min(100, img_size - y))

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": np.random.randint(0, num_classes),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return coco
