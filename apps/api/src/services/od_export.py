"""
Object Detection - Export Service

Exports datasets in various formats:
- YOLO (data.yaml + labels/*.txt + images/)
- COCO (annotations.json + images/)
"""

import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from services.supabase import supabase_service


class ODExportService:
    """Service for exporting OD datasets."""

    def __init__(self):
        self.client = supabase_service.client
        self.storage = supabase_service.client.storage

    async def get_dataset_for_export(
        self,
        dataset_id: str,
        version_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> dict:
        """
        Get dataset with all images, annotations, and classes for export.

        Args:
            dataset_id: Dataset UUID
            version_id: Optional version UUID (for versioned exports)
            split: Optional split filter (train/val/test)

        Returns:
            dict with dataset, images, annotations, and classes
        """
        # Get dataset
        dataset = self.client.table("od_datasets").select("*").eq("id", dataset_id).single().execute()
        if not dataset.data:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Get classes for this dataset
        classes = self.client.table("od_classes").select("*").eq("dataset_id", dataset_id).eq("is_active", True).order("name").execute()

        # Build class name to index mapping (for YOLO format)
        class_mapping = {}
        class_list = []
        for idx, cls in enumerate(classes.data or []):
            class_mapping[cls["id"]] = idx
            class_list.append({
                "id": cls["id"],
                "name": cls["name"],
                "display_name": cls.get("display_name", cls["name"]),
                "color": cls.get("color", "#3B82F6"),
                "index": idx,
            })

        # Get dataset images (with pagination for large datasets)
        all_dataset_images = []
        page_size = 1000
        offset = 0

        while True:
            query = self.client.table("od_dataset_images").select(
                "*, image:od_images(*)"
            ).eq("dataset_id", dataset_id)

            if split:
                query = query.eq("split", split)

            # Only include completed/annotated images for export
            query = query.in_("status", ["completed", "annotating"])
            query = query.range(offset, offset + page_size - 1)

            result = query.execute()
            batch = result.data or []
            all_dataset_images.extend(batch)

            # If we got fewer than page_size, we've reached the end
            if len(batch) < page_size:
                break
            offset += page_size

        # Use combined results
        dataset_images = type('obj', (object,), {'data': all_dataset_images})()

        # Get all annotations for these images (with pagination for large datasets)
        image_ids = [di["image_id"] for di in dataset_images.data or []]

        annotations = []
        if image_ids:
            # Batch fetch annotations with pagination (Supabase limits to ~1000 per request)
            batch_size = 500  # Smaller batch for image_id IN queries
            page_size = 1000  # Rows per page

            # Process image IDs in batches
            for i in range(0, len(image_ids), batch_size):
                batch_image_ids = image_ids[i:i + batch_size]

                # Paginate through annotations for this batch
                offset = 0
                while True:
                    ann_result = (
                        self.client.table("od_annotations")
                        .select("*")
                        .eq("dataset_id", dataset_id)
                        .in_("image_id", batch_image_ids)
                        .range(offset, offset + page_size - 1)
                        .execute()
                    )

                    batch_annotations = ann_result.data or []
                    annotations.extend(batch_annotations)

                    # If we got fewer than page_size, we've reached the end
                    if len(batch_annotations) < page_size:
                        break
                    offset += page_size

        # Group annotations by image_id
        annotations_by_image = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        return {
            "dataset": dataset.data,
            "classes": class_list,
            "class_mapping": class_mapping,
            "images": dataset_images.data or [],
            "annotations_by_image": annotations_by_image,
            "total_annotations": len(annotations),
        }

    def export_yolo(
        self,
        export_data: dict,
        include_images: bool = True,
        image_size: Optional[int] = None,
        train_split: float = 0.8,
        val_split: float = 0.15,
        test_split: float = 0.05,
    ) -> str:
        """
        Export dataset in YOLO format.

        Structure:
        dataset_name/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/

        Returns:
            Path to generated ZIP file
        """
        dataset = export_data["dataset"]
        classes = export_data["classes"]
        images = export_data["images"]
        annotations_by_image = export_data["annotations_by_image"]
        class_mapping = export_data["class_mapping"]

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        dataset_name = dataset["name"].replace(" ", "_").lower()
        export_dir = os.path.join(temp_dir, dataset_name)

        # Create directory structure
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(export_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(export_dir, split, "labels"), exist_ok=True)

        # Assign splits if not already assigned
        import random
        random.seed(42)

        train_images = []
        val_images = []
        test_images = []

        for img_data in images:
            existing_split = img_data.get("split")
            if existing_split:
                if existing_split == "train":
                    train_images.append(img_data)
                elif existing_split == "val":
                    val_images.append(img_data)
                else:
                    test_images.append(img_data)
            else:
                # Random assignment based on split ratios
                r = random.random()
                if r < train_split:
                    train_images.append(img_data)
                elif r < train_split + val_split:
                    val_images.append(img_data)
                else:
                    test_images.append(img_data)

        # Generate data.yaml
        data_yaml = {
            "path": f"./{dataset_name}",
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(classes),
            "names": [cls["name"] for cls in classes],
        }

        with open(os.path.join(export_dir, "data.yaml"), "w") as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Process each split
        for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            for img_data in split_images:
                image = img_data.get("image", {})
                image_id = img_data["image_id"]

                if not image:
                    continue

                # Get image dimensions
                width = image.get("width", 640)
                height = image.get("height", 640)

                # Generate filename (use original or create from id)
                orig_filename = image.get("original_filename", image.get("filename", f"{image_id}.jpg"))
                base_name = os.path.splitext(orig_filename)[0]

                # Create label file
                annotations = annotations_by_image.get(image_id, [])
                label_lines = []

                for ann in annotations:
                    class_id = ann.get("class_id")
                    if class_id not in class_mapping:
                        continue

                    class_idx = class_mapping[class_id]
                    bbox = ann.get("bbox", {})

                    # Convert from x,y,width,height (top-left) to YOLO format (center x, center y, width, height)
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)

                    # YOLO format: center_x, center_y, width, height (all normalized 0-1)
                    center_x = x + w / 2
                    center_y = y + h / 2

                    # Clamp values to 0-1
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    label_lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")

                # Write label file
                label_path = os.path.join(export_dir, split_name, "labels", f"{base_name}.txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

                # Download and save image if requested
                if include_images:
                    image_url = image.get("image_url")
                    if image_url:
                        try:
                            import httpx
                            response = httpx.get(image_url, timeout=30)
                            if response.status_code == 200:
                                ext = os.path.splitext(orig_filename)[1] or ".jpg"
                                image_path = os.path.join(export_dir, split_name, "images", f"{base_name}{ext}")
                                with open(image_path, "wb") as f:
                                    f.write(response.content)
                        except Exception as e:
                            print(f"Failed to download image {image_id}: {e}")

        # Create ZIP file
        zip_path = os.path.join(temp_dir, f"{dataset_name}_yolo.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arc_name)

        return zip_path

    def export_coco(
        self,
        export_data: dict,
        include_images: bool = True,
        split: Optional[str] = None,
    ) -> str:
        """
        Export dataset in COCO format.

        Structure:
        dataset_name/
        ├── annotations/
        │   ├── instances_train.json
        │   ├── instances_val.json
        │   └── instances_test.json
        └── images/
            ├── train/
            ├── val/
            └── test/

        Returns:
            Path to generated ZIP file
        """
        dataset = export_data["dataset"]
        classes = export_data["classes"]
        images = export_data["images"]
        annotations_by_image = export_data["annotations_by_image"]

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        dataset_name = dataset["name"].replace(" ", "_").lower()
        export_dir = os.path.join(temp_dir, dataset_name)

        os.makedirs(os.path.join(export_dir, "annotations"), exist_ok=True)

        # Group images by split with automatic train/val assignment
        images_by_split = {"train": [], "val": [], "test": []}

        import random
        random.seed(42)

        # Default split ratios
        train_ratio = 0.8
        val_ratio = 0.2

        for img_data in images:
            existing_split = img_data.get("split")
            if existing_split and existing_split in images_by_split:
                images_by_split[existing_split].append(img_data)
            else:
                # Auto-assign split based on ratios
                r = random.random()
                if r < train_ratio:
                    images_by_split["train"].append(img_data)
                else:
                    images_by_split["val"].append(img_data)

        # Generate COCO JSON for each split
        for split_name, split_images in images_by_split.items():
            if not split_images:
                continue

            if include_images:
                os.makedirs(os.path.join(export_dir, "images", split_name), exist_ok=True)

            # Build COCO structure
            coco_data = {
                "info": {
                    "description": f"{dataset['name']} - {split_name} split",
                    "version": str(dataset.get("version", 1)),
                    "year": datetime.now().year,
                    "contributor": "BuyBuddy AI",
                    "date_created": datetime.now(timezone.utc).isoformat(),
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [],
            }

            # Add categories
            for cls in classes:
                coco_data["categories"].append({
                    "id": cls["index"] + 1,  # COCO uses 1-indexed
                    "name": cls["name"],
                    "supercategory": cls.get("category", "object"),
                })

            # Add images and annotations
            annotation_id = 1
            for img_idx, img_data in enumerate(split_images, 1):
                image = img_data.get("image", {})
                image_id = img_data["image_id"]

                if not image:
                    continue

                # Image filename
                orig_filename = image.get("original_filename", image.get("filename", f"{image_id}.jpg"))

                # Add image entry
                width = image.get("width", 640)
                height = image.get("height", 640)

                coco_data["images"].append({
                    "id": img_idx,
                    "file_name": orig_filename,
                    "width": width,
                    "height": height,
                    "date_captured": image.get("created_at", datetime.now(timezone.utc).isoformat()),
                })

                # Add annotations
                for ann in annotations_by_image.get(image_id, []):
                    class_id = ann.get("class_id")
                    class_info = next((c for c in classes if c["id"] == class_id), None)
                    if not class_info:
                        continue

                    bbox = ann.get("bbox", {})

                    # Convert normalized bbox to pixel coordinates
                    x = bbox.get("x", 0) * width
                    y = bbox.get("y", 0) * height
                    w = bbox.get("width", 0) * width
                    h = bbox.get("height", 0) * height

                    # COCO format: [x, y, width, height] in pixels
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_id": class_info["index"] + 1,
                        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "area": round(w * h, 2),
                        "iscrowd": 0,
                        "segmentation": [],  # Empty for bbox-only
                    })
                    annotation_id += 1

                # Download image if requested
                if include_images:
                    image_url = image.get("image_url")
                    if image_url:
                        try:
                            import httpx
                            response = httpx.get(image_url, timeout=30)
                            if response.status_code == 200:
                                image_path = os.path.join(export_dir, "images", split_name, orig_filename)
                                with open(image_path, "wb") as f:
                                    f.write(response.content)
                        except Exception as e:
                            print(f"Failed to download image {image_id}: {e}")

            # Write COCO JSON
            json_path = os.path.join(export_dir, "annotations", f"instances_{split_name}.json")
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=2)

        # Create ZIP file
        zip_path = os.path.join(temp_dir, f"{dataset_name}_coco.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arc_name)

        return zip_path

    async def create_export_job(
        self,
        dataset_id: str,
        format: str,
        include_images: bool = True,
        version_id: Optional[str] = None,
        split: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> dict:
        """
        Create an export job record and start processing.

        Args:
            dataset_id: Dataset UUID
            format: Export format (yolo, coco)
            include_images: Whether to include image files
            version_id: Optional version UUID
            split: Optional split filter
            config: Additional config (splits, image_size, etc.)

        Returns:
            Job record with ID and status
        """
        job_id = str(uuid4())

        # Create job record
        job_data = {
            "id": job_id,
            "type": "od_export",
            "status": "pending",
            "config": {
                "dataset_id": dataset_id,
                "format": format,
                "include_images": include_images,
                "version_id": version_id,
                "split": split,
                **(config or {}),
            },
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # For now, process synchronously (can be made async later)
        try:
            job_data["status"] = "processing"

            # Get export data
            export_data = await self.get_dataset_for_export(
                dataset_id=dataset_id,
                version_id=version_id,
                split=split,
            )

            # Generate export
            if format == "yolo":
                zip_path = self.export_yolo(
                    export_data=export_data,
                    include_images=include_images,
                    train_split=config.get("train_split", 0.8) if config else 0.8,
                    val_split=config.get("val_split", 0.15) if config else 0.15,
                    test_split=config.get("test_split", 0.05) if config else 0.05,
                )
            elif format == "coco":
                zip_path = self.export_coco(
                    export_data=export_data,
                    include_images=include_images,
                    split=split,
                )
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Upload to Supabase storage
            bucket = "od-exports"
            file_name = f"{dataset_id}/{job_id}_{format}.zip"

            with open(zip_path, "rb") as f:
                self.storage.from_(bucket).upload(
                    file_name,
                    f.read(),
                    {"content-type": "application/zip"}
                )

            # Get public URL
            download_url = self.storage.from_(bucket).get_public_url(file_name)

            # Clean up temp file
            os.remove(zip_path)

            job_data["status"] = "completed"
            job_data["download_url"] = download_url
            job_data["progress"] = 100
            job_data["result"] = {
                "total_images": len(export_data["images"]),
                "total_annotations": export_data["total_annotations"],
                "total_classes": len(export_data["classes"]),
            }
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            job_data["status"] = "failed"
            job_data["error"] = str(e)

        return job_data


# Singleton instance
od_export_service = ODExportService()
