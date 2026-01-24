"""
Handler for exporting datasets with progress tracking.

This handler wraps the existing od_export_service to provide
real-time progress updates during export operations.
"""

import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Callable, Optional

from services.supabase import supabase_service
from services.od_export import od_export_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class ExportDatasetHandler(BaseJobHandler):
    """
    Handler for exporting OD datasets with progress tracking.

    Config:
        dataset_id: str - Dataset to export
        format: str - Export format (yolo or coco)
        include_images: bool - Whether to include image files (default: True)
        version_id: str (optional) - Version ID for versioned exports
        split: str (optional) - Split filter (train/val/test)
        train_split: float (optional) - Train ratio for YOLO (default: 0.8)
        val_split: float (optional) - Val ratio for YOLO (default: 0.15)
        test_split: float (optional) - Test ratio for YOLO (default: 0.05)

    Result:
        download_url: str - URL to download the export
        total_images: int - Number of images exported
        total_annotations: int - Number of annotations exported
        total_classes: int - Number of classes
    """

    job_type = "local_export_dataset"

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        if not config.get("format"):
            return "format is required"
        if config["format"] not in ["yolo", "coco"]:
            return "format must be 'yolo' or 'coco'"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        dataset_id = config["dataset_id"]
        export_format = config["format"]
        include_images = config.get("include_images", True)
        version_id = config.get("version_id")
        split = config.get("split")

        # Update initial progress
        update_progress(JobProgress(
            progress=0,
            current_step="Initializing export...",
            processed=0,
            total=0,
        ))

        # Verify dataset exists
        dataset = supabase_service.client.table("od_datasets")\
            .select("id, name")\
            .eq("id", dataset_id)\
            .single()\
            .execute()

        if not dataset.data:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get export data
        update_progress(JobProgress(
            progress=5,
            current_step="Fetching dataset data...",
            processed=0,
            total=0,
        ))

        export_data = await od_export_service.get_dataset_for_export(
            dataset_id=dataset_id,
            version_id=version_id,
            split=split,
        )

        total_images = len(export_data["images"])
        total_annotations = export_data["total_annotations"]

        if total_images == 0:
            return {
                "download_url": None,
                "total_images": 0,
                "total_annotations": 0,
                "total_classes": len(export_data["classes"]),
                "message": "No images to export (ensure images are marked as completed)",
            }

        update_progress(JobProgress(
            progress=10,
            current_step=f"Found {total_images} images, {total_annotations} annotations",
            processed=0,
            total=total_images,
        ))

        # Generate export with progress tracking
        if export_format == "yolo":
            zip_path = self._export_yolo_with_progress(
                export_data=export_data,
                include_images=include_images,
                train_split=config.get("train_split", 0.8),
                val_split=config.get("val_split", 0.15),
                test_split=config.get("test_split", 0.05),
                update_progress=update_progress,
                total_images=total_images,
            )
        else:
            zip_path = self._export_coco_with_progress(
                export_data=export_data,
                include_images=include_images,
                split=split,
                update_progress=update_progress,
                total_images=total_images,
            )

        # Upload to storage
        update_progress(JobProgress(
            progress=95,
            current_step="Uploading export file...",
            processed=total_images,
            total=total_images,
        ))

        bucket = "od-exports"
        file_name = f"{dataset_id}/{job_id}_{export_format}.zip"

        with open(zip_path, "rb") as f:
            supabase_service.client.storage.from_(bucket).upload(
                file_name,
                f.read(),
                {"content-type": "application/zip"}
            )

        download_url = supabase_service.client.storage.from_(bucket).get_public_url(file_name)

        # Clean up temp file
        os.remove(zip_path)

        return {
            "download_url": download_url,
            "total_images": total_images,
            "total_annotations": total_annotations,
            "total_classes": len(export_data["classes"]),
            "message": f"Exported {total_images} images with {total_annotations} annotations",
        }

    def _export_yolo_with_progress(
        self,
        export_data: dict,
        include_images: bool,
        train_split: float,
        val_split: float,
        test_split: float,
        update_progress: Callable[[JobProgress], None],
        total_images: int,
    ) -> str:
        """Export in YOLO format with progress updates."""
        import random
        import yaml

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
        for split_name in ["train", "val", "test"]:
            os.makedirs(os.path.join(export_dir, split_name, "images"), exist_ok=True)
            os.makedirs(os.path.join(export_dir, split_name, "labels"), exist_ok=True)

        # Assign splits
        random.seed(42)
        train_images, val_images, test_images = [], [], []

        for img_data in images:
            existing_split = img_data.get("split")
            if existing_split == "train":
                train_images.append(img_data)
            elif existing_split == "val":
                val_images.append(img_data)
            elif existing_split == "test":
                test_images.append(img_data)
            else:
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
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Process images with progress
        processed = 0

        for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            for img_data in split_images:
                image = img_data.get("image", {})
                image_id = img_data["image_id"]

                if not image:
                    processed += 1
                    continue

                # Generate label file
                orig_filename = image.get("original_filename", image.get("filename", f"{image_id}.jpg"))
                base_name = os.path.splitext(orig_filename)[0]

                annotations = annotations_by_image.get(image_id, [])
                label_lines = []

                for ann in annotations:
                    class_id = ann.get("class_id")
                    if class_id not in class_mapping:
                        continue

                    class_idx = class_mapping[class_id]
                    bbox = ann.get("bbox", {})

                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)

                    center_x = max(0, min(1, x + w / 2))
                    center_y = max(0, min(1, y + h / 2))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    label_lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}")

                label_path = os.path.join(export_dir, split_name, "labels", f"{base_name}.txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))

                # Download image if requested
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
                        except Exception:
                            pass

                processed += 1

                # Update progress every 10 images
                if processed % 10 == 0:
                    progress = 10 + calculate_progress(processed, total_images) * 0.80
                    update_progress(JobProgress(
                        progress=int(progress),
                        current_step=f"Exporting images... ({processed}/{total_images})",
                        processed=processed,
                        total=total_images,
                    ))

        # Create ZIP
        update_progress(JobProgress(
            progress=90,
            current_step="Creating ZIP archive...",
            processed=total_images,
            total=total_images,
        ))

        zip_path = os.path.join(temp_dir, f"{dataset_name}_yolo.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arc_name)

        return zip_path

    def _export_coco_with_progress(
        self,
        export_data: dict,
        include_images: bool,
        split: Optional[str],
        update_progress: Callable[[JobProgress], None],
        total_images: int,
    ) -> str:
        """Export in COCO format with progress updates."""
        import json
        import random

        dataset = export_data["dataset"]
        classes = export_data["classes"]
        images = export_data["images"]
        annotations_by_image = export_data["annotations_by_image"]

        temp_dir = tempfile.mkdtemp()
        dataset_name = dataset["name"].replace(" ", "_").lower()
        export_dir = os.path.join(temp_dir, dataset_name)

        os.makedirs(os.path.join(export_dir, "annotations"), exist_ok=True)

        # Group images by split
        random.seed(42)
        images_by_split = {"train": [], "val": [], "test": []}

        for img_data in images:
            existing_split = img_data.get("split")
            if existing_split and existing_split in images_by_split:
                images_by_split[existing_split].append(img_data)
            else:
                r = random.random()
                if r < 0.8:
                    images_by_split["train"].append(img_data)
                else:
                    images_by_split["val"].append(img_data)

        processed = 0

        for split_name, split_images in images_by_split.items():
            if not split_images:
                continue

            if include_images:
                os.makedirs(os.path.join(export_dir, "images", split_name), exist_ok=True)

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
                "categories": [
                    {"id": cls["index"] + 1, "name": cls["name"], "supercategory": cls.get("category", "object")}
                    for cls in classes
                ],
            }

            annotation_id = 1
            for img_idx, img_data in enumerate(split_images, 1):
                image = img_data.get("image", {})
                image_id = img_data["image_id"]

                if not image:
                    processed += 1
                    continue

                orig_filename = image.get("original_filename", image.get("filename", f"{image_id}.jpg"))
                width = image.get("width", 640)
                height = image.get("height", 640)

                coco_data["images"].append({
                    "id": img_idx,
                    "file_name": orig_filename,
                    "width": width,
                    "height": height,
                })

                for ann in annotations_by_image.get(image_id, []):
                    class_id = ann.get("class_id")
                    class_info = next((c for c in classes if c["id"] == class_id), None)
                    if not class_info:
                        continue

                    bbox = ann.get("bbox", {})
                    x = bbox.get("x", 0) * width
                    y = bbox.get("y", 0) * height
                    w = bbox.get("width", 0) * width
                    h = bbox.get("height", 0) * height

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_id": class_info["index"] + 1,
                        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                        "area": round(w * h, 2),
                        "iscrowd": 0,
                    })
                    annotation_id += 1

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
                        except Exception:
                            pass

                processed += 1

                if processed % 10 == 0:
                    progress = 10 + calculate_progress(processed, total_images) * 0.80
                    update_progress(JobProgress(
                        progress=int(progress),
                        current_step=f"Exporting images... ({processed}/{total_images})",
                        processed=processed,
                        total=total_images,
                    ))

            json_path = os.path.join(export_dir, "annotations", f"instances_{split_name}.json")
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=2)

        update_progress(JobProgress(
            progress=90,
            current_step="Creating ZIP archive...",
            processed=total_images,
            total=total_images,
        ))

        zip_path = os.path.join(temp_dir, f"{dataset_name}_coco.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    zf.write(file_path, arc_name)

        return zip_path
