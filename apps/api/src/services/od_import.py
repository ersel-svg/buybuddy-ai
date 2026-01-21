"""
Object Detection - Import Service

Handles importing images and annotations from various formats:
- COCO JSON
- YOLO (data.yaml + labels/)
- Pascal VOC (XML)
- LabelMe JSON
- URL import
- ZIP extraction

Also handles duplicate detection using perceptual hash (pHash).
"""

import io
import json
import zipfile
import tempfile
import httpx
import hashlib
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from uuid import uuid4
from PIL import Image

logger = logging.getLogger(__name__)

# Optional: imagehash for perceptual hashing (not required for basic functionality)
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None
    IMAGEHASH_AVAILABLE = False

from services.supabase import supabase_service


# ===========================================
# Color Generation for New Classes
# ===========================================

# Predefined colors for new classes (distinguishable colors)
DEFAULT_CLASS_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FFD700",
    "#8A2BE2", "#00FA9A", "#FF4500", "#1E90FF", "#FF1493",
]

_color_index = 0

def get_next_class_color() -> str:
    """Get the next color from the predefined color list."""
    global _color_index
    color = DEFAULT_CLASS_COLORS[_color_index % len(DEFAULT_CLASS_COLORS)]
    _color_index += 1
    return color


# ===========================================
# Concurrent Import Protection
# ===========================================

async def check_concurrent_import(dataset_id: str) -> tuple[bool, str | None]:
    """
    Check if there's already an import running for this dataset.

    Returns:
        (is_locked, job_id): is_locked is True if another import is running
    """
    try:
        result = supabase_service.client.table("jobs").select(
            "id, status"
        ).eq("resource_id", dataset_id).in_(
            "status", ["pending", "processing"]
        ).eq("job_type", "roboflow_import").execute()

        if result.data and len(result.data) > 0:
            return True, result.data[0]["id"]
        return False, None
    except Exception:
        # If we can't check, allow the import to proceed
        return False, None


# ===========================================
# Adaptive Rate Limiter for Bulk Uploads
# ===========================================

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that slows down when errors occur.
    Thread-safe for use with concurrent uploads.
    """
    def __init__(self):
        self.consecutive_errors = 0
        self.delay_multiplier = 1.0
        self.max_multiplier = 10.0
        self.base_delay = 0.15  # 150ms base delay between uploads
        self._lock = asyncio.Lock() if asyncio else None

    def reset(self):
        """Reset to default state."""
        self.consecutive_errors = 0
        self.delay_multiplier = 1.0

    def on_error(self):
        """Called when a rate limit error occurs."""
        self.consecutive_errors += 1
        self.delay_multiplier = min(
            1.0 + (self.consecutive_errors * 0.5),
            self.max_multiplier
        )
        logger.info(f"Rate limit hit #{self.consecutive_errors}, delay multiplier: {self.delay_multiplier:.1f}x")

    def on_success(self):
        """Called after successful upload."""
        if self.consecutive_errors > 0:
            self.consecutive_errors = max(0, self.consecutive_errors - 1)
            self.delay_multiplier = max(1.0, self.delay_multiplier - 0.25)

    def get_delay(self) -> float:
        """Get current delay to apply before next upload."""
        return self.base_delay * self.delay_multiplier


# Global rate limiter instance
_rate_limiter = AdaptiveRateLimiter()


# ===========================================
# Storage Upload Helper with Retry (Enhanced)
# ===========================================

def upload_to_storage_with_retry(
    filename: str,
    content: bytes,
    content_type: str,
    max_retries: int = 8,  # Increased from 3 for stability
    base_delay: float = 3.0,  # Increased from 2.0
) -> tuple[bool, str]:
    """
    Upload a file to Supabase Storage with enhanced retry logic.

    Features:
    - Adaptive rate limiting: slows down when errors occur
    - Duplicate detection: treats 409 as "soft success"
    - More retries with longer delays for stability
    - Exponential backoff with jitter

    Args:
        filename: Target filename in storage
        content: File content bytes
        content_type: MIME type
        max_retries: Maximum retry attempts (default 8)
        base_delay: Base delay in seconds (default 3.0)

    Returns:
        Tuple of (success: bool, status: str):
        - (True, "uploaded") - New file uploaded successfully
        - (True, "exists") - File already existed (duplicate, 409)
        - (False, error_message) - Upload failed
    """
    import random
    last_error = None

    # Apply adaptive delay before upload
    adaptive_delay = _rate_limiter.get_delay()
    if adaptive_delay > 0:
        time.sleep(adaptive_delay)

    for attempt in range(max_retries):
        try:
            supabase_service.client.storage.from_("od-images").upload(
                filename,
                content,
                {"content-type": content_type}
            )
            _rate_limiter.on_success()
            return (True, "uploaded")

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            # Check for duplicate (409 Conflict) - treat as soft success
            if "409" in str(e) or "duplicate" in error_msg or "already exists" in error_msg:
                logger.info(f"File {filename} already exists (409), treating as success")
                return (True, "exists")

            # Check if it's a retriable error
            retriable_terms = [
                "timeout", "timed out", "connection", "network", "socket",
                "resource temporarily unavailable", "errno 35", "eagain",
                "temporarily unavailable", "try again", "rate", "throttl",
                "too many requests", "429", "503", "502",
                # Connection reset / broken pipe errors
                "broken pipe", "errno 32", "connection reset", "reset by peer",
                "eof occurred", "incomplete read", "server disconnected",
                "connection aborted", "errno 54", "errno 104",
                # JSON parse errors (server returned non-JSON response, usually temporary)
                "json", "decode", "expecting value", "unterminated string",
                "invalid control character", "jsondecodeerror"
            ]

            is_rate_limit = any(term in error_msg for term in retriable_terms)

            if is_rate_limit:
                _rate_limiter.on_error()

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    base = base_delay * (2 ** attempt) * _rate_limiter.delay_multiplier
                    jitter = random.uniform(0.8, 1.2)
                    delay = min(base * jitter, 120.0)  # Cap at 2 minutes

                    logger.warning(
                        f"Upload attempt {attempt + 1}/{max_retries} failed for {filename}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
            else:
                # Non-retriable error
                return (False, f"Non-retriable error: {str(e)}")

    # All retries exhausted
    return (False, f"Failed after {max_retries} attempts: {last_error}")


async def upload_single_image(
    semaphore: asyncio.Semaphore,
    filename: str,
    content: bytes,
    content_type: str,
    max_retries: int = 8,
) -> tuple[bool, str, Optional[str]]:
    """
    Upload a single image with semaphore-controlled concurrency.

    Returns:
        Tuple of (success, filename, error_message)
    """
    async with semaphore:
        try:
            # Run sync upload in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            success, status = await loop.run_in_executor(
                None,
                lambda: upload_to_storage_with_retry(filename, content, content_type, max_retries)
            )
            if success:
                return (True, filename, None)
            else:
                return (False, filename, status)
        except Exception as e:
            return (False, filename, str(e))


async def parallel_upload_images(
    images: list[tuple[str, bytes, str]],  # List of (filename, content, content_type)
    max_concurrent: int = 2,  # Reduced from 3 for better stability
    progress_callback: Optional[callable] = None,
    batch_size: int = 100,  # Process in batches
    batch_delay: float = 2.0,  # Delay between batches
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Upload multiple images with controlled concurrency and batch processing.

    Enhanced for 10,000+ image uploads:
    - Batch processing: uploads in chunks with delays between batches
    - Lower concurrency (2 concurrent) for stability
    - Adaptive rate limiting
    - Progress tracking
    - Graceful error handling

    Args:
        images: List of (filename, content, content_type) tuples
        max_concurrent: Maximum concurrent uploads (default 2)
        progress_callback: Optional callback(completed, total) for progress
        batch_size: Number of images per batch (default 100)
        batch_delay: Seconds to wait between batches (default 2.0)

    Returns:
        Tuple of (successful_filenames, failed_list[(filename, error)])
    """
    # Reset rate limiter at start of bulk upload
    _rate_limiter.reset()

    semaphore = asyncio.Semaphore(max_concurrent)
    successful = []
    failed = []
    total = len(images)
    completed = 0

    # Process in batches for large uploads
    num_batches = (total + batch_size - 1) // batch_size
    logger.info(f"Starting upload of {total} images in {num_batches} batches (batch_size={batch_size}, concurrent={max_concurrent})")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch = images[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} images)")

        # Create upload tasks for this batch
        tasks = [
            upload_single_image(semaphore, filename, content, content_type)
            for filename, content, content_type in batch
        ]

        # Process batch results
        for coro in asyncio.as_completed(tasks):
            success, filename, error = await coro
            completed += 1

            if success:
                successful.append(filename)
            else:
                failed.append((filename, error))

            # Progress callback every 10 images
            if progress_callback and completed % 10 == 0:
                progress_callback(completed, total)

        # Delay between batches (except for last batch)
        if batch_idx < num_batches - 1:
            # Adaptive delay: increase if we had errors in this batch
            actual_delay = batch_delay * _rate_limiter.delay_multiplier
            logger.info(f"Batch {batch_idx + 1} complete. Waiting {actual_delay:.1f}s before next batch...")
            await asyncio.sleep(actual_delay)

    logger.info(f"Upload complete: {len(successful)} succeeded, {len(failed)} failed out of {total}")
    return successful, failed


# ===========================================
# Data Classes
# ===========================================

@dataclass
class ImportedAnnotation:
    """Represents a single annotation from imported data."""
    class_name: str
    bbox_x: float  # normalized 0-1
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: Optional[float] = None


@dataclass
class ImportedImage:
    """Represents an image with its annotations from imported data."""
    filename: str
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    annotations: list[ImportedAnnotation] = field(default_factory=list)


@dataclass
class ClassMapping:
    """Maps source class name to target class ID."""
    source_name: str
    target_class_id: Optional[str] = None
    create_new: bool = False
    skip: bool = False
    color: Optional[str] = None


@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    images_imported: int = 0
    annotations_imported: int = 0
    images_skipped: int = 0
    duplicates_found: int = 0
    errors: list[str] = field(default_factory=list)
    class_mapping_needed: bool = False
    detected_classes: list[str] = field(default_factory=list)
    preview_images: list[dict] = field(default_factory=list)


@dataclass
class ImportPreview:
    """Preview of what will be imported."""
    format_detected: str
    total_images: int
    total_annotations: int
    classes_found: list[str]
    sample_images: list[dict]  # First 5 images with annotations for preview
    errors: list[str] = field(default_factory=list)


# ===========================================
# Hash-based Duplicate Detection
# ===========================================

def calculate_phash(image_data: bytes) -> str:
    """Calculate perceptual hash of an image (requires imagehash)."""
    if not IMAGEHASH_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(image_data))
        phash = imagehash.phash(img)
        return str(phash)
    except Exception:
        return ""


def calculate_file_hash(content: bytes) -> str:
    """Calculate MD5 hash of file content."""
    return hashlib.md5(content).hexdigest()


async def check_duplicate_by_phash(phash: str, threshold: int = 10) -> list[dict]:
    """
    Check if similar images exist based on pHash.
    Requires imagehash module to be installed.
    """
    if not IMAGEHASH_AVAILABLE or not phash:
        return []

    # Get all images with phash
    result = supabase_service.client.table("od_images").select(
        "id, filename, image_url, phash"
    ).not_.is_("phash", "null").execute()

    similar = []
    for img in result.data or []:
        if img.get("phash"):
            try:
                existing_hash = imagehash.hex_to_hash(img["phash"])
                new_hash = imagehash.hex_to_hash(phash)
                distance = existing_hash - new_hash
                if distance <= threshold:
                    similar.append({
                        "id": img["id"],
                        "filename": img["filename"],
                        "image_url": img["image_url"],
                        "distance": distance,
                        "similarity": 1 - (distance / 64)
                    })
            except Exception:
                continue

    return sorted(similar, key=lambda x: x["distance"])


async def check_duplicate_by_hash(file_hash: str) -> Optional[dict]:
    """Check if exact file exists by MD5 hash."""
    result = supabase_service.client.table("od_images").select(
        "id, filename, image_url"
    ).eq("file_hash", file_hash).limit(1).execute()

    return result.data[0] if result.data else None


# ===========================================
# COCO Format Parser
# ===========================================

def parse_coco_json(annotations_json: dict) -> tuple[list[ImportedImage], list[str]]:
    """
    Parse COCO format annotations.

    Expected structure:
    {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}, ...],
        "categories": [{"id": 1, "name": "class_name"}, ...]
    }

    Returns:
        Tuple of (list of ImportedImage, list of class names)
    """
    images_dict = {}
    categories = {}

    # Build category map
    for cat in annotations_json.get("categories", []):
        categories[cat["id"]] = cat["name"]

    # Build images dict
    for img in annotations_json.get("images", []):
        images_dict[img["id"]] = ImportedImage(
            filename=img["file_name"],
            width=img.get("width"),
            height=img.get("height"),
            annotations=[]
        )

    # Add annotations to images
    for ann in annotations_json.get("annotations", []):
        image_id = ann["image_id"]
        if image_id not in images_dict:
            continue

        img = images_dict[image_id]
        bbox = ann["bbox"]  # COCO bbox is [x, y, width, height] in pixels

        # Normalize bbox to 0-1 if we have image dimensions
        if img.width and img.height:
            norm_x = bbox[0] / img.width
            norm_y = bbox[1] / img.height
            norm_w = bbox[2] / img.width
            norm_h = bbox[3] / img.height
        else:
            # Keep as is, will need to normalize later
            norm_x, norm_y, norm_w, norm_h = bbox[0], bbox[1], bbox[2], bbox[3]

        class_name = categories.get(ann["category_id"], f"class_{ann['category_id']}")

        img.annotations.append(ImportedAnnotation(
            class_name=class_name,
            bbox_x=norm_x,
            bbox_y=norm_y,
            bbox_width=norm_w,
            bbox_height=norm_h,
            confidence=ann.get("score")
        ))

    return list(images_dict.values()), list(categories.values())


# ===========================================
# YOLO Format Parser
# ===========================================

def parse_yolo_labels(
    label_files: dict[str, str],  # filename -> content
    class_names: list[str],
    image_dimensions: dict[str, tuple[int, int]] = None
) -> list[ImportedImage]:
    """
    Parse YOLO format label files.

    YOLO format: class_id center_x center_y width height (all normalized 0-1)

    Args:
        label_files: Dict mapping label filename to its content
        class_names: List of class names from data.yaml
        image_dimensions: Optional dict mapping image filename to (width, height)

    Returns:
        List of ImportedImage with annotations
    """
    images = []

    for label_filename, content in label_files.items():
        # Convert label filename to image filename
        # e.g., "img_001.txt" -> "img_001.jpg" (we'll try common extensions)
        base_name = Path(label_filename).stem

        img = ImportedImage(
            filename=base_name,  # Will be matched to actual image later
            annotations=[]
        )

        for line in content.strip().split("\n"):
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5]) if len(parts) > 5 else None

            # Convert from center format to top-left format
            x = center_x - width / 2
            y = center_y - height / 2

            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            img.annotations.append(ImportedAnnotation(
                class_name=class_name,
                bbox_x=x,
                bbox_y=y,
                bbox_width=width,
                bbox_height=height,
                confidence=confidence
            ))

        if img.annotations:
            images.append(img)

    return images


def parse_yolo_data_yaml(yaml_content: str) -> list[str]:
    """Parse YOLO data.yaml to extract class names."""
    import yaml

    data = yaml.safe_load(yaml_content)

    # YOLO format can have 'names' as list or dict
    names = data.get("names", [])
    if isinstance(names, dict):
        # Convert dict {0: 'class1', 1: 'class2'} to list
        max_id = max(names.keys()) if names else -1
        names = [names.get(i, f"class_{i}") for i in range(max_id + 1)]

    return names


# ===========================================
# Pascal VOC Format Parser
# ===========================================

def parse_voc_xml(xml_content: str) -> ImportedImage:
    """
    Parse Pascal VOC XML annotation file.

    Expected structure:
    <annotation>
        <filename>img.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
        </size>
        <object>
            <name>class_name</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>100</ymin>
                <xmax>200</xmax>
                <ymax>200</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_content)

    filename = root.find("filename").text if root.find("filename") is not None else "unknown.jpg"

    size = root.find("size")
    width = int(size.find("width").text) if size is not None and size.find("width") is not None else None
    height = int(size.find("height").text) if size is not None and size.find("height") is not None else None

    img = ImportedImage(
        filename=filename,
        width=width,
        height=height,
        annotations=[]
    )

    for obj in root.findall("object"):
        class_name = obj.find("name").text if obj.find("name") is not None else "unknown"
        bndbox = obj.find("bndbox")

        if bndbox is not None:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Normalize to 0-1
            if width and height:
                norm_x = xmin / width
                norm_y = ymin / height
                norm_w = (xmax - xmin) / width
                norm_h = (ymax - ymin) / height
            else:
                norm_x, norm_y = xmin, ymin
                norm_w, norm_h = xmax - xmin, ymax - ymin

            img.annotations.append(ImportedAnnotation(
                class_name=class_name,
                bbox_x=norm_x,
                bbox_y=norm_y,
                bbox_width=norm_w,
                bbox_height=norm_h
            ))

    return img


# ===========================================
# LabelMe Format Parser
# ===========================================

def parse_labelme_json(json_content: dict) -> ImportedImage:
    """
    Parse LabelMe JSON annotation format.

    Expected structure:
    {
        "imagePath": "img.jpg",
        "imageWidth": 640,
        "imageHeight": 480,
        "shapes": [
            {
                "label": "class_name",
                "shape_type": "rectangle",
                "points": [[x1, y1], [x2, y2]]
            }
        ]
    }
    """
    filename = json_content.get("imagePath", "unknown.jpg")
    width = json_content.get("imageWidth")
    height = json_content.get("imageHeight")

    img = ImportedImage(
        filename=filename,
        width=width,
        height=height,
        annotations=[]
    )

    for shape in json_content.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue  # Only handle rectangles for bbox

        points = shape.get("points", [])
        if len(points) < 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        # Normalize to 0-1
        if width and height:
            norm_x = xmin / width
            norm_y = ymin / height
            norm_w = (xmax - xmin) / width
            norm_h = (ymax - ymin) / height
        else:
            norm_x, norm_y = xmin, ymin
            norm_w, norm_h = xmax - xmin, ymax - ymin

        img.annotations.append(ImportedAnnotation(
            class_name=shape.get("label", "unknown"),
            bbox_x=norm_x,
            bbox_y=norm_y,
            bbox_width=norm_w,
            bbox_height=norm_h
        ))

    return img


# ===========================================
# ZIP Extraction & Format Detection
# ===========================================

def detect_format_from_zip(zip_file: zipfile.ZipFile) -> tuple[str, dict]:
    """
    Detect annotation format from ZIP contents.

    Returns:
        Tuple of (format_name, relevant_files)
    """
    file_list = zip_file.namelist()

    # Check for COCO format (annotations.json or similar)
    # IMPORTANT: Collect ALL annotation files (train/valid/test splits)
    coco_annotation_files = []
    for name in file_list:
        if name.endswith(".json") and ("annotation" in name.lower() or "instances" in name.lower()):
            try:
                content = zip_file.read(name).decode("utf-8")
                data = json.loads(content)
                if "images" in data and "annotations" in data and "categories" in data:
                    coco_annotation_files.append(name)
            except Exception:
                continue

    if coco_annotation_files:
        # Return all annotation files for merging
        return "coco", {"annotations_files": coco_annotation_files}

    # Check for YOLO format (data.yaml + labels/)
    yaml_file = None
    labels_dir = None
    for name in file_list:
        if name.endswith(("data.yaml", "data.yml")):
            yaml_file = name
        if "/labels/" in name and name.endswith(".txt"):
            labels_dir = "/".join(name.split("/")[:-1])

    if yaml_file:
        return "yolo", {"yaml_file": yaml_file, "labels_dir": labels_dir}

    # Check for Pascal VOC (XML files)
    xml_files = [f for f in file_list if f.endswith(".xml")]
    if xml_files:
        return "voc", {"xml_files": xml_files}

    # Check for LabelMe (JSON per image)
    json_files = [f for f in file_list if f.endswith(".json")]
    if json_files:
        # Check if it's LabelMe format
        for jf in json_files[:1]:
            content = zip_file.read(jf).decode("utf-8")
            data = json.loads(content)
            if "shapes" in data and "imagePath" in data:
                return "labelme", {"json_files": json_files}

    # Just images, no annotations
    image_files = [f for f in file_list if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
    if image_files:
        return "images_only", {"image_files": image_files}

    return "unknown", {}


# ===========================================
# URL Import
# ===========================================

async def import_from_urls(
    urls: list[str],
    folder: Optional[str] = None,
    skip_duplicates: bool = True,
    timeout: float = 30.0
) -> ImportResult:
    """
    Import images from a list of URLs.

    Args:
        urls: List of image URLs to import
        folder: Optional folder to assign imported images to
        skip_duplicates: Whether to skip images with matching pHash
        timeout: Request timeout in seconds

    Returns:
        ImportResult with statistics and any errors
    """
    result = ImportResult(success=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in urls:
            try:
                # Download image
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()

                content = response.content
                content_type = response.headers.get("content-type", "image/jpeg")

                # Validate it's an image
                try:
                    img = Image.open(io.BytesIO(content))
                    width, height = img.size
                except Exception:
                    result.errors.append(f"{url}: Not a valid image")
                    continue

                # Check for duplicates only if requested (skip hash calculation otherwise)
                file_hash = None
                if skip_duplicates:
                    file_hash = calculate_file_hash(content)
                    existing = await check_duplicate_by_hash(file_hash)
                    if existing:
                        result.duplicates_found += 1
                        result.images_skipped += 1
                        continue

                # Generate filename
                ext = url.split(".")[-1].lower()
                if ext not in ["jpg", "jpeg", "png", "webp", "bmp", "gif"]:
                    ext = "jpg"
                unique_filename = f"{uuid4()}.{ext}"

                # Upload to storage with retry
                success, status = upload_to_storage_with_retry(
                    unique_filename,
                    content,
                    content_type,
                    max_retries=8,
                )

                if not success:
                    result.errors.append(f"{url}: Upload failed - {status}")
                    continue

                image_url = supabase_service.client.storage.from_("od-images").get_public_url(unique_filename)

                # Create database record
                image_data = {
                    "filename": unique_filename,
                    "original_filename": url.split("/")[-1],
                    "image_url": image_url,
                    "width": width,
                    "height": height,
                    "file_size_bytes": len(content),
                    "mime_type": content_type,
                    "source": "url",
                    "source_url": url,
                    "folder": folder,
                    "status": "pending",
                    "file_hash": file_hash,
                }

                db_result = supabase_service.client.table("od_images").insert(image_data).execute()

                if db_result.data:
                    result.images_imported += 1
                else:
                    result.errors.append(f"{url}: Failed to save to database")

            except httpx.RequestError as e:
                result.errors.append(f"{url}: Request failed - {str(e)}")
            except Exception as e:
                result.errors.append(f"{url}: {str(e)}")

    return result


# ===========================================
# Main Import Functions
# ===========================================

async def preview_import(
    file_content: bytes,
    filename: str
) -> ImportPreview:
    """
    Preview what will be imported from a file without actually importing.

    Args:
        file_content: The uploaded file content
        filename: Original filename to detect format

    Returns:
        ImportPreview with detected format and contents
    """
    preview = ImportPreview(
        format_detected="unknown",
        total_images=0,
        total_annotations=0,
        classes_found=[],
        sample_images=[]
    )

    try:
        if filename.endswith(".zip"):
            # Handle ZIP file
            with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
                format_type, files_info = detect_format_from_zip(zf)
                preview.format_detected = format_type

                if format_type == "coco":
                    content = zf.read(files_info["annotations_file"]).decode("utf-8")
                    data = json.loads(content)
                    images, classes = parse_coco_json(data)
                    preview.total_images = len(images)
                    preview.total_annotations = sum(len(img.annotations) for img in images)
                    preview.classes_found = classes
                    preview.sample_images = [
                        {
                            "filename": img.filename,
                            "annotation_count": len(img.annotations),
                            "classes": list(set(a.class_name for a in img.annotations))
                        }
                        for img in images[:5]
                    ]

                elif format_type == "yolo":
                    yaml_content = zf.read(files_info["yaml_file"]).decode("utf-8")
                    class_names = parse_yolo_data_yaml(yaml_content)
                    preview.classes_found = class_names

                    # Count label files
                    label_files = [f for f in zf.namelist() if f.endswith(".txt") and "/labels/" in f]
                    preview.total_images = len(label_files)

                    # Parse first few for sample
                    total_anns = 0
                    for lf in label_files[:100]:
                        content = zf.read(lf).decode("utf-8")
                        ann_count = len([l for l in content.strip().split("\n") if l.strip()])
                        total_anns += ann_count
                        if len(preview.sample_images) < 5:
                            preview.sample_images.append({
                                "filename": Path(lf).stem,
                                "annotation_count": ann_count,
                                "classes": []  # Would need to parse to get classes
                            })
                    preview.total_annotations = total_anns

                elif format_type == "voc":
                    xml_files = files_info["xml_files"]
                    preview.total_images = len(xml_files)

                    classes_set = set()
                    total_anns = 0
                    for xf in xml_files[:100]:
                        content = zf.read(xf).decode("utf-8")
                        img = parse_voc_xml(content)
                        total_anns += len(img.annotations)
                        classes_set.update(a.class_name for a in img.annotations)
                        if len(preview.sample_images) < 5:
                            preview.sample_images.append({
                                "filename": img.filename,
                                "annotation_count": len(img.annotations),
                                "classes": list(set(a.class_name for a in img.annotations))
                            })
                    preview.classes_found = list(classes_set)
                    preview.total_annotations = total_anns

                elif format_type == "images_only":
                    preview.total_images = len(files_info["image_files"])

        elif filename.endswith(".json"):
            # Could be COCO or LabelMe
            data = json.loads(file_content.decode("utf-8"))

            if "images" in data and "annotations" in data:
                preview.format_detected = "coco"
                images, classes = parse_coco_json(data)
                preview.total_images = len(images)
                preview.total_annotations = sum(len(img.annotations) for img in images)
                preview.classes_found = classes
                preview.sample_images = [
                    {
                        "filename": img.filename,
                        "annotation_count": len(img.annotations),
                        "classes": list(set(a.class_name for a in img.annotations))
                    }
                    for img in images[:5]
                ]
            elif "shapes" in data:
                preview.format_detected = "labelme"
                img = parse_labelme_json(data)
                preview.total_images = 1
                preview.total_annotations = len(img.annotations)
                preview.classes_found = list(set(a.class_name for a in img.annotations))

    except Exception as e:
        preview.errors.append(str(e))

    return preview


async def import_annotated_dataset(
    zip_content: bytes,
    dataset_id: str,
    class_mapping: list[ClassMapping],
    skip_duplicates: bool = True,
    merge_annotations: bool = False,
    atomic: bool = True,
) -> ImportResult:
    """
    Import an annotated dataset from a ZIP file in memory.

    When atomic=True (default), this function is STRICTLY ATOMIC:
    - Either ALL images upload and import successfully, or nothing is imported
    - If ANY upload fails after retries, the entire import is rolled back
    - This prevents orphan data (images without annotations)

    When atomic=False, partial success is allowed:
    - Successful uploads are imported even if some fail
    - May result in partial data if connection is unstable

    Args:
        zip_content: The ZIP file content
        dataset_id: Target dataset ID
        class_mapping: How to map source classes to target classes
        skip_duplicates: Skip images that already exist (by pHash)
        merge_annotations: If image exists, merge annotations instead of skip
        atomic: If True, rollback on any failure (default True)

    Returns:
        ImportResult with statistics
    """
    result = ImportResult(success=True)

    # Check for concurrent imports (protection against race conditions)
    is_locked, existing_job_id = await check_concurrent_import(dataset_id)
    if is_locked:
        result.success = False
        result.errors.append(f"Another import is already running for this dataset (job: {existing_job_id}). Please wait for it to complete.")
        return result

    # Track what we've created for rollback
    uploaded_files: list[str] = []  # Storage filenames
    inserted_image_ids: list[str] = []  # Database image IDs
    created_class_ids: list[str] = []  # Track newly created classes for potential rollback

    async def rollback():
        """Clean up on failure - delete all created data."""
        # Delete annotations for inserted images
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_annotations").delete().eq("image_id", img_id).execute()
            except Exception:
                pass

        # Delete dataset_images mappings
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_dataset_images").delete().eq("image_id", img_id).execute()
            except Exception:
                pass

        # Delete images from database
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_images").delete().eq("id", img_id).execute()
            except Exception:
                pass

        # Delete uploaded files from storage
        if uploaded_files:
            try:
                supabase_service.client.storage.from_("od-images").remove(uploaded_files)
            except Exception:
                pass

        # Delete newly created classes
        for class_id in created_class_ids:
            try:
                supabase_service.client.table("od_classes").delete().eq("id", class_id).execute()
            except Exception:
                pass

        # Sync dataset image count after rollback
        try:
            count = supabase_service.client.table("od_dataset_images").select(
                "id", count="exact"
            ).eq("dataset_id", dataset_id).execute()
            supabase_service.client.table("od_datasets").update({
                "image_count": count.count or 0
            }).eq("id", dataset_id).execute()
        except Exception:
            pass

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            format_type, files_info = detect_format_from_zip(zf)

            if format_type == "unknown":
                result.success = False
                result.errors.append("Could not detect annotation format")
                return result

            # Parse annotations based on format
            imported_images: list[ImportedImage] = []

            if format_type == "coco":
                # Merge ALL annotation files (train/valid/test splits)
                annotation_files = files_info.get("annotations_files", [])
                # Backward compatibility for single file
                if not annotation_files and "annotations_file" in files_info:
                    annotation_files = [files_info["annotations_file"]]

                logger.info(f"Found {len(annotation_files)} COCO annotation files to merge")
                all_classes = set()

                for ann_file in annotation_files:
                    logger.info(f"Parsing: {ann_file}")
                    content = zf.read(ann_file).decode("utf-8")
                    data = json.loads(content)
                    images, classes = parse_coco_json(data)
                    imported_images.extend(images)
                    all_classes.update(classes)
                    logger.info(f"  -> {len(images)} images, {sum(len(img.annotations) for img in images)} annotations")

                logger.info(f"Total merged: {len(imported_images)} images from {len(annotation_files)} files")

            elif format_type == "yolo":
                yaml_content = zf.read(files_info["yaml_file"]).decode("utf-8")
                class_names = parse_yolo_data_yaml(yaml_content)

                label_files = {}
                for name in zf.namelist():
                    if name.endswith(".txt") and "/labels/" in name:
                        label_files[Path(name).name] = zf.read(name).decode("utf-8")

                imported_images = parse_yolo_labels(label_files, class_names)

            elif format_type == "voc":
                for xf in files_info["xml_files"]:
                    content = zf.read(xf).decode("utf-8")
                    img = parse_voc_xml(content)
                    imported_images.append(img)

            # Build class mapping dict (with create_new support)
            class_map = {}
            for mapping in class_mapping:
                if mapping.skip:
                    continue

                # Handle create_new: create a new class if target_class_id is None
                if mapping.create_new and not mapping.target_class_id:
                    try:
                        color = mapping.color or get_next_class_color()
                        new_class_result = supabase_service.client.table("od_classes").insert({
                            "dataset_id": dataset_id,
                            "name": mapping.source_name,
                            "color": color,
                        }).execute()

                        if new_class_result.data:
                            new_class_id = new_class_result.data[0]["id"]
                            created_class_ids.append(new_class_id)
                            class_map[mapping.source_name] = new_class_id
                            logger.info(f"Created new class '{mapping.source_name}' with ID {new_class_id}")
                        else:
                            result.errors.append(f"Failed to create class '{mapping.source_name}'")
                    except Exception as e:
                        result.errors.append(f"Failed to create class '{mapping.source_name}': {str(e)}")
                else:
                    class_map[mapping.source_name] = mapping.target_class_id

            # Get all image files from ZIP
            image_files = {}
            for name in zf.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    base_name = Path(name).stem
                    image_files[base_name] = name

            # ============================================
            # PHASE 1: Prepare all data (no side effects)
            # ============================================
            prepared_images = []  # List of (imported_img, image_content, image_file_path, file_hash)
            existing_images_for_merge = []  # List of (existing_image_id, imported_img) for merge_annotations

            for imported_img in imported_images:
                # Find matching image file
                base_name = Path(imported_img.filename).stem
                image_file_path = None

                for key in [base_name, base_name.lower(), imported_img.filename]:
                    if key in image_files:
                        image_file_path = image_files[key]
                        break

                if not image_file_path:
                    result.errors.append(f"Image file not found for: {imported_img.filename}")
                    continue

                # Read image
                image_content = zf.read(image_file_path)

                try:
                    pil_img = Image.open(io.BytesIO(image_content))
                    width, height = pil_img.size
                    imported_img.width = width
                    imported_img.height = height
                except Exception:
                    result.errors.append(f"Invalid image: {imported_img.filename}")
                    continue

                # Check for duplicates
                file_hash = None
                if skip_duplicates or merge_annotations:
                    file_hash = calculate_file_hash(image_content)
                    existing = await check_duplicate_by_hash(file_hash)
                    if existing:
                        result.duplicates_found += 1
                        if merge_annotations:
                            # Track existing image for annotation merge
                            existing_images_for_merge.append((existing["id"], imported_img))
                            logger.info(f"Image '{imported_img.filename}' exists, will merge annotations")
                        else:
                            result.images_skipped += 1
                        continue

                prepared_images.append((imported_img, image_content, image_file_path, file_hash))

            if not prepared_images and not existing_images_for_merge:
                # Nothing to import and nothing to merge
                if result.duplicates_found > 0:
                    result.success = True
                    return result
                result.success = False
                result.errors.append("No valid images found to import")
                return result

            # ============================================
            # PHASE 2: Upload all images to storage (PARALLEL)
            # ============================================
            # Prepare upload items: (filename, content, content_type)
            upload_items = []
            upload_metadata = []  # Keep track of (imported_img, file_hash, file_size)

            for imported_img, image_content, image_file_path, file_hash in prepared_images:
                ext = Path(image_file_path).suffix.lower().lstrip(".")
                unique_filename = f"{uuid4()}.{ext}"
                content_type = f"image/{ext}"

                upload_items.append((unique_filename, image_content, content_type))
                upload_metadata.append((unique_filename, imported_img, file_hash, len(image_content)))

            # Parallel upload with batch processing (optimized for large datasets)
            logger.info(f"Starting batch upload of {len(upload_items)} images")
            successful_files, failed_uploads = await parallel_upload_images(
                upload_items,
                max_concurrent=2,  # Lower concurrency for stability
                batch_size=100,    # Process in batches of 100
                batch_delay=2.0,   # 2 second delay between batches
            )

            # Track uploaded files for potential rollback
            uploaded_files.extend(successful_files)

            # Handle upload failures based on atomic mode
            if failed_uploads:
                for filename, error in failed_uploads:
                    result.errors.append(f"Upload failed for {filename}: {error}")
                logger.warning(f"{len(failed_uploads)} images failed to upload")

                # ATOMIC MODE: If ANY upload fails, rollback everything
                if atomic:
                    logger.warning(f"ATOMIC MODE: Rolling back {len(successful_files)} successful uploads due to {len(failed_uploads)} failures")
                    await rollback()
                    result.success = False
                    result.errors.insert(0, f"ATOMIC MODE: {len(failed_uploads)} uploads failed - rolling back all {len(successful_files)} successful uploads to prevent orphan data")
                    return result

            # If ALL uploads failed (non-atomic mode), still fail
            # BUT only if there are no existing images to merge annotations into
            if not successful_files and not existing_images_for_merge:
                await rollback()
                result.success = False
                result.errors.append("All image uploads failed")
                return result

            # Build image_records from successful uploads only
            successful_set = set(successful_files)
            image_records = []
            for unique_filename, imported_img, file_hash, file_size in upload_metadata:
                if unique_filename in successful_set:
                    image_url = supabase_service.client.storage.from_("od-images").get_public_url(unique_filename)
                    image_records.append((unique_filename, image_url, imported_img, file_hash, file_size))

            logger.info(f"Uploaded {len(image_records)} images to storage, {len(existing_images_for_merge)} existing images for annotation merge")

            # ============================================
            # PHASE 3: Batch insert images to database
            # ============================================
            images_to_insert = []
            for unique_filename, image_url, imported_img, file_hash, file_size in image_records:
                images_to_insert.append({
                    "filename": unique_filename,
                    "original_filename": imported_img.filename,
                    "image_url": image_url,
                    "width": imported_img.width,
                    "height": imported_img.height,
                    "file_size_bytes": file_size,
                    "source": "import",
                    "status": "pending",
                    "file_hash": file_hash,
                })

            try:
                db_result = supabase_service.client.table("od_images").insert(images_to_insert).execute()
                if not db_result.data:
                    raise Exception("No data returned from insert")

                for row in db_result.data:
                    inserted_image_ids.append(row["id"])

                result.images_imported = len(db_result.data)
            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Database insert failed: {str(e)}")
                return result

            # Map filename to image_id for annotations
            filename_to_id = {row["filename"]: row["id"] for row in db_result.data}

            # ============================================
            # PHASE 4: Batch insert dataset_images mappings
            # ============================================
            dataset_images_to_insert = []
            for unique_filename, _, _, _, _ in image_records:
                image_id = filename_to_id.get(unique_filename)
                if image_id:
                    dataset_images_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "status": "pending"
                    })

            try:
                if dataset_images_to_insert:
                    supabase_service.client.table("od_dataset_images").insert(dataset_images_to_insert).execute()
            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Dataset mapping failed: {str(e)}")
                return result

            # ============================================
            # PHASE 5: Batch insert annotations
            # ============================================
            annotations_to_insert = []
            skipped_by_class: dict[str, int] = {}  # Track skipped annotations

            # Process annotations for NEW images
            for unique_filename, _, imported_img, _, _ in image_records:
                image_id = filename_to_id.get(unique_filename)
                if not image_id:
                    continue

                width = imported_img.width or 1
                height = imported_img.height or 1

                for ann in imported_img.annotations:
                    target_class_id = class_map.get(ann.class_name)
                    if not target_class_id:
                        # Track skipped annotations for warning
                        skipped_by_class[ann.class_name] = skipped_by_class.get(ann.class_name, 0) + 1
                        continue

                    # Ensure bbox is normalized
                    bbox_x = ann.bbox_x if ann.bbox_x <= 1 else ann.bbox_x / width
                    bbox_y = ann.bbox_y if ann.bbox_y <= 1 else ann.bbox_y / height
                    bbox_w = ann.bbox_width if ann.bbox_width <= 1 else ann.bbox_width / width
                    bbox_h = ann.bbox_height if ann.bbox_height <= 1 else ann.bbox_height / height

                    annotations_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "class_id": target_class_id,
                        "bbox_x": max(0, min(1, bbox_x)),
                        "bbox_y": max(0, min(1, bbox_y)),
                        "bbox_width": max(0.001, min(1, bbox_w)),
                        "bbox_height": max(0.001, min(1, bbox_h)),
                        "is_ai_generated": False,
                        "confidence": ann.confidence,
                    })

            # Process annotations for EXISTING images (merge_annotations)
            merged_annotations_count = 0
            for existing_image_id, imported_img in existing_images_for_merge:
                width = imported_img.width or 1
                height = imported_img.height or 1

                for ann in imported_img.annotations:
                    target_class_id = class_map.get(ann.class_name)
                    if not target_class_id:
                        skipped_by_class[ann.class_name] = skipped_by_class.get(ann.class_name, 0) + 1
                        continue

                    # Ensure bbox is normalized
                    bbox_x = ann.bbox_x if ann.bbox_x <= 1 else ann.bbox_x / width
                    bbox_y = ann.bbox_y if ann.bbox_y <= 1 else ann.bbox_y / height
                    bbox_w = ann.bbox_width if ann.bbox_width <= 1 else ann.bbox_width / width
                    bbox_h = ann.bbox_height if ann.bbox_height <= 1 else ann.bbox_height / height

                    annotations_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": existing_image_id,
                        "class_id": target_class_id,
                        "bbox_x": max(0, min(1, bbox_x)),
                        "bbox_y": max(0, min(1, bbox_y)),
                        "bbox_width": max(0.001, min(1, bbox_w)),
                        "bbox_height": max(0.001, min(1, bbox_h)),
                        "is_ai_generated": False,
                        "confidence": ann.confidence,
                    })
                    merged_annotations_count += 1

            if merged_annotations_count > 0:
                logger.info(f"Merging {merged_annotations_count} annotations to {len(existing_images_for_merge)} existing images")

            # Warn about skipped annotations
            if skipped_by_class:
                total_skipped = sum(skipped_by_class.values())
                classes_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in skipped_by_class.items())
                result.errors.append(f"WARNING: {total_skipped} annotations skipped (no class mapping): {classes_str}")

            try:
                if annotations_to_insert:
                    # Insert in batches of 1000 to avoid payload size limits
                    batch_size = 1000
                    for i in range(0, len(annotations_to_insert), batch_size):
                        batch = annotations_to_insert[i:i + batch_size]
                        ann_result = supabase_service.client.table("od_annotations").insert(batch).execute()
                        result.annotations_imported += len(ann_result.data) if ann_result.data else 0
            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Annotation insert failed: {str(e)}")
                return result

            # ============================================
            # PHASE 6: Update dataset counts
            # ============================================
            try:
                img_count = supabase_service.client.table("od_dataset_images").select(
                    "id", count="exact"
                ).eq("dataset_id", dataset_id).execute()

                ann_count = supabase_service.client.table("od_annotations").select(
                    "id", count="exact"
                ).eq("dataset_id", dataset_id).execute()

                supabase_service.client.table("od_datasets").update({
                    "image_count": img_count.count or 0,
                    "annotation_count": ann_count.count or 0,
                }).eq("id", dataset_id).execute()
            except Exception:
                # Count update failure is not critical, don't rollback
                pass

    except Exception as e:
        await rollback()
        result.success = False
        result.errors.append(f"Import failed: {str(e)}")

    return result


async def get_duplicate_groups(
    threshold: int = 10
) -> list[dict]:
    """
    Get groups of duplicate/similar images based on pHash.
    Requires imagehash module to be installed.
    """
    if not IMAGEHASH_AVAILABLE:
        return []

    # Get all images with pHash
    result = supabase_service.client.table("od_images").select(
        "id, filename, image_url, thumbnail_url, phash, created_at"
    ).not_.is_("phash", "null").order("created_at").execute()

    images = result.data or []
    groups = []
    processed_ids = set()

    for i, img_a in enumerate(images):
        if img_a["id"] in processed_ids:
            continue

        group = {
            "images": [img_a],
            "max_similarity": 1.0
        }

        hash_a = imagehash.hex_to_hash(img_a["phash"])

        for j, img_b in enumerate(images):
            if i >= j or img_b["id"] in processed_ids:
                continue

            hash_b = imagehash.hex_to_hash(img_b["phash"])
            distance = hash_a - hash_b

            if distance <= threshold:
                similarity = 1 - (distance / 64)
                img_b_copy = dict(img_b)
                img_b_copy["similarity"] = round(similarity, 2)
                group["images"].append(img_b_copy)
                group["max_similarity"] = max(group["max_similarity"], similarity)
                processed_ids.add(img_b["id"])

        if len(group["images"]) > 1:
            groups.append(group)
            processed_ids.add(img_a["id"])

    return sorted(groups, key=lambda x: -x["max_similarity"])


async def import_annotated_dataset_from_file(
    zip_file_path: str,
    dataset_id: str,
    class_mapping: list[ClassMapping],
    skip_duplicates: bool = True,
    merge_annotations: bool = False,
    atomic: bool = True,
) -> ImportResult:
    """
    Import an annotated dataset from a ZIP file on disk.

    This is a memory-efficient version that reads directly from disk
    instead of keeping the entire ZIP in memory.

    When atomic=True (default), this function is STRICTLY ATOMIC:
    - Either ALL images upload and import successfully, or nothing is imported
    - If ANY upload fails after retries, the entire import is rolled back
    - This prevents orphan data (images without annotations)

    When atomic=False, partial success is allowed:
    - Successful uploads are imported even if some fail
    - May result in partial data if connection is unstable

    Args:
        zip_file_path: Path to the ZIP file on disk
        dataset_id: Target dataset ID
        class_mapping: How to map source classes to target classes
        skip_duplicates: Skip images that already exist (by pHash)
        merge_annotations: If image exists, merge annotations instead of skip
        atomic: If True, rollback on any failure (default True)

    Returns:
        ImportResult with statistics
    """
    import logging
    logger = logging.getLogger(__name__)

    result = ImportResult(success=True)

    # Check for concurrent imports (protection against race conditions)
    is_locked, existing_job_id = await check_concurrent_import(dataset_id)
    if is_locked:
        result.success = False
        result.errors.append(f"Another import is already running for this dataset (job: {existing_job_id}). Please wait for it to complete.")
        return result

    # Track what we've created for rollback
    uploaded_files: list[str] = []  # Storage filenames
    inserted_image_ids: list[str] = []  # Database image IDs
    created_class_ids: list[str] = []  # Track newly created classes for potential rollback

    async def rollback():
        """Clean up on failure - delete all created data."""
        logger.warning(f"Rolling back import - cleaning up {len(inserted_image_ids)} images and {len(uploaded_files)} files")

        # Delete annotations for inserted images
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_annotations").delete().eq("image_id", img_id).execute()
            except Exception:
                pass

        # Delete dataset_images mappings
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_dataset_images").delete().eq("image_id", img_id).execute()
            except Exception:
                pass

        # Delete images from database
        for img_id in inserted_image_ids:
            try:
                supabase_service.client.table("od_images").delete().eq("id", img_id).execute()
            except Exception:
                pass

        # Delete uploaded files from storage
        if uploaded_files:
            try:
                supabase_service.client.storage.from_("od-images").remove(uploaded_files)
            except Exception:
                pass

        # Delete newly created classes
        for class_id in created_class_ids:
            try:
                supabase_service.client.table("od_classes").delete().eq("id", class_id).execute()
            except Exception:
                pass

        # Sync dataset image count after rollback
        try:
            count = supabase_service.client.table("od_dataset_images").select(
                "id", count="exact"
            ).eq("dataset_id", dataset_id).execute()
            supabase_service.client.table("od_datasets").update({
                "image_count": count.count or 0
            }).eq("id", dataset_id).execute()
            logger.info(f"Updated dataset {dataset_id} image_count to {count.count or 0}")
        except Exception as e:
            logger.warning(f"Failed to sync dataset image count: {e}")

    try:
        # Open ZIP file directly from disk (memory-efficient)
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            format_type, files_info = detect_format_from_zip(zf)

            if format_type == "unknown":
                result.success = False
                result.errors.append("Could not detect annotation format in ZIP file")
                return result

            logger.info(f"Detected format: {format_type}")

            # Parse annotations based on format
            imported_images: list[ImportedImage] = []

            if format_type == "coco":
                # Merge ALL annotation files (train/valid/test splits)
                annotation_files = files_info.get("annotations_files", [])
                # Backward compatibility for single file
                if not annotation_files and "annotations_file" in files_info:
                    annotation_files = [files_info["annotations_file"]]

                logger.info(f"Found {len(annotation_files)} COCO annotation files to merge")

                for ann_file in annotation_files:
                    logger.info(f"Parsing: {ann_file}")
                    content = zf.read(ann_file).decode("utf-8")
                    data = json.loads(content)
                    images, _ = parse_coco_json(data)
                    imported_images.extend(images)
                    logger.info(f"  -> {len(images)} images, {sum(len(img.annotations) for img in images)} annotations")

                logger.info(f"Total merged: {len(imported_images)} images from {len(annotation_files)} files")

            elif format_type == "yolo":
                yaml_content = zf.read(files_info["yaml_file"]).decode("utf-8")
                class_names = parse_yolo_data_yaml(yaml_content)

                label_files = {}
                for name in zf.namelist():
                    if name.endswith(".txt") and "/labels/" in name:
                        label_files[Path(name).name] = zf.read(name).decode("utf-8")

                imported_images = parse_yolo_labels(label_files, class_names)
                logger.info(f"Parsed {len(imported_images)} images from YOLO annotations")

            elif format_type == "voc":
                for xf in files_info["xml_files"]:
                    content = zf.read(xf).decode("utf-8")
                    img = parse_voc_xml(content)
                    imported_images.append(img)
                logger.info(f"Parsed {len(imported_images)} images from VOC annotations")

            # Build class mapping dict (with create_new support)
            class_map = {}
            for mapping in class_mapping:
                if mapping.skip:
                    continue

                # Handle create_new: create a new class if target_class_id is None
                if mapping.create_new and not mapping.target_class_id:
                    try:
                        color = mapping.color or get_next_class_color()
                        new_class_result = supabase_service.client.table("od_classes").insert({
                            "dataset_id": dataset_id,
                            "name": mapping.source_name,
                            "color": color,
                        }).execute()

                        if new_class_result.data:
                            new_class_id = new_class_result.data[0]["id"]
                            created_class_ids.append(new_class_id)
                            class_map[mapping.source_name] = new_class_id
                            logger.info(f"Created new class '{mapping.source_name}' with ID {new_class_id}")
                        else:
                            result.errors.append(f"Failed to create class '{mapping.source_name}'")
                    except Exception as e:
                        result.errors.append(f"Failed to create class '{mapping.source_name}': {str(e)}")
                else:
                    class_map[mapping.source_name] = mapping.target_class_id

            # Get all image files from ZIP
            image_files = {}
            for name in zf.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    base_name = Path(name).stem
                    image_files[base_name] = name

            logger.info(f"Found {len(image_files)} image files in ZIP")

            # ============================================
            # PHASE 1: Prepare all data (no side effects)
            # ============================================
            prepared_images = []  # List of (imported_img, image_content, image_file_path, file_hash)
            existing_images_for_merge = []  # List of (existing_image_id, imported_img) for merge_annotations

            for imported_img in imported_images:
                # Find matching image file
                base_name = Path(imported_img.filename).stem
                image_file_path = None

                for key in [base_name, base_name.lower(), imported_img.filename]:
                    if key in image_files:
                        image_file_path = image_files[key]
                        break

                if not image_file_path:
                    result.errors.append(f"Image file not found for: {imported_img.filename}")
                    continue

                # Read image from ZIP
                image_content = zf.read(image_file_path)

                try:
                    pil_img = Image.open(io.BytesIO(image_content))
                    width, height = pil_img.size
                    imported_img.width = width
                    imported_img.height = height
                except Exception:
                    result.errors.append(f"Invalid image: {imported_img.filename}")
                    continue

                # Check for duplicates
                file_hash = None
                if skip_duplicates or merge_annotations:
                    file_hash = calculate_file_hash(image_content)
                    existing = await check_duplicate_by_hash(file_hash)
                    if existing:
                        result.duplicates_found += 1
                        if merge_annotations:
                            # Track existing image for annotation merge
                            existing_images_for_merge.append((existing["id"], imported_img))
                            logger.info(f"Image '{imported_img.filename}' exists, will merge annotations")
                        else:
                            result.images_skipped += 1
                        continue

                prepared_images.append((imported_img, image_content, image_file_path, file_hash))

            if not prepared_images and not existing_images_for_merge:
                # Nothing to import and nothing to merge
                if result.duplicates_found > 0:
                    result.success = True
                    logger.info(f"No new images to import, {result.duplicates_found} duplicates skipped")
                    return result
                result.success = False
                result.errors.append("No valid images found to import")
                return result

            logger.info(f"Prepared {len(prepared_images)} images for import")

            # ============================================
            # PHASE 2: Upload all images to storage (PARALLEL)
            # ============================================
            # Prepare upload items: (filename, content, content_type)
            upload_items = []
            upload_metadata = []  # Keep track of (imported_img, file_hash, file_size)

            for imported_img, image_content, image_file_path, file_hash in prepared_images:
                ext = Path(image_file_path).suffix.lower().lstrip(".")
                unique_filename = f"{uuid4()}.{ext}"
                content_type = f"image/{ext}"

                upload_items.append((unique_filename, image_content, content_type))
                upload_metadata.append((unique_filename, imported_img, file_hash, len(image_content)))

            # Progress callback for logging
            def log_progress(completed: int, total: int):
                logger.info(f"Uploaded {completed}/{total} images to storage")

            # Parallel upload with batch processing (optimized for large datasets)
            logger.info(f"Starting batch upload of {len(upload_items)} images")
            successful_files, failed_uploads = await parallel_upload_images(
                upload_items,
                max_concurrent=2,  # Lower concurrency for stability
                progress_callback=log_progress,
                batch_size=100,    # Process in batches of 100
                batch_delay=2.0,   # 2 second delay between batches
            )

            # Track uploaded files for potential rollback
            uploaded_files.extend(successful_files)

            # Handle upload failures based on atomic mode
            if failed_uploads:
                for filename, error in failed_uploads:
                    result.errors.append(f"Upload failed for {filename}: {error}")
                logger.warning(f"{len(failed_uploads)} images failed to upload")

                # ATOMIC MODE: If ANY upload fails, rollback everything
                if atomic:
                    logger.warning(f"ATOMIC MODE: Rolling back {len(successful_files)} successful uploads due to {len(failed_uploads)} failures")
                    await rollback()
                    result.success = False
                    result.errors.insert(0, f"ATOMIC MODE: {len(failed_uploads)} uploads failed - rolling back all {len(successful_files)} successful uploads to prevent orphan data")
                    return result

            # If ALL uploads failed (non-atomic mode), still fail
            # BUT only if there are no existing images to merge annotations into
            if not successful_files and not existing_images_for_merge:
                await rollback()
                result.success = False
                result.errors.append("All image uploads failed")
                return result

            # Build image_records from successful uploads only
            successful_set = set(successful_files)
            image_records = []
            for unique_filename, imported_img, file_hash, file_size in upload_metadata:
                if unique_filename in successful_set:
                    image_url = supabase_service.client.storage.from_("od-images").get_public_url(unique_filename)
                    image_records.append((unique_filename, image_url, imported_img, file_hash, file_size))

            logger.info(f"Uploaded {len(image_records)} images to storage, {len(existing_images_for_merge)} existing images for annotation merge")

            # ============================================
            # PHASE 3: Batch insert images to database
            # ============================================
            images_to_insert = []
            for unique_filename, image_url, imported_img, file_hash, file_size in image_records:
                images_to_insert.append({
                    "filename": unique_filename,
                    "original_filename": imported_img.filename,
                    "image_url": image_url,
                    "width": imported_img.width,
                    "height": imported_img.height,
                    "file_size_bytes": file_size,
                    "source": "import",
                    "status": "pending",
                    "file_hash": file_hash,
                })

            try:
                # Insert in batches for large datasets
                batch_size = 500
                all_inserted = []
                for i in range(0, len(images_to_insert), batch_size):
                    batch = images_to_insert[i:i + batch_size]
                    db_result = supabase_service.client.table("od_images").insert(batch).execute()
                    if not db_result.data:
                        raise Exception("No data returned from insert")
                    all_inserted.extend(db_result.data)

                for row in all_inserted:
                    inserted_image_ids.append(row["id"])

                result.images_imported = len(all_inserted)
                logger.info(f"Inserted {result.images_imported} images to database")

            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Database insert failed: {str(e)}")
                return result

            # Map filename to image_id for annotations
            filename_to_id = {row["filename"]: row["id"] for row in all_inserted}

            # ============================================
            # PHASE 4: Batch insert dataset_images mappings
            # ============================================
            dataset_images_to_insert = []
            for unique_filename, _, _, _, _ in image_records:
                image_id = filename_to_id.get(unique_filename)
                if image_id:
                    dataset_images_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "status": "pending"
                    })

            try:
                if dataset_images_to_insert:
                    # Insert in batches
                    batch_size = 500
                    for i in range(0, len(dataset_images_to_insert), batch_size):
                        batch = dataset_images_to_insert[i:i + batch_size]
                        supabase_service.client.table("od_dataset_images").insert(batch).execute()
                    logger.info(f"Created {len(dataset_images_to_insert)} dataset-image mappings")
            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Dataset mapping failed: {str(e)}")
                return result

            # ============================================
            # PHASE 5: Batch insert annotations
            # ============================================
            annotations_to_insert = []
            skipped_by_class: dict[str, int] = {}  # Track skipped annotations

            # Process annotations for NEW images
            for unique_filename, _, imported_img, _, _ in image_records:
                image_id = filename_to_id.get(unique_filename)
                if not image_id:
                    continue

                width = imported_img.width or 1
                height = imported_img.height or 1

                for ann in imported_img.annotations:
                    target_class_id = class_map.get(ann.class_name)
                    if not target_class_id:
                        # Track skipped annotations for warning
                        skipped_by_class[ann.class_name] = skipped_by_class.get(ann.class_name, 0) + 1
                        continue

                    # Ensure bbox is normalized
                    bbox_x = ann.bbox_x if ann.bbox_x <= 1 else ann.bbox_x / width
                    bbox_y = ann.bbox_y if ann.bbox_y <= 1 else ann.bbox_y / height
                    bbox_w = ann.bbox_width if ann.bbox_width <= 1 else ann.bbox_width / width
                    bbox_h = ann.bbox_height if ann.bbox_height <= 1 else ann.bbox_height / height

                    annotations_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": image_id,
                        "class_id": target_class_id,
                        "bbox_x": max(0, min(1, bbox_x)),
                        "bbox_y": max(0, min(1, bbox_y)),
                        "bbox_width": max(0.001, min(1, bbox_w)),
                        "bbox_height": max(0.001, min(1, bbox_h)),
                        "is_ai_generated": False,
                        "confidence": ann.confidence,
                    })

            # Process annotations for EXISTING images (merge_annotations)
            merged_annotations_count = 0
            for existing_image_id, imported_img in existing_images_for_merge:
                width = imported_img.width or 1
                height = imported_img.height or 1

                for ann in imported_img.annotations:
                    target_class_id = class_map.get(ann.class_name)
                    if not target_class_id:
                        skipped_by_class[ann.class_name] = skipped_by_class.get(ann.class_name, 0) + 1
                        continue

                    # Ensure bbox is normalized
                    bbox_x = ann.bbox_x if ann.bbox_x <= 1 else ann.bbox_x / width
                    bbox_y = ann.bbox_y if ann.bbox_y <= 1 else ann.bbox_y / height
                    bbox_w = ann.bbox_width if ann.bbox_width <= 1 else ann.bbox_width / width
                    bbox_h = ann.bbox_height if ann.bbox_height <= 1 else ann.bbox_height / height

                    annotations_to_insert.append({
                        "dataset_id": dataset_id,
                        "image_id": existing_image_id,
                        "class_id": target_class_id,
                        "bbox_x": max(0, min(1, bbox_x)),
                        "bbox_y": max(0, min(1, bbox_y)),
                        "bbox_width": max(0.001, min(1, bbox_w)),
                        "bbox_height": max(0.001, min(1, bbox_h)),
                        "is_ai_generated": False,
                        "confidence": ann.confidence,
                    })
                    merged_annotations_count += 1

            if merged_annotations_count > 0:
                logger.info(f"Merging {merged_annotations_count} annotations to {len(existing_images_for_merge)} existing images")

            # Warn about skipped annotations
            if skipped_by_class:
                total_skipped = sum(skipped_by_class.values())
                classes_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in skipped_by_class.items())
                result.errors.append(f"WARNING: {total_skipped} annotations skipped (no class mapping): {classes_str}")

            try:
                if annotations_to_insert:
                    # Insert in batches of 1000 to avoid payload size limits
                    batch_size = 1000
                    for i in range(0, len(annotations_to_insert), batch_size):
                        batch = annotations_to_insert[i:i + batch_size]
                        ann_result = supabase_service.client.table("od_annotations").insert(batch).execute()
                        result.annotations_imported += len(ann_result.data) if ann_result.data else 0
                    logger.info(f"Inserted {result.annotations_imported} annotations")
            except Exception as e:
                await rollback()
                result.success = False
                result.errors.append(f"Annotation insert failed: {str(e)}")
                return result

            # ============================================
            # PHASE 6: Update dataset counts
            # ============================================
            try:
                img_count = supabase_service.client.table("od_dataset_images").select(
                    "id", count="exact"
                ).eq("dataset_id", dataset_id).execute()

                ann_count = supabase_service.client.table("od_annotations").select(
                    "id", count="exact"
                ).eq("dataset_id", dataset_id).execute()

                supabase_service.client.table("od_datasets").update({
                    "image_count": img_count.count or 0,
                    "annotation_count": ann_count.count or 0,
                }).eq("id", dataset_id).execute()

                logger.info(f"Updated dataset counts: {img_count.count} images, {ann_count.count} annotations")
            except Exception:
                # Count update failure is not critical, don't rollback
                pass

    except zipfile.BadZipFile:
        result.success = False
        result.errors.append("Invalid or corrupted ZIP file")
    except Exception as e:
        await rollback()
        result.success = False
        result.errors.append(f"Import failed: {str(e)}")
        logger.exception(f"Import failed with exception")

    return result
