"""
Workflow Blocks - Transform Blocks

Image transformation operations: Crop, Blur, Draw.
"""

import time
import asyncio
import logging
import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ..base import BaseBlock, BlockResult, ExecutionContext

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def load_image_from_input(image_input: Any) -> Image.Image | None:
    """Load PIL Image from various input formats."""
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        return image_input

    if isinstance(image_input, dict):
        if "image_url" in image_input:
            image_input = image_input["image_url"]
        elif "image_base64" in image_input:
            image_input = image_input["image_base64"]
        elif "url" in image_input:
            image_input = image_input["url"]

    if isinstance(image_input, str):
        if image_input.startswith(("http://", "https://")):
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(image_input)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")

        # Base64
        try:
            if "," in image_input:
                image_input = image_input.split(",")[1]
            image_data = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            pass

    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input.astype("uint8")).convert("RGB")

    return None


class CropBlock(BaseBlock):
    """
    Crop Block

    Crops regions from images based on detection bounding boxes.
    """

    block_type = "crop"
    display_name = "Crop"
    description = "Crop regions from image based on detections"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "crops", "type": "array", "description": "Cropped image regions as base64"},
        {"name": "crop_metadata", "type": "array", "description": "Metadata for each crop"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "padding": {"type": "number", "default": 0, "description": "Padding around bbox (0-1 ratio)"},
            "min_size": {"type": "number", "default": 32, "description": "Minimum crop size in pixels"},
            "max_crops": {"type": "number", "default": 100},
            "filter_classes": {"type": "array", "items": {"type": "string"}},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Crop regions from image."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        detections = inputs.get("detections", [])
        if not detections:
            return BlockResult(
                outputs={"crops": [], "crop_metadata": []},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"crop_count": 0},
            )

        padding = config.get("padding", 0)
        min_size = config.get("min_size", 32)
        max_crops = config.get("max_crops", 100)
        filter_classes = config.get("filter_classes")

        img_width, img_height = image.size
        crops = []
        metadata = []

        for i, det in enumerate(detections[:max_crops]):
            # Filter by class if specified
            if filter_classes and det.get("class_name") not in filter_classes:
                continue

            # Get bounding box
            bbox = det.get("bbox", {})
            if "x1" in bbox:
                # Normalized xyxy format
                x1 = bbox["x1"] * img_width
                y1 = bbox["y1"] * img_height
                x2 = bbox["x2"] * img_width
                y2 = bbox["y2"] * img_height
            elif "x" in bbox:
                # Normalized xywh format
                x1 = bbox["x"] * img_width
                y1 = bbox["y"] * img_height
                x2 = (bbox["x"] + bbox["width"]) * img_width
                y2 = (bbox["y"] + bbox["height"]) * img_height
            else:
                continue

            # Add padding
            if padding > 0:
                pad_w = (x2 - x1) * padding
                pad_h = (y2 - y1) * padding
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(img_width, x2 + pad_w)
                y2 = min(img_height, y2 + pad_h)

            # Check minimum size
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue

            # Crop
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            crop_base64 = f"data:image/jpeg;base64,{image_to_base64(crop)}"

            crops.append(crop_base64)
            metadata.append({
                "index": i,
                "detection": det,
                "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_size": {"width": crop.width, "height": crop.height},
            })

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"crops": crops, "crop_metadata": metadata},
            duration_ms=round(duration, 2),
            metrics={"crop_count": len(crops), "total_detections": len(detections)},
        )


class BlurRegionBlock(BaseBlock):
    """
    Blur Region Block

    Blurs specified regions in an image (for privacy protection, etc.)
    """

    block_type = "blur_region"
    display_name = "Blur Region"
    description = "Blur specified regions in an image"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "regions", "type": "array", "required": True, "description": "Regions to blur (detections)"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Image with blurred regions"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "blur_type": {
                "type": "string",
                "enum": ["gaussian", "pixelate", "black"],
                "default": "gaussian",
            },
            "intensity": {"type": "number", "default": 21, "description": "Blur intensity"},
            "filter_classes": {"type": "array", "items": {"type": "string"}},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Blur regions in image."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        regions = inputs.get("regions", [])
        if not regions:
            # No regions to blur, return original
            return BlockResult(
                outputs={"image": f"data:image/jpeg;base64,{image_to_base64(image)}"},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"blurred_regions": 0},
            )

        blur_type = config.get("blur_type", "gaussian")
        intensity = config.get("intensity", 21)
        filter_classes = config.get("filter_classes")

        img_width, img_height = image.size
        result_image = image.copy()
        blurred_count = 0

        for region in regions:
            # Filter by class if specified
            if filter_classes and region.get("class_name") not in filter_classes:
                continue

            # Get bounding box
            bbox = region.get("bbox", {})
            if "x1" in bbox:
                x1 = int(bbox["x1"] * img_width)
                y1 = int(bbox["y1"] * img_height)
                x2 = int(bbox["x2"] * img_width)
                y2 = int(bbox["y2"] * img_height)
            elif "x" in bbox:
                x1 = int(bbox["x"] * img_width)
                y1 = int(bbox["y"] * img_height)
                x2 = int((bbox["x"] + bbox["width"]) * img_width)
                y2 = int((bbox["y"] + bbox["height"]) * img_height)
            else:
                continue

            # Ensure valid box
            x1, x2 = max(0, x1), min(img_width, x2)
            y1, y2 = max(0, y1), min(img_height, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract region
            region_crop = result_image.crop((x1, y1, x2, y2))

            # Apply blur
            if blur_type == "gaussian":
                # Gaussian blur
                blur_radius = max(1, intensity // 2)
                blurred = region_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            elif blur_type == "pixelate":
                # Pixelate effect
                pixel_size = max(1, intensity)
                small = region_crop.resize(
                    (max(1, region_crop.width // pixel_size), max(1, region_crop.height // pixel_size)),
                    Image.Resampling.NEAREST
                )
                blurred = small.resize(region_crop.size, Image.Resampling.NEAREST)
            elif blur_type == "black":
                # Black out region
                blurred = Image.new("RGB", region_crop.size, (0, 0, 0))
            else:
                blurred = region_crop

            # Paste blurred region back
            result_image.paste(blurred, (x1, y1))
            blurred_count += 1

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"image": f"data:image/jpeg;base64,{image_to_base64(result_image)}"},
            duration_ms=round(duration, 2),
            metrics={"blurred_regions": blurred_count, "total_regions": len(regions)},
        )


class DrawBoxesBlock(BaseBlock):
    """
    Draw Boxes Block

    Draws bounding boxes and labels on images.
    """

    block_type = "draw_boxes"
    display_name = "Draw Boxes"
    description = "Draw bounding boxes on image"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Image with boxes drawn"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "line_width": {"type": "number", "default": 2},
            "show_labels": {"type": "boolean", "default": True},
            "show_confidence": {"type": "boolean", "default": True},
            "color_by_class": {"type": "boolean", "default": True},
            "default_color": {"type": "string", "default": "#00FF00"},
            "font_size": {"type": "number", "default": 12},
        },
    }

    # Color palette for different classes
    COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FF8000", "#8000FF", "#0080FF", "#FF0080", "#80FF00", "#00FF80",
    ]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Draw bounding boxes on image."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        detections = inputs.get("detections", [])
        if not detections:
            return BlockResult(
                outputs={"image": f"data:image/jpeg;base64,{image_to_base64(image)}"},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"boxes_drawn": 0},
            )

        line_width = config.get("line_width", 2)
        show_labels = config.get("show_labels", True)
        show_confidence = config.get("show_confidence", True)
        color_by_class = config.get("color_by_class", True)
        default_color = config.get("default_color", "#00FF00")
        font_size = config.get("font_size", 12)

        img_width, img_height = image.size
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        # Build class -> color mapping
        class_colors = {}
        color_idx = 0

        for det in detections:
            # Get bounding box
            bbox = det.get("bbox", {})
            if "x1" in bbox:
                x1 = int(bbox["x1"] * img_width)
                y1 = int(bbox["y1"] * img_height)
                x2 = int(bbox["x2"] * img_width)
                y2 = int(bbox["y2"] * img_height)
            elif "x" in bbox:
                x1 = int(bbox["x"] * img_width)
                y1 = int(bbox["y"] * img_height)
                x2 = int((bbox["x"] + bbox["width"]) * img_width)
                y2 = int((bbox["y"] + bbox["height"]) * img_height)
            else:
                continue

            # Get color
            class_name = det.get("class_name", "unknown")
            if color_by_class:
                if class_name not in class_colors:
                    class_colors[class_name] = self.COLORS[color_idx % len(self.COLORS)]
                    color_idx += 1
                color = class_colors[class_name]
            else:
                color = default_color

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label
            if show_labels:
                label = class_name
                if show_confidence and "confidence" in det:
                    label = f"{class_name} {det['confidence']:.2f}"

                # Calculate text position
                text_x = x1
                text_y = y1 - font_size - 4
                if text_y < 0:
                    text_y = y2 + 2

                # Draw text background
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
                draw.rectangle(
                    [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                    fill=color
                )
                draw.text((text_x, text_y), label, fill="white", font=font)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"image": f"data:image/jpeg;base64,{image_to_base64(result_image)}"},
            duration_ms=round(duration, 2),
            metrics={"boxes_drawn": len(detections)},
        )


class SegmentationBlock(BaseBlock):
    """
    Segmentation Block

    Segments objects in images using SAM or similar models.
    """

    block_type = "segmentation"
    display_name = "Segmentation"
    description = "Segment objects in images"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "array", "required": False, "description": "Optional detection boxes as prompts"},
    ]
    output_ports = [
        {"name": "masks", "type": "array", "description": "Segmentation masks"},
        {"name": "masked_image", "type": "image", "description": "Image with masks applied"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "default": "sam-base"},
            "mask_threshold": {"type": "number", "default": 0.5},
            "points_per_side": {"type": "number", "default": 32},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Segment objects in image."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        detections = inputs.get("detections", [])

        # For now, return a placeholder implementation
        # Full SAM integration would require loading the model
        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "masks": [],  # Would contain mask data
                "masked_image": f"data:image/jpeg;base64,{image_to_base64(image)}",
            },
            duration_ms=round(duration, 2),
            metrics={
                "segment_count": 0,
                "note": "SAM integration pending",
            },
        )
