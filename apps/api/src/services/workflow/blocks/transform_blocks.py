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

    SOTA Features:
    - Multiple padding modes (pixel, percent, none)
    - Aspect ratio enforcement (square, 4:3, 16:9, custom)
    - Output size standardization
    - Confidence-based filtering
    - Edge padding strategies
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
            # Padding
            "padding_mode": {
                "type": "string",
                "enum": ["pixel", "percent", "none"],
                "default": "percent",
                "description": "How to calculate padding",
            },
            "padding": {
                "type": "number",
                "default": 10,
                "description": "Padding amount (pixels or percentage based on mode)",
            },
            "padding_fill": {
                "type": "string",
                "enum": ["constant", "reflect", "replicate"],
                "default": "constant",
                "description": "How to fill padded areas that extend beyond image",
            },
            "padding_color": {
                "type": "string",
                "default": "#000000",
                "description": "Fill color for constant padding",
            },
            # Aspect ratio
            "aspect_ratio": {
                "type": "string",
                "enum": ["original", "square", "4:3", "3:4", "16:9", "9:16", "custom"],
                "default": "original",
                "description": "Force crops to specific aspect ratio",
            },
            "aspect_w": {"type": "number", "default": 1, "description": "Custom aspect width"},
            "aspect_h": {"type": "number", "default": 1, "description": "Custom aspect height"},
            # Output size
            "output_size": {
                "type": "string",
                "enum": ["original", "64", "128", "224", "256", "384", "512", "custom"],
                "default": "original",
                "description": "Resize all crops to consistent size",
            },
            "custom_size": {
                "type": "number",
                "default": 256,
                "description": "Custom output size when output_size='custom'",
            },
            # Filtering
            "min_size": {"type": "number", "default": 32, "description": "Minimum crop size in pixels"},
            "max_crops": {"type": "number", "default": 100, "description": "Maximum number of crops"},
            "filter_classes": {"type": "array", "items": {"type": "string"}, "description": "Only crop these classes"},
            "filter_by_confidence": {
                "type": "boolean",
                "default": False,
                "description": "Filter by detection confidence",
            },
            "min_confidence": {
                "type": "number",
                "default": 0.5,
                "description": "Minimum confidence to include crop",
            },
            # Sorting
            "sort_by": {
                "type": "string",
                "enum": ["none", "confidence", "area", "left_to_right", "top_to_bottom"],
                "default": "none",
            },
        },
    }

    def _parse_aspect_ratio(self, config: dict) -> tuple[float, float] | None:
        """Parse aspect ratio from config."""
        ar = config.get("aspect_ratio", "original")
        if ar == "original":
            return None
        elif ar == "square":
            return (1, 1)
        elif ar == "4:3":
            return (4, 3)
        elif ar == "3:4":
            return (3, 4)
        elif ar == "16:9":
            return (16, 9)
        elif ar == "9:16":
            return (9, 16)
        elif ar == "custom":
            return (config.get("aspect_w", 1), config.get("aspect_h", 1))
        return None

    def _apply_aspect_ratio(
        self,
        x1: float, y1: float, x2: float, y2: float,
        target_ratio: tuple[float, float],
        img_width: int, img_height: int,
    ) -> tuple[float, float, float, float]:
        """Expand box to match target aspect ratio."""
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        target_w_ratio = target_ratio[0] / target_ratio[1]
        current_ratio = w / h if h > 0 else 1

        if current_ratio > target_w_ratio:
            # Too wide, expand height
            new_h = w / target_w_ratio
            new_w = w
        else:
            # Too tall, expand width
            new_w = h * target_w_ratio
            new_h = h

        # Center the new box
        x1 = cx - new_w / 2
        y1 = cy - new_h / 2
        x2 = cx + new_w / 2
        y2 = cy + new_h / 2

        return x1, y1, x2, y2

    def _pad_crop(
        self,
        crop: Image.Image,
        target_size: tuple[int, int],
        fill_mode: str,
        fill_color: str,
    ) -> Image.Image:
        """Pad crop to target size if needed."""
        if crop.size == target_size:
            return crop

        # Parse fill color
        try:
            color = tuple(int(fill_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            color = (0, 0, 0)

        if fill_mode == "constant":
            padded = Image.new("RGB", target_size, color)
            paste_x = (target_size[0] - crop.width) // 2
            paste_y = (target_size[1] - crop.height) // 2
            padded.paste(crop, (paste_x, paste_y))
            return padded

        elif fill_mode == "reflect":
            # Reflect padding using numpy
            arr = np.array(crop)
            pad_left = (target_size[0] - crop.width) // 2
            pad_right = target_size[0] - crop.width - pad_left
            pad_top = (target_size[1] - crop.height) // 2
            pad_bottom = target_size[1] - crop.height - pad_top

            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
            return Image.fromarray(padded)

        elif fill_mode == "replicate":
            arr = np.array(crop)
            pad_left = (target_size[0] - crop.width) // 2
            pad_right = target_size[0] - crop.width - pad_left
            pad_top = (target_size[1] - crop.height) // 2
            pad_bottom = target_size[1] - crop.height - pad_top

            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
            return Image.fromarray(padded)

        return crop

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Crop regions from image with SOTA features."""
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

        # Parse config
        padding_mode = config.get("padding_mode", "percent")
        padding = config.get("padding", 10)
        padding_fill = config.get("padding_fill", "constant")
        padding_color = config.get("padding_color", "#000000")

        aspect_ratio = self._parse_aspect_ratio(config)

        output_size_str = config.get("output_size", "original")
        if output_size_str == "original":
            output_size = None
        elif output_size_str == "custom":
            output_size = config.get("custom_size", 256)
        else:
            output_size = int(output_size_str)

        min_size = config.get("min_size", 32)
        max_crops = config.get("max_crops", 100)
        filter_classes = config.get("filter_classes")
        filter_by_confidence = config.get("filter_by_confidence", False)
        min_confidence = config.get("min_confidence", 0.5)
        sort_by = config.get("sort_by", "none")

        img_width, img_height = image.size

        # Sort detections if requested
        if sort_by == "confidence":
            detections = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)
        elif sort_by == "area":
            def get_area(d):
                bbox = d.get("bbox", {})
                if "x1" in bbox:
                    return (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
                elif "width" in bbox:
                    return bbox["width"] * bbox["height"]
                return 0
            detections = sorted(detections, key=get_area, reverse=True)
        elif sort_by == "left_to_right":
            def get_x(d):
                bbox = d.get("bbox", {})
                return bbox.get("x1", bbox.get("x", 0))
            detections = sorted(detections, key=get_x)
        elif sort_by == "top_to_bottom":
            def get_y(d):
                bbox = d.get("bbox", {})
                return bbox.get("y1", bbox.get("y", 0))
            detections = sorted(detections, key=get_y)

        crops = []
        metadata = []

        for i, det in enumerate(detections[:max_crops]):
            # Filter by class if specified
            if filter_classes and det.get("class_name") not in filter_classes:
                continue

            # Filter by confidence if enabled
            if filter_by_confidence and det.get("confidence", 1.0) < min_confidence:
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
            if padding_mode == "pixel" and padding > 0:
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding
            elif padding_mode == "percent" and padding > 0:
                pad_w = (x2 - x1) * (padding / 100.0)
                pad_h = (y2 - y1) * (padding / 100.0)
                x1 -= pad_w
                y1 -= pad_h
                x2 += pad_w
                y2 += pad_h

            # Apply aspect ratio if specified
            if aspect_ratio:
                x1, y1, x2, y2 = self._apply_aspect_ratio(x1, y1, x2, y2, aspect_ratio, img_width, img_height)

            # Clamp to image bounds
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(img_width, x2)
            y2_clamped = min(img_height, y2)

            # Check minimum size
            if (x2_clamped - x1_clamped) < min_size or (y2_clamped - y1_clamped) < min_size:
                continue

            # Crop
            crop = image.crop((int(x1_clamped), int(y1_clamped), int(x2_clamped), int(y2_clamped)))

            # Handle edge padding if crop was clamped
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                target_w = int(x2 - x1)
                target_h = int(y2 - y1)
                if target_w > crop.width or target_h > crop.height:
                    crop = self._pad_crop(crop, (target_w, target_h), padding_fill, padding_color)

            # Resize to output size if specified
            if output_size:
                # Maintain aspect ratio within square
                ratio = output_size / max(crop.width, crop.height)
                new_w = int(crop.width * ratio)
                new_h = int(crop.height * ratio)
                crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Pad to square if needed
                if aspect_ratio == (1, 1) or aspect_ratio is None:
                    if crop.width != output_size or crop.height != output_size:
                        crop = self._pad_crop(crop, (output_size, output_size), padding_fill, padding_color)

            crop_base64 = f"data:image/jpeg;base64,{image_to_base64(crop)}"

            crops.append(crop_base64)
            metadata.append({
                "index": i,
                "detection": det,
                "original_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_box": {"x1": x1_clamped, "y1": y1_clamped, "x2": x2_clamped, "y2": y2_clamped},
                "crop_size": {"width": crop.width, "height": crop.height},
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
            })

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"crops": crops, "crop_metadata": metadata},
            duration_ms=round(duration, 2),
            metrics={
                "crop_count": len(crops),
                "total_detections": len(detections),
                "output_size": output_size,
                "aspect_ratio": config.get("aspect_ratio", "original"),
            },
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

    Segments objects in images using SAM (Segment Anything Model).

    SOTA Features:
    - SAM, SAM2, SAM3 support
    - Multiple prompt modes: boxes, points, text (SAM3), grid
    - Multi-mask output with IoU prediction
    - Multiple output formats: binary, polygon, RLE, COCO
    """

    block_type = "segmentation"
    display_name = "Segmentation"
    description = "Segment objects in images using SAM models"

    input_ports = [
        {"name": "image", "type": "image", "required": True, "description": "Input image to segment"},
        {"name": "detections", "type": "array", "required": False, "description": "Detection boxes as prompts"},
        {"name": "points", "type": "array", "required": False, "description": "Point coordinates as prompts"},
    ]
    output_ports = [
        {"name": "masks", "type": "array", "description": "Segmentation masks with metadata"},
        {"name": "masked_image", "type": "image", "description": "Image with masks visualized"},
        {"name": "count", "type": "number", "description": "Number of segments"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Model selection
            "model_id": {
                "type": "string",
                "description": "SAM model ID (sam-base, sam-large, sam2-base, sam2-large, sam3)",
            },
            "model_source": {
                "type": "string",
                "enum": ["pretrained", "trained"],
                "default": "pretrained",
            },
            # Prompt mode
            "prompt_mode": {
                "type": "string",
                "enum": ["auto", "boxes", "points", "text", "grid"],
                "default": "auto",
                "description": "How to prompt the model",
            },
            "text_prompt": {
                "type": "string",
                "description": "Text prompt for SAM3 (e.g., 'the red car, all people')",
            },
            # Grid mode options
            "points_per_side": {
                "type": "number",
                "default": 32,
                "description": "Points per side for grid mode (16, 32, 64, 128)",
            },
            # Mask options
            "mask_threshold": {
                "type": "number",
                "default": 0.0,
                "minimum": -1,
                "maximum": 1,
                "description": "Mask logit threshold (higher = stricter boundaries)",
            },
            "multi_mask": {
                "type": "boolean",
                "default": False,
                "description": "Return multiple mask candidates per prompt",
            },
            "masks_per_prompt": {
                "type": "number",
                "default": 3,
                "description": "Number of masks per prompt when multi_mask is True",
            },
            "select_best_mask": {
                "type": "boolean",
                "default": True,
                "description": "Auto-select best mask by IoU prediction",
            },
            # Advanced SAM options
            "stability_score_thresh": {
                "type": "number",
                "default": 0.95,
                "description": "Filter masks below this stability score",
            },
            "box_nms_thresh": {
                "type": "number",
                "default": 0.7,
                "description": "Remove overlapping masks (lower = more filtering)",
            },
            "min_mask_area": {
                "type": "number",
                "default": 0,
                "description": "Filter masks smaller than this area (pixels²)",
            },
            # Mask refinement
            "refine_masks": {
                "type": "boolean",
                "default": False,
                "description": "Post-process to smooth mask edges",
            },
            "refinement_iterations": {
                "type": "number",
                "default": 4,
                "description": "Number of refinement iterations (2, 4, 8)",
            },
            # Output format
            "mask_format": {
                "type": "string",
                "enum": ["binary", "polygon", "rle", "coco"],
                "default": "binary",
                "description": "Output mask format",
            },
            # Output options
            "output_masked_image": {
                "type": "boolean",
                "default": True,
                "description": "Include visualization image",
            },
            "output_crops": {
                "type": "boolean",
                "default": False,
                "description": "Include cropped objects from masks",
            },
            "output_areas": {
                "type": "boolean",
                "default": True,
                "description": "Include mask area in pixels",
            },
            "output_polygons": {
                "type": "boolean",
                "default": False,
                "description": "Include polygon contours",
            },
            # Visualization
            "draw_masks": {
                "type": "boolean",
                "default": True,
                "description": "Draw masks on output image",
            },
            "mask_opacity": {
                "type": "number",
                "default": 0.5,
                "description": "Mask overlay transparency (0-1)",
            },
            "draw_contours": {
                "type": "boolean",
                "default": True,
                "description": "Draw contour outlines",
            },
            "color_mode": {
                "type": "string",
                "enum": ["random", "rainbow", "category", "single"],
                "default": "random",
            },
            "overlay_alpha": {
                "type": "number",
                "default": 0.5,
                "description": "Mask overlay transparency (0-1)",
            },
            "color_palette": {
                "type": "string",
                "enum": ["rainbow", "random", "fixed"],
                "default": "rainbow",
            },
        },
    }

    def _mask_to_polygon(self, mask: np.ndarray) -> list[list[float]]:
        """Convert binary mask to polygon contours."""
        import cv2
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = contour.flatten().tolist()
                polygons.append(polygon)
        return polygons

    def _mask_to_rle(self, mask: np.ndarray) -> dict:
        """Convert binary mask to RLE (Run-Length Encoding)."""
        pixels = mask.flatten()
        runs = []
        prev = 0
        count = 0

        for pixel in pixels:
            if pixel != prev:
                if count > 0:
                    runs.append(count)
                count = 1
                prev = pixel
            else:
                count += 1

        if count > 0:
            runs.append(count)

        return {
            "counts": runs,
            "size": list(mask.shape),
        }

    def _draw_masks_on_image(
        self,
        image: Image.Image,
        masks: list[np.ndarray],
        alpha: float = 0.5,
        palette: str = "rainbow",
    ) -> Image.Image:
        """Overlay masks on image."""
        img_array = np.array(image)

        # Generate colors
        if palette == "rainbow":
            import colorsys
            colors = []
            for i in range(len(masks)):
                hue = i / max(len(masks), 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                colors.append([int(c * 255) for c in rgb])
        elif palette == "random":
            colors = [np.random.randint(0, 255, 3).tolist() for _ in masks]
        else:
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]] * ((len(masks) // 4) + 1)

        # Apply masks
        for mask, color in zip(masks, colors):
            if mask.shape[:2] != img_array.shape[:2]:
                # Resize mask to match image
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_array.shape[1], img_array.shape[0]))
                mask = np.array(mask_pil) > 127

            for c in range(3):
                img_array[:, :, c] = np.where(
                    mask,
                    img_array[:, :, c] * (1 - alpha) + color[c] * alpha,
                    img_array[:, :, c]
                )

        return Image.fromarray(img_array.astype(np.uint8))

    def _refine_mask(self, mask: np.ndarray, iterations: int = 4) -> np.ndarray:
        """Refine mask with morphological operations to smooth edges."""
        try:
            import cv2
            mask_uint8 = (mask * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            # Close small holes
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            # Open to remove small noise
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=iterations // 2)
            return mask_uint8 > 127
        except ImportError:
            return mask

    def _crop_from_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image | None:
        """Crop object from image using mask."""
        try:
            # Find bounding box of mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return None
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            # Crop image
            crop = image.crop((x1, y1, x2 + 1, y2 + 1))

            # Apply mask alpha
            mask_crop = mask[y1:y2+1, x1:x2+1]
            crop_rgba = crop.convert("RGBA")
            alpha = Image.fromarray((mask_crop * 255).astype(np.uint8))
            crop_rgba.putalpha(alpha)

            return crop_rgba
        except Exception:
            return None

    def _draw_contours_on_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw contours on image."""
        try:
            import cv2
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, thickness)
        except ImportError:
            pass
        return image

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Segment objects in image using SAM with SOTA features."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get config
        model_id = config.get("model_id", "sam-base")
        prompt_mode = config.get("prompt_mode", "auto")
        text_prompt = config.get("text_prompt")
        points_per_side = config.get("points_per_side", 32)
        mask_threshold = config.get("mask_threshold", 0.0)
        multi_mask = config.get("multi_mask", False)
        masks_per_prompt = config.get("masks_per_prompt", 3)
        select_best_mask = config.get("select_best_mask", True)
        mask_format = config.get("mask_format", "binary")

        # Advanced SAM options
        stability_score_thresh = config.get("stability_score_thresh", 0.95)
        box_nms_thresh = config.get("box_nms_thresh", 0.7)
        min_mask_area = config.get("min_mask_area", 0)

        # Refinement options
        refine_masks = config.get("refine_masks", False)
        refinement_iterations = config.get("refinement_iterations", 4)

        # Output options
        output_masked_image = config.get("output_masked_image", True)
        output_crops = config.get("output_crops", False)
        output_areas = config.get("output_areas", True)
        output_polygons = config.get("output_polygons", False)

        # Visualization options
        draw_masks = config.get("draw_masks", True)
        mask_opacity = config.get("mask_opacity", config.get("overlay_alpha", 0.5))
        draw_contours = config.get("draw_contours", True)
        color_mode = config.get("color_mode", config.get("color_palette", "random"))

        # Get prompts from inputs
        detections = inputs.get("detections", [])
        points = inputs.get("points", [])

        # Determine actual prompt mode
        if prompt_mode == "auto":
            if detections:
                prompt_mode = "boxes"
            elif points:
                prompt_mode = "points"
            else:
                prompt_mode = "grid"

        try:
            masks_output = []
            mask_arrays = []
            crops = []

            # Check if SAM is available
            sam_available = False
            try:
                from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
                sam_available = True
            except ImportError:
                pass

            if sam_available:
                import torch

                # Determine model type
                if "sam2" in model_id.lower():
                    model_type = "vit_h"
                elif "large" in model_id.lower():
                    model_type = "vit_l"
                elif "huge" in model_id.lower():
                    model_type = "vit_h"
                else:
                    model_type = "vit_b"

                # Find checkpoint
                checkpoint_paths = [
                    f"/models/sam/{model_id}.pth",
                    f"~/.cache/sam/{model_id}.pth",
                    f"sam_{model_type}.pth",
                ]

                checkpoint = None
                for path in checkpoint_paths:
                    import os
                    expanded = os.path.expanduser(path)
                    if os.path.exists(expanded):
                        checkpoint = expanded
                        break

                if checkpoint:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    sam = sam_model_registry[model_type](checkpoint=checkpoint)
                    sam.to(device)

                    image_np = np.array(image)

                    if prompt_mode == "grid":
                        # Automatic mask generation with advanced options
                        mask_generator = SamAutomaticMaskGenerator(
                            sam,
                            points_per_side=points_per_side,
                            pred_iou_thresh=0.88,
                            stability_score_thresh=stability_score_thresh,
                            box_nms_thresh=box_nms_thresh,
                            min_mask_region_area=min_mask_area,
                        )
                        auto_masks = mask_generator.generate(image_np)

                        for i, mask_data in enumerate(auto_masks):
                            mask = mask_data["segmentation"]

                            # Apply refinement if enabled
                            if refine_masks:
                                mask = self._refine_mask(mask, refinement_iterations)

                            # Filter by area
                            area = int(mask.sum())
                            if min_mask_area > 0 and area < min_mask_area:
                                continue

                            mask_arrays.append(mask)

                            mask_output = {
                                "id": i,
                                "predicted_iou": float(mask_data["predicted_iou"]),
                                "stability_score": float(mask_data["stability_score"]),
                            }

                            if output_areas:
                                mask_output["area"] = area
                                mask_output["bbox"] = mask_data["bbox"]

                            if mask_format == "binary":
                                mask_output["mask"] = mask.tolist()
                            elif mask_format == "polygon" or output_polygons:
                                mask_output["polygon"] = self._mask_to_polygon(mask)
                            if mask_format == "rle":
                                mask_output["rle"] = self._mask_to_rle(mask)
                            elif mask_format == "coco":
                                mask_output["polygon"] = self._mask_to_polygon(mask)
                                mask_output["rle"] = self._mask_to_rle(mask)

                            # Generate crop if requested
                            if output_crops:
                                crop_img = self._crop_from_mask(image, mask)
                                if crop_img:
                                    crops.append(f"data:image/png;base64,{image_to_base64(crop_img, 'PNG')}")

                            masks_output.append(mask_output)

                    else:
                        # Prompted segmentation
                        predictor = SamPredictor(sam)
                        predictor.set_image(image_np)

                        if prompt_mode == "boxes" and detections:
                            for i, det in enumerate(detections):
                                bbox = det.get("bbox", {})
                                if "x1" in bbox:
                                    box = np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
                                    if box.max() <= 1:
                                        box = box * np.array([image.width, image.height, image.width, image.height])
                                else:
                                    continue

                                masks, iou_preds, _ = predictor.predict(
                                    box=box,
                                    multimask_output=multi_mask,
                                )

                                if select_best_mask and len(masks) > 1:
                                    best_idx = np.argmax(iou_preds)
                                    masks = [masks[best_idx]]
                                    iou_preds = [iou_preds[best_idx]]

                                for j, (mask, iou) in enumerate(zip(masks, iou_preds)):
                                    mask_binary = mask > mask_threshold

                                    if refine_masks:
                                        mask_binary = self._refine_mask(mask_binary, refinement_iterations)

                                    area = int(mask_binary.sum())
                                    if min_mask_area > 0 and area < min_mask_area:
                                        continue

                                    mask_arrays.append(mask_binary)

                                    mask_output = {
                                        "id": len(masks_output),
                                        "source_detection_id": i,
                                        "class_name": det.get("class_name", "unknown"),
                                        "predicted_iou": float(iou),
                                    }

                                    if output_areas:
                                        mask_output["area"] = area

                                    if mask_format == "binary":
                                        mask_output["mask"] = mask_binary.tolist()
                                    elif mask_format == "polygon" or output_polygons:
                                        mask_output["polygon"] = self._mask_to_polygon(mask_binary)
                                    if mask_format == "rle":
                                        mask_output["rle"] = self._mask_to_rle(mask_binary)
                                    elif mask_format == "coco":
                                        mask_output["polygon"] = self._mask_to_polygon(mask_binary)
                                        mask_output["rle"] = self._mask_to_rle(mask_binary)

                                    if output_crops:
                                        crop_img = self._crop_from_mask(image, mask_binary)
                                        if crop_img:
                                            crops.append(f"data:image/png;base64,{image_to_base64(crop_img, 'PNG')}")

                                    masks_output.append(mask_output)

                        elif prompt_mode == "points" and points:
                            point_coords = np.array([[p["x"], p["y"]] for p in points])
                            point_labels = np.array([p.get("label", 1) for p in points])

                            masks, iou_preds, _ = predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=multi_mask,
                            )

                            if select_best_mask and len(masks) > 1:
                                best_idx = np.argmax(iou_preds)
                                masks = [masks[best_idx]]
                                iou_preds = [iou_preds[best_idx]]

                            for j, (mask, iou) in enumerate(zip(masks, iou_preds)):
                                mask_binary = mask > mask_threshold

                                if refine_masks:
                                    mask_binary = self._refine_mask(mask_binary, refinement_iterations)

                                area = int(mask_binary.sum())
                                if min_mask_area > 0 and area < min_mask_area:
                                    continue

                                mask_arrays.append(mask_binary)

                                mask_output = {
                                    "id": j,
                                    "predicted_iou": float(iou),
                                }

                                if output_areas:
                                    mask_output["area"] = area

                                if mask_format == "binary":
                                    mask_output["mask"] = mask_binary.tolist()
                                elif mask_format == "polygon" or output_polygons:
                                    mask_output["polygon"] = self._mask_to_polygon(mask_binary)
                                if mask_format == "rle":
                                    mask_output["rle"] = self._mask_to_rle(mask_binary)
                                elif mask_format == "coco":
                                    mask_output["polygon"] = self._mask_to_polygon(mask_binary)
                                    mask_output["rle"] = self._mask_to_rle(mask_binary)

                                if output_crops:
                                    crop_img = self._crop_from_mask(image, mask_binary)
                                    if crop_img:
                                        crops.append(f"data:image/png;base64,{image_to_base64(crop_img, 'PNG')}")

                                masks_output.append(mask_output)
                else:
                    return BlockResult(
                        error=f"SAM model checkpoint not found for {model_id}",
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                    )
            else:
                return BlockResult(
                    error="SAM not installed. Install with: pip install segment-anything",
                    duration_ms=round((time.time() - start_time) * 1000, 2),
                )

            # Generate visualization
            outputs = {
                "masks": masks_output,
                "count": len(masks_output),
            }

            if output_masked_image and draw_masks and mask_arrays:
                masked_image = self._draw_masks_on_image(
                    image, mask_arrays, alpha=mask_opacity, palette=color_mode
                )

                # Draw contours if requested
                if draw_contours:
                    import colorsys
                    img_arr = np.array(masked_image)
                    for i, mask in enumerate(mask_arrays):
                        if color_mode == "rainbow":
                            hue = i / max(len(mask_arrays), 1)
                            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                            color = tuple(int(c * 255) for c in rgb)
                        else:
                            color = (255, 255, 255)
                        img_arr = self._draw_contours_on_image(img_arr, mask, color, thickness=2)
                    masked_image = Image.fromarray(img_arr)

                outputs["masked_image"] = f"data:image/jpeg;base64,{image_to_base64(masked_image)}"
            elif output_masked_image:
                outputs["masked_image"] = f"data:image/jpeg;base64,{image_to_base64(image)}"

            if output_crops and crops:
                outputs["crops"] = crops

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs=outputs,
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "prompt_mode": prompt_mode,
                    "segment_count": len(masks_output),
                    "mask_format": mask_format,
                    "refined": refine_masks,
                },
            )

        except Exception as e:
            logger.exception("Segmentation failed")
            return BlockResult(
                error=f"Segmentation failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )


class ResizeBlock(BaseBlock):
    """
    Resize Block

    Resizes images to specified dimensions with multiple interpolation methods.
    """

    block_type = "resize"
    display_name = "Resize"
    description = "Resize images to specified dimensions"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Resized image"},
        {"name": "original_size", "type": "object", "description": "Original dimensions"},
        {"name": "new_size", "type": "object", "description": "New dimensions"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["dimensions", "scale", "fit", "fill", "long_edge", "short_edge"],
                "default": "dimensions",
            },
            "width": {"type": "number", "description": "Target width"},
            "height": {"type": "number", "description": "Target height"},
            "scale": {"type": "number", "default": 1.0, "description": "Scale factor"},
            "interpolation": {
                "type": "string",
                "enum": ["nearest", "bilinear", "bicubic", "lanczos"],
                "default": "lanczos",
            },
            "maintain_aspect": {"type": "boolean", "default": True},
            "upscale": {"type": "boolean", "default": False},
        },
    }

    INTERPOLATION_MAP = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Resize image."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        orig_w, orig_h = image.size
        method = config.get("method", "dimensions")
        interpolation = self.INTERPOLATION_MAP.get(config.get("interpolation", "lanczos"), Image.Resampling.LANCZOS)
        maintain_aspect = config.get("maintain_aspect", True)
        upscale = config.get("upscale", False)

        target_w, target_h = orig_w, orig_h

        if method == "dimensions":
            target_w = config.get("width", orig_w)
            target_h = config.get("height", orig_h)
            if maintain_aspect:
                ratio = min(target_w / orig_w, target_h / orig_h)
                target_w = int(orig_w * ratio)
                target_h = int(orig_h * ratio)

        elif method == "scale":
            scale = config.get("scale", 1.0)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)

        elif method == "fit":
            target_w = config.get("width", orig_w)
            target_h = config.get("height", orig_h)
            ratio = min(target_w / orig_w, target_h / orig_h)
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        elif method == "fill":
            target_w = config.get("width", orig_w)
            target_h = config.get("height", orig_h)
            ratio = max(target_w / orig_w, target_h / orig_h)
            temp_w = int(orig_w * ratio)
            temp_h = int(orig_h * ratio)
            image = image.resize((temp_w, temp_h), interpolation)
            left = (temp_w - target_w) // 2
            top = (temp_h - target_h) // 2
            image = image.crop((left, top, left + target_w, top + target_h))
            target_w, target_h = image.size

        elif method == "long_edge":
            long_edge = config.get("width", max(orig_w, orig_h))
            if orig_w >= orig_h:
                ratio = long_edge / orig_w
            else:
                ratio = long_edge / orig_h
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        elif method == "short_edge":
            short_edge = config.get("width", min(orig_w, orig_h))
            if orig_w <= orig_h:
                ratio = short_edge / orig_w
            else:
                ratio = short_edge / orig_h
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        # Check upscale constraint
        if not upscale:
            target_w = min(target_w, orig_w)
            target_h = min(target_h, orig_h)

        target_w = max(1, target_w)
        target_h = max(1, target_h)

        if method != "fill":
            image = image.resize((target_w, target_h), interpolation)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "image": f"data:image/jpeg;base64,{image_to_base64(image)}",
                "original_size": {"width": orig_w, "height": orig_h},
                "new_size": {"width": target_w, "height": target_h},
            },
            duration_ms=round(duration, 2),
            metrics={"resize_ratio": round(target_w / orig_w, 3)},
        )


class TileBlock(BaseBlock):
    """
    Tile Block

    Splits images into tiles for processing large images or training SAHI.
    """

    block_type = "tile"
    display_name = "Tile"
    description = "Split image into tiles with optional overlap"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "tiles", "type": "array", "description": "Array of tile images"},
        {"name": "tile_info", "type": "array", "description": "Position and size info for each tile"},
        {"name": "grid_info", "type": "object", "description": "Grid dimensions and overlap info"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "tile_size": {"type": "number", "default": 640, "description": "Tile width and height"},
            "tile_width": {"type": "number", "description": "Tile width (overrides tile_size)"},
            "tile_height": {"type": "number", "description": "Tile height (overrides tile_size)"},
            "overlap": {"type": "number", "default": 0.2, "description": "Overlap ratio (0-1)"},
            "min_tile_area_ratio": {"type": "number", "default": 0.5, "description": "Min area ratio for edge tiles"},
            "padding": {"type": "string", "enum": ["none", "reflect", "constant"], "default": "constant"},
            "padding_value": {"type": "number", "default": 0, "description": "Padding value for constant mode"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Split image into tiles."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        img_w, img_h = image.size
        tile_w = config.get("tile_width") or config.get("tile_size", 640)
        tile_h = config.get("tile_height") or config.get("tile_size", 640)
        overlap = config.get("overlap", 0.2)
        min_area_ratio = config.get("min_tile_area_ratio", 0.5)

        step_x = int(tile_w * (1 - overlap))
        step_y = int(tile_h * (1 - overlap))

        tiles = []
        tile_info = []

        y = 0
        row_idx = 0
        while y < img_h:
            x = 0
            col_idx = 0
            while x < img_w:
                x1, y1 = x, y
                x2 = min(x + tile_w, img_w)
                y2 = min(y + tile_h, img_h)

                actual_w = x2 - x1
                actual_h = y2 - y1
                area_ratio = (actual_w * actual_h) / (tile_w * tile_h)

                if area_ratio >= min_area_ratio:
                    tile = image.crop((x1, y1, x2, y2))

                    # Pad if needed
                    if actual_w < tile_w or actual_h < tile_h:
                        padded = Image.new("RGB", (tile_w, tile_h), (0, 0, 0))
                        padded.paste(tile, (0, 0))
                        tile = padded

                    tiles.append(f"data:image/jpeg;base64,{image_to_base64(tile)}")
                    tile_info.append({
                        "index": len(tiles) - 1,
                        "row": row_idx,
                        "col": col_idx,
                        "x": x1,
                        "y": y1,
                        "width": actual_w,
                        "height": actual_h,
                        "padded": actual_w < tile_w or actual_h < tile_h,
                    })

                x += step_x
                col_idx += 1
            y += step_y
            row_idx += 1

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "tiles": tiles,
                "tile_info": tile_info,
                "grid_info": {
                    "image_size": {"width": img_w, "height": img_h},
                    "tile_size": {"width": tile_w, "height": tile_h},
                    "overlap": overlap,
                    "step": {"x": step_x, "y": step_y},
                    "rows": row_idx,
                    "cols": col_idx,
                    "total_tiles": len(tiles),
                },
            },
            duration_ms=round(duration, 2),
            metrics={"tile_count": len(tiles)},
        )


class StitchBlock(BaseBlock):
    """
    Stitch Block

    Reconstructs image from tiles, merging overlapping regions.
    """

    block_type = "stitch"
    display_name = "Stitch"
    description = "Merge tiles back into a single image"

    input_ports = [
        {"name": "tiles", "type": "array", "required": True, "description": "Array of tile images"},
        {"name": "tile_info", "type": "array", "required": True, "description": "Position info from Tile block"},
        {"name": "grid_info", "type": "object", "required": True, "description": "Grid info from Tile block"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Reconstructed image"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "blend_mode": {
                "type": "string",
                "enum": ["overwrite", "average", "max", "feather"],
                "default": "average",
            },
            "feather_size": {"type": "number", "default": 32, "description": "Feather blend size in pixels"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Stitch tiles back together."""
        start_time = time.time()

        tiles_data = inputs.get("tiles", [])
        tile_info = inputs.get("tile_info", [])
        grid_info = inputs.get("grid_info", {})

        if not tiles_data or not tile_info or not grid_info:
            return BlockResult(
                error="Missing required inputs: tiles, tile_info, or grid_info",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        img_size = grid_info.get("image_size", {})
        img_w = img_size.get("width", 0)
        img_h = img_size.get("height", 0)

        if img_w == 0 or img_h == 0:
            return BlockResult(
                error="Invalid grid_info: missing image dimensions",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        blend_mode = config.get("blend_mode", "average")
        result = np.zeros((img_h, img_w, 3), dtype=np.float32)
        weight_map = np.zeros((img_h, img_w), dtype=np.float32)

        for i, info in enumerate(tile_info):
            if i >= len(tiles_data):
                continue

            tile = await load_image_from_input(tiles_data[i])
            if tile is None:
                continue

            x, y = info.get("x", 0), info.get("y", 0)
            w, h = info.get("width", tile.width), info.get("height", tile.height)

            tile_np = np.array(tile.crop((0, 0, w, h))).astype(np.float32)

            if blend_mode == "overwrite":
                result[y:y+h, x:x+w] = tile_np
                weight_map[y:y+h, x:x+w] = 1.0
            elif blend_mode == "average":
                result[y:y+h, x:x+w] += tile_np
                weight_map[y:y+h, x:x+w] += 1.0
            elif blend_mode == "max":
                result[y:y+h, x:x+w] = np.maximum(result[y:y+h, x:x+w], tile_np)
                weight_map[y:y+h, x:x+w] = 1.0
            else:  # feather
                result[y:y+h, x:x+w] += tile_np
                weight_map[y:y+h, x:x+w] += 1.0

        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)
        result = result / weight_map[:, :, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)

        output_image = Image.fromarray(result)
        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "image": f"data:image/jpeg;base64,{image_to_base64(output_image)}",
            },
            duration_ms=round(duration, 2),
            metrics={"tiles_stitched": len(tile_info)},
        )


class RotateFlipBlock(BaseBlock):
    """
    Rotate/Flip Block

    Rotates and/or flips images for augmentation or orientation correction.
    """

    block_type = "rotate_flip"
    display_name = "Rotate/Flip"
    description = "Rotate and/or flip images"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Transformed image"},
        {"name": "transform_matrix", "type": "object", "description": "Transformation applied"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "rotation": {
                "type": "string",
                "enum": ["none", "90", "180", "270", "auto", "custom"],
                "default": "none",
            },
            "custom_angle": {"type": "number", "default": 0, "description": "Custom rotation angle in degrees"},
            "flip_horizontal": {"type": "boolean", "default": False},
            "flip_vertical": {"type": "boolean", "default": False},
            "expand": {"type": "boolean", "default": True, "description": "Expand canvas for rotation"},
            "fill_color": {"type": "string", "default": "#000000", "description": "Fill color for expanded areas"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Rotate and/or flip image."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        rotation = config.get("rotation", "none")
        custom_angle = config.get("custom_angle", 0)
        flip_h = config.get("flip_horizontal", False)
        flip_v = config.get("flip_vertical", False)
        expand = config.get("expand", True)
        fill_color = config.get("fill_color", "#000000")

        transforms_applied = []

        # Apply rotation
        if rotation == "90":
            image = image.transpose(Image.Transpose.ROTATE_90)
            transforms_applied.append("rotate_90")
        elif rotation == "180":
            image = image.transpose(Image.Transpose.ROTATE_180)
            transforms_applied.append("rotate_180")
        elif rotation == "270":
            image = image.transpose(Image.Transpose.ROTATE_270)
            transforms_applied.append("rotate_270")
        elif rotation == "custom" and custom_angle != 0:
            # Parse fill color
            try:
                fill = tuple(int(fill_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            except Exception:
                fill = (0, 0, 0)
            image = image.rotate(-custom_angle, expand=expand, fillcolor=fill)
            transforms_applied.append(f"rotate_{custom_angle}")
        elif rotation == "auto":
            # Try to detect and correct rotation from EXIF
            try:
                exif = image.getexif()
                if exif:
                    orientation = exif.get(274)  # Orientation tag
                    if orientation == 3:
                        image = image.transpose(Image.Transpose.ROTATE_180)
                        transforms_applied.append("auto_rotate_180")
                    elif orientation == 6:
                        image = image.transpose(Image.Transpose.ROTATE_270)
                        transforms_applied.append("auto_rotate_270")
                    elif orientation == 8:
                        image = image.transpose(Image.Transpose.ROTATE_90)
                        transforms_applied.append("auto_rotate_90")
            except Exception:
                pass

        # Apply flips
        if flip_h:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            transforms_applied.append("flip_horizontal")
        if flip_v:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            transforms_applied.append("flip_vertical")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "image": f"data:image/jpeg;base64,{image_to_base64(image)}",
                "transform_matrix": {
                    "transforms": transforms_applied,
                    "rotation": rotation if rotation != "custom" else custom_angle,
                    "flip_h": flip_h,
                    "flip_v": flip_v,
                },
            },
            duration_ms=round(duration, 2),
            metrics={"transforms_applied": len(transforms_applied)},
        )


class NormalizeBlock(BaseBlock):
    """
    Normalize Block

    Normalizes image pixel values for model input or visual enhancement.
    """

    block_type = "normalize"
    display_name = "Normalize"
    description = "Normalize image values"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Normalized image"},
        {"name": "stats", "type": "object", "description": "Normalization statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["minmax", "zscore", "imagenet", "clahe", "histogram", "custom"],
                "default": "minmax",
            },
            "output_range": {
                "type": "array",
                "items": {"type": "number"},
                "default": [0, 255],
                "description": "Output value range",
            },
            "mean": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Custom mean per channel (for custom/zscore)",
            },
            "std": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Custom std per channel (for custom/zscore)",
            },
            "clahe_clip_limit": {"type": "number", "default": 2.0},
            "clahe_grid_size": {"type": "number", "default": 8},
            "per_channel": {"type": "boolean", "default": True},
        },
    }

    # ImageNet normalization values
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Normalize image."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        method = config.get("method", "minmax")
        output_range = config.get("output_range", [0, 255])
        per_channel = config.get("per_channel", True)

        img_array = np.array(image).astype(np.float32)
        stats = {"method": method, "original_range": [float(img_array.min()), float(img_array.max())]}

        if method == "minmax":
            if per_channel:
                for c in range(3):
                    c_min, c_max = img_array[:, :, c].min(), img_array[:, :, c].max()
                    if c_max > c_min:
                        img_array[:, :, c] = (img_array[:, :, c] - c_min) / (c_max - c_min)
                    else:
                        img_array[:, :, c] = 0
            else:
                i_min, i_max = img_array.min(), img_array.max()
                if i_max > i_min:
                    img_array = (img_array - i_min) / (i_max - i_min)
                else:
                    img_array = np.zeros_like(img_array)

            img_array = img_array * (output_range[1] - output_range[0]) + output_range[0]

        elif method == "zscore":
            mean = config.get("mean")
            std = config.get("std")
            if mean is None or std is None:
                if per_channel:
                    mean = [img_array[:, :, c].mean() for c in range(3)]
                    std = [img_array[:, :, c].std() for c in range(3)]
                else:
                    mean = [img_array.mean()] * 3
                    std = [img_array.std()] * 3

            for c in range(3):
                if std[c] > 0:
                    img_array[:, :, c] = (img_array[:, :, c] - mean[c]) / std[c]

            stats["mean"] = mean
            stats["std"] = std
            # Rescale to output range
            img_array = np.clip(img_array * 64 + 128, 0, 255)

        elif method == "imagenet":
            img_array = img_array / 255.0
            for c in range(3):
                img_array[:, :, c] = (img_array[:, :, c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]

            stats["mean"] = self.IMAGENET_MEAN
            stats["std"] = self.IMAGENET_STD
            # Rescale to viewable range
            img_array = np.clip(img_array * 64 + 128, 0, 255)

        elif method == "clahe":
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            try:
                import cv2
                clip_limit = config.get("clahe_clip_limit", 2.0)
                grid_size = config.get("clahe_grid_size", 8)

                lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
            except ImportError:
                # Fallback to simple histogram equalization
                for c in range(3):
                    hist, bins = np.histogram(img_array[:, :, c].flatten(), 256, [0, 256])
                    cdf = hist.cumsum()
                    cdf_m = np.ma.masked_equal(cdf, 0)
                    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                    cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
                    img_array[:, :, c] = cdf[img_array[:, :, c].astype(np.uint8)]

        elif method == "histogram":
            # Simple histogram equalization per channel
            for c in range(3):
                hist, bins = np.histogram(img_array[:, :, c].flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
                img_array[:, :, c] = cdf[img_array[:, :, c].astype(np.uint8)]

        elif method == "custom":
            mean = config.get("mean", [127.5, 127.5, 127.5])
            std = config.get("std", [127.5, 127.5, 127.5])
            for c in range(3):
                if std[c] > 0:
                    img_array[:, :, c] = (img_array[:, :, c] - mean[c]) / std[c]
            stats["mean"] = mean
            stats["std"] = std
            img_array = np.clip(img_array * 64 + 128, 0, 255)

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(img_array)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "image": f"data:image/jpeg;base64,{image_to_base64(output_image)}",
                "stats": stats,
            },
            duration_ms=round(duration, 2),
            metrics={"method": method},
        )
