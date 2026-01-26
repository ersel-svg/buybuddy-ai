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
        {"name": "detections", "type": "array", "required": False, "description": "Array of detections to crop"},
        {"name": "detection", "type": "object", "required": False, "description": "Single detection (for ForEach patterns)"},
    ]
    output_ports = [
        {"name": "crops", "type": "array", "description": "Cropped image regions as base64"},
        {"name": "crop", "type": "image", "description": "First/single cropped image (for direct Embedding connection)"},
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

        # Support both singular 'detection' and plural 'detections' input
        detections = inputs.get("detections", [])
        single_detection = inputs.get("detection")

        # If singular detection provided (e.g., from ForEach), wrap in list
        if single_detection is not None and not detections:
            detections = [single_detection]

        if not detections:
            return BlockResult(
                outputs={"crops": [], "crop": None, "crop_metadata": []},
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
            outputs={
                "crops": crops,
                "crop": crops[0] if crops else None,  # Singular for direct Embedding connection
                "crop_metadata": metadata,
            },
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

    Resizes images to specified dimensions with SOTA features:
    - Multiple resize modes (exact, fit_within, fit_outside, scale_factor, etc.)
    - Upscale/downscale policies
    - Letterbox padding with configurable position and color
    - Crop position for fit_outside mode
    - Multiple of constraint (for model requirements)
    - Min/max size constraints
    - Anti-aliasing support
    """

    block_type = "resize"
    display_name = "Resize"
    description = "Resize images with SOTA options"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Resized image"},
        {"name": "original_size", "type": "object", "description": "Original dimensions"},
        {"name": "new_size", "type": "object", "description": "New dimensions"},
        {"name": "scale_info", "type": "object", "description": "Scale factors and padding info"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Resize mode
            "mode": {
                "type": "string",
                "enum": ["exact", "fit_within", "fit_outside", "scale_factor", "width_only", "height_only", "longest_edge", "shortest_edge"],
                "default": "fit_within",
                "description": "Resize mode",
            },
            # Target dimensions
            "width": {"type": "number", "description": "Target width"},
            "height": {"type": "number", "description": "Target height"},
            "scale_factor": {"type": "number", "default": 1.0, "description": "Scale factor for scale_factor mode"},
            "edge_size": {"type": "number", "default": 640, "description": "Target edge size for longest/shortest edge modes"},
            # Interpolation
            "interpolation": {
                "type": "string",
                "enum": ["nearest", "bilinear", "bicubic", "lanczos", "area"],
                "default": "bilinear",
                "description": "Interpolation method",
            },
            # Upscale/downscale policies
            "upscale_policy": {
                "type": "string",
                "enum": ["allow", "forbid"],
                "default": "allow",
                "description": "Whether to allow upscaling",
            },
            "downscale_policy": {
                "type": "string",
                "enum": ["allow", "forbid"],
                "default": "allow",
                "description": "Whether to allow downscaling",
            },
            # Padding (for fit_within letterbox)
            "add_padding": {"type": "boolean", "default": False, "description": "Add padding to reach exact target size"},
            "padding_position": {
                "type": "string",
                "enum": ["center", "top_left", "top_right", "bottom_left", "bottom_right"],
                "default": "center",
                "description": "Position of image within padded area",
            },
            "padding_color": {"type": "string", "default": "#000000", "description": "Padding fill color"},
            # Crop position (for fit_outside)
            "crop_position": {
                "type": "string",
                "enum": ["center", "top", "bottom", "left", "right"],
                "default": "center",
                "description": "Crop position for fit_outside mode",
            },
            # Advanced options
            "antialias": {"type": "boolean", "default": True, "description": "Apply anti-aliasing on downscale"},
            "output_scale_info": {"type": "boolean", "default": True, "description": "Include scale info in output"},
            "multiple_of": {"type": "number", "default": 0, "description": "Force dimensions to be multiple of N (0 = off)"},
            "min_size": {"type": "number", "description": "Minimum dimension constraint"},
            "max_size": {"type": "number", "description": "Maximum dimension constraint"},
        },
    }

    INTERPOLATION_MAP = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,  # BOX is best for downscaling (area averaging)
    }

    def _parse_color(self, color_str: str) -> tuple:
        """Parse hex color string to RGB tuple."""
        try:
            color = color_str.lstrip("#")
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

    def _apply_multiple_of(self, value: int, multiple: int) -> int:
        """Round value to nearest multiple."""
        if multiple <= 0:
            return value
        return max(multiple, round(value / multiple) * multiple)

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Resize image with SOTA options."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        orig_w, orig_h = image.size
        mode = config.get("mode", "fit_within")
        interpolation_key = config.get("interpolation", "bilinear")
        interpolation = self.INTERPOLATION_MAP.get(interpolation_key, Image.Resampling.BILINEAR)

        upscale_allowed = config.get("upscale_policy", "allow") == "allow"
        downscale_allowed = config.get("downscale_policy", "allow") == "allow"
        multiple_of = config.get("multiple_of", 0)
        min_size = config.get("min_size")
        max_size = config.get("max_size")

        target_w, target_h = orig_w, orig_h
        canvas_w, canvas_h = orig_w, orig_h
        pad_left, pad_top = 0, 0
        cropped = False

        # Calculate target dimensions based on mode
        if mode == "exact":
            target_w = config.get("width", orig_w)
            target_h = config.get("height", orig_h)

        elif mode == "fit_within":
            target_w = config.get("width", orig_w) or orig_w
            target_h = config.get("height", orig_h) or orig_h
            # Handle 0 values: if one dimension is 0, use only the other for scaling
            if target_w == 0 or target_w == orig_w and config.get("width") == 0:
                # Width is 0, scale by height only
                ratio = target_h / orig_h
            elif target_h == 0 or target_h == orig_h and config.get("height") == 0:
                # Height is 0, scale by width only
                ratio = target_w / orig_w
            else:
                ratio = min(target_w / orig_w, target_h / orig_h)
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        elif mode == "fit_outside":
            target_w = config.get("width", orig_w) or orig_w
            target_h = config.get("height", orig_h) or orig_h
            canvas_w, canvas_h = target_w, target_h
            # Handle 0 values: if one dimension is 0, use only the other for scaling
            if config.get("width") == 0:
                ratio = target_h / orig_h
            elif config.get("height") == 0:
                ratio = target_w / orig_w
            else:
                ratio = max(target_w / orig_w, target_h / orig_h)
            temp_w = int(orig_w * ratio)
            temp_h = int(orig_h * ratio)

            # Resize first
            image = image.resize((temp_w, temp_h), interpolation)

            # Crop based on position
            crop_position = config.get("crop_position", "center")
            if crop_position == "center":
                left = (temp_w - target_w) // 2
                top = (temp_h - target_h) // 2
            elif crop_position == "top":
                left = (temp_w - target_w) // 2
                top = 0
            elif crop_position == "bottom":
                left = (temp_w - target_w) // 2
                top = temp_h - target_h
            elif crop_position == "left":
                left = 0
                top = (temp_h - target_h) // 2
            elif crop_position == "right":
                left = temp_w - target_w
                top = (temp_h - target_h) // 2
            else:
                left = (temp_w - target_w) // 2
                top = (temp_h - target_h) // 2

            image = image.crop((left, top, left + target_w, top + target_h))
            cropped = True

        elif mode == "scale_factor":
            scale = config.get("scale_factor", 1.0)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)

        elif mode == "width_only":
            target_w = config.get("width", orig_w)
            ratio = target_w / orig_w
            target_h = int(orig_h * ratio)

        elif mode == "height_only":
            target_h = config.get("height", orig_h)
            ratio = target_h / orig_h
            target_w = int(orig_w * ratio)

        elif mode == "longest_edge":
            edge_size = config.get("edge_size", 640)
            if orig_w >= orig_h:
                ratio = edge_size / orig_w
            else:
                ratio = edge_size / orig_h
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        elif mode == "shortest_edge":
            edge_size = config.get("edge_size", 640)
            if orig_w <= orig_h:
                ratio = edge_size / orig_w
            else:
                ratio = edge_size / orig_h
            target_w = int(orig_w * ratio)
            target_h = int(orig_h * ratio)

        # Apply upscale/downscale policies
        if not upscale_allowed:
            target_w = min(target_w, orig_w)
            target_h = min(target_h, orig_h)
        if not downscale_allowed:
            target_w = max(target_w, orig_w)
            target_h = max(target_h, orig_h)

        # Apply min/max constraints
        if min_size:
            scale = max(min_size / target_w, min_size / target_h, 1.0)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        if max_size:
            scale = min(max_size / target_w, max_size / target_h, 1.0)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)

        # Apply multiple_of constraint
        if multiple_of > 0:
            target_w = self._apply_multiple_of(target_w, multiple_of)
            target_h = self._apply_multiple_of(target_h, multiple_of)

        # Ensure minimum size
        target_w = max(1, target_w)
        target_h = max(1, target_h)

        # Perform resize (if not already done for fit_outside)
        if not cropped:
            image = image.resize((target_w, target_h), interpolation)

        # Apply letterbox padding for fit_within if requested
        add_padding = config.get("add_padding", False)
        if add_padding and mode == "fit_within":
            canvas_w = config.get("width", target_w)
            canvas_h = config.get("height", target_h)

            # Apply multiple_of to canvas too
            if multiple_of > 0:
                canvas_w = self._apply_multiple_of(canvas_w, multiple_of)
                canvas_h = self._apply_multiple_of(canvas_h, multiple_of)

            padding_color = self._parse_color(config.get("padding_color", "#000000"))
            padding_position = config.get("padding_position", "center")

            padded = Image.new("RGB", (canvas_w, canvas_h), padding_color)

            if padding_position == "center":
                pad_left = (canvas_w - target_w) // 2
                pad_top = (canvas_h - target_h) // 2
            elif padding_position == "top_left":
                pad_left, pad_top = 0, 0
            elif padding_position == "top_right":
                pad_left = canvas_w - target_w
                pad_top = 0
            elif padding_position == "bottom_left":
                pad_left = 0
                pad_top = canvas_h - target_h
            elif padding_position == "bottom_right":
                pad_left = canvas_w - target_w
                pad_top = canvas_h - target_h

            padded.paste(image, (pad_left, pad_top))
            image = padded

        final_w, final_h = image.size
        duration = (time.time() - start_time) * 1000

        # Build scale info
        scale_info = {
            "scale_x": round(final_w / orig_w, 6),
            "scale_y": round(final_h / orig_h, 6),
            "content_scale_x": round(target_w / orig_w, 6),
            "content_scale_y": round(target_h / orig_h, 6),
            "padding": {
                "left": pad_left,
                "top": pad_top,
                "right": final_w - target_w - pad_left if add_padding else 0,
                "bottom": final_h - target_h - pad_top if add_padding else 0,
            },
            "content_box": {
                "x": pad_left,
                "y": pad_top,
                "width": target_w,
                "height": target_h,
            },
        }

        outputs = {
            "image": f"data:image/jpeg;base64,{image_to_base64(image)}",
            "original_size": {"width": orig_w, "height": orig_h},
            "new_size": {"width": final_w, "height": final_h},
        }

        if config.get("output_scale_info", True):
            outputs["scale_info"] = scale_info

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "resize_ratio": round(final_w / orig_w, 4),
                "mode": mode,
            },
        )


class TileBlock(BaseBlock):
    """
    Tile Block - SAHI-style Sliced Inference

    Splits images into tiles for processing large images with SOTA features:
    - Multiple tiling modes (full, auto, grid, adaptive)
    - Configurable overlap ratio
    - Edge tile handling with padding
    - Full image inclusion for multi-scale inference
    - Max tiles limit for memory management
    """

    block_type = "tile"
    display_name = "Tile"
    description = "Split image into tiles (SAHI-style) for small object detection"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "tiles", "type": "array", "description": "Array of tile images"},
        {"name": "tile_info", "type": "array", "description": "Position and size info for each tile"},
        {"name": "grid_info", "type": "object", "description": "Grid dimensions and overlap info"},
        {"name": "full_image", "type": "image", "description": "Downscaled full image (if enabled)"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Tile size
            "tile_size": {"type": "number", "default": 640, "description": "Tile width and height"},
            "tile_width": {"type": "number", "description": "Tile width (overrides tile_size)"},
            "tile_height": {"type": "number", "description": "Tile height (overrides tile_size)"},
            # Overlap
            "overlap_ratio": {"type": "number", "default": 0.2, "description": "Overlap ratio (0-0.5)"},
            # Tiling mode
            "tiling_mode": {
                "type": "string",
                "enum": ["full", "auto", "grid", "adaptive"],
                "default": "full",
                "description": "Tiling strategy",
            },
            # Grid mode settings
            "grid_rows": {"type": "number", "default": 2, "description": "Number of rows for grid mode"},
            "grid_cols": {"type": "number", "default": 2, "description": "Number of columns for grid mode"},
            # Edge handling
            "min_area_ratio": {"type": "number", "default": 0.1, "description": "Min area ratio for edge tiles (0-0.5)"},
            "pad_edges": {"type": "boolean", "default": True, "description": "Pad edge tiles to full size"},
            "padding_color": {"type": "string", "default": "#000000", "description": "Padding fill color"},
            # Full image inclusion (multi-scale)
            "include_full_image": {"type": "boolean", "default": False, "description": "Include downscaled full image"},
            "full_image_scale": {"type": "number", "default": 1.0, "description": "Scale factor for full image"},
            # Output options
            "output_images": {"type": "boolean", "default": True, "description": "Include tile images in output"},
            "max_tiles": {"type": "number", "default": 0, "description": "Max tiles limit (0 = no limit)"},
        },
    }

    def _parse_color(self, color_str: str) -> tuple:
        """Parse hex color string to RGB tuple."""
        try:
            color = color_str.lstrip("#")
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Split image into tiles with SAHI-style options."""
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
        overlap_ratio = config.get("overlap_ratio", 0.2)
        tiling_mode = config.get("tiling_mode", "full")
        min_area_ratio = config.get("min_area_ratio", 0.1)
        pad_edges = config.get("pad_edges", True)
        padding_color = self._parse_color(config.get("padding_color", "#000000"))
        output_images = config.get("output_images", True)
        max_tiles = config.get("max_tiles", 0)

        tiles = []
        tile_info = []

        # Auto mode: skip tiling if image is smaller than tile
        if tiling_mode == "auto" and img_w <= tile_w and img_h <= tile_h:
            # Just return the original image as a single "tile"
            if output_images:
                tiles.append(f"data:image/jpeg;base64,{image_to_base64(image)}")
            tile_info.append({
                "index": 0,
                "row": 0,
                "col": 0,
                "x": 0,
                "y": 0,
                "width": img_w,
                "height": img_h,
                "padded": False,
                "is_full_image": True,
            })

        # Grid mode: fixed number of rows/cols
        elif tiling_mode == "grid":
            grid_rows = max(1, config.get("grid_rows", 2))
            grid_cols = max(1, config.get("grid_cols", 2))

            # Calculate tile size based on grid
            tile_w = img_w // grid_cols
            tile_h = img_h // grid_rows

            # Add overlap
            overlap_w = int(tile_w * overlap_ratio)
            overlap_h = int(tile_h * overlap_ratio)

            for row in range(grid_rows):
                for col in range(grid_cols):
                    if max_tiles > 0 and len(tile_info) >= max_tiles:
                        break

                    x1 = max(0, col * tile_w - overlap_w)
                    y1 = max(0, row * tile_h - overlap_h)
                    x2 = min(img_w, (col + 1) * tile_w + overlap_w)
                    y2 = min(img_h, (row + 1) * tile_h + overlap_h)

                    tile = image.crop((x1, y1, x2, y2))

                    if output_images:
                        tiles.append(f"data:image/jpeg;base64,{image_to_base64(tile)}")

                    tile_info.append({
                        "index": len(tile_info),
                        "row": row,
                        "col": col,
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "padded": False,
                    })

        # Adaptive mode: adjust tile size based on image dimensions
        elif tiling_mode == "adaptive":
            # Calculate number of tiles that fit
            n_cols = max(1, int(np.ceil(img_w / (tile_w * (1 - overlap_ratio)))))
            n_rows = max(1, int(np.ceil(img_h / (tile_h * (1 - overlap_ratio)))))

            # Adjust step to cover image evenly
            step_x = (img_w - tile_w) // max(1, n_cols - 1) if n_cols > 1 else 0
            step_y = (img_h - tile_h) // max(1, n_rows - 1) if n_rows > 1 else 0

            for row in range(n_rows):
                for col in range(n_cols):
                    if max_tiles > 0 and len(tile_info) >= max_tiles:
                        break

                    x1 = min(col * step_x, img_w - tile_w) if step_x > 0 else 0
                    y1 = min(row * step_y, img_h - tile_h) if step_y > 0 else 0
                    x2 = min(x1 + tile_w, img_w)
                    y2 = min(y1 + tile_h, img_h)

                    tile = image.crop((x1, y1, x2, y2))
                    is_padded = False

                    if pad_edges and (x2 - x1 < tile_w or y2 - y1 < tile_h):
                        padded = Image.new("RGB", (tile_w, tile_h), padding_color)
                        padded.paste(tile, (0, 0))
                        tile = padded
                        is_padded = True

                    if output_images:
                        tiles.append(f"data:image/jpeg;base64,{image_to_base64(tile)}")

                    tile_info.append({
                        "index": len(tile_info),
                        "row": row,
                        "col": col,
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "padded": is_padded,
                    })

        # Full mode: standard overlapping grid
        else:
            step_x = max(1, int(tile_w * (1 - overlap_ratio)))
            step_y = max(1, int(tile_h * (1 - overlap_ratio)))

            y = 0
            row_idx = 0
            while y < img_h:
                x = 0
                col_idx = 0
                while x < img_w:
                    if max_tiles > 0 and len(tile_info) >= max_tiles:
                        break

                    x1, y1 = x, y
                    x2 = min(x + tile_w, img_w)
                    y2 = min(y + tile_h, img_h)

                    actual_w = x2 - x1
                    actual_h = y2 - y1
                    area_ratio = (actual_w * actual_h) / (tile_w * tile_h)

                    if area_ratio >= min_area_ratio:
                        tile = image.crop((x1, y1, x2, y2))
                        is_padded = False

                        # Pad edge tiles if needed
                        if pad_edges and (actual_w < tile_w or actual_h < tile_h):
                            padded = Image.new("RGB", (tile_w, tile_h), padding_color)
                            padded.paste(tile, (0, 0))
                            tile = padded
                            is_padded = True

                        if output_images:
                            tiles.append(f"data:image/jpeg;base64,{image_to_base64(tile)}")

                        tile_info.append({
                            "index": len(tile_info),
                            "row": row_idx,
                            "col": col_idx,
                            "x": x1,
                            "y": y1,
                            "width": actual_w,
                            "height": actual_h,
                            "padded": is_padded,
                        })

                    x += step_x
                    col_idx += 1

                if max_tiles > 0 and len(tile_info) >= max_tiles:
                    break

                y += step_y
                row_idx += 1

        # Include full image for multi-scale inference
        full_image_output = None
        include_full = config.get("include_full_image", False)
        if include_full:
            full_scale = config.get("full_image_scale", 1.0)
            if full_scale < 1.0:
                scaled_w = int(img_w * full_scale)
                scaled_h = int(img_h * full_scale)
                full_img = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            else:
                full_img = image

            full_image_output = f"data:image/jpeg;base64,{image_to_base64(full_img)}"

        duration = (time.time() - start_time) * 1000

        # Calculate grid info
        max_row = max((t["row"] for t in tile_info), default=0) + 1 if tile_info else 0
        max_col = max((t["col"] for t in tile_info), default=0) + 1 if tile_info else 0

        outputs = {
            "tiles": tiles,
            "tile_info": tile_info,
            "grid_info": {
                "image_size": {"width": img_w, "height": img_h},
                "tile_size": {"width": tile_w, "height": tile_h},
                "overlap_ratio": overlap_ratio,
                "tiling_mode": tiling_mode,
                "rows": max_row,
                "cols": max_col,
                "total_tiles": len(tile_info),
            },
        }

        if full_image_output:
            outputs["full_image"] = full_image_output

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "tile_count": len(tile_info),
                "tiling_mode": tiling_mode,
            },
        )


class StitchBlock(BaseBlock):
    """
    Stitch Block - SAHI-style Detection Merging

    Merges tiled detection results with SOTA NMS options:
    - Multiple merge modes (NMS, Soft NMS, class-wise NMS, union, greedy)
    - IoU and score threshold filtering
    - Coordinate transformation to original image space
    - Edge box handling (clip, remove, merge)
    - Optional image reconstruction
    """

    block_type = "stitch"
    display_name = "Stitch"
    description = "Merge tiled detection results with NMS"

    input_ports = [
        {"name": "detections", "type": "array", "required": True, "description": "Array of detections from each tile"},
        {"name": "tile_info", "type": "array", "required": True, "description": "Position info from Tile block"},
        {"name": "grid_info", "type": "object", "required": True, "description": "Grid info from Tile block"},
        {"name": "tiles", "type": "array", "required": False, "description": "Tile images (for reconstruction)"},
    ]
    output_ports = [
        {"name": "detections", "type": "array", "description": "Merged detections in original image coordinates"},
        {"name": "image", "type": "image", "description": "Reconstructed image (if enabled)"},
        {"name": "merged", "type": "image", "description": "Alias for image - stitched/merged image"},
        {"name": "stats", "type": "object", "description": "Merge statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Merge mode
            "merge_mode": {
                "type": "string",
                "enum": ["nms", "soft_nms", "nms_class", "union", "greedy"],
                "default": "nms",
                "description": "Detection merge strategy",
            },
            # Thresholds
            "iou_threshold": {"type": "number", "default": 0.5, "description": "IoU threshold for NMS"},
            "score_threshold": {"type": "number", "default": 0.3, "description": "Filter low-confidence detections"},
            # Coordinate transform
            "transform_coords": {"type": "boolean", "default": True, "description": "Transform to original image coords"},
            # Edge handling
            "edge_handling": {
                "type": "string",
                "enum": ["keep", "clip", "remove", "merge"],
                "default": "keep",
                "description": "How to handle boxes touching tile edges",
            },
            # Limits
            "max_detections": {"type": "number", "default": 0, "description": "Max detections (0 = no limit)"},
            # Soft NMS
            "soft_nms_sigma": {"type": "number", "default": 0.5, "description": "Sigma for Soft NMS"},
            # Image reconstruction
            "reconstruct_image": {"type": "boolean", "default": False, "description": "Stitch tile images"},
            "blend_mode": {
                "type": "string",
                "enum": ["overwrite", "average", "max"],
                "default": "average",
                "description": "Image blend mode",
            },
            # Output options
            "include_stats": {"type": "boolean", "default": True, "description": "Include merge statistics"},
            "preserve_tile_info": {"type": "boolean", "default": False, "description": "Add source tile to detections"},
        },
    }

    def _calculate_iou(self, box1: dict, box2: dict) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _apply_nms(self, detections: list, iou_threshold: float, class_wise: bool = False) -> list:
        """Apply NMS to detections."""
        if not detections:
            return []

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

        if class_wise:
            # Group by class
            by_class = {}
            for det in sorted_dets:
                cls = det.get("class", det.get("label", "unknown"))
                if cls not in by_class:
                    by_class[cls] = []
                by_class[cls].append(det)

            # Apply NMS per class
            result = []
            for cls_dets in by_class.values():
                result.extend(self._apply_nms(cls_dets, iou_threshold, class_wise=False))
            return result

        keep = []
        suppressed = set()

        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue

            keep.append(det)

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue

                iou = self._calculate_iou(det, sorted_dets[j])
                if iou > iou_threshold:
                    suppressed.add(j)

        return keep

    def _apply_soft_nms(self, detections: list, iou_threshold: float, sigma: float) -> list:
        """Apply Soft NMS (decay scores instead of suppressing)."""
        if not detections:
            return []

        result = []
        remaining = list(detections)

        while remaining:
            # Find highest scoring
            remaining.sort(key=lambda d: d.get("confidence", 0), reverse=True)
            best = remaining.pop(0)
            result.append(best)

            # Decay overlapping scores
            for det in remaining:
                iou = self._calculate_iou(best, det)
                if iou > 0:
                    # Gaussian decay
                    decay = np.exp(-(iou ** 2) / sigma)
                    det["confidence"] = det.get("confidence", 1.0) * decay

        return result

    def _transform_detection(self, det: dict, tile_info: dict) -> dict:
        """Transform detection coordinates to original image space."""
        transformed = det.copy()
        tile_x = tile_info.get("x", 0)
        tile_y = tile_info.get("y", 0)

        transformed["x"] = det.get("x", 0) + tile_x
        transformed["y"] = det.get("y", 0) + tile_y

        return transformed

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Merge tiled detections with NMS."""
        start_time = time.time()

        detections_per_tile = inputs.get("detections", [])
        tile_info = inputs.get("tile_info", [])
        grid_info = inputs.get("grid_info", {})
        tiles_data = inputs.get("tiles", [])

        if not tile_info or not grid_info:
            return BlockResult(
                error="Missing required inputs: tile_info or grid_info",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        merge_mode = config.get("merge_mode", "nms")
        iou_threshold = config.get("iou_threshold", 0.5)
        score_threshold = config.get("score_threshold", 0.3)
        transform_coords = config.get("transform_coords", True)
        edge_handling = config.get("edge_handling", "keep")
        max_detections = config.get("max_detections", 0)
        soft_nms_sigma = config.get("soft_nms_sigma", 0.5)
        preserve_tile_info = config.get("preserve_tile_info", False)

        img_size = grid_info.get("image_size", {})
        img_w = img_size.get("width", 0)
        img_h = img_size.get("height", 0)
        tile_size = grid_info.get("tile_size", {})
        tile_w = tile_size.get("width", 640)
        tile_h = tile_size.get("height", 640)

        # Collect all detections with coordinate transformation
        all_detections = []
        detections_before = 0

        for i, info in enumerate(tile_info):
            tile_dets = detections_per_tile[i] if i < len(detections_per_tile) else []

            if isinstance(tile_dets, dict):
                tile_dets = tile_dets.get("predictions", tile_dets.get("detections", []))

            if not isinstance(tile_dets, list):
                continue

            detections_before += len(tile_dets)

            for det in tile_dets:
                # Filter by score
                if det.get("confidence", 1.0) < score_threshold:
                    continue

                # Transform coordinates
                if transform_coords:
                    det = self._transform_detection(det, info)

                # Edge handling
                if edge_handling != "keep" and img_w > 0 and img_h > 0:
                    x, y = det.get("x", 0), det.get("y", 0)
                    w, h = det.get("width", 0), det.get("height", 0)

                    touches_edge = (x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h)

                    if touches_edge:
                        if edge_handling == "remove":
                            continue
                        elif edge_handling == "clip":
                            det["x"] = max(0, x)
                            det["y"] = max(0, y)
                            det["width"] = min(w, img_w - det["x"])
                            det["height"] = min(h, img_h - det["y"])

                # Add tile info
                if preserve_tile_info:
                    det["source_tile"] = i
                    det["tile_row"] = info.get("row", 0)
                    det["tile_col"] = info.get("col", 0)

                all_detections.append(det)

        # Apply merge strategy
        if merge_mode == "nms":
            merged = self._apply_nms(all_detections, iou_threshold)
        elif merge_mode == "nms_class":
            merged = self._apply_nms(all_detections, iou_threshold, class_wise=True)
        elif merge_mode == "soft_nms":
            merged = self._apply_soft_nms(all_detections, iou_threshold, soft_nms_sigma)
            # Filter by score again after soft NMS
            merged = [d for d in merged if d.get("confidence", 0) >= score_threshold]
        elif merge_mode == "greedy":
            merged = self._apply_nms(all_detections, iou_threshold * 0.7)  # More aggressive
        else:  # union
            merged = all_detections

        # Apply max detections limit
        if max_detections > 0:
            merged.sort(key=lambda d: d.get("confidence", 0), reverse=True)
            merged = merged[:max_detections]

        # Optional image reconstruction
        output_image = None
        reconstruct = config.get("reconstruct_image", False)
        if reconstruct and tiles_data and img_w > 0 and img_h > 0:
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

            weight_map = np.maximum(weight_map, 1e-8)
            result = result / weight_map[:, :, np.newaxis]
            result = np.clip(result, 0, 255).astype(np.uint8)
            output_image = Image.fromarray(result)

        duration = (time.time() - start_time) * 1000

        outputs = {
            "detections": merged,
        }

        if output_image:
            image_data = f"data:image/jpeg;base64,{image_to_base64(output_image)}"
            outputs["image"] = image_data
            outputs["merged"] = image_data  # Convenience alias

        if config.get("include_stats", True):
            outputs["stats"] = {
                "detections_before_merge": detections_before,
                "detections_after_merge": len(merged),
                "suppressed": detections_before - len(merged),
                "merge_mode": merge_mode,
                "tiles_processed": len(tile_info),
            }

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "detections_merged": len(merged),
                "suppression_rate": round(1 - len(merged) / max(detections_before, 1), 3),
            },
        )


class RotateFlipBlock(BaseBlock):
    """
    Rotate/Flip Block

    Rotates and/or flips images with SOTA options:
    - Fixed rotations (90, 180, 270) and custom angles
    - Auto-rotation from EXIF orientation
    - Horizontal/vertical flip
    - Bounding box transformation support
    - Configurable interpolation and rotation center
    - Transform matrix output for coordinate mapping
    """

    block_type = "rotate_flip"
    display_name = "Rotate/Flip"
    description = "Rotate and/or flip images with bbox support"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "boxes", "type": "array", "required": False, "description": "Bounding boxes to transform"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Transformed image"},
        {"name": "boxes", "type": "array", "description": "Transformed bounding boxes"},
        {"name": "transform_matrix", "type": "object", "description": "Transformation matrix for coordinate mapping"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Rotation
            "rotation": {
                "type": "string",
                "enum": ["none", "90", "180", "270", "auto_exif", "custom"],
                "default": "none",
                "description": "Rotation mode",
            },
            "custom_angle": {"type": "number", "default": 0, "description": "Custom rotation angle in degrees"},
            # Flip
            "flip_horizontal": {"type": "boolean", "default": False, "description": "Mirror left-right"},
            "flip_vertical": {"type": "boolean", "default": False, "description": "Mirror top-bottom"},
            # Bbox transform
            "transform_boxes": {"type": "boolean", "default": True, "description": "Apply transforms to bboxes"},
            # Advanced (for custom rotation)
            "expand_canvas": {"type": "boolean", "default": True, "description": "Expand canvas for rotation"},
            "background_color": {"type": "string", "default": "#000000", "description": "Background fill color"},
            "interpolation": {
                "type": "string",
                "enum": ["nearest", "bilinear", "bicubic"],
                "default": "bilinear",
                "description": "Interpolation method",
            },
            "center": {
                "type": "string",
                "enum": ["center", "top_left", "custom"],
                "default": "center",
                "description": "Rotation center point",
            },
            "center_x": {"type": "number", "description": "Custom center X coordinate"},
            "center_y": {"type": "number", "description": "Custom center Y coordinate"},
            # Output
            "output_matrix": {"type": "boolean", "default": False, "description": "Include transformation matrix"},
        },
    }

    INTERPOLATION_MAP = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }

    def _parse_color(self, color_str: str) -> tuple:
        """Parse hex color string to RGB tuple."""
        try:
            color = color_str.lstrip("#")
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

    def _transform_box_90(self, box: dict, img_w: int, img_h: int) -> dict:
        """Transform box for 90-degree clockwise rotation."""
        x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("width", 0), box.get("height", 0)
        new_box = box.copy()
        new_box["x"] = img_h - y - h
        new_box["y"] = x
        new_box["width"] = h
        new_box["height"] = w
        return new_box

    def _transform_box_180(self, box: dict, img_w: int, img_h: int) -> dict:
        """Transform box for 180-degree rotation."""
        x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("width", 0), box.get("height", 0)
        new_box = box.copy()
        new_box["x"] = img_w - x - w
        new_box["y"] = img_h - y - h
        return new_box

    def _transform_box_270(self, box: dict, img_w: int, img_h: int) -> dict:
        """Transform box for 270-degree (90 CCW) rotation."""
        x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("width", 0), box.get("height", 0)
        new_box = box.copy()
        new_box["x"] = y
        new_box["y"] = img_w - x - w
        new_box["width"] = h
        new_box["height"] = w
        return new_box

    def _transform_box_flip_h(self, box: dict, img_w: int) -> dict:
        """Transform box for horizontal flip."""
        x, w = box.get("x", 0), box.get("width", 0)
        new_box = box.copy()
        new_box["x"] = img_w - x - w
        return new_box

    def _transform_box_flip_v(self, box: dict, img_h: int) -> dict:
        """Transform box for vertical flip."""
        y, h = box.get("y", 0), box.get("height", 0)
        new_box = box.copy()
        new_box["y"] = img_h - y - h
        return new_box

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Rotate and/or flip image with bbox support."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        boxes = inputs.get("boxes", [])
        if isinstance(boxes, dict):
            boxes = boxes.get("predictions", boxes.get("detections", []))
        if not isinstance(boxes, list):
            boxes = []

        orig_w, orig_h = image.size
        rotation = config.get("rotation", "none")
        custom_angle = config.get("custom_angle", 0)
        flip_h = config.get("flip_horizontal", False)
        flip_v = config.get("flip_vertical", False)
        transform_boxes = config.get("transform_boxes", True)
        expand = config.get("expand_canvas", True)
        fill_color = self._parse_color(config.get("background_color", "#000000"))
        interpolation = self.INTERPOLATION_MAP.get(config.get("interpolation", "bilinear"), Image.Resampling.BILINEAR)

        transforms_applied = []
        current_w, current_h = orig_w, orig_h
        transformed_boxes = [b.copy() for b in boxes] if boxes else []

        # Apply rotation
        if rotation == "90":
            image = image.transpose(Image.Transpose.ROTATE_90)
            transforms_applied.append("rotate_90")
            if transform_boxes:
                transformed_boxes = [self._transform_box_90(b, current_w, current_h) for b in transformed_boxes]
            current_w, current_h = current_h, current_w

        elif rotation == "180":
            image = image.transpose(Image.Transpose.ROTATE_180)
            transforms_applied.append("rotate_180")
            if transform_boxes:
                transformed_boxes = [self._transform_box_180(b, current_w, current_h) for b in transformed_boxes]

        elif rotation == "270":
            image = image.transpose(Image.Transpose.ROTATE_270)
            transforms_applied.append("rotate_270")
            if transform_boxes:
                transformed_boxes = [self._transform_box_270(b, current_w, current_h) for b in transformed_boxes]
            current_w, current_h = current_h, current_w

        elif rotation == "custom" and custom_angle != 0:
            # Get rotation center
            center_mode = config.get("center", "center")
            if center_mode == "top_left":
                center = (0, 0)
            elif center_mode == "custom":
                center = (config.get("center_x", current_w // 2), config.get("center_y", current_h // 2))
            else:
                center = None  # Default to center

            image = image.rotate(-custom_angle, expand=expand, fillcolor=fill_color, resample=interpolation, center=center)
            transforms_applied.append(f"rotate_{custom_angle}")
            current_w, current_h = image.size
            # Note: bbox transform for arbitrary angles is complex - would need affine transform

        elif rotation == "auto_exif":
            # Auto-correct from EXIF orientation
            try:
                exif = image.getexif()
                if exif:
                    orientation = exif.get(274)  # Orientation tag
                    if orientation == 3:
                        image = image.transpose(Image.Transpose.ROTATE_180)
                        transforms_applied.append("auto_rotate_180")
                        if transform_boxes:
                            transformed_boxes = [self._transform_box_180(b, current_w, current_h) for b in transformed_boxes]
                    elif orientation == 6:
                        image = image.transpose(Image.Transpose.ROTATE_270)
                        transforms_applied.append("auto_rotate_270")
                        if transform_boxes:
                            transformed_boxes = [self._transform_box_270(b, current_w, current_h) for b in transformed_boxes]
                        current_w, current_h = current_h, current_w
                    elif orientation == 8:
                        image = image.transpose(Image.Transpose.ROTATE_90)
                        transforms_applied.append("auto_rotate_90")
                        if transform_boxes:
                            transformed_boxes = [self._transform_box_90(b, current_w, current_h) for b in transformed_boxes]
                        current_w, current_h = current_h, current_w
            except Exception:
                pass

        # Apply flips
        if flip_h:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            transforms_applied.append("flip_horizontal")
            if transform_boxes:
                transformed_boxes = [self._transform_box_flip_h(b, current_w) for b in transformed_boxes]

        if flip_v:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            transforms_applied.append("flip_vertical")
            if transform_boxes:
                transformed_boxes = [self._transform_box_flip_v(b, current_h) for b in transformed_boxes]

        final_w, final_h = image.size
        duration = (time.time() - start_time) * 1000

        outputs = {
            "image": f"data:image/jpeg;base64,{image_to_base64(image)}",
            "boxes": transformed_boxes,
        }

        # Build transform matrix for coordinate mapping
        if config.get("output_matrix", False):
            outputs["transform_matrix"] = {
                "transforms": transforms_applied,
                "rotation": rotation if rotation != "custom" else custom_angle,
                "flip_h": flip_h,
                "flip_v": flip_v,
                "original_size": {"width": orig_w, "height": orig_h},
                "final_size": {"width": final_w, "height": final_h},
            }

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "transforms_applied": len(transforms_applied),
                "boxes_transformed": len(transformed_boxes),
            },
        )


class NormalizeBlock(BaseBlock):
    """
    Normalize Block - SOTA Model Preprocessing

    Normalizes image pixel values with SOTA presets:
    - ImageNet (most models: ResNet, ViT, etc.)
    - CLIP / SigLIP (OpenAI CLIP models)
    - DINOv2 (Meta's self-supervised models)
    - 0-1 range (divide by 255)
    - -1 to 1 range (common for GANs)
    - Custom mean/std
    - CLAHE / Histogram equalization
    """

    block_type = "normalize"
    display_name = "Normalize"
    description = "Normalize image for model input"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Normalized image (visual preview)"},
        {"name": "tensor", "type": "object", "description": "Normalized tensor data (for models)"},
        {"name": "stats", "type": "object", "description": "Normalization statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Preset selection (matches frontend)
            "preset": {
                "type": "string",
                "enum": ["imagenet", "clip", "dinov2", "0_1", "-1_1", "none", "custom", "clahe", "histogram"],
                "default": "imagenet",
                "description": "Normalization preset",
            },
            # Custom mean/std (per channel RGB)
            "mean_r": {"type": "number", "default": 0.485, "description": "Custom mean for R channel"},
            "mean_g": {"type": "number", "default": 0.456, "description": "Custom mean for G channel"},
            "mean_b": {"type": "number", "default": 0.406, "description": "Custom mean for B channel"},
            "std_r": {"type": "number", "default": 0.229, "description": "Custom std for R channel"},
            "std_g": {"type": "number", "default": 0.224, "description": "Custom std for G channel"},
            "std_b": {"type": "number", "default": 0.225, "description": "Custom std for B channel"},
            # Channel order
            "channel_order": {
                "type": "string",
                "enum": ["RGB", "BGR"],
                "default": "RGB",
                "description": "Output channel order",
            },
            # CLAHE options
            "clahe_clip_limit": {"type": "number", "default": 2.0, "description": "CLAHE clip limit"},
            "clahe_grid_size": {"type": "number", "default": 8, "description": "CLAHE grid size"},
            # Output options
            "output_tensor": {"type": "boolean", "default": False, "description": "Output normalized tensor data"},
            "output_format": {
                "type": "string",
                "enum": ["CHW", "HWC"],
                "default": "CHW",
                "description": "Tensor output format",
            },
        },
    }

    # Preset normalization values
    PRESETS = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "description": "ImageNet (most models)",
        },
        "clip": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "description": "CLIP / SigLIP",
        },
        "dinov2": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "description": "DINOv2 (same as ImageNet)",
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Normalize image with SOTA presets."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        preset = config.get("preset", "imagenet")
        channel_order = config.get("channel_order", "RGB")
        output_tensor = config.get("output_tensor", False)
        output_format = config.get("output_format", "CHW")

        img_array = np.array(image).astype(np.float32)
        original_range = [float(img_array.min()), float(img_array.max())]

        # Apply channel order conversion if needed
        if channel_order == "BGR":
            img_array = img_array[:, :, ::-1]

        normalized_array = None
        stats = {"preset": preset, "original_range": original_range}

        if preset == "none":
            # Keep original values (0-255)
            normalized_array = img_array.copy()
            stats["range"] = [0, 255]

        elif preset == "0_1":
            # Scale to 0-1
            normalized_array = img_array / 255.0
            stats["range"] = [0.0, 1.0]

        elif preset == "-1_1":
            # Scale to -1 to 1
            normalized_array = (img_array / 127.5) - 1.0
            stats["range"] = [-1.0, 1.0]

        elif preset in self.PRESETS:
            # Apply preset mean/std normalization
            preset_config = self.PRESETS[preset]
            mean = preset_config["mean"]
            std = preset_config["std"]

            normalized_array = img_array / 255.0
            for c in range(3):
                normalized_array[:, :, c] = (normalized_array[:, :, c] - mean[c]) / std[c]

            stats["mean"] = mean
            stats["std"] = std
            stats["description"] = preset_config["description"]

        elif preset == "custom":
            # Use custom mean/std values
            mean = [
                config.get("mean_r", 0.485),
                config.get("mean_g", 0.456),
                config.get("mean_b", 0.406),
            ]
            std = [
                config.get("std_r", 0.229),
                config.get("std_g", 0.224),
                config.get("std_b", 0.225),
            ]

            normalized_array = img_array / 255.0
            for c in range(3):
                if std[c] > 0:
                    normalized_array[:, :, c] = (normalized_array[:, :, c] - mean[c]) / std[c]

            stats["mean"] = mean
            stats["std"] = std

        elif preset == "clahe":
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            try:
                import cv2
                clip_limit = config.get("clahe_clip_limit", 2.0)
                grid_size = int(config.get("clahe_grid_size", 8))

                lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                normalized_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
                stats["clahe_clip_limit"] = clip_limit
                stats["clahe_grid_size"] = grid_size
            except ImportError:
                # Fallback to histogram equalization
                normalized_array = self._histogram_equalize(img_array)
                stats["fallback"] = "histogram"

        elif preset == "histogram":
            # Simple histogram equalization per channel
            normalized_array = self._histogram_equalize(img_array)

        else:
            normalized_array = img_array.copy()

        # Create visual preview image (rescaled to 0-255 for display)
        if preset in ["imagenet", "clip", "dinov2", "custom"]:
            # Rescale normalized values back to viewable range
            preview_array = np.clip(normalized_array * 64 + 128, 0, 255).astype(np.uint8)
        elif preset in ["-1_1"]:
            preview_array = np.clip((normalized_array + 1) * 127.5, 0, 255).astype(np.uint8)
        elif preset == "0_1":
            preview_array = np.clip(normalized_array * 255, 0, 255).astype(np.uint8)
        else:
            preview_array = np.clip(normalized_array, 0, 255).astype(np.uint8)

        output_image = Image.fromarray(preview_array)

        duration = (time.time() - start_time) * 1000

        outputs = {
            "image": f"data:image/jpeg;base64,{image_to_base64(output_image)}",
            "stats": stats,
        }

        # Optionally output the normalized tensor
        if output_tensor:
            if output_format == "CHW":
                tensor_data = np.transpose(normalized_array, (2, 0, 1))  # HWC -> CHW
            else:
                tensor_data = normalized_array

            outputs["tensor"] = {
                "shape": list(tensor_data.shape),
                "dtype": "float32",
                "format": output_format,
                "data": tensor_data.tolist(),  # For JSON serialization
            }

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={"preset": preset},
        )

    def _histogram_equalize(self, img_array: np.ndarray) -> np.ndarray:
        """Apply histogram equalization per channel."""
        result = img_array.copy()
        for c in range(3):
            hist, bins = np.histogram(result[:, :, c].flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
            result[:, :, c] = cdf[result[:, :, c].astype(np.uint8)]
        return result


class SmoothingBlock(BaseBlock):
    """
    Smoothing Block - Classical CV Preprocessing

    Applies various smoothing/blur operations for preprocessing:
    - Gaussian blur (noise reduction, edge detection prep)
    - Median blur (salt-pepper noise removal)
    - Bilateral filter (edge-preserving smoothing)
    - Box blur (simple averaging)
    - Motion blur (augmentation/simulation)

    Unlike BlurRegionBlock (privacy-focused region blurring),
    this block applies smoothing to the entire image for preprocessing.
    """

    block_type = "smoothing"
    display_name = "Smoothing"
    description = "Apply smoothing/blur filters for preprocessing"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Smoothed image"},
        {"name": "stats", "type": "object", "description": "Smoothing statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Smoothing type
            "smoothing_type": {
                "type": "string",
                "enum": ["gaussian", "median", "bilateral", "box", "motion"],
                "default": "gaussian",
                "description": "Type of smoothing filter",
            },
            # Common parameters
            "kernel_size": {
                "type": "number",
                "default": 5,
                "minimum": 1,
                "description": "Kernel size (must be odd for most filters)",
            },
            # Gaussian parameters
            "sigma_x": {
                "type": "number",
                "default": 0,
                "description": "Gaussian sigma X (0 = auto from kernel size)",
            },
            "sigma_y": {
                "type": "number",
                "default": 0,
                "description": "Gaussian sigma Y (0 = same as sigma_x)",
            },
            # Bilateral parameters
            "d": {
                "type": "number",
                "default": 9,
                "description": "Bilateral filter diameter (-1 = auto from sigmaSpace)",
            },
            "sigma_color": {
                "type": "number",
                "default": 75,
                "description": "Bilateral color sigma (larger = more colors mixed)",
            },
            "sigma_space": {
                "type": "number",
                "default": 75,
                "description": "Bilateral space sigma (larger = farther pixels influence)",
            },
            # Motion blur parameters
            "motion_angle": {
                "type": "number",
                "default": 0,
                "minimum": 0,
                "maximum": 360,
                "description": "Motion blur angle in degrees (0 = horizontal)",
            },
            "motion_length": {
                "type": "number",
                "default": 15,
                "minimum": 1,
                "description": "Motion blur length in pixels",
            },
            # ROI (optional region of interest)
            "apply_roi": {
                "type": "boolean",
                "default": False,
                "description": "Apply smoothing only to a region",
            },
            "roi_x": {"type": "number", "default": 0, "description": "ROI X (normalized 0-1)"},
            "roi_y": {"type": "number", "default": 0, "description": "ROI Y (normalized 0-1)"},
            "roi_width": {"type": "number", "default": 1, "description": "ROI width (normalized 0-1)"},
            "roi_height": {"type": "number", "default": 1, "description": "ROI height (normalized 0-1)"},
            # Iterations
            "iterations": {
                "type": "number",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
                "description": "Number of times to apply the filter",
            },
        },
    }

    def _ensure_odd(self, value: int) -> int:
        """Ensure kernel size is odd."""
        value = max(1, int(value))
        if value % 2 == 0:
            value += 1
        return value

    def _create_motion_kernel(self, length: int, angle: float) -> np.ndarray:
        """Create a motion blur kernel."""
        kernel = np.zeros((length, length))
        center = length // 2

        # Create line at given angle
        angle_rad = np.deg2rad(angle)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)

        for i in range(length):
            offset = i - center
            x = int(center + offset * cos_val)
            y = int(center + offset * sin_val)
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1

        # Normalize
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel
        return kernel

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Apply smoothing filter to image."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        smoothing_type = config.get("smoothing_type", "gaussian")
        kernel_size = self._ensure_odd(config.get("kernel_size", 5))
        iterations = max(1, min(10, config.get("iterations", 1)))
        apply_roi = config.get("apply_roi", False)

        img_array = np.array(image)
        original_shape = img_array.shape
        img_width, img_height = image.size

        # Handle ROI
        if apply_roi:
            roi_x = int(config.get("roi_x", 0) * img_width)
            roi_y = int(config.get("roi_y", 0) * img_height)
            roi_w = int(config.get("roi_width", 1) * img_width)
            roi_h = int(config.get("roi_height", 1) * img_height)

            # Clamp to valid bounds
            roi_x = max(0, min(roi_x, img_width - 1))
            roi_y = max(0, min(roi_y, img_height - 1))
            roi_w = max(1, min(roi_w, img_width - roi_x))
            roi_h = max(1, min(roi_h, img_height - roi_y))

            # Extract ROI
            roi_region = img_array[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
        else:
            roi_region = img_array
            roi_x, roi_y = 0, 0

        stats = {
            "smoothing_type": smoothing_type,
            "kernel_size": kernel_size,
            "iterations": iterations,
        }

        try:
            import cv2

            result = roi_region.copy()

            for _ in range(iterations):
                if smoothing_type == "gaussian":
                    sigma_x = config.get("sigma_x", 0)
                    sigma_y = config.get("sigma_y", 0)
                    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
                    stats["sigma_x"] = sigma_x if sigma_x > 0 else f"auto ({0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8:.2f})"
                    stats["sigma_y"] = sigma_y if sigma_y > 0 else "same as sigma_x"

                elif smoothing_type == "median":
                    result = cv2.medianBlur(result, kernel_size)

                elif smoothing_type == "bilateral":
                    d = int(config.get("d", 9))
                    sigma_color = config.get("sigma_color", 75)
                    sigma_space = config.get("sigma_space", 75)
                    result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
                    stats["d"] = d
                    stats["sigma_color"] = sigma_color
                    stats["sigma_space"] = sigma_space

                elif smoothing_type == "box":
                    result = cv2.blur(result, (kernel_size, kernel_size))

                elif smoothing_type == "motion":
                    motion_angle = config.get("motion_angle", 0)
                    motion_length = max(1, int(config.get("motion_length", 15)))
                    # Ensure odd length
                    if motion_length % 2 == 0:
                        motion_length += 1

                    kernel = self._create_motion_kernel(motion_length, motion_angle)
                    result = cv2.filter2D(result, -1, kernel)
                    stats["motion_angle"] = motion_angle
                    stats["motion_length"] = motion_length

            # Place result back if ROI was used
            if apply_roi:
                img_array[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = result
                output_array = img_array
                stats["roi"] = {
                    "x": roi_x, "y": roi_y,
                    "width": roi_w, "height": roi_h,
                }
            else:
                output_array = result

        except ImportError:
            # Fallback to PIL if OpenCV not available
            result = image.copy()

            for _ in range(iterations):
                if smoothing_type == "gaussian":
                    from PIL import ImageFilter
                    radius = kernel_size // 2
                    result = result.filter(ImageFilter.GaussianBlur(radius=radius))
                    stats["note"] = "Using PIL fallback (OpenCV recommended)"

                elif smoothing_type == "box":
                    from PIL import ImageFilter
                    result = result.filter(ImageFilter.BoxBlur(radius=kernel_size // 2))
                    stats["note"] = "Using PIL fallback (OpenCV recommended)"

                elif smoothing_type == "median":
                    from PIL import ImageFilter
                    result = result.filter(ImageFilter.MedianFilter(size=kernel_size))
                    stats["note"] = "Using PIL fallback (OpenCV recommended)"

                else:
                    # Bilateral and motion blur need OpenCV
                    return BlockResult(
                        error=f"{smoothing_type} blur requires OpenCV. Install with: pip install opencv-python",
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                    )

            output_array = np.array(result)

        output_image = Image.fromarray(output_array.astype(np.uint8))
        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "image": f"data:image/jpeg;base64,{image_to_base64(output_image)}",
                "stats": stats,
            },
            duration_ms=round(duration, 2),
            metrics={
                "smoothing_type": smoothing_type,
                "kernel_size": kernel_size,
                "iterations": iterations,
            },
        )


class DrawMasksBlock(BaseBlock):
    """
    Draw Masks Block - SOTA Segmentation Visualization

    Visualizes segmentation masks on images with advanced rendering options:
    - Multiple color palettes (rainbow, category, custom)
    - Overlay with adjustable opacity
    - Contour drawing
    - Instance vs semantic modes
    - Mask labels and areas
    """

    block_type = "draw_masks"
    display_name = "Draw Masks"
    description = "Visualize segmentation masks on images"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "masks", "type": "array", "required": True, "description": "Segmentation masks from SegmentationBlock"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Image with masks visualized"},
        {"name": "legend", "type": "object", "description": "Color legend for masks"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Rendering mode
            "mode": {
                "type": "string",
                "enum": ["overlay", "contour", "filled", "side_by_side"],
                "default": "overlay",
                "description": "How to render masks",
            },
            # Color palette
            "palette": {
                "type": "string",
                "enum": ["rainbow", "category", "pastel", "viridis", "custom"],
                "default": "rainbow",
                "description": "Color palette for masks",
            },
            "custom_colors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Custom colors as hex codes",
            },
            # Opacity
            "opacity": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Mask overlay opacity",
            },
            # Contours
            "draw_contours": {
                "type": "boolean",
                "default": True,
                "description": "Draw mask contours/edges",
            },
            "contour_thickness": {
                "type": "number",
                "default": 2,
                "description": "Contour line thickness",
            },
            "contour_color": {
                "type": "string",
                "enum": ["same", "white", "black", "custom"],
                "default": "same",
                "description": "Contour color mode",
            },
            # Labels
            "show_labels": {
                "type": "boolean",
                "default": False,
                "description": "Show mask labels/class names",
            },
            "show_areas": {
                "type": "boolean",
                "default": False,
                "description": "Show mask areas",
            },
            "show_iou": {
                "type": "boolean",
                "default": False,
                "description": "Show IoU scores if available",
            },
            "label_position": {
                "type": "string",
                "enum": ["center", "top", "bottom"],
                "default": "center",
            },
            # Filtering
            "min_area": {
                "type": "number",
                "default": 0,
                "description": "Minimum mask area to display",
            },
            "max_masks": {
                "type": "number",
                "default": 100,
                "description": "Maximum number of masks to display",
            },
            "filter_classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only show these classes",
            },
            # Output
            "output_legend": {
                "type": "boolean",
                "default": True,
                "description": "Generate color legend",
            },
        },
    }

    # Color palettes
    PALETTES = {
        "rainbow": [
            (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
            (0, 0, 255), (75, 0, 130), (148, 0, 211), (255, 0, 127),
            (0, 255, 255), (255, 0, 255), (127, 255, 0), (0, 127, 255),
        ],
        "category": [
            (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
        ],
        "pastel": [
            (255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201),
            (186, 225, 255), (219, 186, 255), (255, 186, 243), (186, 255, 255),
        ],
        "viridis": [
            (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
            (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
            (180, 222, 44), (253, 231, 37),
        ],
    }

    def _get_color(self, index: int, palette: str, custom_colors: list | None) -> tuple:
        """Get color for mask index."""
        if palette == "custom" and custom_colors:
            try:
                hex_color = custom_colors[index % len(custom_colors)]
                hex_color = hex_color.lstrip("#")
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            except Exception:
                pass

        colors = self.PALETTES.get(palette, self.PALETTES["rainbow"])
        return colors[index % len(colors)]

    def _get_mask_centroid(self, mask: np.ndarray) -> tuple[int, int]:
        """Get centroid of a binary mask."""
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return 0, 0
        return int(np.mean(x_coords)), int(np.mean(y_coords))

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Draw segmentation masks on image."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        masks_data = inputs.get("masks", [])
        if not masks_data:
            return BlockResult(
                outputs={"image": f"data:image/jpeg;base64,{image_to_base64(image)}", "legend": {}},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"masks_drawn": 0},
            )

        # Config
        mode = config.get("mode", "overlay")
        palette = config.get("palette", "rainbow")
        custom_colors = config.get("custom_colors")
        opacity = config.get("opacity", 0.5)
        draw_contours = config.get("draw_contours", True)
        contour_thickness = int(config.get("contour_thickness", 2))
        contour_color_mode = config.get("contour_color", "same")
        show_labels = config.get("show_labels", False)
        show_areas = config.get("show_areas", False)
        show_iou = config.get("show_iou", False)
        label_position = config.get("label_position", "center")
        min_area = config.get("min_area", 0)
        max_masks = config.get("max_masks", 100)
        filter_classes = config.get("filter_classes")
        output_legend = config.get("output_legend", True)

        img_array = np.array(image).astype(np.float32)
        overlay = img_array.copy()
        legend = {}
        masks_drawn = 0

        try:
            import cv2

            for i, mask_info in enumerate(masks_data[:max_masks]):
                # Get mask array
                if isinstance(mask_info, dict):
                    mask = mask_info.get("mask")
                    class_name = mask_info.get("class_name", f"mask_{i}")
                    iou_score = mask_info.get("predicted_iou", mask_info.get("iou"))
                    area = mask_info.get("area")
                else:
                    mask = mask_info
                    class_name = f"mask_{i}"
                    iou_score = None
                    area = None

                if mask is None:
                    continue

                # Convert mask to numpy array
                if isinstance(mask, list):
                    mask = np.array(mask, dtype=np.uint8)
                elif not isinstance(mask, np.ndarray):
                    continue

                # Ensure binary mask
                mask = (mask > 0).astype(np.uint8)

                # Filter by class
                if filter_classes and class_name not in filter_classes:
                    continue

                # Filter by area
                mask_area = int(mask.sum()) if area is None else area
                if min_area > 0 and mask_area < min_area:
                    continue

                # Get color
                color = self._get_color(i, palette, custom_colors)
                legend[class_name] = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                # Resize mask if needed
                if mask.shape[:2] != (image.height, image.width):
                    mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)

                # Apply overlay
                if mode in ("overlay", "filled"):
                    mask_3d = np.stack([mask] * 3, axis=-1)
                    color_mask = np.zeros_like(img_array)
                    color_mask[:, :] = color
                    overlay = np.where(mask_3d > 0, overlay * (1 - opacity) + color_mask * opacity, overlay)

                # Draw contours
                if draw_contours or mode == "contour":
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contour_color_mode == "same":
                        c_color = color
                    elif contour_color_mode == "white":
                        c_color = (255, 255, 255)
                    elif contour_color_mode == "black":
                        c_color = (0, 0, 0)
                    else:
                        c_color = color

                    cv2.drawContours(overlay.astype(np.uint8), contours, -1, c_color, contour_thickness)
                    overlay = overlay.astype(np.float32)

                # Draw labels
                if show_labels or show_areas or show_iou:
                    cx, cy = self._get_mask_centroid(mask)

                    label_parts = []
                    if show_labels:
                        label_parts.append(class_name)
                    if show_areas:
                        label_parts.append(f"{mask_area:,}px")
                    if show_iou and iou_score is not None:
                        label_parts.append(f"IoU:{iou_score:.2f}")

                    label_text = " | ".join(label_parts)

                    if label_position == "top":
                        y_coords = np.where(mask > 0)[0]
                        cy = int(y_coords.min()) - 10 if len(y_coords) > 0 else cy
                    elif label_position == "bottom":
                        y_coords = np.where(mask > 0)[0]
                        cy = int(y_coords.max()) + 20 if len(y_coords) > 0 else cy

                    overlay_uint8 = overlay.astype(np.uint8)
                    cv2.putText(overlay_uint8, label_text, (cx - 50, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(overlay_uint8, label_text, (cx - 50, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    overlay = overlay_uint8.astype(np.float32)

                masks_drawn += 1

        except ImportError:
            # PIL-only fallback (limited functionality)
            from PIL import ImageDraw

            result = image.copy()
            draw = ImageDraw.Draw(result, "RGBA")

            for i, mask_info in enumerate(masks_data[:max_masks]):
                if isinstance(mask_info, dict):
                    mask = mask_info.get("mask")
                    class_name = mask_info.get("class_name", f"mask_{i}")
                else:
                    mask = mask_info
                    class_name = f"mask_{i}"

                if mask is None:
                    continue

                if isinstance(mask, list):
                    mask = np.array(mask, dtype=np.uint8)

                color = self._get_color(i, palette, custom_colors)
                legend[class_name] = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                # Create colored overlay
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                if mask_img.size != image.size:
                    mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)

                color_layer = Image.new("RGBA", image.size, (*color, int(opacity * 255)))
                result.paste(color_layer, mask=mask_img)
                masks_drawn += 1

            overlay = np.array(result).astype(np.float32)

        output_image = Image.fromarray(overlay.astype(np.uint8))

        # Side by side mode
        if mode == "side_by_side":
            combined = Image.new("RGB", (image.width * 2, image.height))
            combined.paste(image, (0, 0))
            combined.paste(output_image, (image.width, 0))
            output_image = combined

        duration = (time.time() - start_time) * 1000

        outputs = {"image": f"data:image/jpeg;base64,{image_to_base64(output_image)}"}
        if output_legend:
            outputs["legend"] = legend

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={"masks_drawn": masks_drawn, "mode": mode, "palette": palette},
        )


class HeatmapBlock(BaseBlock):
    """
    Heatmap Block - SOTA Attention/Activation Visualization

    Generates heatmap overlays for:
    - Attention maps (from transformers)
    - Activation maps (GradCAM, CAM)
    - Saliency maps
    - Custom 2D data

    SOTA Features:
    - Multiple colormaps (jet, viridis, hot, cool, plasma)
    - Adjustable opacity and blending
    - Thresholding and normalization
    - Multi-layer visualization
    - Colorbar generation
    """

    block_type = "heatmap"
    display_name = "Heatmap"
    description = "Generate heatmap overlays for attention/activation visualization"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "heatmap", "type": "array", "required": True, "description": "2D heatmap data (HxW)"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Image with heatmap overlay"},
        {"name": "result", "type": "image", "description": "Alias for image - heatmap result"},
        {"name": "colorbar", "type": "image", "description": "Colorbar legend"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Colormap
            "colormap": {
                "type": "string",
                "enum": ["jet", "viridis", "hot", "cool", "plasma", "inferno", "magma", "turbo"],
                "default": "jet",
                "description": "Colormap for heatmap",
            },
            # Opacity
            "opacity": {
                "type": "number",
                "default": 0.6,
                "minimum": 0,
                "maximum": 1,
                "description": "Heatmap overlay opacity",
            },
            # Blending mode
            "blend_mode": {
                "type": "string",
                "enum": ["overlay", "multiply", "screen", "add"],
                "default": "overlay",
                "description": "How to blend heatmap with image",
            },
            # Normalization
            "normalize": {
                "type": "boolean",
                "default": True,
                "description": "Normalize heatmap values to 0-1",
            },
            "normalize_method": {
                "type": "string",
                "enum": ["minmax", "percentile", "zscore"],
                "default": "minmax",
            },
            "percentile_low": {
                "type": "number",
                "default": 2,
                "description": "Low percentile for percentile normalization",
            },
            "percentile_high": {
                "type": "number",
                "default": 98,
                "description": "High percentile for percentile normalization",
            },
            # Thresholding
            "threshold": {
                "type": "number",
                "default": 0,
                "minimum": 0,
                "maximum": 1,
                "description": "Hide values below this threshold (after normalization)",
            },
            # Smoothing
            "smooth": {
                "type": "boolean",
                "default": True,
                "description": "Apply Gaussian smoothing to heatmap",
            },
            "smooth_sigma": {
                "type": "number",
                "default": 2,
                "description": "Smoothing sigma",
            },
            # Output
            "output_colorbar": {
                "type": "boolean",
                "default": False,
                "description": "Generate colorbar image",
            },
            "colorbar_position": {
                "type": "string",
                "enum": ["right", "bottom", "separate"],
                "default": "separate",
            },
        },
    }

    # Colormaps (simplified versions for PIL fallback)
    COLORMAPS = {
        "jet": [(0, 0, 128), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0), (128, 0, 0)],
        "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
        "hot": [(0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 128, 0), (255, 255, 0), (255, 255, 255)],
        "cool": [(0, 255, 255), (128, 128, 255), (255, 0, 255)],
        "plasma": [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
        "inferno": [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66), (245, 125, 21), (252, 255, 164)],
        "magma": [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191)],
        "turbo": [(48, 18, 59), (86, 91, 198), (35, 161, 205), (18, 213, 136), (103, 237, 63), (205, 231, 51), (254, 185, 56), (251, 105, 48), (181, 32, 38)],
    }

    def _interpolate_color(self, value: float, colormap: list) -> tuple:
        """Interpolate color from colormap."""
        n = len(colormap) - 1
        idx = value * n
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, n)
        t = idx - lower_idx

        c1 = colormap[lower_idx]
        c2 = colormap[upper_idx]

        return tuple(int(c1[i] + t * (c2[i] - c1[i])) for i in range(3))

    def _apply_colormap(self, heatmap: np.ndarray, colormap_name: str) -> np.ndarray:
        """Apply colormap to normalized heatmap."""
        try:
            import cv2
            # Use OpenCV colormaps
            cmap_dict = {
                "jet": cv2.COLORMAP_JET,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "hot": cv2.COLORMAP_HOT,
                "cool": cv2.COLORMAP_COOL,
                "plasma": cv2.COLORMAP_PLASMA,
                "inferno": cv2.COLORMAP_INFERNO,
                "magma": cv2.COLORMAP_MAGMA,
                "turbo": cv2.COLORMAP_TURBO,
            }
            cmap = cmap_dict.get(colormap_name, cv2.COLORMAP_JET)
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            colored = cv2.applyColorMap(heatmap_uint8, cmap)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        except ImportError:
            # PIL fallback
            colormap = self.COLORMAPS.get(colormap_name, self.COLORMAPS["jet"])
            h, w = heatmap.shape
            result = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    result[i, j] = self._interpolate_color(heatmap[i, j], colormap)
            return result

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Generate heatmap overlay."""
        start_time = time.time()

        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        heatmap_data = inputs.get("heatmap")
        if heatmap_data is None:
            return BlockResult(
                error="No heatmap data provided",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Convert to numpy
        if isinstance(heatmap_data, list):
            heatmap = np.array(heatmap_data, dtype=np.float32)
        else:
            heatmap = np.array(heatmap_data, dtype=np.float32)

        # Ensure 2D
        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()
        if heatmap.ndim != 2:
            return BlockResult(
                error=f"Heatmap must be 2D, got shape {heatmap.shape}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Config
        colormap = config.get("colormap", "jet")
        opacity = config.get("opacity", 0.6)
        blend_mode = config.get("blend_mode", "overlay")
        normalize = config.get("normalize", True)
        normalize_method = config.get("normalize_method", "minmax")
        threshold = config.get("threshold", 0)
        smooth = config.get("smooth", True)
        smooth_sigma = config.get("smooth_sigma", 2)
        output_colorbar = config.get("output_colorbar", False)

        # Resize heatmap to image size
        try:
            import cv2
            heatmap = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            heatmap_img = Image.fromarray(heatmap)
            heatmap_img = heatmap_img.resize((image.width, image.height), Image.Resampling.BILINEAR)
            heatmap = np.array(heatmap_img)

        # Smooth
        if smooth:
            try:
                import cv2
                ksize = int(smooth_sigma * 4) | 1  # Ensure odd
                heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), smooth_sigma)
            except ImportError:
                pass

        # Normalize
        if normalize:
            if normalize_method == "minmax":
                hmin, hmax = heatmap.min(), heatmap.max()
                if hmax > hmin:
                    heatmap = (heatmap - hmin) / (hmax - hmin)
                else:
                    heatmap = np.zeros_like(heatmap)
            elif normalize_method == "percentile":
                plow = config.get("percentile_low", 2)
                phigh = config.get("percentile_high", 98)
                vmin = np.percentile(heatmap, plow)
                vmax = np.percentile(heatmap, phigh)
                heatmap = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
            elif normalize_method == "zscore":
                heatmap = (heatmap - heatmap.mean()) / (heatmap.std() + 1e-8)
                heatmap = np.clip((heatmap + 3) / 6, 0, 1)  # Map -3 to 3 std to 0-1

        # Apply threshold
        if threshold > 0:
            heatmap = np.where(heatmap >= threshold, heatmap, 0)

        # Apply colormap
        colored_heatmap = self._apply_colormap(heatmap, colormap)

        # Blend with image
        img_array = np.array(image).astype(np.float32)

        if blend_mode == "overlay":
            result = img_array * (1 - opacity) + colored_heatmap.astype(np.float32) * opacity
        elif blend_mode == "multiply":
            result = img_array * (colored_heatmap.astype(np.float32) / 255)
        elif blend_mode == "screen":
            result = 255 - (255 - img_array) * (255 - colored_heatmap.astype(np.float32)) / 255
        elif blend_mode == "add":
            result = np.clip(img_array + colored_heatmap.astype(np.float32) * opacity, 0, 255)
        else:
            result = img_array * (1 - opacity) + colored_heatmap.astype(np.float32) * opacity

        # Apply only where heatmap is non-zero (if threshold applied)
        if threshold > 0:
            mask = (heatmap > 0)[:, :, np.newaxis]
            result = np.where(mask, result, img_array)

        output_image = Image.fromarray(result.astype(np.uint8))
        duration = (time.time() - start_time) * 1000

        heatmap_image_data = f"data:image/jpeg;base64,{image_to_base64(output_image)}"
        outputs = {
            "image": heatmap_image_data,
            "result": heatmap_image_data,  # Convenience alias
        }

        # Generate colorbar
        if output_colorbar:
            colorbar_height = 20
            colorbar_width = image.width
            colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
            for x in range(colorbar_width):
                value = x / colorbar_width
                colorbar[:, x] = self._interpolate_color(value, self.COLORMAPS.get(colormap, self.COLORMAPS["jet"]))
            colorbar_img = Image.fromarray(colorbar)
            outputs["colorbar"] = f"data:image/png;base64,{image_to_base64(colorbar_img, 'PNG')}"

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={"colormap": colormap, "opacity": opacity, "threshold": threshold},
        )


class ComparisonBlock(BaseBlock):
    """
    Comparison Block - SOTA Visual Comparison

    Creates visual comparisons between images:
    - Side by side
    - Overlay with slider
    - Difference map
    - Grid layout
    - Animated GIF

    SOTA Features:
    - Multiple layout modes
    - Difference visualization (absolute, signed, edge)
    - Labels and annotations
    - Synchronized sizing
    """

    block_type = "comparison"
    display_name = "Comparison"
    description = "Create visual comparisons between images"

    input_ports = [
        {"name": "image_a", "type": "image", "required": True, "description": "First image (before/original)"},
        {"name": "image_b", "type": "image", "required": True, "description": "Second image (after/processed)"},
        {"name": "images", "type": "array", "required": False, "description": "Multiple images for grid layout"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Comparison visualization"},
        {"name": "result", "type": "image", "description": "Alias for image - comparison result"},
        {"name": "diff_stats", "type": "object", "description": "Difference statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Layout mode
            "mode": {
                "type": "string",
                "enum": ["side_by_side", "top_bottom", "overlay", "difference", "checkerboard", "grid"],
                "default": "side_by_side",
                "description": "Comparison layout mode",
            },
            # Labels
            "show_labels": {
                "type": "boolean",
                "default": True,
                "description": "Show image labels",
            },
            "label_a": {
                "type": "string",
                "default": "Before",
                "description": "Label for first image",
            },
            "label_b": {
                "type": "string",
                "default": "After",
                "description": "Label for second image",
            },
            "label_position": {
                "type": "string",
                "enum": ["top", "bottom"],
                "default": "top",
            },
            # Overlay mode
            "overlay_opacity": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Opacity for overlay mode",
            },
            "slider_position": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Slider position (0=left, 1=right)",
            },
            # Difference mode
            "diff_type": {
                "type": "string",
                "enum": ["absolute", "signed", "edge", "ssim"],
                "default": "absolute",
                "description": "Type of difference visualization",
            },
            "diff_amplify": {
                "type": "number",
                "default": 1,
                "minimum": 1,
                "maximum": 10,
                "description": "Amplify difference values",
            },
            "diff_colorize": {
                "type": "boolean",
                "default": True,
                "description": "Colorize difference (green=same, red=different)",
            },
            # Checkerboard
            "checker_size": {
                "type": "number",
                "default": 50,
                "description": "Size of checkerboard tiles in pixels",
            },
            # Grid mode
            "grid_cols": {
                "type": "number",
                "default": 2,
                "description": "Number of columns in grid mode",
            },
            # Border
            "show_border": {
                "type": "boolean",
                "default": True,
                "description": "Show border between images",
            },
            "border_width": {
                "type": "number",
                "default": 2,
            },
            "border_color": {
                "type": "string",
                "default": "#ffffff",
            },
            # Output stats
            "compute_diff_stats": {
                "type": "boolean",
                "default": False,
                "description": "Compute difference statistics (MSE, PSNR, SSIM)",
            },
        },
    }

    def _add_label(self, image: Image.Image, label: str, position: str = "top") -> Image.Image:
        """Add label to image."""
        from PIL import ImageDraw, ImageFont

        label_height = 30
        new_height = image.height + label_height
        new_image = Image.new("RGB", (image.width, new_height), (40, 40, 40))

        if position == "top":
            new_image.paste(image, (0, label_height))
            text_y = 5
        else:
            new_image.paste(image, (0, 0))
            text_y = image.height + 5

        draw = ImageDraw.Draw(new_image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_x = (image.width - (text_bbox[2] - text_bbox[0])) // 2
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

        return new_image

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Create comparison visualization."""
        start_time = time.time()

        image_a = await load_image_from_input(inputs.get("image_a"))
        image_b = await load_image_from_input(inputs.get("image_b"))

        if image_a is None or image_b is None:
            return BlockResult(
                error="Failed to load input images",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Config
        mode = config.get("mode", "side_by_side")
        show_labels = config.get("show_labels", True)
        label_a = config.get("label_a", "Before")
        label_b = config.get("label_b", "After")
        label_position = config.get("label_position", "top")
        show_border = config.get("show_border", True)
        border_width = config.get("border_width", 2)
        border_color = config.get("border_color", "#ffffff")
        compute_diff_stats = config.get("compute_diff_stats", False)

        # Ensure same size
        target_w = max(image_a.width, image_b.width)
        target_h = max(image_a.height, image_b.height)

        if image_a.size != (target_w, target_h):
            image_a = image_a.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if image_b.size != (target_w, target_h):
            image_b = image_b.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Add labels if requested
        if show_labels:
            image_a_labeled = self._add_label(image_a, label_a, label_position)
            image_b_labeled = self._add_label(image_b, label_b, label_position)
        else:
            image_a_labeled = image_a
            image_b_labeled = image_b

        # Parse border color
        try:
            bc = border_color.lstrip("#")
            border_rgb = tuple(int(bc[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            border_rgb = (255, 255, 255)

        result = None
        diff_stats = {}

        if mode == "side_by_side":
            total_width = image_a_labeled.width + image_b_labeled.width
            if show_border:
                total_width += border_width

            result = Image.new("RGB", (total_width, image_a_labeled.height), border_rgb)
            result.paste(image_a_labeled, (0, 0))
            paste_x = image_a_labeled.width + (border_width if show_border else 0)
            result.paste(image_b_labeled, (paste_x, 0))

        elif mode == "top_bottom":
            total_height = image_a_labeled.height + image_b_labeled.height
            if show_border:
                total_height += border_width

            result = Image.new("RGB", (image_a_labeled.width, total_height), border_rgb)
            result.paste(image_a_labeled, (0, 0))
            paste_y = image_a_labeled.height + (border_width if show_border else 0)
            result.paste(image_b_labeled, (0, paste_y))

        elif mode == "overlay":
            opacity = config.get("overlay_opacity", 0.5)
            arr_a = np.array(image_a).astype(np.float32)
            arr_b = np.array(image_b).astype(np.float32)
            blended = arr_a * (1 - opacity) + arr_b * opacity
            result = Image.fromarray(blended.astype(np.uint8))

        elif mode == "difference":
            diff_type = config.get("diff_type", "absolute")
            diff_amplify = config.get("diff_amplify", 1)
            diff_colorize = config.get("diff_colorize", True)

            arr_a = np.array(image_a).astype(np.float32)
            arr_b = np.array(image_b).astype(np.float32)

            if diff_type == "absolute":
                diff = np.abs(arr_a - arr_b)
            elif diff_type == "signed":
                diff = (arr_a - arr_b + 255) / 2
            elif diff_type == "edge":
                try:
                    import cv2
                    gray_a = cv2.cvtColor(arr_a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    gray_b = cv2.cvtColor(arr_b.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    edges_a = cv2.Canny(gray_a, 50, 150)
                    edges_b = cv2.Canny(gray_b, 50, 150)
                    diff = np.abs(edges_a.astype(np.float32) - edges_b.astype(np.float32))
                    diff = np.stack([diff] * 3, axis=-1)
                except ImportError:
                    diff = np.abs(arr_a - arr_b)
            else:
                diff = np.abs(arr_a - arr_b)

            # Amplify
            diff = np.clip(diff * diff_amplify, 0, 255)

            # Colorize
            if diff_colorize and diff_type != "signed":
                diff_gray = np.mean(diff, axis=2)
                colored_diff = np.zeros_like(arr_a)
                colored_diff[:, :, 0] = diff_gray  # Red for differences
                colored_diff[:, :, 1] = 255 - diff_gray  # Green for similarities
                diff = colored_diff

            result = Image.fromarray(diff.astype(np.uint8))

        elif mode == "checkerboard":
            checker_size = config.get("checker_size", 50)
            arr_a = np.array(image_a)
            arr_b = np.array(image_b)
            result_arr = np.zeros_like(arr_a)

            for y in range(0, target_h, checker_size):
                for x in range(0, target_w, checker_size):
                    tile_y = y // checker_size
                    tile_x = x // checker_size
                    is_a = (tile_x + tile_y) % 2 == 0

                    y_end = min(y + checker_size, target_h)
                    x_end = min(x + checker_size, target_w)

                    if is_a:
                        result_arr[y:y_end, x:x_end] = arr_a[y:y_end, x:x_end]
                    else:
                        result_arr[y:y_end, x:x_end] = arr_b[y:y_end, x:x_end]

            result = Image.fromarray(result_arr)

        elif mode == "grid":
            images = inputs.get("images", [])
            if not images:
                images = [image_a, image_b]
            else:
                loaded_images = []
                for img in images:
                    loaded = await load_image_from_input(img)
                    if loaded:
                        loaded_images.append(loaded.resize((target_w, target_h), Image.Resampling.LANCZOS))
                images = loaded_images if loaded_images else [image_a, image_b]

            grid_cols = config.get("grid_cols", 2)
            grid_rows = (len(images) + grid_cols - 1) // grid_cols

            cell_w = target_w + (border_width if show_border else 0)
            cell_h = target_h + (border_width if show_border else 0)

            result = Image.new("RGB", (cell_w * grid_cols, cell_h * grid_rows), border_rgb)

            for i, img in enumerate(images):
                row = i // grid_cols
                col = i % grid_cols
                x = col * cell_w
                y = row * cell_h
                result.paste(img, (x, y))

        # Compute difference statistics
        if compute_diff_stats:
            arr_a = np.array(image_a).astype(np.float32)
            arr_b = np.array(image_b).astype(np.float32)

            # MSE
            mse = np.mean((arr_a - arr_b) ** 2)
            diff_stats["mse"] = round(float(mse), 4)

            # PSNR
            if mse > 0:
                psnr = 10 * np.log10(255**2 / mse)
                diff_stats["psnr"] = round(float(psnr), 2)
            else:
                diff_stats["psnr"] = float("inf")

            # Mean absolute difference
            mad = np.mean(np.abs(arr_a - arr_b))
            diff_stats["mean_abs_diff"] = round(float(mad), 4)

            # Percentage of changed pixels
            threshold = 5
            changed_pixels = np.sum(np.abs(arr_a - arr_b).max(axis=2) > threshold)
            total_pixels = arr_a.shape[0] * arr_a.shape[1]
            diff_stats["changed_percent"] = round(100 * changed_pixels / total_pixels, 2)

        duration = (time.time() - start_time) * 1000

        comparison_image_data = f"data:image/jpeg;base64,{image_to_base64(result)}"
        outputs = {
            "image": comparison_image_data,
            "result": comparison_image_data,  # Convenience alias
        }
        if compute_diff_stats:
            outputs["diff_stats"] = diff_stats

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={"mode": mode, "size": f"{result.width}x{result.height}"},
        )
