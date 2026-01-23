"""
Workflow Blocks - Input Blocks

Blocks for workflow inputs (image, parameters, etc.)
SOTA features: Auto-resize, EXIF rotation, format validation, size limits
"""

import time
import base64
import httpx
import logging
from typing import Any, Optional
from io import BytesIO

from ..base import BaseBlock, BlockResult, ExecutionContext

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "HEIC", "HEIF"}


def apply_exif_rotation(image) -> "Image.Image":
    """
    Apply EXIF orientation to image (fix phone photo rotations).

    EXIF orientation values:
    1 = Normal
    2 = Mirrored horizontal
    3 = Rotated 180
    4 = Mirrored vertical
    5 = Mirrored horizontal then rotated 270
    6 = Rotated 90 CW
    7 = Mirrored horizontal then rotated 90
    8 = Rotated 270 CW
    """
    from PIL import Image, ExifTags

    try:
        # Get EXIF data
        exif = image._getexif()
        if exif is None:
            return image

        # Find orientation tag
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break

        if orientation_key is None or orientation_key not in exif:
            return image

        orientation = exif[orientation_key]

        # Apply transformations based on orientation
        if orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 7:
            image = image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)

        return image
    except Exception as e:
        logger.debug(f"Could not apply EXIF rotation: {e}")
        return image


def resize_image(image, max_dimension: int) -> "Image.Image":
    """
    Resize image keeping aspect ratio so largest dimension = max_dimension.
    Only downscales, never upscales.
    """
    from PIL import Image

    width, height = image.size

    # Don't upscale
    if width <= max_dimension and height <= max_dimension:
        return image

    # Calculate new size maintaining aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    # Use high-quality downsampling
    return image.resize((new_width, new_height), Image.LANCZOS)


class ImageInputBlock(BaseBlock):
    """
    Image Input Block - SOTA

    The entry point for image-based workflows.
    Accepts image URL, base64, or file upload.

    SOTA Features:
    - Auto-resize large images (memory optimization)
    - EXIF rotation correction (phone photos)
    - Format validation
    - Min/Max dimension validation
    - File size limits
    - RGB normalization
    """

    block_type = "image_input"
    display_name = "Image Input"
    description = "Input block for images. Accepts URL or base64 with preprocessing options."

    input_ports = []  # No inputs - this is a source block
    output_ports = [
        {"name": "image", "type": "image", "description": "The processed input image"},
        {"name": "image_url", "type": "string", "description": "URL of the image (if provided)"},
        {"name": "width", "type": "number", "description": "Final image width in pixels"},
        {"name": "height", "type": "number", "description": "Final image height in pixels"},
        {"name": "original_width", "type": "number", "description": "Original image width"},
        {"name": "original_height", "type": "number", "description": "Original image height"},
        {"name": "format", "type": "string", "description": "Image format (JPEG, PNG, etc.)"},
        {"name": "was_resized", "type": "boolean", "description": "Whether image was resized"},
        {"name": "was_rotated", "type": "boolean", "description": "Whether EXIF rotation was applied"},
    ]

    config_schema = {
        "type": "object",
        "properties": {
            "auto_resize": {
                "type": "boolean",
                "default": False,
                "description": "Automatically resize large images",
            },
            "max_dimension": {
                "type": "integer",
                "default": 1920,
                "minimum": 256,
                "maximum": 4096,
                "description": "Maximum dimension (width or height) in pixels",
            },
            "apply_exif_rotation": {
                "type": "boolean",
                "default": True,
                "description": "Auto-correct phone photo orientation",
            },
            "convert_to_rgb": {
                "type": "boolean",
                "default": True,
                "description": "Convert all images to RGB format",
            },
            "validate_format": {
                "type": "boolean",
                "default": False,
                "description": "Only accept specific image formats",
            },
            "allowed_formats": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["JPEG", "PNG", "WEBP"],
                "description": "Allowed image formats (when validate_format is true)",
            },
            "min_dimension": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "description": "Minimum dimension requirement (0 = no limit)",
            },
            "max_file_size_mb": {
                "type": "number",
                "default": 0,
                "minimum": 0,
                "description": "Maximum file size in MB (0 = no limit)",
            },
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Load and preprocess image with SOTA features."""
        start_time = time.time()

        image_url = context.inputs.get("image_url")
        image_base64 = context.inputs.get("image_base64")

        if not image_url and not image_base64:
            return BlockResult(
                error="No image provided. Provide either image_url or image_base64.",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get config options
        auto_resize = config.get("auto_resize", False)
        max_dimension = config.get("max_dimension", 1920)
        apply_exif = config.get("apply_exif_rotation", True)
        convert_rgb = config.get("convert_to_rgb", True)
        validate_format = config.get("validate_format", False)
        allowed_formats = config.get("allowed_formats", ["JPEG", "PNG", "WEBP"])
        min_dimension = config.get("min_dimension", 0)
        max_file_size_mb = config.get("max_file_size_mb", 0)

        try:
            from PIL import Image

            image_data = None
            file_size_bytes = 0

            if image_url:
                # Fetch image from URL (follow redirects for CDN/storage URLs)
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(image_url, timeout=30.0)
                    response.raise_for_status()
                    image_data = BytesIO(response.content)
                    file_size_bytes = len(response.content)
            else:
                # Decode base64
                image_bytes = base64.b64decode(image_base64)
                image_data = BytesIO(image_bytes)
                file_size_bytes = len(image_bytes)

            # Check file size limit
            if max_file_size_mb > 0:
                file_size_mb = file_size_bytes / (1024 * 1024)
                if file_size_mb > max_file_size_mb:
                    return BlockResult(
                        error=f"Image file size ({file_size_mb:.1f}MB) exceeds limit ({max_file_size_mb}MB)",
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                    )

            # Load image
            image = Image.open(image_data)
            image_format = image.format or "UNKNOWN"

            # Validate format
            if validate_format:
                format_upper = image_format.upper()
                allowed_upper = [f.upper() for f in allowed_formats]
                if format_upper not in allowed_upper:
                    return BlockResult(
                        error=f"Image format '{image_format}' not allowed. Allowed: {', '.join(allowed_formats)}",
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                    )

            # Store original dimensions
            original_width, original_height = image.size

            # Validate minimum dimension
            if min_dimension > 0:
                if original_width < min_dimension or original_height < min_dimension:
                    return BlockResult(
                        error=f"Image too small ({original_width}x{original_height}). Minimum: {min_dimension}px",
                        duration_ms=round((time.time() - start_time) * 1000, 2),
                    )

            # Apply EXIF rotation (before resize for correct dimensions)
            was_rotated = False
            if apply_exif:
                rotated_image = apply_exif_rotation(image)
                if rotated_image.size != image.size:
                    was_rotated = True
                image = rotated_image

            # Auto-resize if enabled
            was_resized = False
            if auto_resize:
                resized_image = resize_image(image, max_dimension)
                if resized_image.size != image.size:
                    was_resized = True
                image = resized_image

            # Convert to RGB if needed
            if convert_rgb and image.mode != "RGB":
                # Handle transparency (RGBA, LA, P with transparency)
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    # Create white background and paste image
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                    image = background
                else:
                    image = image.convert("RGB")

            final_width, final_height = image.size
            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={
                    "image": image,
                    "image_url": image_url,
                    "width": final_width,
                    "height": final_height,
                    "original_width": original_width,
                    "original_height": original_height,
                    "format": image_format,
                    "was_resized": was_resized,
                    "was_rotated": was_rotated,
                },
                duration_ms=round(duration, 2),
                metrics={
                    "original_size": f"{original_width}x{original_height}",
                    "final_size": f"{final_width}x{final_height}",
                    "format": image_format,
                    "file_size_kb": round(file_size_bytes / 1024, 1),
                    "was_resized": was_resized,
                    "was_rotated": was_rotated,
                },
            )

        except httpx.HTTPError as e:
            return BlockResult(
                error=f"Failed to fetch image: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            logger.exception("Image input processing failed")
            return BlockResult(
                error=f"Failed to load image: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )


class ParameterInputBlock(BaseBlock):
    """
    Parameter Input Block

    Provides access to workflow parameters.
    """

    block_type = "parameter_input"
    display_name = "Parameters"
    description = "Access workflow input parameters"

    input_ports = []
    output_ports = [
        {"name": "parameters", "type": "object", "description": "All input parameters"},
    ]

    config_schema = {
        "type": "object",
        "properties": {
            "parameter_name": {
                "type": "string",
                "description": "Specific parameter to extract (optional)",
            },
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Extract parameters from workflow inputs."""
        start_time = time.time()

        parameters = context.inputs.get("parameters", {})
        param_name = config.get("parameter_name")

        if param_name:
            value = parameters.get(param_name)
            outputs = {"value": value, "parameters": parameters}
        else:
            outputs = {"parameters": parameters}

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
        )
