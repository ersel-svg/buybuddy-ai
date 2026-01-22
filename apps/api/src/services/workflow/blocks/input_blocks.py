"""
Workflow Blocks - Input Blocks

Blocks for workflow inputs (image, parameters, etc.)
"""

import time
import base64
import httpx
from typing import Any, Optional
from io import BytesIO

from ..base import BaseBlock, BlockResult, ExecutionContext


class ImageInputBlock(BaseBlock):
    """
    Image Input Block

    The entry point for image-based workflows.
    Accepts either an image URL or base64 encoded image.
    """

    block_type = "image_input"
    display_name = "Image Input"
    description = "Input block for images. Accepts URL or base64."

    input_ports = []  # No inputs - this is a source block
    output_ports = [
        {"name": "image", "type": "image", "description": "The input image"},
        {"name": "image_url", "type": "string", "description": "URL of the image (if provided)"},
        {"name": "width", "type": "number", "description": "Image width in pixels"},
        {"name": "height", "type": "number", "description": "Image height in pixels"},
    ]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Load image from URL or base64."""
        start_time = time.time()

        image_url = context.inputs.get("image_url")
        image_base64 = context.inputs.get("image_base64")

        if not image_url and not image_base64:
            return BlockResult(
                error="No image provided. Provide either image_url or image_base64."
            )

        try:
            from PIL import Image

            if image_url:
                # Fetch image from URL
                async with httpx.AsyncClient() as client:
                    response = await client.get(image_url, timeout=30.0)
                    response.raise_for_status()
                    image_data = BytesIO(response.content)
                    image = Image.open(image_data)
            else:
                # Decode base64
                image_bytes = base64.b64decode(image_base64)
                image_data = BytesIO(image_bytes)
                image = Image.open(image_data)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={
                    "image": image,
                    "image_url": image_url,
                    "width": width,
                    "height": height,
                },
                duration_ms=round(duration, 2),
                metrics={"image_size": f"{width}x{height}"},
            )

        except httpx.HTTPError as e:
            return BlockResult(error=f"Failed to fetch image: {str(e)}")
        except Exception as e:
            return BlockResult(error=f"Failed to load image: {str(e)}")


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
