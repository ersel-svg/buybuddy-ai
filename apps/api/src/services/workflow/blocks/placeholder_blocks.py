"""
Workflow Blocks - Placeholder Blocks

Placeholder implementations for blocks that will be fully implemented in later weeks.
These provide the structure and interface while returning mock data.
"""

import time
from typing import Any

from ..base import BaseBlock, BlockResult, ExecutionContext, ModelBlock


class DetectionBlock(ModelBlock):
    """
    Detection Block - Placeholder

    Will support both pretrained (YOLO) and trained (RF-DETR, RT-DETR) models.
    Full implementation in Week 2.
    """

    block_type = "detection"
    display_name = "Object Detection"
    description = "Detect objects in images using YOLO or trained models"
    model_type = "detection"

    input_ports = [
        {"name": "image", "type": "image", "required": True, "description": "Input image"},
    ]
    output_ports = [
        {"name": "detections", "type": "array", "description": "List of detected objects"},
        {"name": "annotated_image", "type": "image", "description": "Image with bounding boxes"},
        {"name": "count", "type": "number", "description": "Number of detections"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "Model ID"},
            "model_source": {"type": "string", "enum": ["pretrained", "trained"]},
            "confidence": {"type": "number", "default": 0.5},
            "nms_threshold": {"type": "number", "default": 0.4},
            "classes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["model_id", "model_source"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock detection data."""
        start_time = time.time()

        # TODO: Implement actual detection in Week 2
        # For now, return placeholder data

        image = inputs.get("image")
        model_id = config.get("model_id", "unknown")

        mock_detections = [
            {
                "class_name": "placeholder",
                "confidence": 0.95,
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            }
        ]

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "detections": mock_detections,
                "annotated_image": image,  # Pass through for now
                "count": len(mock_detections),
            },
            duration_ms=round(duration, 2),
            metrics={
                "model_id": model_id,
                "detection_count": len(mock_detections),
                "placeholder": True,
            },
        )


class ClassificationBlock(ModelBlock):
    """
    Classification Block - Placeholder

    Will support trained classification models (ViT, ConvNeXt, etc.)
    Full implementation in Week 2.
    """

    block_type = "classification"
    display_name = "Classification"
    description = "Classify images using trained models"
    model_type = "classification"

    input_ports = [
        {"name": "image", "type": "image", "required": False},
        {"name": "images", "type": "array", "required": False, "description": "Array of images"},
    ]
    output_ports = [
        {"name": "predictions", "type": "array", "description": "Classification predictions"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "model_source": {"type": "string", "enum": ["pretrained", "trained"]},
            "top_k": {"type": "number", "default": 1},
        },
        "required": ["model_id"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock classification data."""
        start_time = time.time()

        images = inputs.get("images", [])
        if not images and inputs.get("image"):
            images = [inputs["image"]]

        mock_predictions = []
        for _ in images:
            mock_predictions.append({
                "class_name": "placeholder",
                "confidence": 0.9,
                "top_k": [{"class_name": "placeholder", "confidence": 0.9}],
            })

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"predictions": mock_predictions},
            duration_ms=round(duration, 2),
            metrics={"placeholder": True},
        )


class EmbeddingBlock(ModelBlock):
    """
    Embedding Block - Placeholder

    Will support pretrained (DINOv2, CLIP) and trained embedding models.
    Full implementation in Week 2.
    """

    block_type = "embedding"
    display_name = "Embedding"
    description = "Extract embeddings from images"
    model_type = "embedding"

    input_ports = [
        {"name": "image", "type": "image", "required": False},
        {"name": "images", "type": "array", "required": False},
    ]
    output_ports = [
        {"name": "embeddings", "type": "array", "description": "Embedding vectors"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "model_id": {"type": "string"},
            "model_source": {"type": "string", "enum": ["pretrained", "trained"]},
        },
        "required": ["model_id"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock embeddings."""
        start_time = time.time()

        images = inputs.get("images", [])
        if not images and inputs.get("image"):
            images = [inputs["image"]]

        # Mock embeddings (768-dim for DINOv2-base)
        mock_embeddings = [[0.0] * 768 for _ in images]

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"embeddings": mock_embeddings},
            duration_ms=round(duration, 2),
            metrics={"embedding_count": len(mock_embeddings), "placeholder": True},
        )


class SimilaritySearchBlock(BaseBlock):
    """
    Similarity Search Block - Placeholder

    Will search Qdrant for similar products.
    Full implementation in Week 2.
    """

    block_type = "similarity_search"
    display_name = "Similarity Search"
    description = "Search for similar products in vector database"

    input_ports = [
        {"name": "embeddings", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "matches", "type": "array", "description": "Matching products"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "top_k": {"type": "number", "default": 5},
            "threshold": {"type": "number", "default": 0.7},
        },
        "required": ["collection"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock matches."""
        start_time = time.time()

        embeddings = inputs.get("embeddings", [])

        mock_matches = []
        for _ in embeddings:
            mock_matches.append({
                "product_id": "placeholder-id",
                "similarity": 0.85,
                "product_info": {"name": "Placeholder Product"},
                "identifiers": {},
            })

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"matches": mock_matches},
            duration_ms=round(duration, 2),
            metrics={"matched_count": len(mock_matches), "placeholder": True},
        )


class CropBlock(BaseBlock):
    """
    Crop Block - Placeholder

    Will crop regions from images based on detections.
    Full implementation in Week 3.
    """

    block_type = "crop"
    display_name = "Crop"
    description = "Crop regions from image based on detections"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "crops", "type": "array", "description": "Cropped image regions"},
        {"name": "crop_metadata", "type": "array", "description": "Metadata for each crop"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "padding": {"type": "number", "default": 0},
            "min_size": {"type": "number", "default": 32},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock crops."""
        start_time = time.time()

        detections = inputs.get("detections", [])
        image = inputs.get("image")

        # For placeholder, just duplicate the image for each detection
        crops = [image] * len(detections)
        metadata = [{"index": i, "detection": d} for i, d in enumerate(detections)]

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"crops": crops, "crop_metadata": metadata},
            duration_ms=round(duration, 2),
            metrics={"crop_count": len(crops), "placeholder": True},
        )


class SegmentationBlock(ModelBlock):
    """
    Segmentation Block - Placeholder

    Will support SAM and YOLO-seg models.
    Full implementation in Week 3.
    """

    block_type = "segmentation"
    display_name = "Segmentation"
    description = "Segment objects in images"
    model_type = "segmentation"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "array", "required": False},
    ]
    output_ports = [
        {"name": "masks", "type": "array", "description": "Segmentation masks"},
        {"name": "masked_image", "type": "image", "description": "Image with masks applied"},
    ]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns mock segmentation data."""
        start_time = time.time()

        image = inputs.get("image")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"masks": [], "masked_image": image},
            duration_ms=round(duration, 2),
            metrics={"placeholder": True},
        )


class BlurRegionBlock(BaseBlock):
    """
    Blur Region Block - Placeholder

    Will blur specified regions (for privacy, etc.)
    Full implementation in Week 3.
    """

    block_type = "blur_region"
    display_name = "Blur Region"
    description = "Blur specified regions in an image"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "regions", "type": "array", "required": True, "description": "Regions to blur (detections or masks)"},
    ]
    output_ports = [
        {"name": "image", "type": "image", "description": "Image with blurred regions"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "blur_type": {"type": "string", "enum": ["gaussian", "pixelate", "black"], "default": "gaussian"},
            "intensity": {"type": "number", "default": 21},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns image unchanged."""
        start_time = time.time()

        image = inputs.get("image")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"image": image},
            duration_ms=round(duration, 2),
            metrics={"placeholder": True},
        )


class ConditionBlock(BaseBlock):
    """
    Condition Block - Placeholder

    Will implement if-else branching.
    Full implementation in Week 4.
    """

    block_type = "condition"
    display_name = "Condition"
    description = "If-else branching based on expression"

    input_ports = [
        {"name": "value", "type": "any", "required": True},
    ]
    output_ports = [
        {"name": "true_output", "type": "any", "description": "Output when condition is true"},
        {"name": "false_output", "type": "any", "description": "Output when condition is false"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Expression to evaluate"},
        },
        "required": ["expression"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Always returns true path."""
        start_time = time.time()

        value = inputs.get("value")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"true_output": value, "false_output": None},
            duration_ms=round(duration, 2),
            metrics={"condition_result": True, "placeholder": True},
        )


class FilterBlock(BaseBlock):
    """
    Filter Block - Placeholder

    Will filter arrays based on expression.
    Full implementation in Week 4.
    """

    block_type = "filter"
    display_name = "Filter"
    description = "Filter array items based on expression"

    input_ports = [
        {"name": "items", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "passed", "type": "array", "description": "Items that passed filter"},
        {"name": "rejected", "type": "array", "description": "Items that failed filter"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Passes all items."""
        start_time = time.time()

        items = inputs.get("items", [])

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"passed": items, "rejected": []},
            duration_ms=round(duration, 2),
            metrics={"passed_count": len(items), "placeholder": True},
        )


class DrawBoxesBlock(BaseBlock):
    """
    Draw Boxes Block - Placeholder

    Will draw bounding boxes on images.
    Full implementation in Week 3.
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

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns image unchanged."""
        start_time = time.time()

        image = inputs.get("image")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"image": image},
            duration_ms=round(duration, 2),
            metrics={"placeholder": True},
        )


class GridBuilderBlock(BaseBlock):
    """
    Grid Builder Block - Placeholder

    Will build realogram/planogram grid from detection results.
    Full implementation in Week 4.
    """

    block_type = "grid_builder"
    display_name = "Grid Builder"
    description = "Build realogram/planogram grid from detections"

    input_ports = [
        {"name": "shelves", "type": "array", "required": False},
        {"name": "slots", "type": "array", "required": False},
        {"name": "matches", "type": "array", "required": False},
        {"name": "voids", "type": "array", "required": False},
    ]
    output_ports = [
        {"name": "grid", "type": "array", "description": "2D grid representation"},
        {"name": "realogram", "type": "object", "description": "Full realogram data"},
    ]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Placeholder: Returns empty grid."""
        start_time = time.time()

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "grid": [],
                "realogram": {
                    "rows": 0,
                    "cols": 0,
                    "cells": [],
                    "total_products": 0,
                    "total_voids": 0,
                },
            },
            duration_ms=round(duration, 2),
            metrics={"placeholder": True},
        )


class JsonOutputBlock(BaseBlock):
    """
    JSON Output Block - Placeholder

    Will format output as JSON.
    """

    block_type = "json_output"
    display_name = "JSON Output"
    description = "Format output as JSON"

    input_ports = [
        {"name": "data", "type": "any", "required": True},
    ]
    output_ports = [
        {"name": "json", "type": "object", "description": "JSON formatted output"},
    ]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Pass through data as JSON."""
        start_time = time.time()

        data = inputs.get("data")

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"json": data},
            duration_ms=round(duration, 2),
        )
