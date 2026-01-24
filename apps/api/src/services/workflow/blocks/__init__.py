"""
Workflow Blocks Registry

All available workflow blocks are registered here.
"""

from typing import Dict, Type
from ..base import BaseBlock

# Import block implementations
from .input_blocks import ImageInputBlock, ParameterInputBlock

# Model blocks (real implementations)
from .model_blocks import (
    DetectionBlock,
    ClassificationBlock,
    EmbeddingBlock,
    SimilaritySearchBlock,
)

# Transform blocks (real implementations)
from .transform_blocks import (
    CropBlock,
    BlurRegionBlock,
    DrawBoxesBlock,
    SegmentationBlock,
    ResizeBlock,
    TileBlock,
    StitchBlock,
    RotateFlipBlock,
    NormalizeBlock,
    SmoothingBlock,
    DrawMasksBlock,
    HeatmapBlock,
    ComparisonBlock,
)

# Logic and output blocks
from .placeholder_blocks import (
    ConditionBlock,
    FilterBlock,
    ForEachBlock,
    CollectBlock,
    MapBlock,
    GridBuilderBlock,
    JsonOutputBlock,
    APIResponseBlock,
    WebhookBlock,
    AggregationBlock,
)


# Block registry
_BLOCK_REGISTRY: Dict[str, Type[BaseBlock]] = {
    # Input blocks
    "image_input": ImageInputBlock,
    "parameter_input": ParameterInputBlock,

    # Model blocks
    "detection": DetectionBlock,
    "classification": ClassificationBlock,
    "embedding": EmbeddingBlock,
    "segmentation": SegmentationBlock,
    "similarity_search": SimilaritySearchBlock,

    # Transform blocks
    "crop": CropBlock,
    "blur_region": BlurRegionBlock,
    "resize": ResizeBlock,
    "tile": TileBlock,
    "stitch": StitchBlock,
    "rotate_flip": RotateFlipBlock,
    "normalize": NormalizeBlock,
    "smoothing": SmoothingBlock,

    # Visualization blocks
    "draw_boxes": DrawBoxesBlock,
    "draw_masks": DrawMasksBlock,
    "heatmap": HeatmapBlock,
    "comparison": ComparisonBlock,

    # Logic blocks
    "condition": ConditionBlock,
    "filter": FilterBlock,
    "foreach": ForEachBlock,
    "collect": CollectBlock,
    "map": MapBlock,
    "grid_builder": GridBuilderBlock,

    # Output blocks
    "json_output": JsonOutputBlock,
    "api_response": APIResponseBlock,
    "webhook": WebhookBlock,
    "aggregation": AggregationBlock,
}


def get_all_blocks() -> Dict[str, Type[BaseBlock]]:
    """Get all registered block types."""
    return _BLOCK_REGISTRY.copy()


def get_block(block_type: str) -> Type[BaseBlock]:
    """Get a specific block type."""
    if block_type not in _BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type: {block_type}")
    return _BLOCK_REGISTRY[block_type]


def register_block(block_type: str, block_class: Type[BaseBlock]):
    """Register a new block type."""
    _BLOCK_REGISTRY[block_type] = block_class


def get_block_metadata() -> Dict[str, dict]:
    """Get metadata for all blocks (for frontend block palette)."""
    metadata = {}
    for block_type, block_class in _BLOCK_REGISTRY.items():
        block = block_class()
        metadata[block_type] = {
            "type": block.block_type,
            "name": block.display_name,
            "description": block.description,
            "inputs": block.input_ports,
            "outputs": block.output_ports,
            "config_schema": block.config_schema,
        }
    return metadata


# Block categories for UI organization
BLOCK_CATEGORIES = {
    "input": {
        "name": "Input",
        "color": "#3B82F6",  # Blue
        "blocks": ["image_input", "parameter_input"],
    },
    "model": {
        "name": "Models",
        "color": "#8B5CF6",  # Purple
        "blocks": ["detection", "classification", "embedding", "segmentation", "similarity_search"],
    },
    "transform": {
        "name": "Transform",
        "color": "#10B981",  # Green
        "blocks": ["crop", "resize", "tile", "stitch", "rotate_flip", "normalize", "smoothing", "blur_region"],
    },
    "logic": {
        "name": "Logic",
        "color": "#F59E0B",  # Yellow/Orange
        "blocks": ["condition", "filter", "foreach", "collect", "map", "grid_builder"],
    },
    "visualization": {
        "name": "Visualization",
        "color": "#EC4899",  # Pink
        "blocks": ["draw_boxes", "draw_masks", "heatmap", "comparison"],
    },
    "output": {
        "name": "Output",
        "color": "#EF4444",  # Red
        "blocks": ["json_output", "api_response", "webhook", "aggregation"],
    },
}
