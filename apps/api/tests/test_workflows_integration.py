#!/usr/bin/env python3
"""
Integration Tests for Workflow Engine

Tests the ACTUAL workflow engine with REAL blocks, mocking only external dependencies:
- ML model inference (detection, classification, embedding, segmentation)
- Qdrant vector search

This validates:
- Real topological sorting
- Real edge resolution ($nodes.x.y format)
- Real ForEach → Collect iteration loops
- Real block execute() methods
- Real input/output port validation
"""

import sys
import os
import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import numpy as np

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.workflow.engine import WorkflowEngine
from services.workflow.base import BlockResult


def create_test_image(width=640, height=480, color=(255, 0, 0)) -> str:
    """Create a real base64-encoded test image."""
    img = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def create_mock_detections(count=5):
    """Create realistic mock detection results."""
    detections = []
    for i in range(count):
        detections.append({
            "class_name": f"product_{i % 3}",
            "class_id": i % 3,
            "confidence": 0.85 + (i * 0.02),
            "bbox": {
                "x1": 50 + i * 100,
                "y1": 50 + i * 50,
                "x2": 150 + i * 100,
                "y2": 150 + i * 50,
                "width": 100,
                "height": 100,
            },
        })
    return detections


def create_mock_embedding(dim=768):
    """Create a realistic mock embedding vector."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # Normalize
    return vec.tolist()


def create_mock_similarity_matches(count=3):
    """Create realistic mock similarity search results."""
    matches = []
    for i in range(count):
        matches.append({
            "id": f"product_{i}",
            "similarity": 0.95 - (i * 0.05),
            "product_id": f"SKU{1000 + i}",
            "product_info": {
                "name": f"Test Product {i}",
                "upc": f"0000000000{i}",
                "sku": f"SKU{1000 + i}",
            },
            "metadata": {
                "product_id": f"SKU{1000 + i}",
                "category": "test",
            },
        })
    return matches


class MockModelLoader:
    """Mock model loader that returns mock models."""

    async def load_detection_model(self, model_id, source):
        model = MagicMock()
        processor = MagicMock()
        model_info = MagicMock()
        model_info.model_type = "detection"
        model_info.classes = ["product_0", "product_1", "product_2"]
        model_info.config = {}
        return model, processor, model_info

    async def load_classification_model(self, model_id, source):
        model = MagicMock()
        processor = MagicMock()
        model_info = MagicMock()
        model_info.model_type = "classification"
        model_info.classes = ["class_a", "class_b", "class_c"]
        return model, processor, model_info

    async def load_embedding_model(self, model_id, source):
        model = MagicMock()
        processor = MagicMock()
        model_info = MagicMock()
        model_info.embedding_dim = 768
        return model, processor, model_info

    async def load_segmentation_model(self, model_id, source):
        model = MagicMock()
        processor = MagicMock()
        model_info = MagicMock()
        model_info.model_type = "segmentation"
        return model, processor, model_info


class MockQdrantService:
    """Mock Qdrant service for similarity search."""

    def is_configured(self):
        return True

    async def search(self, collection_name, query_vector, limit=5, score_threshold=0.7, filter_conditions=None):
        return [
            {
                "id": f"item_{i}",
                "score": 0.95 - (i * 0.05),
                "payload": {
                    "product_id": f"SKU{1000 + i}",
                    "product_name": f"Product {i}",
                    "upc": f"0000000000{i}",
                    "sku": f"SKU{1000 + i}",
                    "category": "test",
                },
            }
            for i in range(min(limit, 5))
        ]


async def mock_detection_execute(self, inputs, config, context):
    """Mock detection block execute method."""
    import time
    start_time = time.time()

    image = inputs.get("image")
    if not image:
        return BlockResult(error="No image provided")

    detections = create_mock_detections(5)

    return BlockResult(
        outputs={
            "detections": detections,
            "first_detection": detections[0] if detections else None,
            "annotated_image": image,  # Return same image for simplicity
            "count": len(detections),
        },
        duration_ms=round((time.time() - start_time) * 1000 + 50, 2),  # Simulate inference time
        metrics={"model_id": config.get("model_id", "mock"), "detection_count": len(detections)},
    )


async def mock_classification_execute(self, inputs, config, context):
    """Mock classification block execute method."""
    import time
    start_time = time.time()

    return BlockResult(
        outputs={
            "predictions": [
                {"class_name": "class_a", "confidence": 0.92},
                {"class_name": "class_b", "confidence": 0.05},
                {"class_name": "class_c", "confidence": 0.03},
            ],
            "top_class": "class_a",
            "top_confidence": 0.92,
        },
        duration_ms=round((time.time() - start_time) * 1000 + 30, 2),
        metrics={"model_id": config.get("model_id", "mock")},
    )


async def mock_embedding_execute(self, inputs, config, context):
    """Mock embedding block execute method."""
    import time
    start_time = time.time()

    embedding = create_mock_embedding(768)

    return BlockResult(
        outputs={
            "embeddings": [embedding],
            "embedding": embedding,
            "metadata": {"embedding_dim": 768, "count": 1, "normalized": True},
        },
        duration_ms=round((time.time() - start_time) * 1000 + 40, 2),
        metrics={"model_id": config.get("model_id", "mock"), "embedding_count": 1},
    )


async def mock_segmentation_execute(self, inputs, config, context):
    """Mock segmentation block execute method."""
    import time
    start_time = time.time()

    image = inputs.get("image", "")

    return BlockResult(
        outputs={
            "masks": [
                {"class_name": "shelf", "class_id": 0, "mask": "mock_mask_data", "area_ratio": 0.3},
                {"class_name": "product", "class_id": 1, "mask": "mock_mask_data", "area_ratio": 0.5},
            ],
            "annotated_image": image,
            "mask_count": 2,
        },
        duration_ms=round((time.time() - start_time) * 1000 + 60, 2),
        metrics={"model_id": config.get("model_id", "mock"), "mask_count": 2},
    )


async def mock_similarity_search_execute(self, inputs, config, context):
    """Mock similarity search block execute method."""
    import time
    start_time = time.time()

    matches = create_mock_similarity_matches(config.get("top_k", 5))

    return BlockResult(
        outputs={
            "matches": [matches],
            "flat_matches": matches,
            "best_match": matches[0] if matches else None,
            "match_count": len(matches),
        },
        duration_ms=round((time.time() - start_time) * 1000 + 20, 2),
        metrics={"collection": config.get("collection", "test"), "total_matches": len(matches)},
    )


def get_engine_with_mocks():
    """Create a WorkflowEngine with mocked ML dependencies."""
    engine = WorkflowEngine()

    # Patch the model block execute methods
    from services.workflow.blocks.model_blocks import (
        DetectionBlock,
        ClassificationBlock,
        EmbeddingBlock,
        SimilaritySearchBlock,
    )
    from services.workflow.blocks.transform_blocks import SegmentationBlock

    # Store original methods
    original_detection = DetectionBlock.execute
    original_classification = ClassificationBlock.execute
    original_embedding = EmbeddingBlock.execute
    original_segmentation = SegmentationBlock.execute
    original_similarity = SimilaritySearchBlock.execute

    # Replace with mocks
    DetectionBlock.execute = mock_detection_execute
    ClassificationBlock.execute = mock_classification_execute
    EmbeddingBlock.execute = mock_embedding_execute
    SegmentationBlock.execute = mock_segmentation_execute
    SimilaritySearchBlock.execute = mock_similarity_search_execute

    return engine


# ============================================================
# TEST WORKFLOWS
# ============================================================

def create_simple_detection_workflow():
    """Simple workflow: Image → Detection → Output"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n", "confidence_threshold": 0.5}},
            {"id": "output_1", "type": "json_output", "config": {"include_all_inputs": True}},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "output_1", "sourceHandle": "detections", "targetHandle": "data"},
        ],
        "outputs": [
            {"name": "detections", "source": "$nodes.detect_1.detections"},
            {"name": "count", "source": "$nodes.detect_1.count"},
        ],
    }


def create_detection_with_filter_workflow():
    """Workflow: Image → Detection → Filter (confidence > 0.8)"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
            {"id": "filter_1", "type": "filter", "config": {
                "conditions": [{"field": "confidence", "operator": ">=", "value": 0.8}],
                "logic": "and",
            }},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "filter_1", "sourceHandle": "detections", "targetHandle": "items"},
        ],
        "outputs": [
            {"name": "filtered", "source": "$nodes.filter_1.passed"},
            {"name": "rejected", "source": "$nodes.filter_1.rejected"},
        ],
    }


def create_foreach_embedding_workflow():
    """Workflow: Detection → ForEach → Crop → Embedding → Collect"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
            {"id": "foreach_1", "type": "foreach", "config": {}},
            {"id": "crop_1", "type": "crop", "config": {}},
            {"id": "embed_1", "type": "embedding", "config": {"model_id": "dinov2-base"}},
            {"id": "collect_1", "type": "collect", "config": {}},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "foreach_1", "sourceHandle": "detections", "targetHandle": "items"},
            {"source": "input_1", "target": "foreach_1", "sourceHandle": "image", "targetHandle": "context"},
            {"source": "foreach_1", "target": "crop_1", "sourceHandle": "item", "targetHandle": "detection"},
            {"source": "foreach_1", "target": "crop_1", "sourceHandle": "context", "targetHandle": "image"},
            {"source": "crop_1", "target": "embed_1", "sourceHandle": "cropped", "targetHandle": "image"},
            {"source": "embed_1", "target": "collect_1", "sourceHandle": "embedding", "targetHandle": "item"},
        ],
        "outputs": [
            {"name": "embeddings", "source": "$nodes.collect_1.results"},
            {"name": "count", "source": "$nodes.collect_1.count"},
        ],
    }


def create_similarity_search_workflow():
    """Workflow: Image → Embedding → SimilaritySearch"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "embed_1", "type": "embedding", "config": {"model_id": "dinov2-base"}},
            {"id": "search_1", "type": "similarity_search", "config": {"collection": "products", "top_k": 5}},
        ],
        "edges": [
            {"source": "input_1", "target": "embed_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "embed_1", "target": "search_1", "sourceHandle": "embedding", "targetHandle": "embedding"},
        ],
        "outputs": [
            {"name": "matches", "source": "$nodes.search_1.flat_matches"},
            {"name": "best_match", "source": "$nodes.search_1.best_match"},
        ],
    }


def create_full_pipeline_workflow():
    """Full pipeline: Detection → ForEach → Crop → Embed → Search → Collect → Aggregate"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
            {"id": "foreach_1", "type": "foreach", "config": {}},
            {"id": "crop_1", "type": "crop", "config": {}},
            {"id": "embed_1", "type": "embedding", "config": {"model_id": "dinov2-base"}},
            {"id": "search_1", "type": "similarity_search", "config": {"collection": "products", "top_k": 3}},
            {"id": "collect_1", "type": "collect", "config": {}},
            {"id": "agg_1", "type": "aggregation", "config": {"operation": "flatten", "flatten_config": {"child_field": "matches"}}},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "foreach_1", "sourceHandle": "detections", "targetHandle": "items"},
            {"source": "input_1", "target": "foreach_1", "sourceHandle": "image", "targetHandle": "context"},
            {"source": "foreach_1", "target": "crop_1", "sourceHandle": "item", "targetHandle": "detection"},
            {"source": "foreach_1", "target": "crop_1", "sourceHandle": "context", "targetHandle": "image"},
            {"source": "crop_1", "target": "embed_1", "sourceHandle": "cropped", "targetHandle": "image"},
            {"source": "embed_1", "target": "search_1", "sourceHandle": "embedding", "targetHandle": "embedding"},
            {"source": "search_1", "target": "collect_1", "sourceHandle": "flat_matches", "targetHandle": "item"},
            {"source": "collect_1", "target": "agg_1", "sourceHandle": "results", "targetHandle": "data"},
        ],
        "outputs": [
            {"name": "matches", "source": "$nodes.agg_1.result"},
            {"name": "detection_count", "source": "$nodes.detect_1.count"},
        ],
    }


def create_conditional_workflow():
    """Workflow with condition: Detection → Condition (count > 0) → Filter or Empty"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
            {"id": "cond_1", "type": "condition", "config": {
                "conditions": [{"field": "length", "operator": ">", "value": 0}],
            }},
            {"id": "filter_1", "type": "filter", "config": {
                "conditions": [{"field": "confidence", "operator": ">=", "value": 0.85}],
            }},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "cond_1", "sourceHandle": "detections", "targetHandle": "value"},
            {"source": "detect_1", "target": "filter_1", "sourceHandle": "detections", "targetHandle": "items"},
        ],
        "outputs": [
            {"name": "has_detections", "source": "$nodes.cond_1.passed"},
            {"name": "high_confidence", "source": "$nodes.filter_1.passed"},
        ],
    }


def create_visualization_workflow():
    """Workflow: Detection → DrawBoxes (visualization)"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
            {"id": "draw_1", "type": "draw_boxes", "config": {"show_labels": True, "show_confidence": True}},
        ],
        "edges": [
            {"source": "input_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "input_1", "target": "draw_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "detect_1", "target": "draw_1", "sourceHandle": "detections", "targetHandle": "detections"},
        ],
        "outputs": [
            {"name": "annotated", "source": "$nodes.draw_1.image"},
            {"name": "detection_count", "source": "$nodes.detect_1.count"},
        ],
    }


def create_transform_workflow():
    """Workflow: Resize → Detection (simpler)"""
    return {
        "nodes": [
            {"id": "input_1", "type": "image_input", "config": {}},
            {"id": "resize_1", "type": "resize", "config": {"width": 640, "height": 480, "mode": "fit"}},
            {"id": "detect_1", "type": "detection", "config": {"model_id": "yolov8n"}},
        ],
        "edges": [
            {"source": "input_1", "target": "resize_1", "sourceHandle": "image", "targetHandle": "image"},
            {"source": "resize_1", "target": "detect_1", "sourceHandle": "image", "targetHandle": "image"},
        ],
        "outputs": [
            {"name": "detections", "source": "$nodes.detect_1.detections"},
            {"name": "resize_info", "source": "$nodes.resize_1.scale_info"},
        ],
    }


# ============================================================
# TEST RUNNER
# ============================================================

async def run_workflow_test(engine, workflow, name, inputs=None):
    """Run a single workflow test."""
    if inputs is None:
        # ImageInputBlock expects image_base64 (raw base64, without data:image prefix)
        base64_image = create_test_image()
        if ";base64," in base64_image:
            base64_data = base64_image.split(";base64,")[1]
        else:
            base64_data = base64_image
        inputs = {"image_base64": base64_data}

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    try:
        result = await engine.execute(
            workflow=workflow,
            inputs=inputs,
            workflow_id="test-workflow-id",
            execution_id="test-execution-id",
        )

        if result.get("error"):
            print(f"  ✗ FAILED: {result['error']}")
            if result.get("error_node_id"):
                print(f"    Node: {result['error_node_id']}")
            return False
        else:
            print(f"  ✓ PASSED")
            print(f"    Duration: {result['duration_ms']:.2f}ms")
            print(f"    Outputs: {list(result['outputs'].keys())}")

            # Print metrics summary
            if result.get("metrics"):
                node_count = len(result["metrics"])
                print(f"    Nodes executed: {node_count}")

            return True

    except Exception as e:
        print(f"  ✗ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("WORKFLOW ENGINE INTEGRATION TESTS")
    print("Using REAL blocks with mocked ML dependencies")
    print("="*60)

    engine = get_engine_with_mocks()

    tests = [
        ("Simple Detection", create_simple_detection_workflow()),
        ("Detection with Filter", create_detection_with_filter_workflow()),
        ("ForEach + Embedding", create_foreach_embedding_workflow()),
        ("Similarity Search", create_similarity_search_workflow()),
        ("Full Pipeline", create_full_pipeline_workflow()),
        ("Conditional Logic", create_conditional_workflow()),
        ("Visualization", create_visualization_workflow()),
        ("Transform Pipeline", create_transform_workflow()),
    ]

    passed = 0
    failed = 0

    for name, workflow in tests:
        success = await run_workflow_test(engine, workflow, name)
        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
