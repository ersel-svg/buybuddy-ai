#!/usr/bin/env python3
"""
Test Retail Scenarios with Real Workflow Engine

This tests all 40 retail scenarios (20 basic + 20 advanced) using:
- Real WorkflowEngine with topological sorting
- Real block execution with mocked ML dependencies
- Real edge resolution ($nodes.x.y format)
- Real ForEach → Collect iteration loops

This is the ultimate validation that the scenarios actually work.
"""

import sys
import os
import asyncio
import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import numpy as np
import time

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
    return base64.b64encode(buffer.read()).decode("utf-8")


def create_mock_detections(count=5):
    """Create realistic mock detection results with all bbox formats."""
    return [
        {
            "class_name": f"product_{i % 3}",
            "class_id": i % 3,
            "confidence": 0.85 + (i * 0.02),
            "bbox": {
                "x": 50 + i * 100,
                "y": 50 + i * 50,
                "x1": 50 + i * 100,
                "y1": 50 + i * 50,
                "x2": 150 + i * 100,
                "y2": 150 + i * 50,
                "width": 100,
                "height": 100,
            },
            # Flat bbox fields for blocks that expect them at top level
            "x": 50 + i * 100,
            "y": 50 + i * 50,
            "width": 100,
            "height": 100,
        }
        for i in range(count)
    ]


def create_mock_embedding(dim=768):
    """Create a realistic mock embedding vector."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def create_mock_similarity_matches(count=3):
    """Create realistic mock similarity search results."""
    return [
        {
            "id": f"product_{i}",
            "similarity": 0.95 - (i * 0.05),
            "product_id": f"SKU{1000 + i}",
            "product_info": {"name": f"Test Product {i}"},
            "metadata": {"product_id": f"SKU{1000 + i}", "category": "test"},
        }
        for i in range(count)
    ]


# Mock execute methods for ML blocks
async def mock_detection_execute(self, inputs, config, context):
    start_time = time.time()
    image = inputs.get("image")
    if not image:
        return BlockResult(error="No image provided")
    detections = create_mock_detections(5)
    return BlockResult(
        outputs={
            "detections": detections,
            "first_detection": detections[0] if detections else None,
            "annotated_image": image,
            "count": len(detections),
        },
        duration_ms=round((time.time() - start_time) * 1000 + 50, 2),
        metrics={"model_id": config.get("model_id", "mock"), "detection_count": len(detections)},
    )


async def mock_classification_execute(self, inputs, config, context):
    start_time = time.time()
    return BlockResult(
        outputs={
            "predictions": [
                {"class_name": "class_a", "confidence": 0.92},
                {"class_name": "class_b", "confidence": 0.05},
            ],
            "top_class": "class_a",
            "top_confidence": 0.92,
        },
        duration_ms=round((time.time() - start_time) * 1000 + 30, 2),
        metrics={"model_id": config.get("model_id", "mock")},
    )


async def mock_embedding_execute(self, inputs, config, context):
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
    start_time = time.time()
    image = inputs.get("image", "")
    return BlockResult(
        outputs={
            "masks": [
                {"class_name": "shelf", "class_id": 0, "mask": "mock_mask", "area_ratio": 0.3},
                {"class_name": "product", "class_id": 1, "mask": "mock_mask", "area_ratio": 0.5},
            ],
            "annotated_image": image,
            "mask_count": 2,
        },
        duration_ms=round((time.time() - start_time) * 1000 + 60, 2),
        metrics={"model_id": config.get("model_id", "mock"), "mask_count": 2},
    )


async def mock_similarity_search_execute(self, inputs, config, context):
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

    from services.workflow.blocks.model_blocks import (
        DetectionBlock,
        ClassificationBlock,
        EmbeddingBlock,
        SimilaritySearchBlock,
    )
    from services.workflow.blocks.transform_blocks import SegmentationBlock

    DetectionBlock.execute = mock_detection_execute
    ClassificationBlock.execute = mock_classification_execute
    EmbeddingBlock.execute = mock_embedding_execute
    SegmentationBlock.execute = mock_segmentation_execute
    SimilaritySearchBlock.execute = mock_similarity_search_execute

    return engine


def convert_scenario_to_workflow(scenario: dict) -> dict:
    """Convert a scenario definition to a workflow definition."""
    # Scenarios have nodes/edges inside 'pipeline' key
    pipeline = scenario.get("pipeline", scenario)

    nodes = []
    for node in pipeline.get("nodes", []):
        nodes.append({
            "id": node["id"],
            "type": node["type"],
            "config": node.get("config", {}),
        })

    edges = []
    for edge in pipeline.get("edges", []):
        edges.append({
            "source": edge["source"],
            "target": edge["target"],
            "sourceHandle": edge.get("sourceHandle", "output"),
            "targetHandle": edge.get("targetHandle", "input"),
        })

    outputs = pipeline.get("outputs", [])

    return {
        "nodes": nodes,
        "edges": edges,
        "outputs": outputs,
    }


async def run_scenario_test(engine, scenario, inputs):
    """Run a single scenario test."""
    workflow = convert_scenario_to_workflow(scenario)

    try:
        result = await engine.execute(
            workflow=workflow,
            inputs=inputs,
            workflow_id="test-workflow-id",
            execution_id="test-execution-id",
        )

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "error_node_id": result.get("error_node_id"),
                "duration_ms": result.get("duration_ms", 0),
            }
        else:
            return {
                "success": True,
                "duration_ms": result.get("duration_ms", 0),
                "outputs": list(result.get("outputs", {}).keys()),
                "nodes_executed": len(result.get("metrics", {})),
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "exception": True,
        }


async def run_all_scenarios():
    """Run all retail scenarios with the real workflow engine."""
    print("=" * 80)
    print("RETAIL SCENARIOS - REAL WORKFLOW ENGINE TEST")
    print("=" * 80)

    # Import scenarios
    try:
        from workflow_scenarios_basic import BASIC_RETAIL_SCENARIOS
    except ImportError:
        BASIC_RETAIL_SCENARIOS = []
        print("Warning: Could not import BASIC_RETAIL_SCENARIOS")

    try:
        from workflow_scenarios_advanced import ADVANCED_RETAIL_SCENARIOS
    except ImportError:
        ADVANCED_RETAIL_SCENARIOS = []
        print("Warning: Could not import ADVANCED_RETAIL_SCENARIOS")

    all_scenarios = BASIC_RETAIL_SCENARIOS + ADVANCED_RETAIL_SCENARIOS

    if not all_scenarios:
        print("ERROR: No scenarios found!")
        return False

    print(f"\nTotal Scenarios: {len(all_scenarios)}")
    print(f"  - Basic: {len(BASIC_RETAIL_SCENARIOS)}")
    print(f"  - Advanced: {len(ADVANCED_RETAIL_SCENARIOS)}")

    # Create engine with mocks
    engine = get_engine_with_mocks()

    # Create test inputs
    inputs = {"image_base64": create_test_image()}

    # Run tests
    passed = 0
    failed = 0
    errors = []

    print("\n" + "-" * 40)
    print("Running Tests...")
    print("-" * 40)

    start_time = time.time()

    for scenario in all_scenarios:
        scenario_id = scenario.get("id", "unknown")
        scenario_name = scenario.get("name", "Unnamed")

        result = await run_scenario_test(engine, scenario, inputs)

        if result["success"]:
            print(f"  ✓ {scenario_id}: {scenario_name}")
            passed += 1
        else:
            print(f"  ✗ {scenario_id}: {scenario_name}")
            print(f"      Error: {result.get('error', 'Unknown error')}")
            if result.get("error_node_id"):
                print(f"      Node: {result['error_node_id']}")
            failed += 1
            errors.append({
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "error": result.get("error"),
                "error_node_id": result.get("error_node_id"),
            })

    total_time = (time.time() - start_time) * 1000

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Passed: {passed}/{len(all_scenarios)}")
    print(f"  Failed: {failed}/{len(all_scenarios)}")
    print(f"  Total Duration: {total_time:.2f} ms")
    print(f"  Average Duration: {total_time / len(all_scenarios):.2f} ms per scenario")

    if errors:
        print("\n" + "-" * 40)
        print("FAILURES:")
        print("-" * 40)
        for err in errors[:10]:  # Show first 10 failures
            print(f"\n  {err['scenario_id']}: {err['scenario_name']}")
            print(f"    Error: {err['error']}")
            if err.get("error_node_id"):
                print(f"    Node: {err['error_node_id']}")

        if len(errors) > 10:
            print(f"\n  ... and {len(errors) - 10} more failures")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_scenarios())
    sys.exit(0 if success else 1)
