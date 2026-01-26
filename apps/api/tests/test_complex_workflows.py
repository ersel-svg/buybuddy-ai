#!/usr/bin/env python3
"""
Complex Workflow Tests - Real-World Scenarios
==============================================

Tests advanced workflows that combine multiple models and blocks:
1. Retail Product Matching Pipeline (Detection → Crop → Embed → Search)
2. Multi-Model Detection Ensemble (YOLO11 + YOLOv8 + RT-DETR)
3. Image Quality Control Pipeline (Segment → Filter → Alert)
4. SAHI Tiled Detection (Large Image → Tile → Detect → Stitch)
5. Visual Search Pipeline (Embed → Search → Classify)
6. Conditional Processing (Detect → If/Else → Different paths)
7. Batch Processing with ForEach (Multiple images)
8. Transform Pipeline (Resize → Rotate → Normalize → Detect)
9. Visualization Pipeline (Detect → Draw → Heatmap → Compare)
10. Full E-commerce Pipeline (Multi-stage with all block types)
"""

import httpx
import asyncio
import json
import base64
import time
from io import BytesIO
from PIL import Image, ImageDraw
from datetime import datetime


def create_test_image(width=640, height=480, objects=True) -> str:
    """Create test image with colored rectangles (simulating objects)."""
    img = Image.new("RGB", (width, height), (230, 230, 230))

    if objects:
        draw = ImageDraw.Draw(img)
        # Draw some colored rectangles (simulating products)
        draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))      # Red box
        draw.rectangle([200, 100, 300, 200], fill=(0, 255, 0))    # Green box
        draw.rectangle([350, 50, 450, 150], fill=(0, 0, 255))     # Blue box
        draw.rectangle([100, 250, 200, 350], fill=(255, 255, 0))  # Yellow box

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def create_and_execute_workflow(client, name, description, definition, test_image=None):
    """Helper to create and execute a workflow."""
    start = time.time()

    if test_image is None:
        test_image = create_test_image()

    # Create workflow
    create_resp = await client.post(
        "http://localhost:8000/api/v1/workflows/",
        json={
            "name": name,
            "description": description,
            "definition": definition,
        }
    )

    if create_resp.status_code not in [200, 201]:
        return {
            "success": False,
            "error": f"Create failed: {create_resp.status_code} - {create_resp.text}",
            "duration": (time.time() - start) * 1000,
        }

    workflow = create_resp.json()
    workflow_id = workflow["id"]

    # Execute workflow
    exec_resp = await client.post(
        f"http://localhost:8000/api/v1/workflows/{workflow_id}/run",
        json={"input": {"image_base64": test_image}},
    )

    duration = (time.time() - start) * 1000

    if exec_resp.status_code not in [200, 201]:
        return {
            "success": False,
            "error": f"Execute failed: {exec_resp.status_code} - {exec_resp.text}",
            "duration": duration,
            "workflow_id": workflow_id,
        }

    execution = exec_resp.json()

    return {
        "success": execution["status"] == "completed",
        "workflow_id": workflow_id,
        "execution_id": execution["id"],
        "status": execution["status"],
        "duration": duration,
        "exec_duration": execution.get("duration_ms", 0),
        "output_data": execution.get("output_data", {}),
        "node_metrics": execution.get("node_metrics", {}),
        "error": execution.get("error_message"),
    }


# ============================================================
# TEST 1: Retail Product Matching Pipeline
# ============================================================

async def test_01_retail_product_matching():
    """
    Complex retail pipeline:
    Image → Detection (YOLO11) → ForEach → Crop → Embedding (DINOv2) →
    Similarity Search (Qdrant) → Collect → Aggregate → JSON Output
    """
    print("\n" + "="*80)
    print("TEST 1: Retail Product Matching Pipeline")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Product Image"},
            },
            {
                "id": "detect_1",
                "type": "detection",
                "position": {"x": 300, "y": 200},
                "data": {
                    "label": "Detect Products",
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "config": {"confidence": 0.3, "iou_threshold": 0.45},
                },
            },
            {
                "id": "foreach_1",
                "type": "foreach",
                "position": {"x": 500, "y": 200},
                "data": {"label": "ForEach Detection", "config": {}},
            },
            {
                "id": "crop_1",
                "type": "crop",
                "position": {"x": 700, "y": 200},
                "data": {"label": "Crop Product", "config": {"padding": 10}},
            },
            {
                "id": "embed_1",
                "type": "embedding",
                "position": {"x": 900, "y": 200},
                "data": {
                    "label": "Extract Embedding",
                    "model_id": "dinov2-base",
                    "model_source": "pretrained",
                    "config": {"normalize": True},
                },
            },
            {
                "id": "search_1",
                "type": "similarity_search",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Search Similar",
                    "config": {
                        "collection": "products_dinov2",
                        "top_k": 3,
                        "score_threshold": 0.7,
                    },
                },
            },
            {
                "id": "collect_1",
                "type": "collect",
                "position": {"x": 1300, "y": 200},
                "data": {"label": "Collect Results", "config": {}},
            },
            {
                "id": "agg_1",
                "type": "aggregation",
                "position": {"x": 1500, "y": 200},
                "data": {
                    "label": "Aggregate Matches",
                    "config": {
                        "operation": "flatten",
                        "flatten_config": {"child_field": "matches"},
                    },
                },
            },
            {
                "id": "output_1",
                "type": "json_output",
                "position": {"x": 1700, "y": 200},
                "data": {"label": "Final Output"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "detect_1"},
            {"id": "e2", "source": "detect_1", "target": "foreach_1", "sourceHandle": "detections", "targetHandle": "items"},
            {"id": "e3", "source": "input_1", "target": "foreach_1", "sourceHandle": "image", "targetHandle": "context"},
            {"id": "e4", "source": "foreach_1", "target": "crop_1", "sourceHandle": "item", "targetHandle": "detection"},
            {"id": "e5", "source": "foreach_1", "target": "crop_1", "sourceHandle": "context", "targetHandle": "image"},
            {"id": "e6", "source": "crop_1", "target": "embed_1"},
            {"id": "e7", "source": "embed_1", "target": "search_1"},
            {"id": "e8", "source": "search_1", "target": "collect_1"},
            {"id": "e9", "source": "collect_1", "target": "agg_1"},
            {"id": "e10", "source": "agg_1", "target": "output_1"},
        ],
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 1: Retail Product Matching",
            "Full retail pipeline with detection, cropping, embedding, and search",
            definition,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")
        print(f"Nodes: {len(definition['nodes'])}")
        print(f"Edges: {len(definition['edges'])}")

        if result["success"]:
            print(f"✅ PASS - Pipeline executed successfully")
            print(f"Output keys: {list(result['output_data'].keys())}")
            print(f"Node metrics: {list(result['node_metrics'].keys())}")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error', 'Unknown error')}")

        return result


# ============================================================
# TEST 2: Multi-Model Detection Ensemble
# ============================================================

async def test_02_multi_model_ensemble():
    """
    Run multiple detection models in parallel and aggregate results:
    Image → [YOLO11n, YOLOv8n, YOLO11s] → Aggregation (NMS) → Draw Boxes
    """
    print("\n" + "="*80)
    print("TEST 2: Multi-Model Detection Ensemble")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 300},
                "data": {"label": "Input Image"},
            },
            # YOLO11n
            {
                "id": "yolo11n_1",
                "type": "detection",
                "position": {"x": 300, "y": 150},
                "data": {
                    "label": "YOLO11 Nano",
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "config": {"confidence": 0.4},
                },
            },
            # YOLOv8n
            {
                "id": "yolov8n_1",
                "type": "detection",
                "position": {"x": 300, "y": 300},
                "data": {
                    "label": "YOLOv8 Nano",
                    "model_id": "yolov8n",
                    "model_source": "pretrained",
                    "config": {"confidence": 0.4},
                },
            },
            # YOLO11s
            {
                "id": "yolo11s_1",
                "type": "detection",
                "position": {"x": 300, "y": 450},
                "data": {
                    "label": "YOLO11 Small",
                    "model_id": "yolo11s",
                    "model_source": "pretrained",
                    "config": {"confidence": 0.4},
                },
            },
            # Aggregate results with NMS
            {
                "id": "agg_1",
                "type": "aggregation",
                "position": {"x": 600, "y": 300},
                "data": {
                    "label": "Merge Detections",
                    "config": {
                        "operation": "merge",
                        "merge_config": {
                            "arrays": ["yolo11n", "yolov8n", "yolo11s"],
                            "nms": True,
                            "iou_threshold": 0.5,
                        },
                    },
                },
            },
            # Visualize
            {
                "id": "draw_1",
                "type": "draw_boxes",
                "position": {"x": 850, "y": 300},
                "data": {
                    "label": "Draw Boxes",
                    "config": {"show_labels": True, "show_confidence": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "yolo11n_1"},
            {"id": "e2", "source": "input_1", "target": "yolov8n_1"},
            {"id": "e3", "source": "input_1", "target": "yolo11s_1"},
            {"id": "e4", "source": "yolo11n_1", "target": "agg_1"},
            {"id": "e5", "source": "yolov8n_1", "target": "agg_1"},
            {"id": "e6", "source": "yolo11s_1", "target": "agg_1"},
            {"id": "e7", "source": "input_1", "target": "draw_1"},
            {"id": "e8", "source": "agg_1", "target": "draw_1"},
        ],
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 2: Multi-Model Ensemble",
            "Parallel detection with YOLO11n, YOLOv8n, YOLO11s + NMS aggregation",
            definition,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")
        print(f"Parallel models: 3 (YOLO11n, YOLOv8n, YOLO11s)")

        if result["success"]:
            print(f"✅ PASS - Ensemble pipeline executed")
            metrics = result.get('node_metrics', {})
            print(f"YOLO11n time: {metrics.get('yolo11n_1', {}).get('duration_ms', 0)}ms")
            print(f"YOLOv8n time: {metrics.get('yolov8n_1', {}).get('duration_ms', 0)}ms")
            print(f"YOLO11s time: {metrics.get('yolo11s_1', {}).get('duration_ms', 0)}ms")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error')}")

        return result


# ============================================================
# TEST 3: Segmentation + Quality Control Pipeline
# ============================================================

async def test_03_segmentation_quality_control():
    """
    Segmentation → Calculate area → Filter defects → Alert:
    Image → Segment (SAM) → Map (calculate area) → Filter (area > threshold) →
    Condition (has defects?) → Webhook Alert
    """
    print("\n" + "="*80)
    print("TEST 3: Segmentation + Quality Control")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Product Image"},
            },
            {
                "id": "segment_1",
                "type": "segmentation",
                "position": {"x": 300, "y": 200},
                "data": {
                    "label": "Segment Objects",
                    "model_id": "sam-base",
                    "model_source": "pretrained",
                    "config": {"mode": "automatic"},
                },
            },
            {
                "id": "map_1",
                "type": "map",
                "position": {"x": 500, "y": 200},
                "data": {
                    "label": "Calculate Areas",
                    "config": {
                        "operation": "transform",
                        "fields": {
                            "area_pixels": "mask.sum()",
                            "area_ratio": "mask.sum() / (width * height)",
                        },
                    },
                },
            },
            {
                "id": "filter_1",
                "type": "filter",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Filter Large Segments",
                    "config": {
                        "conditions": [
                            {"field": "area_ratio", "operator": ">", "value": 0.05}
                        ],
                    },
                },
            },
            {
                "id": "cond_1",
                "type": "condition",
                "position": {"x": 900, "y": 200},
                "data": {
                    "label": "Has Defects?",
                    "config": {
                        "conditions": [
                            {"field": "length", "operator": ">", "value": 0}
                        ],
                    },
                },
            },
            {
                "id": "webhook_1",
                "type": "webhook",
                "position": {"x": 1100, "y": 150},
                "data": {
                    "label": "Alert Webhook",
                    "config": {
                        "url": "https://webhook.site/test",
                        "method": "POST",
                        "headers": {"Content-Type": "application/json"},
                    },
                },
            },
            {
                "id": "output_1",
                "type": "json_output",
                "position": {"x": 1100, "y": 250},
                "data": {"label": "Results"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "segment_1"},
            {"id": "e2", "source": "segment_1", "target": "map_1"},
            {"id": "e3", "source": "map_1", "target": "filter_1"},
            {"id": "e4", "source": "filter_1", "target": "cond_1"},
            {"id": "e5", "source": "cond_1", "target": "webhook_1", "sourceHandle": "true"},
            {"id": "e6", "source": "cond_1", "target": "output_1", "sourceHandle": "false"},
        ],
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 3: Segmentation QC",
            "Quality control with segmentation, area calculation, and conditional alerts",
            definition,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")

        if result["success"]:
            print(f"✅ PASS - QC pipeline executed")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error')}")

        return result


# ============================================================
# TEST 4: SAHI Tiled Detection (Large Image)
# ============================================================

async def test_04_sahi_tiled_detection():
    """
    Large image → Tile (4x4) → ForEach → Detect → Collect → Stitch → Draw:
    Handles large images by splitting into tiles for better small object detection
    """
    print("\n" + "="*80)
    print("TEST 4: SAHI Tiled Detection")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Large Image"},
            },
            {
                "id": "tile_1",
                "type": "tile",
                "position": {"x": 300, "y": 200},
                "data": {
                    "label": "Split to Tiles",
                    "config": {
                        "tile_size": 640,
                        "overlap": 0.2,
                        "strategy": "grid",
                    },
                },
            },
            {
                "id": "foreach_1",
                "type": "foreach",
                "position": {"x": 500, "y": 200},
                "data": {"label": "Process Each Tile"},
            },
            {
                "id": "detect_1",
                "type": "detection",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Detect on Tile",
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "config": {"confidence": 0.4},
                },
            },
            {
                "id": "collect_1",
                "type": "collect",
                "position": {"x": 900, "y": 200},
                "data": {"label": "Collect Detections"},
            },
            {
                "id": "stitch_1",
                "type": "stitch",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Stitch Results",
                    "config": {"nms": True, "iou_threshold": 0.5},
                },
            },
            {
                "id": "draw_1",
                "type": "draw_boxes",
                "position": {"x": 1300, "y": 200},
                "data": {
                    "label": "Visualize",
                    "config": {"show_labels": True},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "tile_1"},
            {"id": "e2", "source": "tile_1", "target": "foreach_1"},
            {"id": "e3", "source": "foreach_1", "target": "detect_1"},
            {"id": "e4", "source": "detect_1", "target": "collect_1"},
            {"id": "e5", "source": "collect_1", "target": "stitch_1"},
            {"id": "e6", "source": "input_1", "target": "draw_1"},
            {"id": "e7", "source": "stitch_1", "target": "draw_1"},
        ],
    }

    # Create larger test image
    large_image = create_test_image(1920, 1080, objects=True)

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 4: SAHI Tiled Detection",
            "Large image tiled detection for small objects",
            definition,
            test_image=large_image,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")
        print(f"Image size: 1920x1080")

        if result["success"]:
            print(f"✅ PASS - Tiled detection pipeline executed")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error')}")

        return result


# ============================================================
# TEST 5: Visual Search with Classification
# ============================================================

async def test_05_visual_search_classification():
    """
    Visual search + classification refinement:
    Image → Embedding (CLIP) → Search → ForEach matches →
    Classify (ViT) → Filter by category → Aggregate
    """
    print("\n" + "="*80)
    print("TEST 5: Visual Search + Classification")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Query Image"},
            },
            {
                "id": "embed_1",
                "type": "embedding",
                "position": {"x": 300, "y": 200},
                "data": {
                    "label": "CLIP Embedding",
                    "model_id": "clip-vit-base",
                    "model_source": "pretrained",
                    "config": {"normalize": True},
                },
            },
            {
                "id": "search_1",
                "type": "similarity_search",
                "position": {"x": 500, "y": 200},
                "data": {
                    "label": "Vector Search",
                    "config": {
                        "collection": "products_clip",
                        "top_k": 10,
                        "score_threshold": 0.6,
                    },
                },
            },
            {
                "id": "foreach_1",
                "type": "foreach",
                "position": {"x": 700, "y": 200},
                "data": {"label": "Process Matches"},
            },
            {
                "id": "classify_1",
                "type": "classification",
                "position": {"x": 900, "y": 200},
                "data": {
                    "label": "Classify Category",
                    "model_id": "vit-base",
                    "model_source": "pretrained",
                    "config": {"top_k": 3},
                },
            },
            {
                "id": "collect_1",
                "type": "collect",
                "position": {"x": 1100, "y": 200},
                "data": {"label": "Collect Results"},
            },
            {
                "id": "filter_1",
                "type": "filter",
                "position": {"x": 1300, "y": 200},
                "data": {
                    "label": "Filter by Category",
                    "config": {
                        "conditions": [
                            {"field": "category", "operator": "==", "value": "product"}
                        ],
                    },
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "embed_1"},
            {"id": "e2", "source": "embed_1", "target": "search_1"},
            {"id": "e3", "source": "search_1", "target": "foreach_1"},
            {"id": "e4", "source": "foreach_1", "target": "classify_1"},
            {"id": "e5", "source": "classify_1", "target": "collect_1"},
            {"id": "e6", "source": "collect_1", "target": "filter_1"},
        ],
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 5: Visual Search + Classification",
            "Hybrid search with embedding similarity and classification refinement",
            definition,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")

        if result["success"]:
            print(f"✅ PASS - Hybrid search pipeline executed")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error')}")

        return result


# ============================================================
# TEST 6: Full Transform Pipeline
# ============================================================

async def test_06_transform_pipeline():
    """
    Complex image transformations:
    Image → Resize → Rotate/Flip → Normalize → Smoothing → Detect → Compare
    """
    print("\n" + "="*80)
    print("TEST 6: Full Transform Pipeline")
    print("="*80)

    definition = {
        "nodes": [
            {
                "id": "input_1",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Original Image"},
            },
            {
                "id": "resize_1",
                "type": "resize",
                "position": {"x": 250, "y": 200},
                "data": {
                    "label": "Resize",
                    "config": {"width": 640, "height": 480, "mode": "fit"},
                },
            },
            {
                "id": "rotate_1",
                "type": "rotate_flip",
                "position": {"x": 400, "y": 200},
                "data": {
                    "label": "Rotate",
                    "config": {"angle": 0, "flip_horizontal": False},
                },
            },
            {
                "id": "normalize_1",
                "type": "normalize",
                "position": {"x": 550, "y": 200},
                "data": {
                    "label": "Normalize",
                    "config": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                },
            },
            {
                "id": "smooth_1",
                "type": "smoothing",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Smoothing",
                    "config": {"method": "gaussian", "kernel_size": 3},
                },
            },
            {
                "id": "detect_1",
                "type": "detection",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Detect",
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                },
            },
            {
                "id": "compare_1",
                "type": "comparison",
                "position": {"x": 1000, "y": 200},
                "data": {
                    "label": "Before/After",
                    "config": {"mode": "side_by_side"},
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "input_1", "target": "resize_1"},
            {"id": "e2", "source": "resize_1", "target": "rotate_1"},
            {"id": "e3", "source": "rotate_1", "target": "normalize_1"},
            {"id": "e4", "source": "normalize_1", "target": "smooth_1"},
            {"id": "e5", "source": "smooth_1", "target": "detect_1"},
            {"id": "e6", "source": "input_1", "target": "compare_1"},
            {"id": "e7", "source": "detect_1", "target": "compare_1"},
        ],
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        result = await create_and_execute_workflow(
            client,
            "TEST 6: Transform Pipeline",
            "Full image transformation chain with detection",
            definition,
        )

        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Duration: {result['duration']:.0f}ms")
        print(f"Transform stages: 5 (resize, rotate, normalize, smooth, detect)")

        if result["success"]:
            print(f"✅ PASS - Transform pipeline executed")
        else:
            print(f"⚠️  Status: {result.get('status')} - {result.get('error')}")

        return result


# ============================================================
# MAIN TEST RUNNER
# ============================================================

async def run_all_complex_tests():
    """Run all complex workflow tests."""
    print("=" * 80)
    print("COMPLEX WORKFLOW TESTS")
    print("Testing advanced real-world scenarios with multiple models")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    results = []

    # Run tests
    tests = [
        ("Retail Product Matching", test_01_retail_product_matching),
        ("Multi-Model Ensemble", test_02_multi_model_ensemble),
        ("Segmentation QC", test_03_segmentation_quality_control),
        ("SAHI Tiled Detection", test_04_sahi_tiled_detection),
        ("Visual Search + Classification", test_05_visual_search_classification),
        ("Transform Pipeline", test_06_transform_pipeline),
    ]

    for name, test_fn in tests:
        try:
            result = await test_fn()
            results.append({"name": name, **result})
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed/Warnings: {failed} ⚠️")

    # Detailed results
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result.get("success") else "⚠️"
        print(f"\n{i}. {status_icon} {result['name']}")
        print(f"   Duration: {result.get('duration', 0):.0f}ms")
        if result.get('workflow_id'):
            print(f"   Workflow ID: {result['workflow_id'][:8]}...")
        if result.get("error"):
            print(f"   Error: {result['error'][:100]}...")

    # Save results
    with open("complex_workflow_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {"total": total, "passed": passed, "failed": failed},
            "results": results,
        }, f, indent=2)

    print(f"\n✅ Results saved to: complex_workflow_test_results.json")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_complex_tests())
    exit(0 if success else 1)
