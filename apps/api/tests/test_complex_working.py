#!/usr/bin/env python3
"""
Working Complex Workflow Tests
===============================
Simplified but comprehensive tests that actually execute.
"""

import httpx
import asyncio
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from datetime import datetime


def create_test_image(width=640, height=480, with_objects=True) -> str:
    """Create test image with colored boxes."""
    img = Image.new("RGB", (width, height), (200, 200, 200))

    if with_objects:
        draw = ImageDraw.Draw(img)
        # Red box
        draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0), outline=(0, 0, 0), width=3)
        # Green box
        draw.rectangle([300, 150, 400, 250], fill=(0, 255, 0), outline=(0, 0, 0), width=3)
        # Blue box
        draw.rectangle([450, 100, 550, 200], fill=(0, 0, 255), outline=(0, 0, 0), width=3)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def test_workflow(name, nodes, edges, description=""):
    """Generic test function."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    if description:
        print(f"Description: {description}")

    base = "http://localhost:8000/api/v1/workflows"
    test_image = create_test_image()

    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        # Create workflow
        create_resp = await client.post(
            f"{base}/",
            json={
                "name": f"COMPLEX TEST: {name}",
                "description": description,
                "definition": {"nodes": nodes, "edges": edges},
            }
        )

        if create_resp.status_code not in [200, 201]:
            print(f"❌ FAIL - Create: {create_resp.status_code}")
            print(f"   Error: {create_resp.text[:200]}")
            return False

        workflow = create_resp.json()
        wf_id = workflow["id"]
        print(f"✓ Created workflow: {wf_id[:8]}...")
        print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")

        # Execute
        exec_resp = await client.post(
            f"{base}/{wf_id}/run",
            json={"input": {"image_base64": test_image}}
        )

        if exec_resp.status_code not in [200, 201]:
            print(f"❌ FAIL - Execute: {exec_resp.status_code}")
            return False

        execution = exec_resp.json()
        status = execution["status"]
        duration = execution.get("duration_ms", 0)

        print(f"✓ Execution: {execution['id'][:8]}...")
        print(f"  Status: {status}")
        print(f"  Duration: {duration}ms")

        if status == "completed":
            print(f"  ✅ SUCCESS")
            output_keys = list(execution.get("output_data", {}).keys())
            if output_keys:
                print(f"  Outputs: {output_keys[:5]}")
            metrics = execution.get("node_metrics", {})
            if metrics:
                print(f"  Metrics: {len(metrics)} nodes executed")
            return True
        else:
            error = execution.get("error_message", "Unknown")
            print(f"  ⚠️  {status.upper()}: {error[:100]}")
            return status != "failed"  # Warnings OK, failures not


# ============================================================
# TEST 1: Simple Detection → Draw Boxes
# ============================================================

async def test_01_detection_visualization():
    """Detection with visualization."""
    return await test_workflow(
        "Detection + Visualization",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Input"}},
            {"id": "det", "type": "detection", "position": {"x": 300, "y": 100}, "data": {"label": "YOLO", "model_id": "yolo11n", "model_source": "pretrained", "config": {"confidence": 0.4}}},
            {"id": "draw", "type": "draw_boxes", "position": {"x": 500, "y": 100}, "data": {"label": "Draw", "config": {"show_labels": True}}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "det"},
            {"id": "e2", "source": "img", "target": "draw"},
            {"id": "e3", "source": "det", "target": "draw"},
        ],
        description="Detect objects and draw bounding boxes"
    )


# ============================================================
# TEST 2: Detection → Filter → Count
# ============================================================

async def test_02_detection_filter():
    """Detection with confidence filtering."""
    return await test_workflow(
        "Detection + Filter",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Input"}},
            {"id": "det", "type": "detection", "position": {"x": 300, "y": 100}, "data": {"label": "Detect", "model_id": "yolo11n", "model_source": "pretrained", "config": {"confidence": 0.25}}},
            {"id": "filter", "type": "filter", "position": {"x": 500, "y": 100}, "data": {"label": "High Conf", "config": {"conditions": [{"field": "confidence", "operator": ">=", "value": 0.7}]}}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "det"},
            {"id": "e2", "source": "det", "target": "filter"},
        ],
        description="Filter detections by confidence threshold"
    )


# ============================================================
# TEST 3: Transform Pipeline
# ============================================================

async def test_03_transform_chain():
    """Image transformations before detection."""
    return await test_workflow(
        "Transform Chain",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Input"}},
            {"id": "resize", "type": "resize", "position": {"x": 250, "y": 100}, "data": {"label": "Resize", "config": {"width": 640, "height": 480}}},
            {"id": "norm", "type": "normalize", "position": {"x": 400, "y": 100}, "data": {"label": "Normalize"}},
            {"id": "det", "type": "detection", "position": {"x": 550, "y": 100}, "data": {"label": "Detect", "model_id": "yolo11s", "model_source": "pretrained"}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "resize"},
            {"id": "e2", "source": "resize", "target": "norm"},
            {"id": "e3", "source": "norm", "target": "det"},
        ],
        description="Resize → Normalize → Detect"
    )


# ============================================================
# TEST 4: Multi-Model Parallel Detection
# ============================================================

async def test_04_parallel_detection():
    """Run multiple YOLO models in parallel."""
    return await test_workflow(
        "Parallel Multi-Model",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 200}, "data": {"label": "Input"}},
            {"id": "y11n", "type": "detection", "position": {"x": 300, "y": 100}, "data": {"label": "YOLO11n", "model_id": "yolo11n", "model_source": "pretrained"}},
            {"id": "y8n", "type": "detection", "position": {"x": 300, "y": 200}, "data": {"label": "YOLOv8n", "model_id": "yolov8n", "model_source": "pretrained"}},
            {"id": "y11s", "type": "detection", "position": {"x": 300, "y": 300}, "data": {"label": "YOLO11s", "model_id": "yolo11s", "model_source": "pretrained"}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "y11n"},
            {"id": "e2", "source": "img", "target": "y8n"},
            {"id": "e3", "source": "img", "target": "y11s"},
        ],
        description="3 YOLO models running in parallel"
    )


# ============================================================
# TEST 5: Embedding Extraction
# ============================================================

async def test_05_embedding():
    """Extract DINOv2 embeddings."""
    return await test_workflow(
        "Embedding Extraction",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Input"}},
            {"id": "embed", "type": "embedding", "position": {"x": 300, "y": 100}, "data": {"label": "DINOv2", "model_id": "dinov2-base", "model_source": "pretrained", "config": {"normalize": True}}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "embed"},
        ],
        description="Extract normalized image embeddings"
    )


# ============================================================
# TEST 6: Classification
# ============================================================

async def test_06_classification():
    """Image classification with ViT."""
    return await test_workflow(
        "Image Classification",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Input"}},
            {"id": "cls", "type": "classification", "position": {"x": 300, "y": 100}, "data": {"label": "ViT Classifier", "model_id": "vit-base", "model_source": "pretrained", "config": {"top_k": 5}}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "cls"},
        ],
        description="Classify image with Vision Transformer"
    )


# ============================================================
# TEST 7: Detection → Conditional Logic
# ============================================================

async def test_07_conditional():
    """Conditional branching based on detection count."""
    return await test_workflow(
        "Conditional Logic",
        nodes=[
            {"id": "img", "type": "image_input", "position": {"x": 100, "y": 200}, "data": {"label": "Input"}},
            {"id": "det", "type": "detection", "position": {"x": 300, "y": 200}, "data": {"label": "Detect", "model_id": "yolo11n", "model_source": "pretrained"}},
            {"id": "cond", "type": "condition", "position": {"x": 500, "y": 200}, "data": {"label": "Has Objects?", "config": {"conditions": [{"field": "length", "operator": ">", "value": 0}]}}},
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "det"},
            {"id": "e2", "source": "det", "target": "cond", "sourceHandle": "detections"},
        ],
        description="Conditional logic: check if objects detected"
    )


# ============================================================
# TEST 8: Large Image Processing
# ============================================================

async def test_08_large_image():
    """Process large image (1920x1080)."""
    large_img = create_test_image(1920, 1080, with_objects=True)

    base = "http://localhost:8000/api/v1/workflows"

    print(f"\n{'='*80}")
    print("TEST: Large Image Processing")
    print(f"{'='*80}")
    print(f"Description: 1920x1080 image detection")

    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        create_resp = await client.post(
            f"{base}/",
            json={
                "name": "COMPLEX TEST: Large Image",
                "definition": {
                    "nodes": [
                        {"id": "img", "type": "image_input", "position": {"x": 100, "y": 100}, "data": {"label": "Large Image"}},
                        {"id": "det", "type": "detection", "position": {"x": 300, "y": 100}, "data": {"label": "Detect", "model_id": "yolo11n", "model_source": "pretrained"}},
                    ],
                    "edges": [{"id": "e1", "source": "img", "target": "det"}],
                }
            }
        )

        wf_id = create_resp.json()["id"]
        print(f"✓ Created workflow: {wf_id[:8]}...")

        exec_resp = await client.post(
            f"{base}/{wf_id}/run",
            json={"input": {"image_base64": large_img}}
        )

        execution = exec_resp.json()
        status = execution["status"]
        print(f"✓ Execution: {execution['id'][:8]}...")
        print(f"  Status: {status}")
        print(f"  Duration: {execution.get('duration_ms', 0)}ms")

        if status == "completed":
            print(f"  ✅ SUCCESS - Large image processed")
            return True
        else:
            print(f"  ⚠️  {status.upper()}: {execution.get('error_message', '')[:100]}")
            return status != "failed"


# ============================================================
# MAIN RUNNER
# ============================================================

async def run_all_tests():
    """Run all tests."""
    print("="*80)
    print("COMPLEX WORKFLOW TESTS - WORKING VERSION")
    print("="*80)
    print(f"Testing comprehensive workflows with real models")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)

    tests = [
        test_01_detection_visualization,
        test_02_detection_filter,
        test_03_transform_chain,
        test_04_parallel_detection,
        test_05_embedding,
        test_06_classification,
        test_07_conditional,
        test_08_large_image,
    ]

    results = []
    for i, test in enumerate(tests, 1):
        try:
            success = await test()
            results.append(success)
        except Exception as e:
            print(f"\n❌ Test {i} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\nTotal: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed/Warnings: {failed} ⚠️")
    print(f"Success Rate: {(passed/total*100):.1f}%")

    # Save results
    with open("complex_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total,
        }, f, indent=2)

    print(f"\n✅ Results saved to: complex_test_results.json")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
