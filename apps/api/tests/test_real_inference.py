#!/usr/bin/env python3
"""
Real Inference Tests with RunPod Worker
========================================
Tests workflows with actual GPU inference using RunPod worker.
Endpoint: yz9lgcxh1rdj9o
"""

import httpx
import asyncio
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from datetime import datetime
from typing import Dict, Any, List


def create_product_image(width=800, height=600) -> str:
    """Create realistic product image for testing."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Simulate product (bottle)
    draw.rectangle([250, 150, 350, 450], fill=(220, 20, 60), outline=(0, 0, 0), width=3)
    draw.ellipse([240, 140, 360, 180], fill=(220, 20, 60), outline=(0, 0, 0), width=3)

    # Simulate label
    draw.rectangle([270, 250, 330, 320], fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    # Background shelf
    draw.rectangle([50, 400, 750, 550], fill=(139, 69, 19), outline=(0, 0, 0), width=2)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def create_test_scene(width=640, height=480) -> str:
    """Create test scene with multiple objects."""
    img = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(img)

    # Object 1: Red box
    draw.rectangle([80, 80, 160, 160], fill=(255, 0, 0), outline=(0, 0, 0), width=3)

    # Object 2: Green circle
    draw.ellipse([250, 90, 350, 190], fill=(0, 255, 0), outline=(0, 0, 0), width=3)

    # Object 3: Blue triangle (approximated with polygon)
    draw.polygon([(450, 180), (520, 60), (590, 180)], fill=(0, 0, 255), outline=(0, 0, 0))

    # Object 4: Yellow rectangle
    draw.rectangle([100, 300, 250, 400], fill=(255, 255, 0), outline=(0, 0, 0), width=3)

    # Object 5: Cyan box
    draw.rectangle([400, 300, 550, 420], fill=(0, 255, 255), outline=(0, 0, 0), width=3)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def test_workflow(
    name: str,
    nodes: List[Dict],
    edges: List[Dict],
    input_data: Dict[str, Any],
    description: str = "",
    timeout: int = 180
) -> Dict[str, Any]:
    """Execute workflow test with real inference."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print(f"{'='*80}")
    if description:
        print(f"Description: {description}")

    base = "http://localhost:8000/api/v1/workflows"

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        # Create workflow
        create_resp = await client.post(
            f"{base}/",
            json={
                "name": f"REAL INFERENCE: {name}",
                "description": description,
                "definition": {"nodes": nodes, "edges": edges},
            }
        )

        if create_resp.status_code not in [200, 201]:
            print(f"❌ FAIL - Create: {create_resp.status_code}")
            print(f"   Error: {create_resp.text[:300]}")
            return {"success": False, "error": "Failed to create workflow"}

        workflow = create_resp.json()
        wf_id = workflow["id"]
        print(f"✓ Created workflow: {wf_id[:8]}...")
        print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")

        # Execute with real inference
        print(f"  Executing with RunPod worker (GPU inference)...")
        exec_start = datetime.now()

        exec_resp = await client.post(
            f"{base}/{wf_id}/run",
            json={"input": input_data}
        )

        exec_duration = (datetime.now() - exec_start).total_seconds() * 1000

        if exec_resp.status_code not in [200, 201]:
            print(f"❌ FAIL - Execute: {exec_resp.status_code}")
            print(f"   Error: {exec_resp.text[:300]}")
            return {"success": False, "error": "Failed to execute workflow"}

        execution = exec_resp.json()
        exec_id = execution["id"]
        status = execution["status"]
        duration = execution.get("duration_ms", 0)

        print(f"✓ Execution: {exec_id[:8]}...")
        print(f"  Status: {status}")
        print(f"  API Duration: {exec_duration:.0f}ms")
        print(f"  Execution Duration: {duration}ms")

        result = {
            "success": status == "completed",
            "status": status,
            "duration_ms": duration,
            "execution_id": exec_id,
            "workflow_id": wf_id,
        }

        if status == "completed":
            print(f"  ✅ SUCCESS - Real inference completed!")

            output_data = execution.get("output_data", {})
            node_metrics = execution.get("node_metrics", {})

            print(f"\n  📊 Output Summary:")
            for key, value in output_data.items():
                if isinstance(value, list):
                    print(f"    - {key}: {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"      Sample: {str(value[0])[:100]}...")
                elif isinstance(value, dict):
                    print(f"    - {key}: {len(value)} keys")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"    - {key}: {len(value)} chars (image/large data)")
                else:
                    print(f"    - {key}: {value}")

            if node_metrics:
                print(f"\n  ⏱️  Node Performance:")
                for node_id, metrics in node_metrics.items():
                    node_dur = metrics.get("duration_ms", 0)
                    output_count = metrics.get("output_count", 0)
                    print(f"    - {node_id}: {node_dur}ms (outputs: {output_count})")

            result["output_data"] = output_data
            result["node_metrics"] = node_metrics

        elif status == "failed":
            error = execution.get("error_message", "Unknown")
            error_node = execution.get("error_node_id", "Unknown")
            print(f"  ❌ FAILED")
            print(f"    Error: {error}")
            print(f"    Failed Node: {error_node}")
            result["error"] = error
            result["error_node"] = error_node
        else:
            print(f"  ⚠️  {status.upper()}")
            if "error_message" in execution:
                print(f"    Message: {execution['error_message']}")

        return result


# ============================================================
# TEST 1: Simple YOLO Detection
# ============================================================

async def test_01_simple_detection():
    """Basic YOLO detection with real GPU inference."""
    return await test_workflow(
        "Simple YOLO Detection",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input Image"},
                "config": {}
            },
            {
                "id": "detect",
                "type": "detection",
                "position": {"x": 350, "y": 100},
                "data": {"label": "YOLO11n Detection"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.4,
                    "iou_threshold": 0.45
                }
            }
        ],
        edges=[
            {
                "id": "e1",
                "source": "img",
                "target": "detect",
                "sourceHandle": "image",
                "targetHandle": "image"
            }
        ],
        input_data={"image_base64": create_test_scene()},
        description="Test basic YOLO11n object detection on GPU"
    )


# ============================================================
# TEST 2: Detection + Visualization
# ============================================================

async def test_02_detection_visualization():
    """Detection with bounding box visualization."""
    return await test_workflow(
        "Detection + Visualization",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "detect",
                "type": "detection",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Detect"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            },
            {
                "id": "draw",
                "type": "draw_boxes",
                "position": {"x": 500, "y": 100},
                "data": {"label": "Draw Boxes"},
                "config": {
                    "show_labels": True,
                    "show_confidence": True,
                    "box_thickness": 2
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e2", "source": "img", "target": "draw", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e3", "source": "detect", "target": "draw", "sourceHandle": "detections", "targetHandle": "detections"}
        ],
        input_data={"image_base64": create_test_scene()},
        description="Detect objects and draw bounding boxes with labels"
    )


# ============================================================
# TEST 3: DINOv2 Embedding Extraction
# ============================================================

async def test_03_embedding_extraction():
    """Extract DINOv2 embeddings from image."""
    return await test_workflow(
        "DINOv2 Embedding",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "embed",
                "type": "embedding",
                "position": {"x": 350, "y": 100},
                "data": {"label": "DINOv2 Base"},
                "config": {
                    "model_id": "dinov2-base",
                    "model_source": "pretrained",
                    "normalize": True
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "embed", "sourceHandle": "image", "targetHandle": "image"}
        ],
        input_data={"image_base64": create_product_image()},
        description="Extract normalized DINOv2 embeddings"
    )


# ============================================================
# TEST 4: Multi-Model Parallel Detection
# ============================================================

async def test_04_parallel_models():
    """Run multiple YOLO models in parallel."""
    return await test_workflow(
        "Parallel Multi-Model",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 200},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "y11n",
                "type": "detection",
                "position": {"x": 350, "y": 100},
                "data": {"label": "YOLO11n"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            },
            {
                "id": "y8n",
                "type": "detection",
                "position": {"x": 350, "y": 200},
                "data": {"label": "YOLOv8n"},
                "config": {
                    "model_id": "yolov8n",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            },
            {
                "id": "y11s",
                "type": "detection",
                "position": {"x": 350, "y": 300},
                "data": {"label": "YOLO11s"},
                "config": {
                    "model_id": "yolo11s",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "y11n", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e2", "source": "img", "target": "y8n", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e3", "source": "img", "target": "y11s", "sourceHandle": "image", "targetHandle": "image"}
        ],
        input_data={"image_base64": create_test_scene()},
        description="Run YOLO11n, YOLOv8n, and YOLO11s in parallel",
        timeout=240
    )


# ============================================================
# TEST 5: Image Classification
# ============================================================

async def test_05_classification():
    """ViT image classification."""
    return await test_workflow(
        "ViT Classification",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "cls",
                "type": "classification",
                "position": {"x": 350, "y": 100},
                "data": {"label": "ViT Classifier"},
                "config": {
                    "model_id": "vit-base",
                    "model_source": "pretrained",
                    "top_k": 5
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "cls", "sourceHandle": "image", "targetHandle": "image"}
        ],
        input_data={"image_base64": create_product_image()},
        description="Classify image with Vision Transformer (top-5 predictions)"
    )


# ============================================================
# TEST 6: Detection → Filter → Count
# ============================================================

async def test_06_filter_workflow():
    """Detection with confidence filtering."""
    return await test_workflow(
        "Detection + Filter",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "detect",
                "type": "detection",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Detect All"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.25
                }
            },
            {
                "id": "filter",
                "type": "filter",
                "position": {"x": 500, "y": 100},
                "data": {"label": "High Confidence"},
                "config": {
                    "conditions": [
                        {"field": "confidence", "operator": ">=", "value": 0.7}
                    ]
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e2", "source": "detect", "target": "filter", "sourceHandle": "detections", "targetHandle": "items"}
        ],
        input_data={"image_base64": create_test_scene()},
        description="Detect objects and filter by confidence >= 0.7"
    )


# ============================================================
# TEST 7: Transform Pipeline
# ============================================================

async def test_07_transform_pipeline():
    """Image transformations before detection."""
    return await test_workflow(
        "Transform → Detect",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "resize",
                "type": "resize",
                "position": {"x": 250, "y": 100},
                "data": {"label": "Resize"},
                "config": {"width": 640, "height": 640}
            },
            {
                "id": "norm",
                "type": "normalize",
                "position": {"x": 400, "y": 100},
                "data": {"label": "Normalize"},
                "config": {}
            },
            {
                "id": "detect",
                "type": "detection",
                "position": {"x": 550, "y": 100},
                "data": {"label": "Detect"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "resize", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e2", "source": "resize", "target": "norm", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e3", "source": "norm", "target": "detect", "sourceHandle": "image", "targetHandle": "image"}
        ],
        input_data={"image_base64": create_product_image(1024, 768)},
        description="Resize → Normalize → Detect pipeline"
    )


# ============================================================
# TEST 8: Conditional Logic
# ============================================================

async def test_08_conditional():
    """Conditional branching based on detection count."""
    return await test_workflow(
        "Conditional Logic",
        nodes=[
            {
                "id": "img",
                "type": "image_input",
                "position": {"x": 100, "y": 100},
                "data": {"label": "Input"},
                "config": {}
            },
            {
                "id": "detect",
                "type": "detection",
                "position": {"x": 300, "y": 100},
                "data": {"label": "Detect"},
                "config": {
                    "model_id": "yolo11n",
                    "model_source": "pretrained",
                    "confidence": 0.5
                }
            },
            {
                "id": "cond",
                "type": "condition",
                "position": {"x": 500, "y": 100},
                "data": {"label": "Has Objects?"},
                "config": {
                    "conditions": [
                        {"field": "length", "operator": ">", "value": 0}
                    ]
                }
            }
        ],
        edges=[
            {"id": "e1", "source": "img", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
            {"id": "e2", "source": "detect", "target": "cond", "sourceHandle": "detections", "targetHandle": "value"}
        ],
        input_data={"image_base64": create_test_scene()},
        description="Check if any objects were detected"
    )


# ============================================================
# MAIN TEST RUNNER
# ============================================================

async def run_all_tests():
    """Run all real inference tests."""
    print("="*80)
    print("REAL INFERENCE TESTS - RUNPOD GPU WORKER")
    print("="*80)
    print(f"Endpoint: yz9lgcxh1rdj9o")
    print(f"GPU: 24GB Pro")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)

    tests = [
        ("Simple Detection", test_01_simple_detection),
        ("Detection + Visualization", test_02_detection_visualization),
        ("DINOv2 Embedding", test_03_embedding_extraction),
        ("Parallel Models", test_04_parallel_models),
        ("ViT Classification", test_05_classification),
        ("Detection + Filter", test_06_filter_workflow),
        ("Transform Pipeline", test_07_transform_pipeline),
        ("Conditional Logic", test_08_conditional),
    ]

    results = []

    for i, (test_name, test_func) in enumerate(tests, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST {i}/{len(tests)}")
        print(f"{'#'*80}")

        try:
            result = await test_func()
            results.append({
                "name": test_name,
                "success": result.get("success", False),
                "status": result.get("status", "unknown"),
                "duration_ms": result.get("duration_ms", 0),
                "execution_id": result.get("execution_id", ""),
                "error": result.get("error"),
            })
        except Exception as e:
            print(f"\n❌ Test {i} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": test_name,
                "success": False,
                "status": "crashed",
                "error": str(e),
            })

    # Summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY - REAL INFERENCE TESTS")
    print("="*80)

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/total*100):.1f}%")

    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        print(f"\n{i}. {result['name']} {status_icon}")
        print(f"   Status: {result['status']}")
        if result.get('duration_ms'):
            print(f"   Duration: {result['duration_ms']}ms")
        if result.get('error'):
            print(f"   Error: {result['error'][:100]}")
        if result.get('execution_id'):
            print(f"   Execution: {result['execution_id'][:8]}...")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "yz9lgcxh1rdj9o",
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "success_rate": passed / total,
        "results": results,
    }

    with open("real_inference_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Results saved to: real_inference_results.json")
    print(f"\nTest completed at: {datetime.now().isoformat()}")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
