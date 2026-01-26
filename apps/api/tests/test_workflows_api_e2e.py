#!/usr/bin/env python3
"""
Real API E2E Workflow Tests
============================

Tests workflows via the actual API endpoints:
1. Create workflow via API
2. Execute workflow via API
3. Check execution status
4. Validate results

Using real inference worker and real models.
"""

import httpx
import asyncio
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

API_BASE = "http://localhost:8000/api/v1/workflows"
test_results = []


def log_test(name: str, status: str, duration: float, details: dict):
    """Log test result."""
    result = {
        "name": name,
        "status": status,
        "duration_ms": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
        **details,
    }
    test_results.append(result)

    status_emoji = "✅" if status == "PASS" else "❌"
    print(f"\n{status_emoji} {name}")
    print(f"   Duration: {duration:.2f}ms")
    if details:
        for key, value in details.items():
            if key not in ["error", "exception"]:
                print(f"   {key}: {value}")
        if "error" in details:
            print(f"   ERROR: {details['error']}")


def create_test_image_base64(width=640, height=480, color=(100, 150, 200)) -> str:
    """Create a test image as base64."""
    img = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def cleanup_workflows(client: httpx.AsyncClient, prefix: str = "TEST"):
    """Delete all test workflows."""
    try:
        response = await client.get(f"{API_BASE}/workflows")
        if response.status_code == 200:
            data = response.json()
            for workflow in data.get("workflows", []):
                if workflow["name"].startswith(prefix):
                    await client.delete(f"{API_BASE}/workflows/{workflow['id']}")
                    print(f"   Cleaned up: {workflow['name']}")
    except Exception as e:
        print(f"   Cleanup error: {e}")


# ============================================================
# TEST 1: Create Simple Detection Workflow
# ============================================================

async def test_01_create_workflow():
    """Create a simple YOLO detection workflow via API."""
    start = time.time()

    workflow_data = {
        "name": "TEST 1: Simple YOLO Detection",
        "description": "Basic object detection test",
        "definition": {
            "nodes": [
                {
                    "id": "input_1",
                    "type": "image_input",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Image Input"},
                },
                {
                    "id": "detect_1",
                    "type": "detection",
                    "position": {"x": 350, "y": 100},
                    "data": {
                        "label": "YOLO Detection",
                        "model_id": "yolo11n",
                        "model_source": "pretrained",
                        "config": {
                            "confidence": 0.5,
                            "iou_threshold": 0.45,
                        },
                    },
                },
                {
                    "id": "output_1",
                    "type": "json_output",
                    "position": {"x": 600, "y": 100},
                    "data": {"label": "JSON Output"},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "input_1",
                    "target": "detect_1",
                    "sourceHandle": "image",
                    "targetHandle": "image",
                },
                {
                    "id": "e2",
                    "source": "detect_1",
                    "target": "output_1",
                    "sourceHandle": "detections",
                    "targetHandle": "data",
                },
            ],
            "parameters": [],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE}/workflows",
                json=workflow_data,
            )

            duration = (time.time() - start) * 1000

            if response.status_code in [200, 201]:
                workflow = response.json()
                log_test(
                    "Test 1: Create Workflow",
                    "PASS",
                    duration,
                    {
                        "workflow_id": workflow["id"],
                        "status": workflow["status"],
                        "node_count": len(workflow["definition"]["nodes"]),
                    },
                )
                return workflow["id"]
            else:
                log_test(
                    "Test 1: Create Workflow",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}: {response.text}"},
                )
                return None
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 1: Create Workflow", "FAIL", duration, {"exception": str(e)})
        return None


# ============================================================
# TEST 2: Execute Simple Detection Workflow
# ============================================================

async def test_02_execute_detection(workflow_id: str):
    """Execute the detection workflow with a test image."""
    start = time.time()

    test_image = create_test_image_base64()

    execution_data = {
        "input": {
            "image_base64": test_image,
            "parameters": {},
        }
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE}/workflows/{workflow_id}/run",
                json=execution_data,
            )

            duration = (time.time() - start) * 1000

            if response.status_code in [200, 201]:
                execution = response.json()

                log_test(
                    "Test 2: Execute Detection",
                    "PASS" if execution["status"] == "completed" else "FAIL",
                    duration,
                    {
                        "execution_id": execution["id"],
                        "status": execution["status"],
                        "duration_ms": execution.get("duration_ms", 0),
                        "has_output": bool(execution.get("output_data")),
                        "error": execution.get("error_message"),
                    },
                )
                return execution["id"]
            else:
                log_test(
                    "Test 2: Execute Detection",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}: {response.text}"},
                )
                return None
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 2: Execute Detection", "FAIL", duration, {"exception": str(e)})
        return None


# ============================================================
# TEST 3: Create and Execute DINOv2 Embedding
# ============================================================

async def test_03_embedding_workflow():
    """Create and execute DINOv2 embedding workflow."""
    start = time.time()

    workflow_data = {
        "name": "TEST 3: DINOv2 Embedding",
        "description": "Extract image embeddings",
        "definition": {
            "nodes": [
                {
                    "id": "input_1",
                    "type": "image_input",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Image Input"},
                },
                {
                    "id": "embed_1",
                    "type": "embedding",
                    "position": {"x": 350, "y": 100},
                    "data": {
                        "label": "DINOv2 Embedding",
                        "model_id": "dinov2-base",
                        "model_source": "pretrained",
                        "config": {"normalize": True},
                    },
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "input_1",
                    "target": "embed_1",
                    "sourceHandle": "image",
                    "targetHandle": "image",
                },
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create workflow
            create_response = await client.post(
                f"{API_BASE}/workflows",
                json=workflow_data,
            )

            if create_response.status_code not in [200, 201]:
                duration = (time.time() - start) * 1000
                log_test(
                    "Test 3: Embedding Workflow",
                    "FAIL",
                    duration,
                    {"error": f"Create failed: {create_response.text}"},
                )
                return

            workflow = create_response.json()
            workflow_id = workflow["id"]

            # Execute workflow
            test_image = create_test_image_base64()
            exec_response = await client.post(
                f"{API_BASE}/workflows/{workflow_id}/run",
                json={"input": {"image_base64": test_image}},
            )

            duration = (time.time() - start) * 1000

            if exec_response.status_code in [200, 201]:
                execution = exec_response.json()
                log_test(
                    "Test 3: Embedding Workflow",
                    "PASS" if execution["status"] == "completed" else "FAIL",
                    duration,
                    {
                        "workflow_id": workflow_id,
                        "execution_status": execution["status"],
                        "duration_ms": execution.get("duration_ms", 0),
                        "has_embedding": "embed_1" in execution.get("output_data", {}),
                        "error": execution.get("error_message"),
                    },
                )
            else:
                log_test(
                    "Test 3: Embedding Workflow",
                    "FAIL",
                    duration,
                    {"error": f"Execute failed: {exec_response.text}"},
                )
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 3: Embedding Workflow", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 4: Detection + Filter Workflow
# ============================================================

async def test_04_detection_filter():
    """Create and execute detection with filtering."""
    start = time.time()

    workflow_data = {
        "name": "TEST 4: Detection + Filter",
        "description": "Detect objects and filter by confidence",
        "definition": {
            "nodes": [
                {
                    "id": "input_1",
                    "type": "image_input",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Image"},
                },
                {
                    "id": "detect_1",
                    "type": "detection",
                    "position": {"x": 300, "y": 100},
                    "data": {
                        "label": "Detection",
                        "model_id": "yolo11n",
                        "model_source": "pretrained",
                        "config": {"confidence": 0.25},
                    },
                },
                {
                    "id": "filter_1",
                    "type": "filter",
                    "position": {"x": 500, "y": 100},
                    "data": {
                        "label": "High Confidence Filter",
                        "config": {
                            "conditions": [
                                {"field": "confidence", "operator": ">=", "value": 0.7}
                            ]
                        },
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "input_1", "target": "detect_1"},
                {"id": "e2", "source": "detect_1", "target": "filter_1"},
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create
            create_resp = await client.post(f"{API_BASE}/workflows", json=workflow_data)
            if create_resp.status_code not in [200, 201]:
                duration = (time.time() - start) * 1000
                log_test("Test 4: Detection + Filter", "FAIL", duration, {"error": create_resp.text})
                return

            workflow_id = create_resp.json()["id"]

            # Execute
            exec_resp = await client.post(
                f"{API_BASE}/workflows/{workflow_id}/run",
                json={"input": {"image_base64": create_test_image_base64()}},
            )

            duration = (time.time() - start) * 1000

            if exec_resp.status_code in [200, 201]:
                execution = exec_resp.json()
                log_test(
                    "Test 4: Detection + Filter",
                    "PASS" if execution["status"] == "completed" else "FAIL",
                    duration,
                    {
                        "status": execution["status"],
                        "duration_ms": execution.get("duration_ms", 0),
                        "filter_worked": "filter_1" in execution.get("output_data", {}),
                        "error": execution.get("error_message"),
                    },
                )
            else:
                log_test("Test 4: Detection + Filter", "FAIL", duration, {"error": exec_resp.text})
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 4: Detection + Filter", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 5: List Workflows
# ============================================================

async def test_05_list_workflows():
    """Test listing workflows."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_BASE}/workflows")

            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                workflows = data.get("workflows", [])
                test_workflows = [w for w in workflows if w["name"].startswith("TEST")]

                log_test(
                    "Test 5: List Workflows",
                    "PASS",
                    duration,
                    {
                        "total_workflows": data.get("total", 0),
                        "test_workflows": len(test_workflows),
                        "has_pagination": "total" in data,
                    },
                )
            else:
                log_test(
                    "Test 5: List Workflows",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}"},
                )
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 5: List Workflows", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 6: List Executions
# ============================================================

async def test_06_list_executions():
    """Test listing all executions."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_BASE}/workflows/executions")

            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                executions = data.get("executions", [])

                log_test(
                    "Test 6: List Executions",
                    "PASS",
                    duration,
                    {
                        "total_executions": data.get("total", 0),
                        "recent_executions": len(executions),
                        "has_pagination": "total" in data,
                    },
                )
            else:
                log_test(
                    "Test 6: List Executions",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}"},
                )
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 6: List Executions", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 7: Get Block Registry
# ============================================================

async def test_07_block_registry():
    """Test fetching available blocks."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_BASE}/workflows/blocks")

            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                blocks = response.json()

                categories = {}
                for block in blocks:
                    cat = block.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1

                log_test(
                    "Test 7: Block Registry",
                    "PASS",
                    duration,
                    {
                        "total_blocks": len(blocks),
                        "categories": list(categories.keys()),
                        "category_counts": categories,
                    },
                )
            else:
                log_test(
                    "Test 7: Block Registry",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}"},
                )
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 7: Block Registry", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 8: Transform Pipeline (Resize + Detection)
# ============================================================

async def test_08_transform_pipeline():
    """Create and execute transform + detection pipeline."""
    start = time.time()

    workflow_data = {
        "name": "TEST 8: Transform Pipeline",
        "description": "Resize image then detect",
        "definition": {
            "nodes": [
                {
                    "id": "input_1",
                    "type": "image_input",
                    "position": {"x": 100, "y": 100},
                    "data": {"label": "Image"},
                },
                {
                    "id": "resize_1",
                    "type": "resize",
                    "position": {"x": 300, "y": 100},
                    "data": {
                        "label": "Resize",
                        "config": {"width": 640, "height": 480, "mode": "fit"},
                    },
                },
                {
                    "id": "detect_1",
                    "type": "detection",
                    "position": {"x": 500, "y": 100},
                    "data": {
                        "label": "Detection",
                        "model_id": "yolo11n",
                        "model_source": "pretrained",
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "input_1", "target": "resize_1"},
                {"id": "e2", "source": "resize_1", "target": "detect_1"},
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Create
            create_resp = await client.post(f"{API_BASE}/workflows", json=workflow_data)
            if create_resp.status_code not in [200, 201]:
                duration = (time.time() - start) * 1000
                log_test("Test 8: Transform Pipeline", "FAIL", duration, {"error": create_resp.text})
                return

            workflow_id = create_resp.json()["id"]

            # Execute with larger image
            large_image = create_test_image_base64(800, 600)
            exec_resp = await client.post(
                f"{API_BASE}/workflows/{workflow_id}/run",
                json={"input": {"image_base64": large_image}},
            )

            duration = (time.time() - start) * 1000

            if exec_resp.status_code in [200, 201]:
                execution = exec_resp.json()
                log_test(
                    "Test 8: Transform Pipeline",
                    "PASS" if execution["status"] == "completed" else "FAIL",
                    duration,
                    {
                        "status": execution["status"],
                        "duration_ms": execution.get("duration_ms", 0),
                        "resize_applied": "resize_1" in execution.get("node_metrics", {}),
                        "error": execution.get("error_message"),
                    },
                )
            else:
                log_test("Test 8: Transform Pipeline", "FAIL", duration, {"error": exec_resp.text})
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 8: Transform Pipeline", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 9: Workflow Update
# ============================================================

async def test_09_update_workflow():
    """Test updating a workflow."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create simple workflow
            create_resp = await client.post(
                f"{API_BASE}/workflows",
                json={
                    "name": "TEST 9: Update Test",
                    "description": "Original description",
                    "definition": {"nodes": [], "edges": []},
                },
            )

            if create_resp.status_code not in [200, 201]:
                duration = (time.time() - start) * 1000
                log_test("Test 9: Update Workflow", "FAIL", duration, {"error": create_resp.text})
                return

            workflow_id = create_resp.json()["id"]

            # Update workflow
            update_resp = await client.patch(
                f"{API_BASE}/workflows/{workflow_id}",
                json={
                    "name": "TEST 9: Updated Workflow",
                    "description": "Updated description",
                    "status": "active",
                },
            )

            duration = (time.time() - start) * 1000

            if update_resp.status_code == 200:
                updated = update_resp.json()
                log_test(
                    "Test 9: Update Workflow",
                    "PASS",
                    duration,
                    {
                        "name_updated": updated["name"] == "TEST 9: Updated Workflow",
                        "status_updated": updated["status"] == "active",
                        "description_updated": updated["description"] == "Updated description",
                    },
                )
            else:
                log_test("Test 9: Update Workflow", "FAIL", duration, {"error": update_resp.text})
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 9: Update Workflow", "FAIL", duration, {"exception": str(e)})


# ============================================================
# TEST 10: Get Pretrained Models
# ============================================================

async def test_10_pretrained_models():
    """Test fetching pretrained models."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_BASE}/workflows/models/pretrained")

            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                models = response.json()

                by_type = {}
                for model in models:
                    model_type = model.get("model_type", "unknown")
                    by_type[model_type] = by_type.get(model_type, 0) + 1

                log_test(
                    "Test 10: Pretrained Models",
                    "PASS",
                    duration,
                    {
                        "total_models": len(models),
                        "model_types": list(by_type.keys()),
                        "type_counts": by_type,
                    },
                )
            else:
                log_test(
                    "Test 10: Pretrained Models",
                    "FAIL",
                    duration,
                    {"error": f"HTTP {response.status_code}"},
                )
    except Exception as e:
        duration = (time.time() - start) * 1000
        log_test("Test 10: Pretrained Models", "FAIL", duration, {"exception": str(e)})


# ============================================================
# MAIN TEST RUNNER
# ============================================================

async def run_all_tests():
    """Run all API E2E tests."""
    print("=" * 80)
    print("WORKFLOW API END-TO-END TESTS")
    print("=" * 80)
    print(f"API: {API_BASE}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    # Cleanup previous test workflows
    async with httpx.AsyncClient(timeout=30.0) as client:
        await cleanup_workflows(client)

    # Run tests
    workflow_id = None

    print("\n📝 Test 1: Create Workflow")
    workflow_id = await test_01_create_workflow()

    if workflow_id:
        print("\n🚀 Test 2: Execute Detection Workflow")
        await test_02_execute_detection(workflow_id)

    print("\n🧬 Test 3: Embedding Workflow")
    await test_03_embedding_workflow()

    print("\n🔍 Test 4: Detection + Filter")
    await test_04_detection_filter()

    print("\n📋 Test 5: List Workflows")
    await test_05_list_workflows()

    print("\n📊 Test 6: List Executions")
    await test_06_list_executions()

    print("\n🧩 Test 7: Block Registry")
    await test_07_block_registry()

    print("\n🔄 Test 8: Transform Pipeline")
    await test_08_transform_pipeline()

    print("\n✏️ Test 9: Update Workflow")
    await test_09_update_workflow()

    print("\n🤖 Test 10: Pretrained Models")
    await test_10_pretrained_models()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    total_duration = sum(r["duration_ms"] for r in test_results)

    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Total Duration: {total_duration:.2f}ms")
    print(f"Average Duration: {total_duration / len(test_results) if test_results else 0:.2f}ms")

    # Save results
    output_file = Path(__file__).parent / "workflow_api_test_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "api_base": API_BASE,
                "summary": {
                    "total": len(test_results),
                    "passed": passed,
                    "failed": failed,
                    "total_duration_ms": total_duration,
                },
                "tests": test_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    # Cleanup after tests
    print("\n🧹 Cleaning up test workflows...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        await cleanup_workflows(client)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
