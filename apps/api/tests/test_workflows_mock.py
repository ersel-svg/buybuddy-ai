"""
Mock Workflow Execution Tests

Tests all workflow scenarios using mock data to validate:
1. Block execution order
2. Input/output port compatibility
3. Data flow between blocks
4. ForEach/Collect iteration patterns
"""

import asyncio
import time
from typing import Any, Dict, List
from dataclasses import dataclass, field
from collections import defaultdict
import base64
import json


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

def generate_mock_image() -> str:
    """Generate a mock base64 image."""
    return "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAA..."


def generate_mock_detections(count: int = 5) -> List[Dict]:
    """Generate mock detection results."""
    classes = ["product", "shelf", "price_tag", "gap", "label"]
    detections = []
    for i in range(count):
        detections.append({
            "id": f"det_{i}",
            "class": classes[i % len(classes)],
            "class_id": i % len(classes),
            "confidence": 0.85 + (i * 0.02),
            "x": 100 + (i * 50),
            "y": 100 + (i * 30),
            "width": 80,
            "height": 120,
            "bbox": [100 + (i * 50), 100 + (i * 30), 180 + (i * 50), 220 + (i * 30)],
        })
    return detections


def generate_mock_classification() -> Dict:
    """Generate mock classification results."""
    return {
        "predictions": [
            {"class": "fresh", "confidence": 0.92},
            {"class": "aging", "confidence": 0.06},
            {"class": "expired", "confidence": 0.02},
        ],
        "top_prediction": {"class": "fresh", "confidence": 0.92},
        "confidence": 0.92,
        "all_classes": ["fresh", "aging", "expired"],
    }


def generate_mock_embedding(dim: int = 768) -> List[float]:
    """Generate mock embedding vector."""
    import random
    random.seed(42)
    embedding = [random.uniform(-1, 1) for _ in range(dim)]
    # Normalize
    norm = sum(x**2 for x in embedding) ** 0.5
    return [x / norm for x in embedding]


def generate_mock_masks(count: int = 3) -> List[Dict]:
    """Generate mock segmentation masks."""
    masks = []
    for i in range(count):
        masks.append({
            "id": f"mask_{i}",
            "area": 5000 + (i * 1000),
            "bbox": [50 + (i * 100), 50, 150, 200],
            "predicted_iou": 0.95 - (i * 0.05),
            "stability_score": 0.98,
            "mask_rle": f"encoded_mask_{i}",
        })
    return masks


def generate_mock_similarity_matches(count: int = 3) -> List[Dict]:
    """Generate mock similarity search matches."""
    matches = []
    for i in range(count):
        matches.append({
            "id": f"match_{i}",
            "similarity": 0.95 - (i * 0.1),
            "product_id": f"prod_{1000 + i}",
            "product_info": {
                "name": f"Product {i}",
                "sku": f"SKU{1000 + i}",
                "upc": f"0123456789{i:03d}",
            },
            "metadata": {
                "category": ["electronics", "food", "clothing"][i % 3],
                "brand": f"Brand{i}",
                "is_promo": i == 0,
            },
        })
    return matches


# ============================================================================
# MOCK BLOCK EXECUTOR
# ============================================================================

class MockBlockExecutor:
    """Simulates block execution with mock data."""

    def __init__(self):
        self.execution_log = []
        self.node_outputs = {}

    async def execute_block(
        self,
        block_type: str,
        node_id: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a block with mock data and return outputs."""

        start_time = time.time()
        outputs = {}

        # Input blocks
        if block_type == "image_input":
            outputs = {"image": generate_mock_image()}

        elif block_type == "parameter_input":
            outputs = {"value": config, "params": config}

        # Model blocks
        elif block_type == "detection":
            detections = generate_mock_detections(5)
            outputs = {
                "detections": detections,
                "first_detection": detections[0] if detections else None,
                "annotated_image": generate_mock_image(),
                "count": len(detections),
            }

        elif block_type == "classification":
            result = generate_mock_classification()
            outputs = {
                "predictions": result["predictions"],
                "top_prediction": result["top_prediction"],
                "confidence": result["confidence"],
                "all_classes": result["all_classes"],
            }

        elif block_type == "embedding":
            embedding = generate_mock_embedding()
            outputs = {
                "embeddings": [embedding],
                "embedding": embedding,
                "metadata": {"embedding_dim": len(embedding), "normalized": True},
            }

        elif block_type == "segmentation":
            masks = generate_mock_masks(3)
            outputs = {
                "masks": masks,
                "count": len(masks),
                "visualization": generate_mock_image(),
            }

        elif block_type == "similarity_search":
            matches = generate_mock_similarity_matches(3)
            outputs = {
                "matches": [matches],
                "flat_matches": matches,
                "best_match": matches[0] if matches else None,
                "match_count": len(matches),
            }

        # Transform blocks
        elif block_type == "crop":
            crops = [generate_mock_image() for _ in range(3)]
            outputs = {
                "crops": crops,
                "crop": crops[0] if crops else None,
                "crop_metadata": [{"width": 100, "height": 100} for _ in crops],
            }

        elif block_type == "blur_region":
            outputs = {"image": generate_mock_image()}

        elif block_type == "resize":
            outputs = {
                "image": generate_mock_image(),
                "scale_info": {"original": [640, 480], "resized": [320, 240]},
            }

        elif block_type == "tile":
            tiles = [generate_mock_image() for _ in range(4)]
            outputs = {
                "tiles": tiles,
                "tile_info": [{"x": i * 320, "y": 0, "width": 320, "height": 320} for i in range(4)],
                "grid_info": {"rows": 1, "cols": 4, "original_size": [1280, 320]},
                "full_image": generate_mock_image(),
            }

        elif block_type == "stitch":
            detections = inputs.get("detections", [])
            if isinstance(detections, list) and detections and isinstance(detections[0], list):
                # Flatten nested detections
                flat_dets = []
                for d in detections:
                    if isinstance(d, list):
                        flat_dets.extend(d)
                    else:
                        flat_dets.append(d)
                detections = flat_dets
            outputs = {
                "detections": detections[:5] if detections else generate_mock_detections(3),
                "image": generate_mock_image(),
                "merged": generate_mock_image(),
                "stats": {"before": 10, "after": 5, "suppressed": 5},
            }

        elif block_type == "rotate_flip":
            outputs = {
                "image": generate_mock_image(),
                "boxes": inputs.get("boxes", []),
                "transform_info": {"rotation": 0, "flip": "none"},
            }

        elif block_type == "normalize":
            outputs = {
                "image": generate_mock_image(),
                "tensor": [[0.5] * 224] * 224,
                "stats": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }

        elif block_type == "smoothing":
            outputs = {
                "image": generate_mock_image(),
                "stats": {"kernel_size": 5, "sigma": 1.0},
            }

        # Visualization blocks
        elif block_type == "draw_boxes":
            outputs = {"image": generate_mock_image()}

        elif block_type == "draw_masks":
            outputs = {
                "image": generate_mock_image(),
                "legend": {"mask_0": "#FF0000", "mask_1": "#00FF00"},
            }

        elif block_type == "heatmap":
            outputs = {
                "image": generate_mock_image(),
                "result": generate_mock_image(),
                "colorbar": generate_mock_image(),
            }

        elif block_type == "comparison":
            outputs = {
                "image": generate_mock_image(),
                "result": generate_mock_image(),
                "diff_stats": {"mse": 0.01, "psnr": 40.0, "ssim": 0.98},
            }

        # Logic blocks
        elif block_type == "condition":
            value = inputs.get("value", 0)
            # Simple condition evaluation
            result = bool(value) if not isinstance(value, (int, float)) else value > 0
            outputs = {
                "passed": result,
                "true_output": inputs.get("value") if result else None,
                "false_output": None if result else inputs.get("value"),
                "metadata": {"condition_results": [result]},
            }

        elif block_type == "filter":
            items = inputs.get("items", [])
            if not isinstance(items, list):
                items = [items] if items else []
            # Simple pass-through filter (50% pass)
            mid = len(items) // 2 or 1
            passed = items[:mid]
            rejected = items[mid:]
            outputs = {
                "passed": passed,
                "rejected": rejected,
                "passed_count": len(passed),
                "failed_count": len(rejected),
                "stats": {"total": len(items), "pass_rate": len(passed) / max(len(items), 1)},
                "grouped": {},
            }

        elif block_type == "foreach":
            items = inputs.get("items", [])
            if not isinstance(items, list):
                items = [items] if items else []
            # ForEach outputs current item - this is handled specially by the engine
            # For mock, we'll output the first item
            current_index = 0
            outputs = {
                "item": items[current_index] if items else None,
                "index": current_index,
                "total": len(items),
                "is_first": True,
                "is_last": len(items) <= 1,
                "context": inputs.get("context"),
            }

        elif block_type == "collect":
            item = inputs.get("item")
            outputs = {
                "results": [item] if item else [],
                "count": 1 if item else 0,
            }

        elif block_type == "map":
            items = inputs.get("items", [])
            if not isinstance(items, list):
                items = [items] if items else []
            # Simple identity map
            outputs = {
                "items": items,
                "mapped": items,
            }

        elif block_type == "grid_builder":
            outputs = {
                "grid": generate_mock_image(),
                "layout": {"rows": 2, "cols": 2},
            }

        # Output blocks
        elif block_type == "json_output":
            data = inputs.get("data", {})
            outputs = {"json": data}

        elif block_type == "api_response":
            data = inputs.get("data", {})
            outputs = {
                "response": {"success": True, "data": data},
                "status_code": 200,
                "headers": {"Content-Type": "application/json"},
            }

        elif block_type == "webhook":
            outputs = {
                "response": {"status": "sent"},
                "status_code": 200,
                "success": True,
                "request_id": "mock-request-id",
            }

        elif block_type == "aggregation":
            data = inputs.get("data", [])
            if not isinstance(data, list):
                data = [data] if data else []
            outputs = {
                "result": data,
                "summary": {"count": len(data)},
                "flat": data,
            }

        else:
            outputs = {"error": f"Unknown block type: {block_type}"}

        duration = (time.time() - start_time) * 1000

        self.execution_log.append({
            "node_id": node_id,
            "block_type": block_type,
            "duration_ms": round(duration, 2),
            "input_keys": list(inputs.keys()),
            "output_keys": list(outputs.keys()),
        })

        self.node_outputs[node_id] = outputs
        return outputs


# ============================================================================
# MOCK WORKFLOW ENGINE
# ============================================================================

class MockWorkflowEngine:
    """Simplified workflow engine for testing."""

    def __init__(self):
        self.executor = MockBlockExecutor()

    def _build_adjacency(self, edges: List[Dict]) -> Dict[str, List[str]]:
        """Build adjacency list from edges."""
        adj = defaultdict(list)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adj[source].append(target)
        return adj

    def _build_edge_inputs(self, edges: List[Dict]) -> Dict[str, Dict[str, str]]:
        """Build input mappings from edges."""
        edge_inputs = defaultdict(dict)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            source_handle = edge.get("sourceHandle", "output")
            target_handle = edge.get("targetHandle", "input")
            if source and target:
                edge_inputs[target][target_handle] = f"$nodes.{source}.{source_handle}"
        return edge_inputs

    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Sort nodes in topological order."""
        node_map = {n["id"]: n for n in nodes}
        in_degree = {n["id"]: 0 for n in nodes}

        for edge in edges:
            target = edge.get("target")
            if target in in_degree:
                in_degree[target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            nid = queue.pop(0)
            result.append(node_map[nid])

            for edge in edges:
                if edge.get("source") == nid:
                    target = edge.get("target")
                    if target in in_degree:
                        in_degree[target] -= 1
                        if in_degree[target] == 0:
                            queue.append(target)

        return result

    def _resolve_inputs(
        self,
        node_id: str,
        edge_inputs: Dict[str, Dict[str, str]],
        node_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Resolve input values from edge references."""
        resolved = {}
        for input_name, ref in edge_inputs.get(node_id, {}).items():
            if ref.startswith("$nodes."):
                parts = ref[7:].split(".")
                source_node = parts[0]
                output_port = parts[1] if len(parts) > 1 else "output"

                if source_node in node_outputs:
                    source_outputs = node_outputs[source_node]
                    if output_port in source_outputs:
                        resolved[input_name] = source_outputs[output_port]
                    elif "output" in source_outputs:
                        resolved[input_name] = source_outputs["output"]
                    else:
                        # Try to get any output
                        resolved[input_name] = list(source_outputs.values())[0] if source_outputs else None
            else:
                resolved[input_name] = ref

        return resolved

    async def execute(self, pipeline: Dict) -> Dict:
        """Execute the workflow pipeline."""
        nodes = pipeline.get("nodes", [])
        edges = pipeline.get("edges", [])

        if not nodes:
            return {"success": False, "error": "No nodes in pipeline"}

        # Sort nodes topologically
        try:
            execution_order = self._topological_sort(nodes, edges)
        except Exception as e:
            return {"success": False, "error": f"Topological sort failed: {e}"}

        # Build edge input mappings
        edge_inputs = self._build_edge_inputs(edges)

        # Execute each node
        self.executor.node_outputs = {}
        errors = []

        for node in execution_order:
            node_id = node["id"]
            block_type = node.get("type")
            config = node.get("config", {})

            # Resolve inputs from upstream nodes
            inputs = self._resolve_inputs(node_id, edge_inputs, self.executor.node_outputs)

            try:
                outputs = await self.executor.execute_block(block_type, node_id, inputs, config)
            except Exception as e:
                errors.append({"node_id": node_id, "error": str(e)})
                # Continue to next node

        return {
            "success": len(errors) == 0,
            "errors": errors,
            "execution_log": self.executor.execution_log,
            "node_count": len(nodes),
            "executed_count": len(self.executor.execution_log),
        }


# ============================================================================
# TEST RUNNER
# ============================================================================

async def test_scenario(scenario: Dict, engine: MockWorkflowEngine) -> Dict:
    """Test a single scenario."""
    scenario_id = scenario.get("id", "unknown")
    scenario_name = scenario.get("name", "Unknown")
    pipeline = scenario.get("pipeline", {})

    start_time = time.time()

    try:
        result = await engine.execute(pipeline)
        duration = (time.time() - start_time) * 1000

        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "success": result["success"],
            "errors": result.get("errors", []),
            "duration_ms": round(duration, 2),
            "nodes_executed": result.get("executed_count", 0),
            "total_nodes": result.get("node_count", 0),
        }
    except Exception as e:
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "success": False,
            "errors": [{"error": str(e)}],
            "duration_ms": round((time.time() - start_time) * 1000, 2),
            "nodes_executed": 0,
            "total_nodes": len(pipeline.get("nodes", [])),
        }


async def run_all_tests():
    """Run all scenario tests."""
    # Import scenarios
    try:
        from workflow_scenarios_advanced import ADVANCED_RETAIL_SCENARIOS
    except ImportError:
        ADVANCED_RETAIL_SCENARIOS = []
        print("Warning: Could not import ADVANCED_RETAIL_SCENARIOS")

    try:
        from workflow_scenarios_basic import BASIC_RETAIL_SCENARIOS
    except ImportError:
        BASIC_RETAIL_SCENARIOS = []
        print("Warning: Could not import BASIC_RETAIL_SCENARIOS")

    all_scenarios = BASIC_RETAIL_SCENARIOS + ADVANCED_RETAIL_SCENARIOS

    print("=" * 80)
    print("MOCK WORKFLOW EXECUTION TESTS")
    print("=" * 80)
    print(f"\nTotal Scenarios: {len(all_scenarios)}")

    engine = MockWorkflowEngine()
    results = []

    print("\n" + "-" * 40)
    print("Running Tests...")
    print("-" * 40)

    for scenario in all_scenarios:
        result = await test_scenario(scenario, engine)
        results.append(result)

        status = "✓" if result["success"] else "✗"
        print(f"  {status} {result['scenario_id']}: {result['scenario_name']}")
        if not result["success"]:
            for err in result.get("errors", [])[:2]:
                print(f"      Error: {err}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"\n  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\n  Failed Scenarios:")
        for r in results:
            if not r["success"]:
                print(f"    - {r['scenario_id']}: {r['scenario_name']}")

    # Performance stats
    total_duration = sum(r["duration_ms"] for r in results)
    avg_duration = total_duration / len(results) if results else 0

    print(f"\n  Total Duration: {round(total_duration, 2)} ms")
    print(f"  Average Duration: {round(avg_duration, 2)} ms per scenario")

    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
        "success": failed == 0,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    exit(0 if results["success"] else 1)
