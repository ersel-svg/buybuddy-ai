"""
Test Advanced Retail Workflow Scenarios Against Available Blocks

This script validates that all 20 advanced scenarios:
1. Use only available block types
2. Have valid edge connections (source/target nodes exist)
3. Reference valid input/output ports
4. Cover all model block types (detection, classification, embedding, segmentation, similarity_search)
"""

import sys
sys.path.insert(0, '.')

# Import available blocks
AVAILABLE_BLOCKS = [
    # Input blocks
    "image_input",
    "parameter_input",

    # Model blocks
    "detection",
    "classification",
    "embedding",
    "segmentation",
    "similarity_search",

    # Transform blocks
    "crop",
    "blur_region",
    "resize",
    "tile",
    "stitch",
    "rotate_flip",
    "normalize",
    "smoothing",

    # Visualization blocks
    "draw_boxes",
    "draw_masks",
    "heatmap",
    "comparison",

    # Logic blocks
    "condition",
    "filter",
    "foreach",
    "collect",
    "map",
    "grid_builder",

    # Output blocks
    "json_output",
    "api_response",
    "webhook",
    "aggregation",
]

# Define known output ports for each block type
BLOCK_OUTPUT_PORTS = {
    "image_input": ["image"],
    "parameter_input": ["value", "params"],

    "detection": ["detections", "first_detection", "annotated_image", "count"],
    "classification": ["predictions", "top_prediction", "confidence", "all_classes"],
    "embedding": ["embeddings", "embedding", "metadata"],
    "segmentation": ["masks", "count", "visualization"],
    "similarity_search": ["matches", "flat_matches", "best_match", "match_count"],

    "crop": ["crops", "crop", "crop_metadata"],
    "blur_region": ["image"],
    "resize": ["image", "scale_info"],
    "tile": ["tiles", "tile_info", "grid_info", "full_image"],
    "stitch": ["detections", "image", "merged", "stats"],
    "rotate_flip": ["image", "boxes", "transform_info"],
    "normalize": ["image", "tensor", "stats"],
    "smoothing": ["image", "stats"],

    "draw_boxes": ["image"],
    "draw_masks": ["image", "legend"],
    "heatmap": ["image", "result", "colorbar"],
    "comparison": ["image", "result", "diff_stats"],

    "condition": ["passed", "true_output", "false_output", "metadata"],
    "filter": ["passed", "rejected", "passed_count", "failed_count", "stats", "grouped"],
    "foreach": ["item", "index", "total", "is_first", "is_last", "context"],
    "collect": ["results", "count"],
    "map": ["items", "mapped"],
    "grid_builder": ["grid", "layout"],

    "json_output": ["json"],
    "api_response": ["response", "status_code", "headers"],
    "webhook": ["response", "status_code", "success", "request_id"],
    "aggregation": ["result", "summary", "flat"],
}

# Define known input ports for each block type
BLOCK_INPUT_PORTS = {
    "image_input": [],
    "parameter_input": [],

    "detection": ["image"],
    "classification": ["image"],
    "embedding": ["image", "images"],
    "segmentation": ["image", "boxes"],
    "similarity_search": ["embeddings", "embedding", "text_queries"],

    "crop": ["image", "detections", "detection"],
    "blur_region": ["image", "regions"],
    "resize": ["image"],
    "tile": ["image"],
    "stitch": ["detections", "tile_info", "grid_info", "tiles"],
    "rotate_flip": ["image", "boxes"],
    "normalize": ["image"],
    "smoothing": ["image"],

    "draw_boxes": ["image", "detections"],
    "draw_masks": ["image", "masks"],
    "heatmap": ["image", "heatmap"],
    "comparison": ["image_a", "image_b", "images"],

    "condition": ["value"],
    "filter": ["items"],
    "foreach": ["items", "context"],
    "collect": ["item"],
    "map": ["items"],
    "grid_builder": ["images", "items"],

    "json_output": ["data"],
    "api_response": ["data", "total_count", "errors"],
    "webhook": ["data", "url_override"],
    "aggregation": ["data"],
}


def test_scenario_blocks(scenario: dict) -> dict:
    """Test if all blocks in scenario are available."""
    pipeline = scenario.get("pipeline", {})
    nodes = pipeline.get("nodes", [])

    node_types = [node.get("type") for node in nodes]
    missing = [t for t in node_types if t not in AVAILABLE_BLOCKS]

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "node_count": len(nodes),
        "missing_blocks": missing,
        "passed": len(missing) == 0,
    }


def test_scenario_edges(scenario: dict) -> dict:
    """Test if all edges reference valid nodes."""
    pipeline = scenario.get("pipeline", {})
    nodes = pipeline.get("nodes", [])
    edges = pipeline.get("edges", [])

    node_ids = {node.get("id") for node in nodes}
    node_types = {node.get("id"): node.get("type") for node in nodes}

    errors = []

    for i, edge in enumerate(edges):
        source = edge.get("source")
        target = edge.get("target")
        source_handle = edge.get("sourceHandle")
        target_handle = edge.get("targetHandle")

        # Check node existence
        if source not in node_ids:
            errors.append(f"Edge {i}: Source node '{source}' not found")
        if target not in node_ids:
            errors.append(f"Edge {i}: Target node '{target}' not found")

        # Check output port validity
        if source in node_types:
            source_type = node_types[source]
            valid_outputs = BLOCK_OUTPUT_PORTS.get(source_type, [])
            if source_handle and source_handle not in valid_outputs:
                errors.append(f"Edge {i}: Invalid output port '{source_handle}' for {source_type} (valid: {valid_outputs})")

        # Check input port validity
        if target in node_types:
            target_type = node_types[target]
            valid_inputs = BLOCK_INPUT_PORTS.get(target_type, [])
            if target_handle and target_handle not in valid_inputs:
                errors.append(f"Edge {i}: Invalid input port '{target_handle}' for {target_type} (valid: {valid_inputs})")

    return {
        "scenario_id": scenario["id"],
        "edge_count": len(edges),
        "errors": errors,
        "passed": len(errors) == 0,
    }


def test_model_coverage(scenarios: list) -> dict:
    """Test if all model blocks are used across scenarios."""
    model_blocks = ["detection", "classification", "embedding", "segmentation", "similarity_search"]

    coverage = {block: 0 for block in model_blocks}

    for scenario in scenarios:
        pipeline = scenario.get("pipeline", {})
        nodes = pipeline.get("nodes", [])
        node_types = [node.get("type") for node in nodes]

        for block in model_blocks:
            if block in node_types:
                coverage[block] += 1

    all_covered = all(count > 0 for count in coverage.values())

    return {
        "coverage": coverage,
        "all_models_used": all_covered,
        "passed": all_covered,
    }


def test_use_case_distribution(scenarios: list) -> dict:
    """Check distribution of scenarios across use cases."""
    use_cases = {}
    for scenario in scenarios:
        uc = scenario.get("use_case", "Unknown")
        if uc not in use_cases:
            use_cases[uc] = 0
        use_cases[uc] += 1

    return {
        "use_cases": use_cases,
        "total_use_cases": len(use_cases),
        "passed": len(use_cases) >= 3,  # At least 3 different use cases
    }


def run_all_tests():
    """Run all tests on the advanced scenarios."""
    from workflow_scenarios_advanced import ADVANCED_RETAIL_SCENARIOS

    print("=" * 80)
    print("ADVANCED RETAIL WORKFLOW SCENARIOS - COMPATIBILITY TEST")
    print("=" * 80)
    print(f"\nTotal Scenarios: {len(ADVANCED_RETAIL_SCENARIOS)}")
    print(f"Available Blocks: {len(AVAILABLE_BLOCKS)}")

    # Test 1: Block Availability
    print("\n" + "-" * 40)
    print("TEST 1: Block Availability")
    print("-" * 40)

    block_results = []
    for scenario in ADVANCED_RETAIL_SCENARIOS:
        result = test_scenario_blocks(scenario)
        block_results.append(result)
        status = "✓" if result["passed"] else "✗"
        if not result["passed"]:
            print(f"  {status} {result['scenario_name']}")
            print(f"      Missing: {result['missing_blocks']}")

    block_passed = sum(1 for r in block_results if r["passed"])
    print(f"\n  Passed: {block_passed}/{len(ADVANCED_RETAIL_SCENARIOS)}")

    # Test 2: Edge Validity
    print("\n" + "-" * 40)
    print("TEST 2: Edge Validity")
    print("-" * 40)

    edge_results = []
    for scenario in ADVANCED_RETAIL_SCENARIOS:
        result = test_scenario_edges(scenario)
        edge_results.append(result)
        status = "✓" if result["passed"] else "✗"
        if not result["passed"]:
            print(f"  {status} {result['scenario_id']}: {len(result['errors'])} error(s)")
            for err in result["errors"][:3]:  # Show first 3 errors
                print(f"      - {err}")
            if len(result["errors"]) > 3:
                print(f"      ... and {len(result['errors']) - 3} more")

    edge_passed = sum(1 for r in edge_results if r["passed"])
    print(f"\n  Passed: {edge_passed}/{len(ADVANCED_RETAIL_SCENARIOS)}")

    # Test 3: Model Coverage
    print("\n" + "-" * 40)
    print("TEST 3: Model Block Coverage")
    print("-" * 40)

    coverage_result = test_model_coverage(ADVANCED_RETAIL_SCENARIOS)
    for model, count in coverage_result["coverage"].items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {model}: {count} scenarios")

    print(f"\n  All Models Used: {'Yes' if coverage_result['all_models_used'] else 'No'}")

    # Test 4: Use Case Distribution
    print("\n" + "-" * 40)
    print("TEST 4: Use Case Distribution")
    print("-" * 40)

    distribution_result = test_use_case_distribution(ADVANCED_RETAIL_SCENARIOS)
    for uc, count in distribution_result["use_cases"].items():
        print(f"  • {uc}: {count} scenarios")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_tests = 4
    tests_passed = sum([
        block_passed == len(ADVANCED_RETAIL_SCENARIOS),
        edge_passed == len(ADVANCED_RETAIL_SCENARIOS),
        coverage_result["passed"],
        distribution_result["passed"],
    ])

    print(f"\n  Block Availability: {block_passed}/{len(ADVANCED_RETAIL_SCENARIOS)} scenarios OK")
    print(f"  Edge Validity: {edge_passed}/{len(ADVANCED_RETAIL_SCENARIOS)} scenarios OK")
    print(f"  Model Coverage: {'PASS' if coverage_result['passed'] else 'FAIL'}")
    print(f"  Use Case Distribution: {'PASS' if distribution_result['passed'] else 'FAIL'}")

    print(f"\n  Overall: {tests_passed}/{total_tests} tests passed")

    # Return detailed results
    return {
        "block_results": block_results,
        "edge_results": edge_results,
        "coverage_result": coverage_result,
        "distribution_result": distribution_result,
        "all_passed": tests_passed == total_tests,
    }


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with appropriate code
    import sys
    sys.exit(0 if results["all_passed"] else 1)
