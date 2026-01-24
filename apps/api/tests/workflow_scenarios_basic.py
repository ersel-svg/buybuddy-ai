"""
Basic Retail Workflow Scenarios (20 scenarios)

These are foundational retail workflow patterns that test
basic block connectivity and simple pipelines.
"""

BASIC_RETAIL_SCENARIOS = [
    # ==========================================================================
    # BASIC DETECTION & IDENTIFICATION (1-5)
    # ==========================================================================

    {
        "id": "basic_scenario_1",
        "name": "Simple Product Detection",
        "description": "Detect products on shelf and output results",
        "use_case": "Basic Detection",
        "blocks_used": ["detection", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "output", "sourceHandle": "detections", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_2",
        "name": "Detection with Visualization",
        "description": "Detect and draw bounding boxes",
        "use_case": "Basic Detection",
        "blocks_used": ["detection", "draw_boxes", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "draw", "type": "draw_boxes"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "input", "target": "draw", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "draw", "sourceHandle": "detections", "targetHandle": "detections"},
                {"source": "draw", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_3",
        "name": "Single Product Identification",
        "description": "Detect first product and identify via similarity search",
        "use_case": "Product Identification",
        "blocks_used": ["detection", "crop", "embedding", "similarity_search", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "crop", "type": "crop"},
                {"id": "embed", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "search", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "crop", "sourceHandle": "first_detection", "targetHandle": "detection"},
                {"source": "input", "target": "crop", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "crop", "target": "embed", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed", "target": "search", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "search", "target": "output", "sourceHandle": "best_match", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_4",
        "name": "Batch Product Identification",
        "description": "Detect all products and identify each",
        "use_case": "Product Identification",
        "blocks_used": ["detection", "foreach", "crop", "embedding", "similarity_search", "collect", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "foreach", "type": "foreach"},
                {"id": "crop", "type": "crop"},
                {"id": "embed", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "search", "type": "similarity_search", "config": {"collection": "products"}},
                {"id": "collect", "type": "collect"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "foreach", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "input", "target": "foreach", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach", "target": "crop", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach", "target": "crop", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop", "target": "embed", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed", "target": "search", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "search", "target": "collect", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect", "target": "output", "sourceHandle": "results", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_5",
        "name": "Product Classification",
        "description": "Classify detected products into categories",
        "use_case": "Classification",
        "blocks_used": ["detection", "foreach", "crop", "classification", "collect", "aggregation", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "foreach", "type": "foreach"},
                {"id": "crop", "type": "crop"},
                {"id": "classify", "type": "classification", "config": {"model_id": "product-classifier"}},
                {"id": "collect", "type": "collect"},
                {"id": "aggregate", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "class"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "foreach", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "input", "target": "foreach", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach", "target": "crop", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach", "target": "crop", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop", "target": "classify", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "classify", "target": "collect", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "collect", "target": "aggregate", "sourceHandle": "results", "targetHandle": "data"},
                {"source": "aggregate", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
    },

    # ==========================================================================
    # FILTERING & CONDITIONS (6-10)
    # ==========================================================================

    {
        "id": "basic_scenario_6",
        "name": "Filter High Confidence Detections",
        "description": "Filter detections by confidence threshold",
        "use_case": "Filtering",
        "blocks_used": ["detection", "filter", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "filter", "type": "filter", "config": {"conditions": [{"field": "confidence", "operator": "gte", "value": 0.8}]}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "filter", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "filter", "target": "output", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_7",
        "name": "Conditional Processing",
        "description": "Check if products found and branch accordingly",
        "use_case": "Conditional Logic",
        "blocks_used": ["detection", "condition", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "check", "type": "condition", "config": {"conditions": [{"field": "count", "operator": "gt", "value": 0}]}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "check", "sourceHandle": "count", "targetHandle": "value"},
                {"source": "check", "target": "output", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_8",
        "name": "Filter by Class",
        "description": "Filter detections to specific product classes",
        "use_case": "Filtering",
        "blocks_used": ["detection", "filter", "draw_boxes", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "filter", "type": "filter", "config": {"conditions": [{"field": "class", "operator": "in", "value": ["product", "bottle"]}]}},
                {"id": "draw", "type": "draw_boxes"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "filter", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "input", "target": "draw", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "filter", "target": "draw", "sourceHandle": "passed", "targetHandle": "detections"},
                {"source": "draw", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_9",
        "name": "Count Products per Category",
        "description": "Aggregate products by detected class",
        "use_case": "Aggregation",
        "blocks_used": ["detection", "aggregation", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "aggregate", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "class", "agg_func": "count"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "aggregate", "sourceHandle": "detections", "targetHandle": "data"},
                {"source": "aggregate", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_10",
        "name": "Top K Detections",
        "description": "Get top K most confident detections",
        "use_case": "Filtering",
        "blocks_used": ["detection", "aggregation", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "top_k", "type": "aggregation", "config": {"operation": "top_n", "top_n_config": {"n": 5, "by": "confidence", "order": "desc"}}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "top_k", "sourceHandle": "detections", "targetHandle": "data"},
                {"source": "top_k", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
    },

    # ==========================================================================
    # IMAGE TRANSFORMS (11-15)
    # ==========================================================================

    {
        "id": "basic_scenario_11",
        "name": "Resize and Detect",
        "description": "Resize image before detection",
        "use_case": "Image Processing",
        "blocks_used": ["resize", "detection", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "resize", "type": "resize", "config": {"width": 640, "height": 640}},
                {"id": "detect", "type": "detection"},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "resize", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "resize", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "output", "sourceHandle": "detections", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_12",
        "name": "Tile Large Image",
        "description": "Tile large image for better detection",
        "use_case": "Image Processing",
        "blocks_used": ["tile", "foreach", "detection", "collect", "stitch", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "tile", "type": "tile", "config": {"tile_size": 640}},
                {"id": "foreach", "type": "foreach"},
                {"id": "detect", "type": "detection"},
                {"id": "collect", "type": "collect"},
                {"id": "stitch", "type": "stitch", "config": {"merge_mode": "nms"}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "tile", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "tile", "target": "foreach", "sourceHandle": "tiles", "targetHandle": "items"},
                {"source": "foreach", "target": "detect", "sourceHandle": "item", "targetHandle": "image"},
                {"source": "detect", "target": "collect", "sourceHandle": "detections", "targetHandle": "item"},
                {"source": "collect", "target": "stitch", "sourceHandle": "results", "targetHandle": "detections"},
                {"source": "tile", "target": "stitch", "sourceHandle": "tile_info", "targetHandle": "tile_info"},
                {"source": "tile", "target": "stitch", "sourceHandle": "grid_info", "targetHandle": "grid_info"},
                {"source": "stitch", "target": "output", "sourceHandle": "detections", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_13",
        "name": "Blur Sensitive Regions",
        "description": "Detect and blur price tags",
        "use_case": "Image Processing",
        "blocks_used": ["detection", "filter", "blur_region", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "filter", "type": "filter", "config": {"conditions": [{"field": "class", "operator": "eq", "value": "price_tag"}]}},
                {"id": "blur", "type": "blur_region"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "filter", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "input", "target": "blur", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "filter", "target": "blur", "sourceHandle": "passed", "targetHandle": "regions"},
                {"source": "blur", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_14",
        "name": "Normalize for ML",
        "description": "Normalize image for model input",
        "use_case": "Image Processing",
        "blocks_used": ["normalize", "embedding", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "normalize", "type": "normalize", "config": {"preset": "imagenet"}},
                {"id": "embed", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "input", "target": "normalize", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "normalize", "target": "embed", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "embed", "target": "output", "sourceHandle": "embedding", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_15",
        "name": "Smoothing Before Detection",
        "description": "Apply smoothing to reduce noise",
        "use_case": "Image Processing",
        "blocks_used": ["smoothing", "detection", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "smooth", "type": "smoothing", "config": {"smoothing_type": "gaussian", "kernel_size": 3}},
                {"id": "detect", "type": "detection"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "smooth", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "smooth", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "output", "sourceHandle": "detections", "targetHandle": "data"},
            ],
        },
    },

    # ==========================================================================
    # OUTPUT & INTEGRATION (16-20)
    # ==========================================================================

    {
        "id": "basic_scenario_16",
        "name": "Webhook Alert",
        "description": "Send detection results to webhook",
        "use_case": "Integration",
        "blocks_used": ["detection", "filter", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "filter", "type": "filter", "config": {"conditions": [{"field": "confidence", "operator": "gte", "value": 0.9}]}},
                {"id": "webhook", "type": "webhook", "config": {"url": "https://api.example.com/alerts"}},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "filter", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "filter", "target": "webhook", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_17",
        "name": "API Response with Pagination",
        "description": "Format results as paginated API response",
        "use_case": "Output",
        "blocks_used": ["detection", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "output", "type": "api_response", "config": {"pagination_enabled": True, "per_page": 10}},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "output", "sourceHandle": "detections", "targetHandle": "data"},
                {"source": "detect", "target": "output", "sourceHandle": "count", "targetHandle": "total_count"},
            ],
        },
    },

    {
        "id": "basic_scenario_18",
        "name": "Segmentation Visualization",
        "description": "Segment products and draw masks",
        "use_case": "Visualization",
        "blocks_used": ["segmentation", "draw_masks", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "segment", "type": "segmentation", "config": {"model_id": "sam-base"}},
                {"id": "draw", "type": "draw_masks"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "segment", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "input", "target": "draw", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "segment", "target": "draw", "sourceHandle": "masks", "targetHandle": "masks"},
                {"source": "draw", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_19",
        "name": "Image Comparison",
        "description": "Compare two shelf images",
        "use_case": "Visualization",
        "blocks_used": ["comparison", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "image_a", "type": "image_input"},
                {"id": "image_b", "type": "image_input"},
                {"id": "compare", "type": "comparison", "config": {"mode": "side_by_side"}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "image_a", "target": "compare", "sourceHandle": "image", "targetHandle": "image_a"},
                {"source": "image_b", "target": "compare", "sourceHandle": "image", "targetHandle": "image_b"},
                {"source": "compare", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
    },

    {
        "id": "basic_scenario_20",
        "name": "Full Retail Pipeline",
        "description": "Complete pipeline: detect, identify, filter, visualize, output",
        "use_case": "Complete Pipeline",
        "blocks_used": ["detection", "foreach", "crop", "embedding", "similarity_search", "collect", "filter", "draw_boxes", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "input", "type": "image_input"},
                {"id": "detect", "type": "detection"},
                {"id": "foreach", "type": "foreach"},
                {"id": "crop", "type": "crop"},
                {"id": "embed", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "search", "type": "similarity_search", "config": {"collection": "products"}},
                {"id": "collect", "type": "collect"},
                {"id": "filter", "type": "filter", "config": {"conditions": [{"field": "similarity", "operator": "gte", "value": 0.8}]}},
                {"id": "draw", "type": "draw_boxes"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "input", "target": "detect", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "foreach", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "input", "target": "foreach", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach", "target": "crop", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach", "target": "crop", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop", "target": "embed", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed", "target": "search", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "search", "target": "collect", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect", "target": "filter", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "input", "target": "draw", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect", "target": "draw", "sourceHandle": "detections", "targetHandle": "detections"},
                {"source": "filter", "target": "output", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
    },
]


if __name__ == "__main__":
    print(f"Total Basic Scenarios: {len(BASIC_RETAIL_SCENARIOS)}")

    # Count block usage
    block_usage = {}
    for scenario in BASIC_RETAIL_SCENARIOS:
        for block in scenario.get("blocks_used", []):
            block_usage[block] = block_usage.get(block, 0) + 1

    print("\nBlock Usage:")
    for block, count in sorted(block_usage.items(), key=lambda x: -x[1]):
        print(f"  {block}: {count} scenarios")
