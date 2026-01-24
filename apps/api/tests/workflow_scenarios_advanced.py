"""
Advanced Retail Workflow Scenarios - Testing Complex Multi-Model Pipelines

These scenarios test real-world retail use cases combining:
- Detection + Classification + Embedding + SimilaritySearch
- Planogram compliance checking
- Stock level management
- Product availability monitoring
- Multi-stage visual inspection
"""

ADVANCED_RETAIL_SCENARIOS = [
    # ==========================================================================
    # PLANOGRAM COMPLIANCE SCENARIOS (1-5)
    # ==========================================================================

    {
        "id": "adv_scenario_1",
        "name": "Full Planogram Compliance Check",
        "description": "Detect products, identify them via similarity search, compare with planogram positions",
        "use_case": "Planogram Management",
        "blocks_used": ["detection", "crop", "embedding", "similarity_search", "filter", "condition", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "planogram_params", "type": "parameter_input", "config": {"planogram_id": "string", "tolerance_cm": "number"}},
                # Detect all products on shelf
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail", "confidence_threshold": 0.5}},
                # Process each detection
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop", "config": {"padding": 5}},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base", "normalize": True}},
                {"id": "find_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1, "threshold": 0.85}},
                {"id": "collect_matches", "type": "collect"},
                # Filter successful identifications
                {"id": "filter_identified", "type": "filter", "config": {"conditions": [{"field": "best_match", "operator": "exists"}]}},
                # Check if positions match planogram
                {"id": "check_compliance", "type": "condition", "config": {"conditions": [{"field": "passed_count", "operator": "gte", "value": "$nodes.detect_products.count * 0.9"}]}},
                {"id": "output", "type": "api_response", "config": {"format": "standard"}},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "find_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "find_product", "target": "collect_matches", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_matches", "target": "filter_identified", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_identified", "target": "check_compliance", "sourceHandle": "passed_count", "targetHandle": "value"},
                {"source": "filter_identified", "target": "output", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → embedding → similarity_search → collect → filter → condition → api_response",
    },

    {
        "id": "adv_scenario_2",
        "name": "Planogram Gap Detection with Segmentation",
        "description": "Use segmentation to find empty shelf spaces, classify gap severity",
        "use_case": "Planogram Management",
        "blocks_used": ["detection", "segmentation", "classification", "filter", "aggregation", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Detect shelf structure
                {"id": "detect_shelves", "type": "detection", "config": {"model_id": "shelf-detector", "confidence_threshold": 0.6}},
                # Segment empty areas
                {"id": "segment_gaps", "type": "segmentation", "config": {"model_id": "sam-base", "prompt_mode": "auto"}},
                # Classify gap severity (small/medium/large/critical)
                {"id": "foreach_gap", "type": "foreach"},
                {"id": "crop_gap", "type": "crop", "config": {"padding": 10}},
                {"id": "classify_severity", "type": "classification", "config": {"model_id": "gap-severity-classifier"}},
                {"id": "collect_classified", "type": "collect"},
                # Filter critical gaps
                {"id": "filter_critical", "type": "filter", "config": {"conditions": [{"field": "class", "operator": "in", "value": ["large", "critical"]}]}},
                # Aggregate statistics
                {"id": "aggregate_stats", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "class", "agg_func": "count"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_shelves", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "shelf_image", "target": "segment_gaps", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "segment_gaps", "target": "foreach_gap", "sourceHandle": "masks", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_gap", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_gap", "target": "crop_gap", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_gap", "target": "crop_gap", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_gap", "target": "classify_severity", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "classify_severity", "target": "collect_classified", "sourceHandle": "predictions", "targetHandle": "item"},
                {"source": "collect_classified", "target": "filter_critical", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_critical", "target": "aggregate_stats", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "aggregate_stats", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection + segmentation → foreach → crop → classification → collect → filter → aggregation → json_output",
    },

    {
        "id": "adv_scenario_3",
        "name": "Planogram Position Accuracy Heatmap",
        "description": "Generate heatmap showing correct vs incorrect product placements",
        "use_case": "Planogram Management",
        "blocks_used": ["detection", "embedding", "similarity_search", "map", "heatmap", "comparison"],
        "pipeline": {
            "nodes": [
                {"id": "current_shelf", "type": "image_input"},
                {"id": "reference_planogram", "type": "image_input"},
                # Detect and identify products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "match_product", "type": "similarity_search", "config": {"collection": "planogram_products", "top_k": 1}},
                {"id": "collect_matches", "type": "collect"},
                # Map to compliance scores (1.0 = correct position, 0.0 = wrong)
                {"id": "map_scores", "type": "map", "config": {"expression": "item.similarity > 0.9 ? 1.0 : 0.0", "output_field": "compliance_score"}},
                # Generate heatmap
                {"id": "create_heatmap", "type": "heatmap", "config": {"colormap": "turbo", "opacity": 0.7}},
                # Compare with reference
                {"id": "compare_views", "type": "comparison", "config": {"mode": "side_by_side", "label_a": "Current", "label_b": "Reference"}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "current_shelf", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "current_shelf", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "match_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "match_product", "target": "collect_matches", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_matches", "target": "map_scores", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "map_scores", "target": "create_heatmap", "sourceHandle": "items", "targetHandle": "heatmap"},
                {"source": "current_shelf", "target": "create_heatmap", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "create_heatmap", "target": "compare_views", "sourceHandle": "image", "targetHandle": "image_a"},
                {"source": "reference_planogram", "target": "compare_views", "sourceHandle": "image", "targetHandle": "image_b"},
                {"source": "compare_views", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → embedding → similarity_search → collect → map → heatmap → comparison → json_output",
    },

    {
        "id": "adv_scenario_4",
        "name": "Multi-Shelf Planogram Audit with Tiling",
        "description": "Handle large shelf images using tiling, detect products across multiple shelves",
        "use_case": "Planogram Management",
        "blocks_used": ["tile", "detection", "stitch", "embedding", "similarity_search", "aggregation", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "full_aisle", "type": "image_input"},
                # Tile large image for better detection
                {"id": "tile_image", "type": "tile", "config": {"tile_size": 640, "overlap_ratio": 0.2}},
                # Detect products in each tile
                {"id": "foreach_tile", "type": "foreach"},
                {"id": "detect_tile", "type": "detection", "config": {"model_id": "yolov8-retail", "confidence_threshold": 0.4}},
                {"id": "collect_detections", "type": "collect", "config": {"flatten": True}},
                # Stitch detections back together with NMS
                {"id": "stitch_results", "type": "stitch", "config": {"merge_mode": "nms", "iou_threshold": 0.5}},
                # Identify each product
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "clip-vit-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "product_catalog", "top_k": 1}},
                {"id": "collect_identified", "type": "collect"},
                # Aggregate by product category
                {"id": "aggregate_categories", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "best_match.metadata.category", "agg_func": "count"}}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "full_aisle", "target": "tile_image", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "tile_image", "target": "foreach_tile", "sourceHandle": "tiles", "targetHandle": "items"},
                {"source": "foreach_tile", "target": "detect_tile", "sourceHandle": "item", "targetHandle": "image"},
                {"source": "detect_tile", "target": "collect_detections", "sourceHandle": "detections", "targetHandle": "item"},
                {"source": "collect_detections", "target": "stitch_results", "sourceHandle": "results", "targetHandle": "detections"},
                {"source": "tile_image", "target": "stitch_results", "sourceHandle": "tile_info", "targetHandle": "tile_info"},
                {"source": "tile_image", "target": "stitch_results", "sourceHandle": "grid_info", "targetHandle": "grid_info"},
                {"source": "stitch_results", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "full_aisle", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_product", "target": "collect_identified", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_identified", "target": "aggregate_categories", "sourceHandle": "results", "targetHandle": "data"},
                {"source": "aggregate_categories", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → tile → foreach → detection → collect → stitch → foreach → crop → embedding → similarity_search → collect → aggregation → api_response",
    },

    {
        "id": "adv_scenario_5",
        "name": "Facing Count Validation",
        "description": "Count product facings, validate against planogram requirements",
        "use_case": "Planogram Management",
        "blocks_used": ["detection", "embedding", "similarity_search", "aggregation", "filter", "condition", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"store_id": "string", "shelf_id": "string"}},
                # Detect all product facings
                {"id": "detect_facings", "type": "detection", "config": {"model_id": "yolov8-retail", "confidence_threshold": 0.5}},
                # Identify each facing
                {"id": "foreach_facing", "type": "foreach"},
                {"id": "crop_facing", "type": "crop"},
                {"id": "embed_facing", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_facing", "type": "similarity_search", "config": {"collection": "products", "top_k": 1, "threshold": 0.8}},
                {"id": "collect_identified", "type": "collect"},
                # Count facings per product
                {"id": "count_facings", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "best_match.product_info.sku", "agg_func": "count"}}},
                # Filter products below minimum facing requirement
                {"id": "filter_low_facings", "type": "filter", "config": {"conditions": [{"field": "count", "operator": "lt", "value": 2}]}},
                # Check if any violations
                {"id": "check_violations", "type": "condition", "config": {"conditions": [{"field": "failed_count", "operator": "gt", "value": 0}]}},
                # Alert if violations found
                {"id": "send_alert", "type": "webhook", "config": {"url": "https://api.store.com/alerts", "method": "POST"}},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_facings", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_facings", "target": "foreach_facing", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_facing", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_facing", "target": "crop_facing", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_facing", "target": "crop_facing", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_facing", "target": "embed_facing", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_facing", "target": "identify_facing", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_facing", "target": "collect_identified", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_identified", "target": "count_facings", "sourceHandle": "results", "targetHandle": "data"},
                {"source": "count_facings", "target": "filter_low_facings", "sourceHandle": "result", "targetHandle": "items"},
                {"source": "filter_low_facings", "target": "check_violations", "sourceHandle": "failed_count", "targetHandle": "value"},
                {"source": "filter_low_facings", "target": "send_alert", "sourceHandle": "rejected", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → embedding → similarity_search → collect → aggregation → filter → condition → webhook",
    },

    # ==========================================================================
    # STOCK MANAGEMENT SCENARIOS (6-10)
    # ==========================================================================

    {
        "id": "adv_scenario_6",
        "name": "Real-time Stock Level Estimation",
        "description": "Estimate stock levels by counting visible products and calculating depth",
        "use_case": "Stock Management",
        "blocks_used": ["detection", "classification", "embedding", "similarity_search", "map", "aggregation", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"shelf_depth": "number", "avg_product_depth": "number"}},
                # Detect products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Classify stock level appearance (full/partial/low)
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "classify_stock", "type": "classification", "config": {"model_id": "stock-level-classifier"}},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_data", "type": "collect"},
                # Map to estimated quantities
                {"id": "map_quantities", "type": "map", "config": {"expression": "item.stock_class == 'full' ? 10 : (item.stock_class == 'partial' ? 5 : 2)", "output_field": "estimated_qty"}},
                # Aggregate by product
                {"id": "aggregate_stock", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "product_id", "agg_field": "estimated_qty", "agg_func": "sum"}}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_stock", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_stock", "target": "collect_data", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_data", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_data", "target": "map_quantities", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "map_quantities", "target": "aggregate_stock", "sourceHandle": "items", "targetHandle": "data"},
                {"source": "aggregate_stock", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + embedding → similarity_search → collect → map → aggregation → api_response",
    },

    {
        "id": "adv_scenario_7",
        "name": "Out-of-Stock Detection with Segmentation",
        "description": "Use segmentation to identify empty shelf areas and map to products that should be there",
        "use_case": "Stock Management",
        "blocks_used": ["detection", "segmentation", "embedding", "similarity_search", "filter", "aggregation", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"planogram_id": "string", "alert_threshold": "number"}},
                # Detect shelf structure and products
                {"id": "detect_structure", "type": "detection", "config": {"model_id": "shelf-structure-detector"}},
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Segment empty areas
                {"id": "segment_empty", "type": "segmentation", "config": {"model_id": "sam-base", "prompt_mode": "box"}},
                # Find which products should be in empty areas
                {"id": "foreach_empty", "type": "foreach"},
                {"id": "crop_empty_area", "type": "crop"},
                {"id": "embed_area", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "find_expected", "type": "similarity_search", "config": {"collection": "planogram_positions", "top_k": 3}},
                {"id": "collect_missing", "type": "collect"},
                # Filter high-confidence missing products
                {"id": "filter_confident", "type": "filter", "config": {"conditions": [{"field": "best_match.similarity", "operator": "gte", "value": 0.7}]}},
                # Aggregate missing products
                {"id": "aggregate_missing", "type": "aggregation", "config": {"operation": "dedupe", "dedupe_config": {"by": "product_id", "keep": "best"}}},
                # Send OOS alerts
                {"id": "send_oos_alert", "type": "webhook", "config": {"url": "https://api.store.com/oos-alerts", "method": "POST"}},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_structure", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_structure", "target": "segment_empty", "sourceHandle": "detections", "targetHandle": "boxes"},
                {"source": "shelf_image", "target": "segment_empty", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "segment_empty", "target": "foreach_empty", "sourceHandle": "masks", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_empty", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_empty", "target": "crop_empty_area", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_empty", "target": "crop_empty_area", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_empty_area", "target": "embed_area", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_area", "target": "find_expected", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "find_expected", "target": "collect_missing", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_missing", "target": "filter_confident", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_confident", "target": "aggregate_missing", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "aggregate_missing", "target": "send_oos_alert", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection × 2 → segmentation → foreach → crop → embedding → similarity_search → collect → filter → aggregation → webhook",
    },

    {
        "id": "adv_scenario_8",
        "name": "Inventory Count with Multiple Detection Passes",
        "description": "Use multiple detection models to ensure accurate product counting",
        "use_case": "Stock Management",
        "blocks_used": ["detection", "embedding", "similarity_search", "aggregation", "stitch", "filter", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Run multiple detection models
                {"id": "detect_yolo", "type": "detection", "config": {"model_id": "yolov8-retail", "confidence_threshold": 0.4}},
                {"id": "detect_rtdetr", "type": "detection", "config": {"model_id": "rt-detr-retail", "confidence_threshold": 0.4}},
                # Combine detections
                {"id": "collect_all", "type": "collect"},
                # Use stitch/NMS to deduplicate
                {"id": "dedupe_detections", "type": "stitch", "config": {"merge_mode": "nms_class", "iou_threshold": 0.6}},
                # Identify each unique product
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1, "threshold": 0.75}},
                {"id": "collect_identified", "type": "collect"},
                # Filter unidentified
                {"id": "filter_identified", "type": "filter", "config": {"conditions": [{"field": "best_match", "operator": "exists"}]}},
                # Count per product
                {"id": "count_products", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "best_match.product_info.sku", "agg_func": "count"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_yolo", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "shelf_image", "target": "detect_rtdetr", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_yolo", "target": "collect_all", "sourceHandle": "detections", "targetHandle": "item"},
                {"source": "detect_rtdetr", "target": "collect_all", "sourceHandle": "detections", "targetHandle": "item"},
                {"source": "collect_all", "target": "dedupe_detections", "sourceHandle": "results", "targetHandle": "detections"},
                {"source": "dedupe_detections", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_product", "target": "collect_identified", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_identified", "target": "filter_identified", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_identified", "target": "count_products", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "count_products", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection × 2 (parallel) → collect → stitch → foreach → crop → embedding → similarity_search → collect → filter → aggregation → json_output",
    },

    {
        "id": "adv_scenario_9",
        "name": "Low Stock Early Warning System",
        "description": "Detect products with visually low stock and trigger replenishment orders",
        "use_case": "Stock Management",
        "blocks_used": ["detection", "classification", "embedding", "similarity_search", "filter", "map", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"store_id": "string", "priority_threshold": "number"}},
                # Detect all products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Classify each for stock level
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "classify_stock", "type": "classification", "config": {"model_id": "stock-level-classifier", "top_k": 1}},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_all", "type": "collect"},
                # Filter for low stock items
                {"id": "filter_low", "type": "filter", "config": {"conditions": [{"field": "stock_class", "operator": "in", "value": ["low", "critical"]}]}},
                # Map to replenishment orders
                {"id": "map_orders", "type": "map", "config": {"expression": "{product_id: item.best_match.product_id, priority: item.stock_class == 'critical' ? 'urgent' : 'normal', quantity: item.stock_class == 'critical' ? 20 : 10}", "output_field": "order"}},
                # Send to order system
                {"id": "create_orders", "type": "webhook", "config": {"url": "https://api.store.com/replenishment", "method": "POST", "batch_enabled": True}},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_stock", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_stock", "target": "collect_all", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_all", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_all", "target": "filter_low", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_low", "target": "map_orders", "sourceHandle": "passed", "targetHandle": "items"},
                {"source": "map_orders", "target": "create_orders", "sourceHandle": "items", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + embedding → similarity_search → collect → filter → map → webhook",
    },

    {
        "id": "adv_scenario_10",
        "name": "Expiry Date Visibility Check",
        "description": "Detect products and check if expiry dates are visible/front-facing",
        "use_case": "Stock Management",
        "blocks_used": ["detection", "classification", "embedding", "similarity_search", "filter", "aggregation", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Detect products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Classify if expiry date visible
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "classify_expiry_visible", "type": "classification", "config": {"model_id": "expiry-visibility-classifier"}},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_all", "type": "collect"},
                # Filter products with non-visible expiry
                {"id": "filter_hidden_expiry", "type": "filter", "config": {"conditions": [{"field": "expiry_visible", "operator": "eq", "value": False}]}},
                # Aggregate statistics
                {"id": "aggregate_stats", "type": "aggregation", "config": {"operation": "stats", "stats_config": {"fields": ["confidence"]}}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_expiry_visible", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_expiry_visible", "target": "collect_all", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_all", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_all", "target": "filter_hidden_expiry", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_hidden_expiry", "target": "aggregate_stats", "sourceHandle": "rejected", "targetHandle": "data"},
                {"source": "aggregate_stats", "target": "output", "sourceHandle": "summary", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + embedding → similarity_search → collect → filter → aggregation → api_response",
    },

    # ==========================================================================
    # PRODUCT AVAILABILITY SCENARIOS (11-15)
    # ==========================================================================

    {
        "id": "adv_scenario_11",
        "name": "Cross-Store Product Availability Comparison",
        "description": "Compare product availability across multiple store images",
        "use_case": "Product Availability",
        "blocks_used": ["detection", "embedding", "similarity_search", "aggregation", "comparison", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "store_a_image", "type": "image_input"},
                {"id": "store_b_image", "type": "image_input"},
                # Detect products in both stores
                {"id": "detect_store_a", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "detect_store_b", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Identify products in store A
                {"id": "foreach_a", "type": "foreach"},
                {"id": "crop_a", "type": "crop"},
                {"id": "embed_a", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_a", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_a", "type": "collect"},
                # Identify products in store B
                {"id": "foreach_b", "type": "foreach"},
                {"id": "crop_b", "type": "crop"},
                {"id": "embed_b", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_b", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_b", "type": "collect"},
                # Aggregate each store's products
                {"id": "aggregate_a", "type": "aggregation", "config": {"operation": "dedupe", "dedupe_config": {"by": "product_id", "keep": "best"}}},
                {"id": "aggregate_b", "type": "aggregation", "config": {"operation": "dedupe", "dedupe_config": {"by": "product_id", "keep": "best"}}},
                # Visual comparison
                {"id": "draw_boxes_a", "type": "draw_boxes"},
                {"id": "draw_boxes_b", "type": "draw_boxes"},
                {"id": "compare_stores", "type": "comparison", "config": {"mode": "side_by_side", "label_a": "Store A", "label_b": "Store B"}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "store_a_image", "target": "detect_store_a", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "store_b_image", "target": "detect_store_b", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_store_a", "target": "foreach_a", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "store_a_image", "target": "foreach_a", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_a", "target": "crop_a", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_a", "target": "crop_a", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_a", "target": "embed_a", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_a", "target": "identify_a", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_a", "target": "collect_a", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "detect_store_b", "target": "foreach_b", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "store_b_image", "target": "foreach_b", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_b", "target": "crop_b", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_b", "target": "crop_b", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_b", "target": "embed_b", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_b", "target": "identify_b", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_b", "target": "collect_b", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_a", "target": "aggregate_a", "sourceHandle": "results", "targetHandle": "data"},
                {"source": "collect_b", "target": "aggregate_b", "sourceHandle": "results", "targetHandle": "data"},
                {"source": "store_a_image", "target": "draw_boxes_a", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_store_a", "target": "draw_boxes_a", "sourceHandle": "detections", "targetHandle": "detections"},
                {"source": "store_b_image", "target": "draw_boxes_b", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_store_b", "target": "draw_boxes_b", "sourceHandle": "detections", "targetHandle": "detections"},
                {"source": "draw_boxes_a", "target": "compare_stores", "sourceHandle": "image", "targetHandle": "image_a"},
                {"source": "draw_boxes_b", "target": "compare_stores", "sourceHandle": "image", "targetHandle": "image_b"},
                {"source": "compare_stores", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input × 2 → detection × 2 → foreach × 2 → crop × 2 → embedding × 2 → similarity_search × 2 → collect × 2 → aggregation × 2 → draw_boxes × 2 → comparison → json_output",
    },

    {
        "id": "adv_scenario_12",
        "name": "New Product Detection in Store",
        "description": "Identify products not in the known catalog (potential new SKUs)",
        "use_case": "Product Availability",
        "blocks_used": ["detection", "embedding", "similarity_search", "filter", "classification", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Detect all products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Try to identify each
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "search_catalog", "type": "similarity_search", "config": {"collection": "products", "top_k": 1, "threshold": 0.8}},
                {"id": "collect_searches", "type": "collect"},
                # Filter for unidentified (low similarity = unknown product)
                {"id": "filter_unknown", "type": "filter", "config": {"conditions": [{"field": "best_match.similarity", "operator": "lt", "value": 0.8}]}},
                # Classify unknown products into categories
                {"id": "foreach_unknown", "type": "foreach"},
                {"id": "crop_unknown", "type": "crop"},
                {"id": "classify_category", "type": "classification", "config": {"model_id": "product-category-classifier"}},
                {"id": "collect_classified", "type": "collect"},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "search_catalog", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "search_catalog", "target": "collect_searches", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "foreach_product", "target": "collect_searches", "sourceHandle": "item", "targetHandle": "item"},
                {"source": "collect_searches", "target": "filter_unknown", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_unknown", "target": "foreach_unknown", "sourceHandle": "passed", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_unknown", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_unknown", "target": "crop_unknown", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_unknown", "target": "crop_unknown", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_unknown", "target": "classify_category", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "classify_category", "target": "collect_classified", "sourceHandle": "predictions", "targetHandle": "item"},
                {"source": "collect_classified", "target": "output", "sourceHandle": "results", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → embedding → similarity_search → collect → filter → foreach → crop → classification → collect → api_response",
    },

    {
        "id": "adv_scenario_13",
        "name": "Promotional Display Verification",
        "description": "Verify promotional products are correctly displayed in designated areas",
        "use_case": "Product Availability",
        "blocks_used": ["detection", "segmentation", "embedding", "similarity_search", "condition", "filter", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "display_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"promo_id": "string", "expected_products": "array"}},
                # Detect promotional display area
                {"id": "detect_promo_area", "type": "detection", "config": {"model_id": "promo-display-detector"}},
                # Segment the promotional zone
                {"id": "segment_promo", "type": "segmentation", "config": {"model_id": "sam-base", "prompt_mode": "box"}},
                # Detect products within promo area
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Identify products
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "promo_products", "top_k": 1}},
                {"id": "collect_identified", "type": "collect"},
                # Filter to only expected promo products
                {"id": "filter_promo", "type": "filter", "config": {"conditions": [{"field": "best_match.metadata.is_promo", "operator": "eq", "value": True}]}},
                # Check if enough promo products visible
                {"id": "check_compliance", "type": "condition", "config": {"conditions": [{"field": "passed_count", "operator": "gte", "value": 3}]}},
                # Alert if non-compliant
                {"id": "alert_compliance", "type": "webhook", "config": {"url": "https://api.store.com/promo-alerts"}},
            ],
            "edges": [
                {"source": "display_image", "target": "detect_promo_area", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_promo_area", "target": "segment_promo", "sourceHandle": "first_detection", "targetHandle": "boxes"},
                {"source": "display_image", "target": "segment_promo", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "display_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "display_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_product", "target": "collect_identified", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_identified", "target": "filter_promo", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_promo", "target": "check_compliance", "sourceHandle": "passed_count", "targetHandle": "value"},
                {"source": "check_compliance", "target": "alert_compliance", "sourceHandle": "false_output", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → segmentation → detection → foreach → crop → embedding → similarity_search → collect → filter → condition → webhook",
    },

    {
        "id": "adv_scenario_14",
        "name": "Competitor Product Detection",
        "description": "Identify competitor products that may have been placed in wrong location",
        "use_case": "Product Availability",
        "blocks_used": ["detection", "embedding", "similarity_search", "classification", "filter", "blur_region", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"our_brand": "string", "competitor_brands": "array"}},
                # Detect all products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Identify and classify brand
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "clip-vit-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "all_brands", "top_k": 1}},
                {"id": "classify_brand", "type": "classification", "config": {"model_id": "brand-classifier"}},
                {"id": "collect_products", "type": "collect"},
                # Filter competitor products
                {"id": "filter_competitors", "type": "filter", "config": {"conditions": [{"field": "brand_class", "operator": "neq", "value": "$params.our_brand"}]}},
                # Blur competitor products for visualization
                {"id": "blur_competitors", "type": "blur_region", "config": {"blur_type": "pixelate", "intensity": 20}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_brand", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_brand", "target": "collect_products", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_products", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_products", "target": "filter_competitors", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "shelf_image", "target": "blur_competitors", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "filter_competitors", "target": "blur_competitors", "sourceHandle": "passed", "targetHandle": "regions"},
                {"source": "blur_competitors", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → embedding + classification → similarity_search → collect → filter → blur_region → api_response",
    },

    {
        "id": "adv_scenario_15",
        "name": "Price Tag Product Matching",
        "description": "Match detected price tags to their corresponding products",
        "use_case": "Product Availability",
        "blocks_used": ["detection", "crop", "embedding", "similarity_search", "map", "aggregation", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Detect products and price tags separately
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "detect_price_tags", "type": "detection", "config": {"model_id": "price-tag-detector"}},
                # Identify products
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_products", "type": "collect"},
                # Map products to nearest price tags (by position)
                {"id": "map_to_tags", "type": "map", "config": {"expression": "find_nearest(item.bbox, $nodes.detect_price_tags.detections)", "output_field": "price_tag"}},
                # Aggregate matched pairs
                {"id": "aggregate_pairs", "type": "aggregation", "config": {"operation": "flatten", "flatten_config": {"child_field": "price_tag"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "shelf_image", "target": "detect_price_tags", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "identify_product", "target": "collect_products", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "foreach_product", "target": "collect_products", "sourceHandle": "item", "targetHandle": "item"},
                {"source": "collect_products", "target": "map_to_tags", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "map_to_tags", "target": "aggregate_pairs", "sourceHandle": "items", "targetHandle": "data"},
                {"source": "aggregate_pairs", "target": "output", "sourceHandle": "flat", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection × 2 → foreach → crop → embedding → similarity_search → collect → map → aggregation → json_output",
    },

    # ==========================================================================
    # QUALITY & DAMAGE DETECTION SCENARIOS (16-20)
    # ==========================================================================

    {
        "id": "adv_scenario_16",
        "name": "Damaged Product Detection Pipeline",
        "description": "Detect products with visible damage (dents, tears, stains)",
        "use_case": "Quality Control",
        "blocks_used": ["detection", "classification", "segmentation", "embedding", "similarity_search", "filter", "draw_masks", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "product_image", "type": "image_input"},
                # Detect products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Check each for damage
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                # Classify damage type
                {"id": "classify_damage", "type": "classification", "config": {"model_id": "damage-classifier", "top_k": 3}},
                # Segment damage areas
                {"id": "segment_damage", "type": "segmentation", "config": {"model_id": "sam-base", "prompt_mode": "text", "text_prompt": "damaged area"}},
                # Identify product
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_all", "type": "collect"},
                # Filter damaged products
                {"id": "filter_damaged", "type": "filter", "config": {"conditions": [{"field": "damage_class", "operator": "neq", "value": "none"}]}},
                # Visualize damage areas
                {"id": "draw_damage", "type": "draw_masks", "config": {"mode": "overlay", "opacity": 0.6, "palette": "reds"}},
                # Alert for damaged products
                {"id": "alert_damage", "type": "webhook", "config": {"url": "https://api.store.com/damage-reports"}},
            ],
            "edges": [
                {"source": "product_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "product_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_damage", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "segment_damage", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_damage", "target": "collect_all", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "segment_damage", "target": "collect_all", "sourceHandle": "masks", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_all", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_all", "target": "filter_damaged", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "product_image", "target": "draw_damage", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "filter_damaged", "target": "draw_damage", "sourceHandle": "passed", "targetHandle": "masks"},
                {"source": "filter_damaged", "target": "alert_damage", "sourceHandle": "passed", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + segmentation + embedding → similarity_search → collect → filter → draw_masks + webhook",
    },

    {
        "id": "adv_scenario_17",
        "name": "Freshness Detection for Perishables",
        "description": "Assess freshness of produce/perishables using visual cues",
        "use_case": "Quality Control",
        "blocks_used": ["detection", "classification", "embedding", "similarity_search", "filter", "aggregation", "condition", "webhook"],
        "pipeline": {
            "nodes": [
                {"id": "produce_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"freshness_threshold": "number", "alert_email": "string"}},
                # Detect produce items
                {"id": "detect_produce", "type": "detection", "config": {"model_id": "produce-detector"}},
                # Assess each item
                {"id": "foreach_item", "type": "foreach"},
                {"id": "crop_item", "type": "crop"},
                # Classify freshness (fresh/aging/expired)
                {"id": "classify_freshness", "type": "classification", "config": {"model_id": "freshness-classifier"}},
                # Identify produce type
                {"id": "embed_item", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_produce", "type": "similarity_search", "config": {"collection": "produce_catalog", "top_k": 1}},
                {"id": "collect_assessed", "type": "collect"},
                # Filter items needing attention
                {"id": "filter_expired", "type": "filter", "config": {"conditions": [{"field": "freshness_class", "operator": "in", "value": ["aging", "expired"]}]}},
                # Aggregate by freshness status
                {"id": "aggregate_status", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "freshness_class", "agg_func": "count"}}},
                # Check if action needed
                {"id": "check_action", "type": "condition", "config": {"conditions": [{"field": "failed_count", "operator": "gt", "value": 0}]}},
                # Send alerts
                {"id": "send_alert", "type": "webhook", "config": {"url": "https://api.store.com/freshness-alerts"}},
            ],
            "edges": [
                {"source": "produce_image", "target": "detect_produce", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_produce", "target": "foreach_item", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "produce_image", "target": "foreach_item", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_item", "target": "crop_item", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_item", "target": "crop_item", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_item", "target": "classify_freshness", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_item", "target": "embed_item", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_item", "target": "identify_produce", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_freshness", "target": "collect_assessed", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_produce", "target": "collect_assessed", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_assessed", "target": "filter_expired", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_expired", "target": "aggregate_status", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "filter_expired", "target": "check_action", "sourceHandle": "failed_count", "targetHandle": "value"},
                {"source": "aggregate_status", "target": "send_alert", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + embedding → similarity_search → collect → filter → aggregation → condition → webhook",
    },

    {
        "id": "adv_scenario_18",
        "name": "Package Integrity Verification",
        "description": "Check packages for tampering, open seals, or damage",
        "use_case": "Quality Control",
        "blocks_used": ["detection", "classification", "segmentation", "filter", "draw_boxes", "comparison", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "package_image", "type": "image_input"},
                {"id": "reference_image", "type": "image_input"},
                # Detect package
                {"id": "detect_package", "type": "detection", "config": {"model_id": "package-detector"}},
                # Segment seal areas
                {"id": "segment_seals", "type": "segmentation", "config": {"model_id": "sam-base", "prompt_mode": "text", "text_prompt": "seal, cap, lid"}},
                # Classify seal status
                {"id": "foreach_seal", "type": "foreach"},
                {"id": "crop_seal", "type": "crop"},
                {"id": "classify_seal", "type": "classification", "config": {"model_id": "seal-integrity-classifier"}},
                {"id": "collect_seals", "type": "collect"},
                # Filter compromised seals
                {"id": "filter_compromised", "type": "filter", "config": {"conditions": [{"field": "seal_status", "operator": "neq", "value": "intact"}]}},
                # Visualize issues
                {"id": "draw_issues", "type": "draw_boxes", "config": {"line_width": 3, "default_color": "#FF0000"}},
                # Compare with reference
                {"id": "compare_reference", "type": "comparison", "config": {"mode": "side_by_side", "label_a": "Inspected", "label_b": "Reference"}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "package_image", "target": "detect_package", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "package_image", "target": "segment_seals", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "segment_seals", "target": "foreach_seal", "sourceHandle": "masks", "targetHandle": "items"},
                {"source": "package_image", "target": "foreach_seal", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_seal", "target": "crop_seal", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_seal", "target": "crop_seal", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_seal", "target": "classify_seal", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "classify_seal", "target": "collect_seals", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "collect_seals", "target": "filter_compromised", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "package_image", "target": "draw_issues", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "filter_compromised", "target": "draw_issues", "sourceHandle": "passed", "targetHandle": "detections"},
                {"source": "draw_issues", "target": "compare_reference", "sourceHandle": "image", "targetHandle": "image_a"},
                {"source": "reference_image", "target": "compare_reference", "sourceHandle": "image", "targetHandle": "image_b"},
                {"source": "compare_reference", "target": "output", "sourceHandle": "image", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input × 2 → detection + segmentation → foreach → crop → classification → collect → filter → draw_boxes → comparison → api_response",
    },

    {
        "id": "adv_scenario_19",
        "name": "Label Orientation and Readability Check",
        "description": "Verify product labels are properly oriented and readable",
        "use_case": "Quality Control",
        "blocks_used": ["detection", "classification", "rotate_flip", "embedding", "similarity_search", "filter", "aggregation", "json_output"],
        "pipeline": {
            "nodes": [
                {"id": "shelf_image", "type": "image_input"},
                # Detect products
                {"id": "detect_products", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                # Check each product
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                # Classify orientation
                {"id": "classify_orientation", "type": "classification", "config": {"model_id": "orientation-classifier"}},
                # Auto-correct orientation for identification
                {"id": "correct_orientation", "type": "rotate_flip", "config": {"auto_orient": True}},
                # Identify product
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_checked", "type": "collect"},
                # Filter improperly oriented
                {"id": "filter_misoriented", "type": "filter", "config": {"conditions": [{"field": "orientation_class", "operator": "neq", "value": "correct"}]}},
                # Aggregate by orientation status
                {"id": "aggregate_orientation", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "orientation_class", "agg_func": "count"}}},
                {"id": "output", "type": "json_output"},
            ],
            "edges": [
                {"source": "shelf_image", "target": "detect_products", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "detect_products", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "shelf_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_orientation", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "correct_orientation", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "correct_orientation", "target": "embed_product", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_orientation", "target": "collect_checked", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_checked", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_checked", "target": "filter_misoriented", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_misoriented", "target": "aggregate_orientation", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "aggregate_orientation", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → detection → foreach → crop → classification + rotate_flip → embedding → similarity_search → collect → filter → aggregation → json_output",
    },

    {
        "id": "adv_scenario_20",
        "name": "Complete Store Audit Pipeline",
        "description": "Full store audit combining planogram, stock, quality, and availability checks",
        "use_case": "Comprehensive Audit",
        "blocks_used": ["tile", "detection", "stitch", "classification", "segmentation", "embedding", "similarity_search", "filter", "aggregation", "condition", "webhook", "api_response"],
        "pipeline": {
            "nodes": [
                {"id": "store_image", "type": "image_input"},
                {"id": "params", "type": "parameter_input", "config": {"store_id": "string", "audit_type": "string", "threshold": "number"}},
                # Tile for large store image
                {"id": "tile_store", "type": "tile", "config": {"tile_size": 800, "overlap_ratio": 0.15}},
                # Detect in each tile
                {"id": "foreach_tile", "type": "foreach"},
                {"id": "detect_tile", "type": "detection", "config": {"model_id": "yolov8-retail"}},
                {"id": "collect_tile_detections", "type": "collect", "config": {"flatten": True}},
                # Stitch detections
                {"id": "stitch_detections", "type": "stitch", "config": {"merge_mode": "nms", "iou_threshold": 0.5}},
                # Process each product
                {"id": "foreach_product", "type": "foreach"},
                {"id": "crop_product", "type": "crop"},
                # Multi-task analysis
                {"id": "classify_stock", "type": "classification", "config": {"model_id": "stock-level-classifier"}},
                {"id": "classify_quality", "type": "classification", "config": {"model_id": "quality-classifier"}},
                {"id": "embed_product", "type": "embedding", "config": {"model_id": "dinov2-base"}},
                {"id": "identify_product", "type": "similarity_search", "config": {"collection": "products", "top_k": 1}},
                {"id": "collect_analyzed", "type": "collect"},
                # Filter products with issues
                {"id": "filter_stock_issues", "type": "filter", "config": {"conditions": [{"field": "stock_level", "operator": "in", "value": ["low", "critical"]}]}},
                {"id": "filter_quality_issues", "type": "filter", "config": {"conditions": [{"field": "quality_class", "operator": "neq", "value": "good"}]}},
                # Aggregate all issues
                {"id": "aggregate_stock", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "stock_level", "agg_func": "count"}}},
                {"id": "aggregate_quality", "type": "aggregation", "config": {"operation": "group", "group_config": {"by": "quality_class", "agg_func": "count"}}},
                # Check if audit passes
                {"id": "check_audit", "type": "condition", "config": {"conditions": [{"field": "total_issues", "operator": "lte", "value": 5}]}},
                # Send audit results
                {"id": "send_report", "type": "webhook", "config": {"url": "https://api.store.com/audit-reports"}},
                {"id": "output", "type": "api_response"},
            ],
            "edges": [
                {"source": "store_image", "target": "tile_store", "sourceHandle": "image", "targetHandle": "image"},
                {"source": "tile_store", "target": "foreach_tile", "sourceHandle": "tiles", "targetHandle": "items"},
                {"source": "foreach_tile", "target": "detect_tile", "sourceHandle": "item", "targetHandle": "image"},
                {"source": "detect_tile", "target": "collect_tile_detections", "sourceHandle": "detections", "targetHandle": "item"},
                {"source": "collect_tile_detections", "target": "stitch_detections", "sourceHandle": "results", "targetHandle": "detections"},
                {"source": "tile_store", "target": "stitch_detections", "sourceHandle": "tile_info", "targetHandle": "tile_info"},
                {"source": "tile_store", "target": "stitch_detections", "sourceHandle": "grid_info", "targetHandle": "grid_info"},
                {"source": "stitch_detections", "target": "foreach_product", "sourceHandle": "detections", "targetHandle": "items"},
                {"source": "store_image", "target": "foreach_product", "sourceHandle": "image", "targetHandle": "context"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "item", "targetHandle": "detection"},
                {"source": "foreach_product", "target": "crop_product", "sourceHandle": "context", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_stock", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "classify_quality", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "crop_product", "target": "embed_product", "sourceHandle": "crop", "targetHandle": "image"},
                {"source": "embed_product", "target": "identify_product", "sourceHandle": "embedding", "targetHandle": "embedding"},
                {"source": "classify_stock", "target": "collect_analyzed", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "classify_quality", "target": "collect_analyzed", "sourceHandle": "top_prediction", "targetHandle": "item"},
                {"source": "identify_product", "target": "collect_analyzed", "sourceHandle": "best_match", "targetHandle": "item"},
                {"source": "collect_analyzed", "target": "filter_stock_issues", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "collect_analyzed", "target": "filter_quality_issues", "sourceHandle": "results", "targetHandle": "items"},
                {"source": "filter_stock_issues", "target": "aggregate_stock", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "filter_quality_issues", "target": "aggregate_quality", "sourceHandle": "passed", "targetHandle": "data"},
                {"source": "aggregate_stock", "target": "check_audit", "sourceHandle": "summary", "targetHandle": "value"},
                {"source": "aggregate_quality", "target": "send_report", "sourceHandle": "result", "targetHandle": "data"},
                {"source": "aggregate_stock", "target": "output", "sourceHandle": "result", "targetHandle": "data"},
            ],
        },
        "expected_flow": "image_input → tile → foreach → detection → collect → stitch → foreach → crop → classification × 2 + embedding → similarity_search → collect → filter × 2 → aggregation × 2 → condition → webhook + api_response",
    },
]


def validate_scenario_blocks(scenario: dict, available_blocks: list) -> dict:
    """Validate that all blocks in a scenario are available."""
    blocks_used = scenario.get("blocks_used", [])
    pipeline_nodes = scenario.get("pipeline", {}).get("nodes", [])

    # Get actual block types from nodes
    node_types = set(node.get("type") for node in pipeline_nodes)

    missing_blocks = []
    for block_type in node_types:
        if block_type not in available_blocks:
            missing_blocks.append(block_type)

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "blocks_claimed": blocks_used,
        "blocks_actual": list(node_types),
        "missing_blocks": missing_blocks,
        "valid": len(missing_blocks) == 0,
    }


def validate_scenario_edges(scenario: dict) -> dict:
    """Validate that all edges reference valid nodes and ports."""
    pipeline = scenario.get("pipeline", {})
    nodes = pipeline.get("nodes", [])
    edges = pipeline.get("edges", [])

    node_ids = set(node.get("id") for node in nodes)

    edge_errors = []
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in node_ids:
            edge_errors.append(f"Source node '{source}' not found")
        if target not in node_ids:
            edge_errors.append(f"Target node '{target}' not found")

    return {
        "scenario_id": scenario["id"],
        "total_edges": len(edges),
        "errors": edge_errors,
        "valid": len(edge_errors) == 0,
    }


def get_model_coverage(scenarios: list) -> dict:
    """Analyze which model blocks are covered by scenarios."""
    model_blocks = ["detection", "classification", "embedding", "segmentation", "similarity_search"]

    coverage = {block: [] for block in model_blocks}

    for scenario in scenarios:
        blocks_used = scenario.get("blocks_used", [])
        for block in model_blocks:
            if block in blocks_used:
                coverage[block].append(scenario["id"])

    return {
        "model_blocks": model_blocks,
        "coverage": coverage,
        "all_models_used": all(len(scenarios) > 0 for scenarios in coverage.values()),
    }


if __name__ == "__main__":
    print(f"Total Advanced Scenarios: {len(ADVANCED_RETAIL_SCENARIOS)}")
    print("\nScenarios by Use Case:")

    use_cases = {}
    for scenario in ADVANCED_RETAIL_SCENARIOS:
        uc = scenario.get("use_case", "Unknown")
        if uc not in use_cases:
            use_cases[uc] = []
        use_cases[uc].append(scenario["name"])

    for uc, scenarios in use_cases.items():
        print(f"\n{uc}:")
        for s in scenarios:
            print(f"  - {s}")

    print("\n\nModel Coverage:")
    coverage = get_model_coverage(ADVANCED_RETAIL_SCENARIOS)
    for model, scenarios in coverage["coverage"].items():
        print(f"  {model}: {len(scenarios)} scenarios")
