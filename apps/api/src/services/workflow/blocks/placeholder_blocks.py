"""
Workflow Blocks - Logic and Output Blocks

Logic and control flow blocks.
"""

import time
from typing import Any

from ..base import BaseBlock, BlockResult, ExecutionContext


class ConditionBlock(BaseBlock):
    """
    Condition Block

    If-else branching based on expression evaluation.
    """

    block_type = "condition"
    display_name = "Condition"
    description = "If-else branching based on expression"

    input_ports = [
        {"name": "value", "type": "any", "required": True},
    ]
    output_ports = [
        {"name": "true_output", "type": "any", "description": "Output when condition is true"},
        {"name": "false_output", "type": "any", "description": "Output when condition is false"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Expression to evaluate"},
            "operator": {
                "type": "string",
                "enum": ["equals", "not_equals", "greater_than", "less_than", "contains", "is_empty", "is_not_empty"],
                "default": "is_not_empty",
            },
            "compare_value": {"type": "string", "description": "Value to compare against"},
        },
        "required": ["operator"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Evaluate condition and route output."""
        start_time = time.time()

        value = inputs.get("value")
        operator = config.get("operator", "is_not_empty")
        compare_value = config.get("compare_value")

        # Evaluate condition
        result = False

        if operator == "equals":
            result = str(value) == str(compare_value)
        elif operator == "not_equals":
            result = str(value) != str(compare_value)
        elif operator == "greater_than":
            try:
                result = float(value) > float(compare_value)
            except (ValueError, TypeError):
                result = False
        elif operator == "less_than":
            try:
                result = float(value) < float(compare_value)
            except (ValueError, TypeError):
                result = False
        elif operator == "contains":
            result = str(compare_value) in str(value) if value else False
        elif operator == "is_empty":
            result = not value or (isinstance(value, (list, dict)) and len(value) == 0)
        elif operator == "is_not_empty":
            result = bool(value) and not (isinstance(value, (list, dict)) and len(value) == 0)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "true_output": value if result else None,
                "false_output": None if result else value,
            },
            duration_ms=round(duration, 2),
            metrics={
                "condition_result": result,
                "operator": operator,
            },
        )


class FilterBlock(BaseBlock):
    """
    Filter Block

    Filter array items based on field value or expression.
    """

    block_type = "filter"
    display_name = "Filter"
    description = "Filter array items based on expression"

    input_ports = [
        {"name": "items", "type": "array", "required": True},
    ]
    output_ports = [
        {"name": "passed", "type": "array", "description": "Items that passed filter"},
        {"name": "rejected", "type": "array", "description": "Items that failed filter"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "field": {"type": "string", "description": "Field to filter on (e.g., 'confidence')"},
            "operator": {
                "type": "string",
                "enum": ["equals", "not_equals", "greater_than", "less_than", "contains", "in_list"],
                "default": "greater_than",
            },
            "value": {"type": "string", "description": "Value to compare against"},
            "class_filter": {"type": "array", "items": {"type": "string"}, "description": "Filter by class names"},
        },
        "required": ["operator"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Filter items based on criteria."""
        start_time = time.time()

        items = inputs.get("items", [])
        if not items:
            return BlockResult(
                outputs={"passed": [], "rejected": []},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"passed_count": 0, "rejected_count": 0},
            )

        field = config.get("field")
        operator = config.get("operator", "greater_than")
        compare_value = config.get("value")
        class_filter = config.get("class_filter")

        passed = []
        rejected = []

        for item in items:
            # Class filter
            if class_filter:
                item_class = item.get("class_name") if isinstance(item, dict) else None
                if item_class not in class_filter:
                    rejected.append(item)
                    continue

            # Field filter
            if field:
                item_value = item.get(field) if isinstance(item, dict) else None

                if operator == "equals":
                    passes = str(item_value) == str(compare_value)
                elif operator == "not_equals":
                    passes = str(item_value) != str(compare_value)
                elif operator == "greater_than":
                    try:
                        passes = float(item_value) > float(compare_value)
                    except (ValueError, TypeError):
                        passes = False
                elif operator == "less_than":
                    try:
                        passes = float(item_value) < float(compare_value)
                    except (ValueError, TypeError):
                        passes = False
                elif operator == "contains":
                    passes = str(compare_value) in str(item_value) if item_value else False
                elif operator == "in_list":
                    list_values = [v.strip() for v in str(compare_value).split(",")]
                    passes = str(item_value) in list_values
                else:
                    passes = True

                if passes:
                    passed.append(item)
                else:
                    rejected.append(item)
            else:
                # No field specified, pass all
                passed.append(item)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"passed": passed, "rejected": rejected},
            duration_ms=round(duration, 2),
            metrics={"passed_count": len(passed), "rejected_count": len(rejected)},
        )


class GridBuilderBlock(BaseBlock):
    """
    Grid Builder Block

    Builds realogram/planogram grid from detection and matching results.
    """

    block_type = "grid_builder"
    display_name = "Grid Builder"
    description = "Build realogram/planogram grid from detections"

    input_ports = [
        {"name": "shelves", "type": "array", "required": False, "description": "Shelf detections"},
        {"name": "slots", "type": "array", "required": False, "description": "Slot detections"},
        {"name": "matches", "type": "array", "required": False, "description": "Product matches"},
        {"name": "voids", "type": "array", "required": False, "description": "Empty slot detections"},
    ]
    output_ports = [
        {"name": "grid", "type": "array", "description": "2D grid representation"},
        {"name": "realogram", "type": "object", "description": "Full realogram data"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "sort_by": {"type": "string", "enum": ["position", "confidence"], "default": "position"},
            "group_by_shelf": {"type": "boolean", "default": True},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Build grid from detections and matches."""
        start_time = time.time()

        shelves = inputs.get("shelves", [])
        slots = inputs.get("slots", [])
        matches = inputs.get("matches", [])
        voids = inputs.get("voids", [])

        grid = []
        cells = []

        # Sort shelves by y position (top to bottom)
        sorted_shelves = sorted(
            shelves,
            key=lambda s: s.get("bbox", {}).get("y1", 0) if isinstance(s.get("bbox"), dict) else 0
        )

        for shelf_idx, shelf in enumerate(sorted_shelves):
            shelf_bbox = shelf.get("bbox", {})
            shelf_y1 = shelf_bbox.get("y1", 0) if isinstance(shelf_bbox, dict) else 0
            shelf_y2 = shelf_bbox.get("y2", 1) if isinstance(shelf_bbox, dict) else 1

            # Find slots within this shelf
            shelf_slots = []
            for slot in slots:
                slot_bbox = slot.get("bbox", {})
                slot_y = (slot_bbox.get("y1", 0) + slot_bbox.get("y2", 0)) / 2 if isinstance(slot_bbox, dict) else 0
                if shelf_y1 <= slot_y <= shelf_y2:
                    shelf_slots.append(slot)

            # Sort slots by x position (left to right)
            shelf_slots.sort(
                key=lambda s: s.get("bbox", {}).get("x1", 0) if isinstance(s.get("bbox"), dict) else 0
            )

            # Build row
            row = []
            for slot_idx, slot in enumerate(shelf_slots):
                # Find matching product
                slot_match = None
                for match_group in matches:
                    if isinstance(match_group, list) and len(match_group) > 0:
                        slot_match = match_group[0]
                        break
                    elif isinstance(match_group, dict):
                        slot_match = match_group
                        break

                cell = {
                    "row": shelf_idx,
                    "col": slot_idx,
                    "slot_id": slot.get("id", slot_idx),
                    "product": slot_match.get("product_id") if slot_match else None,
                    "similarity": slot_match.get("similarity") if slot_match else None,
                    "is_void": slot.get("class_name") == "void" or slot_match is None,
                }
                row.append(cell)
                cells.append(cell)

            if row:
                grid.append(row)

        # Count totals
        total_products = sum(1 for c in cells if c.get("product"))
        total_voids = sum(1 for c in cells if c.get("is_void"))

        realogram = {
            "rows": len(grid),
            "cols": max(len(row) for row in grid) if grid else 0,
            "cells": cells,
            "total_products": total_products,
            "total_voids": total_voids,
            "shelves": len(shelves),
            "slots": len(slots),
        }

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"grid": grid, "realogram": realogram},
            duration_ms=round(duration, 2),
            metrics={
                "grid_rows": len(grid),
                "total_cells": len(cells),
                "products": total_products,
                "voids": total_voids,
            },
        )


class JsonOutputBlock(BaseBlock):
    """
    JSON Output Block

    Formats data as JSON output.
    """

    block_type = "json_output"
    display_name = "JSON Output"
    description = "Format output as JSON"

    input_ports = [
        {"name": "data", "type": "any", "required": True},
    ]
    output_ports = [
        {"name": "json", "type": "object", "description": "JSON formatted output"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "include_metadata": {"type": "boolean", "default": False},
            "flatten": {"type": "boolean", "default": False},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Format data as JSON."""
        start_time = time.time()

        data = inputs.get("data")
        include_metadata = config.get("include_metadata", False)

        output = data

        if include_metadata:
            output = {
                "data": data,
                "metadata": {
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id,
                },
            }

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={"json": output},
            duration_ms=round(duration, 2),
        )
