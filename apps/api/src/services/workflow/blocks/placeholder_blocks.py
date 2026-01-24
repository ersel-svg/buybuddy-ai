"""
Workflow Blocks - Logic and Output Blocks

Logic and control flow blocks.
"""

import time
from typing import Any

from ..base import BaseBlock, BlockResult, ExecutionContext


class ConditionBlock(BaseBlock):
    """
    Condition Block - SOTA

    If-else branching based on expression evaluation.
    Supports multiple conditions, various operators, and logic modes.
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
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {"type": "string"},
                        "type": {"type": "string", "enum": ["auto", "string", "number", "boolean", "field"]},
                    },
                },
            },
            "logic_mode": {
                "type": "string",
                "enum": ["and", "or", "xor", "nand", "custom"],
                "default": "and",
            },
            "custom_expression": {"type": "string", "description": "Custom logic expression like (c1 AND c2) OR c3"},
            "invert": {"type": "boolean", "default": False},
            "default_branch": {"type": "string", "enum": ["true", "false", "error"], "default": "false"},
            "fallback_value": {"type": "string"},
            "add_metadata": {"type": "boolean", "default": False},
        },
    }

    def _get_field_value(self, data: Any, field_path: str) -> Any:
        """Get nested field value using dot notation and array indexing."""
        if not field_path:
            return data

        current = data
        parts = field_path.replace("[", ".").replace("]", "").split(".")

        for part in parts:
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current

    def _parse_value(self, value_str: str, value_type: str, data: Any) -> Any:
        """Parse value based on type specification."""
        if value_type == "field":
            return self._get_field_value(data, value_str)
        if value_type == "number":
            try:
                return float(value_str)
            except (ValueError, TypeError):
                return 0
        if value_type == "boolean":
            return value_str.lower() in ("true", "1", "yes")
        if value_type == "string":
            return str(value_str)
        # Auto-detect
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return value_str

    def _evaluate_single(self, field_value: Any, operator: str, compare_value: Any) -> bool:
        """Evaluate a single condition."""
        import re

        # Type check operators (no compare value needed)
        if operator == "exists":
            return field_value is not None
        if operator == "not_exists":
            return field_value is None
        if operator == "is_true":
            return bool(field_value) is True
        if operator == "is_false":
            return bool(field_value) is False
        if operator == "array_empty":
            return isinstance(field_value, (list, tuple)) and len(field_value) == 0
        if operator == "array_not_empty":
            return isinstance(field_value, (list, tuple)) and len(field_value) > 0

        # Comparison operators
        if operator == "equals":
            return field_value == compare_value or str(field_value) == str(compare_value)
        if operator == "not_equals":
            return field_value != compare_value and str(field_value) != str(compare_value)

        # Numeric comparisons
        try:
            fv = float(field_value) if field_value is not None else 0
            cv = float(compare_value) if compare_value is not None else 0

            if operator == "greater_than":
                return fv > cv
            if operator == "greater_equal":
                return fv >= cv
            if operator == "less_than":
                return fv < cv
            if operator == "less_equal":
                return fv <= cv
        except (ValueError, TypeError):
            pass

        # Range operators (value should be "min, max")
        if operator in ("between", "not_between"):
            try:
                parts = str(compare_value).split(",")
                if len(parts) >= 2:
                    min_v, max_v = float(parts[0].strip()), float(parts[1].strip())
                    fv = float(field_value)
                    in_range = min_v <= fv <= max_v
                    return in_range if operator == "between" else not in_range
            except (ValueError, TypeError):
                pass
            return False

        # String operators
        fv_str = str(field_value) if field_value is not None else ""
        cv_str = str(compare_value) if compare_value is not None else ""

        if operator == "contains":
            return cv_str in fv_str
        if operator == "not_contains":
            return cv_str not in fv_str
        if operator == "starts_with":
            return fv_str.startswith(cv_str)
        if operator == "ends_with":
            return fv_str.endswith(cv_str)
        if operator == "regex_match":
            try:
                return bool(re.search(cv_str, fv_str))
            except re.error:
                return False

        # Array operators
        if operator == "in_array":
            items = [x.strip() for x in cv_str.split(",")]
            return str(field_value) in items
        if operator == "not_in_array":
            items = [x.strip() for x in cv_str.split(",")]
            return str(field_value) not in items
        if operator == "array_contains":
            if isinstance(field_value, (list, tuple)):
                return compare_value in field_value or str(compare_value) in [str(x) for x in field_value]
            return False
        if operator == "array_length_equals":
            if isinstance(field_value, (list, tuple)):
                try:
                    return len(field_value) == int(compare_value)
                except (ValueError, TypeError):
                    return False
            return False
        if operator == "array_length_greater":
            if isinstance(field_value, (list, tuple)):
                try:
                    return len(field_value) > int(compare_value)
                except (ValueError, TypeError):
                    return False
            return False

        # Type check
        if operator == "is_type":
            type_map = {
                "string": str,
                "number": (int, float),
                "boolean": bool,
                "array": (list, tuple),
                "object": dict,
                "null": type(None),
            }
            expected_type = type_map.get(cv_str.lower())
            if expected_type:
                return isinstance(field_value, expected_type)
            return False

        return False

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Evaluate condition(s) and route output."""
        start_time = time.time()

        value = inputs.get("value")
        conditions = config.get("conditions", [])
        logic_mode = config.get("logic_mode", "and")
        invert = config.get("invert", False)
        default_branch = config.get("default_branch", "false")
        add_metadata = config.get("add_metadata", False)

        # Legacy support: single condition
        if not conditions:
            operator = config.get("operator", "is_not_empty")
            compare_value = config.get("compare_value")
            conditions = [{"field": "", "operator": operator, "value": compare_value, "type": "auto"}]

        condition_results = []
        try:
            for cond in conditions:
                field = cond.get("field", "")
                operator = cond.get("operator", "exists")
                compare_str = cond.get("value", "")
                value_type = cond.get("type", "auto")

                # Get field value
                field_value = self._get_field_value(value, field) if field else value

                # Parse compare value
                compare_value = self._parse_value(compare_str, value_type, value) if compare_str else None

                # Evaluate
                cond_result = self._evaluate_single(field_value, operator, compare_value)
                condition_results.append(cond_result)

            # Apply logic mode
            if logic_mode == "and":
                result = all(condition_results) if condition_results else True
            elif logic_mode == "or":
                result = any(condition_results) if condition_results else False
            elif logic_mode == "xor":
                result = sum(condition_results) == 1 if condition_results else False
            elif logic_mode == "nand":
                result = not all(condition_results) if condition_results else True
            elif logic_mode == "custom":
                # Parse custom expression like "(c1 AND c2) OR c3"
                expr = config.get("custom_expression", "")
                if expr:
                    # Replace c1, c2, etc. with actual results
                    for i, r in enumerate(condition_results):
                        expr = expr.replace(f"c{i + 1}", str(r))
                    expr = expr.upper().replace("AND", " and ").replace("OR", " or ").replace("NOT", " not ")
                    try:
                        result = eval(expr, {"__builtins__": {}}, {"True": True, "False": False})
                    except Exception:
                        result = default_branch == "true"
                else:
                    result = all(condition_results) if condition_results else True
            else:
                result = all(condition_results) if condition_results else True

        except Exception:
            if default_branch == "error":
                raise
            result = default_branch == "true"

        # Apply invert
        if invert:
            result = not result

        duration = (time.time() - start_time) * 1000

        outputs = {
            "true_output": value if result else None,
            "false_output": None if result else value,
        }

        # Add metadata if requested
        if add_metadata:
            outputs["metadata"] = {
                "condition_results": condition_results,
                "logic_mode": logic_mode,
                "final_result": result,
                "inverted": invert,
            }

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "condition_result": result,
                "conditions_count": len(condition_results),
                "conditions_passed": sum(condition_results),
                "logic_mode": logic_mode,
            },
        )


class FilterBlock(BaseBlock):
    """
    Filter Block - SOTA

    Filter array items based on multiple conditions with advanced operators.
    Supports sorting, grouping, statistics, and top-N limiting.
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
        {"name": "stats", "type": "object", "description": "Filter statistics"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {},
                    },
                },
            },
            "logic_mode": {
                "type": "string",
                "enum": ["all", "any", "none"],
                "default": "all",
            },
            "invert": {"type": "boolean", "default": False},
            "sort_enabled": {"type": "boolean", "default": False},
            "sort_by": {"type": "string"},
            "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
            "group_enabled": {"type": "boolean", "default": False},
            "group_by": {"type": "string"},
            "top_n": {"type": "number", "default": 0},
            "include_stats": {"type": "boolean", "default": False},
        },
    }

    def _get_field_value(self, item: Any, field: str) -> Any:
        """Get nested field value using dot notation."""
        if not field or not isinstance(item, dict):
            return item

        current = item
        parts = field.replace("[", ".").replace("]", "").split(".")

        for part in parts:
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, (list, tuple)):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return current

    def _evaluate_condition(self, item: Any, condition: dict) -> bool:
        """Evaluate a single condition against an item."""
        import re

        field = condition.get("field", "")
        operator = condition.get("operator", "greater_than")
        compare_value = condition.get("value")

        item_value = self._get_field_value(item, field)

        # Null check operators
        if operator == "is_null":
            return item_value is None or item_value == "" or (isinstance(item_value, (list, dict)) and len(item_value) == 0)
        if operator == "is_not_null":
            return item_value is not None and item_value != "" and not (isinstance(item_value, (list, dict)) and len(item_value) == 0)

        # Equality operators
        if operator == "equals":
            return item_value == compare_value or str(item_value) == str(compare_value)
        if operator == "not_equals":
            return item_value != compare_value and str(item_value) != str(compare_value)

        # Numeric comparisons
        try:
            iv = float(item_value) if item_value is not None else 0
            if isinstance(compare_value, dict):
                # Range comparison
                cv_min = float(compare_value.get("min", 0))
                cv_max = float(compare_value.get("max", float("inf")))
            else:
                cv = float(compare_value) if compare_value is not None else 0

            if operator == "greater_than":
                return iv > cv
            if operator == "greater_equal":
                return iv >= cv
            if operator == "less_than":
                return iv < cv
            if operator == "less_equal":
                return iv <= cv
            if operator in ("between", "in_range"):
                return cv_min <= iv <= cv_max
        except (ValueError, TypeError):
            pass

        # String operators
        iv_str = str(item_value) if item_value is not None else ""
        cv_str = str(compare_value) if compare_value is not None else ""

        if operator == "contains":
            return cv_str.lower() in iv_str.lower()
        if operator == "not_contains":
            return cv_str.lower() not in iv_str.lower()
        if operator == "starts_with":
            return iv_str.lower().startswith(cv_str.lower())
        if operator == "ends_with":
            return iv_str.lower().endswith(cv_str.lower())
        if operator == "regex":
            try:
                return bool(re.search(cv_str, iv_str))
            except re.error:
                return False

        # List operators
        if operator in ("in", "in_list"):
            if isinstance(compare_value, list):
                list_values = compare_value
            else:
                list_values = [v.strip() for v in cv_str.split(",")]
            return str(item_value) in list_values or item_value in list_values
        if operator in ("not_in", "not_in_list"):
            if isinstance(compare_value, list):
                list_values = compare_value
            else:
                list_values = [v.strip() for v in cv_str.split(",")]
            return str(item_value) not in list_values and item_value not in list_values

        return True

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
                outputs={"passed": [], "rejected": [], "stats": {"total": 0, "passed": 0, "rejected": 0}},
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"passed_count": 0, "rejected_count": 0},
            )

        conditions = config.get("conditions", [])
        logic_mode = config.get("logic_mode", "all")
        invert = config.get("invert", False)
        sort_enabled = config.get("sort_enabled", False)
        sort_by = config.get("sort_by", "confidence")
        sort_order = config.get("sort_order", "desc")
        group_enabled = config.get("group_enabled", False)
        group_by = config.get("group_by")
        top_n = config.get("top_n", 0)
        include_stats = config.get("include_stats", False)

        # Legacy support
        if not conditions:
            field = config.get("field")
            operator = config.get("operator", "greater_than")
            value = config.get("value")
            class_filter = config.get("class_filter")
            if field or operator:
                conditions = [{"field": field or "confidence", "operator": operator, "value": value}]
            # Add class filter as condition
            if class_filter:
                conditions.append({"field": "class_name", "operator": "in", "value": class_filter})

        passed = []
        rejected = []

        for item in items:
            if not conditions:
                passed.append(item)
                continue

            # Evaluate all conditions
            results = [self._evaluate_condition(item, cond) for cond in conditions]

            # Apply logic mode
            if logic_mode == "all":
                item_passes = all(results)
            elif logic_mode == "any":
                item_passes = any(results)
            elif logic_mode == "none":
                item_passes = not any(results)
            else:
                item_passes = all(results)

            # Apply invert
            if invert:
                item_passes = not item_passes

            if item_passes:
                passed.append(item)
            else:
                rejected.append(item)

        # Sort results
        if sort_enabled and sort_by and passed:
            try:
                passed = sorted(
                    passed,
                    key=lambda x: self._get_field_value(x, sort_by) or 0,
                    reverse=(sort_order == "desc"),
                )
            except (TypeError, ValueError):
                pass

        # Group results
        grouped = None
        if group_enabled and group_by and passed:
            grouped = {}
            for item in passed:
                key = str(self._get_field_value(item, group_by) or "unknown")
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item)

        # Top-N limiting
        if top_n > 0 and passed:
            passed = passed[:top_n]

        # Calculate statistics
        stats = {
            "total": len(items),
            "passed": len(passed),
            "rejected": len(rejected),
            "pass_rate": round(len(passed) / len(items) * 100, 2) if items else 0,
        }

        if include_stats and passed:
            # Add field statistics for numeric fields
            numeric_fields = ["confidence", "area", "width", "height", "score", "distance"]
            for field in numeric_fields:
                values = []
                for item in passed:
                    val = self._get_field_value(item, field)
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            pass
                if values:
                    stats[f"{field}_min"] = round(min(values), 4)
                    stats[f"{field}_max"] = round(max(values), 4)
                    stats[f"{field}_avg"] = round(sum(values) / len(values), 4)

        duration = (time.time() - start_time) * 1000

        outputs = {
            "passed": passed,
            "rejected": rejected,
            "stats": stats,
        }
        if grouped:
            outputs["grouped"] = grouped

        return BlockResult(
            outputs=outputs,
            duration_ms=round(duration, 2),
            metrics={
                "passed_count": len(passed),
                "rejected_count": len(rejected),
                "conditions_count": len(conditions),
            },
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


class ForEachBlock(BaseBlock):
    """
    ForEach Block

    Iterates over an array and applies downstream blocks to each item.
    Used for batch processing like: detections -> embed each -> similarity search each.
    """

    block_type = "foreach"
    display_name = "For Each"
    description = "Iterate over array items for batch processing"

    input_ports = [
        {"name": "items", "type": "array", "required": True, "description": "Array of items to iterate"},
        {"name": "context", "type": "any", "required": False, "description": "Additional context passed to each iteration"},
    ]
    output_ports = [
        {"name": "item", "type": "any", "description": "Current item in iteration"},
        {"name": "index", "type": "number", "description": "Current index (0-based)"},
        {"name": "total", "type": "number", "description": "Total number of items"},
        {"name": "context", "type": "any", "description": "Passed through context"},
        {"name": "is_first", "type": "boolean", "description": "True if first item"},
        {"name": "is_last", "type": "boolean", "description": "True if last item"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "max_items": {"type": "number", "description": "Limit number of items to process (0 = no limit)"},
            "start_index": {"type": "number", "default": 0, "description": "Start from this index"},
            "parallel": {"type": "boolean", "default": False, "description": "Process items in parallel"},
            "batch_size": {"type": "number", "default": 1, "description": "Process in batches of this size"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """
        ForEach execution.

        Note: The actual iteration is handled by the workflow engine.
        This block sets up the iteration context and the engine handles
        running downstream blocks for each item.
        """
        start_time = time.time()

        items = inputs.get("items", [])
        ctx = inputs.get("context")
        max_items = config.get("max_items", 0)
        start_index = config.get("start_index", 0)

        if not isinstance(items, list):
            items = [items] if items else []

        # Apply limits
        if start_index > 0:
            items = items[start_index:]
        if max_items > 0:
            items = items[:max_items]

        # For the first execution, return the first item
        # The engine will call this block multiple times with _iteration_index
        iteration_index = context.variables.get("_iteration_index", 0)
        total = len(items)

        if iteration_index >= total:
            # Iteration complete
            return BlockResult(
                outputs={
                    "item": None,
                    "index": iteration_index,
                    "total": total,
                    "context": ctx,
                    "is_first": False,
                    "is_last": True,
                },
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"iteration_complete": True, "total_items": total},
            )

        current_item = items[iteration_index]

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "item": current_item,
                "index": iteration_index,
                "total": total,
                "context": ctx,
                "is_first": iteration_index == 0,
                "is_last": iteration_index == total - 1,
            },
            duration_ms=round(duration, 2),
            metrics={
                "current_index": iteration_index,
                "total_items": total,
                "items_remaining": total - iteration_index - 1,
            },
        )


class CollectBlock(BaseBlock):
    """
    Collect Block

    Collects results from ForEach iterations back into an array.
    """

    block_type = "collect"
    display_name = "Collect"
    description = "Collect iteration results into array"

    input_ports = [
        {"name": "item", "type": "any", "required": True, "description": "Item from each iteration"},
        {"name": "index", "type": "number", "required": False, "description": "Index from ForEach"},
    ]
    output_ports = [
        {"name": "results", "type": "array", "description": "Collected results array"},
        {"name": "count", "type": "number", "description": "Number of collected items"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "filter_nulls": {"type": "boolean", "default": True, "description": "Exclude null/None values"},
            "flatten": {"type": "boolean", "default": False, "description": "Flatten nested arrays"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Collect iteration results."""
        start_time = time.time()

        item = inputs.get("item")
        filter_nulls = config.get("filter_nulls", True)
        flatten = config.get("flatten", False)

        # Get or initialize results array from context
        results_key = f"_collect_{context.execution_id}"
        results = context.variables.get(results_key, [])

        # Add item
        if not (filter_nulls and item is None):
            if flatten and isinstance(item, list):
                results.extend(item)
            else:
                results.append(item)

        # Store back in context
        context.variables[results_key] = results

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "results": results,
                "count": len(results),
            },
            duration_ms=round(duration, 2),
            metrics={"collected_count": len(results)},
        )


class MapBlock(BaseBlock):
    """
    Map Block

    Applies a transformation to each item in an array.
    Simpler than ForEach for basic transformations.
    """

    block_type = "map"
    display_name = "Map"
    description = "Transform each item in array"

    input_ports = [
        {"name": "items", "type": "array", "required": True, "description": "Array of items"},
    ]
    output_ports = [
        {"name": "results", "type": "array", "description": "Transformed items"},
        {"name": "count", "type": "number", "description": "Number of items"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "extract_field": {"type": "string", "description": "Extract this field from each item"},
            "add_field": {"type": "string", "description": "Add a field to each item"},
            "add_value": {"type": "string", "description": "Value for added field"},
            "transform": {
                "type": "string",
                "enum": ["none", "to_string", "to_number", "to_bbox_array", "extract_crop"],
                "default": "none",
            },
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Transform array items."""
        start_time = time.time()

        items = inputs.get("items", [])
        if not isinstance(items, list):
            items = [items] if items else []

        extract_field = config.get("extract_field")
        add_field = config.get("add_field")
        add_value = config.get("add_value")
        transform = config.get("transform", "none")

        results = []

        for item in items:
            result = item

            # Extract field
            if extract_field and isinstance(item, dict):
                result = item.get(extract_field)

            # Add field
            if add_field and isinstance(result, dict):
                result = {**result, add_field: add_value}
            elif add_field and result is not None:
                result = {add_field: add_value, "value": result}

            # Transform
            if transform == "to_string":
                result = str(result) if result is not None else None
            elif transform == "to_number":
                try:
                    result = float(result) if result is not None else None
                except (ValueError, TypeError):
                    result = None
            elif transform == "to_bbox_array" and isinstance(item, dict):
                # Convert detection to [x1, y1, x2, y2] array
                bbox = item.get("bbox", item.get("box", {}))
                if isinstance(bbox, dict):
                    result = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
                elif isinstance(bbox, list):
                    result = bbox
            elif transform == "extract_crop" and isinstance(item, dict):
                # Get crop image from detection
                result = item.get("crop", item.get("image"))

            results.append(result)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "results": results,
                "count": len(results),
            },
            duration_ms=round(duration, 2),
            metrics={"item_count": len(results)},
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
