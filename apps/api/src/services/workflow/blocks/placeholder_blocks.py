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
        {"name": "result", "type": "boolean", "description": "Condition result (true/false)"},
        {"name": "passed", "type": "boolean", "description": "Alias for result (backward compat)"},
        {"name": "true_output", "type": "any", "description": "Output when condition is true"},
        {"name": "false_output", "type": "any", "description": "Output when condition is false"},
        {"name": "matched_conditions", "type": "array", "description": "Which conditions matched"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Single condition fields (frontend format)
            "field": {"type": "string", "description": "Field to evaluate (dot notation)"},
            "operator": {
                "type": "string",
                "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "in", "nin", "contains", "matches", "exists", "empty",
                         # Legacy operators (also supported)
                         "equals", "not_equals", "greater_than", "greater_equal", "less_than", "less_equal",
                         "is_true", "is_false", "array_empty", "array_not_empty", "array_contains"],
                "default": "gt",
            },
            "value": {"type": "string", "description": "Value to compare against"},
            "true_value": {"type": "string", "default": "pass", "description": "Output when true"},
            "false_value": {"type": "string", "default": "fail", "description": "Output when false"},
            # Multi-condition support (advanced)
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

    # Operator mapping from frontend short names to backend names
    OPERATOR_MAP = {
        "eq": "equals",
        "neq": "not_equals",
        "gt": "greater_than",
        "gte": "greater_equal",
        "lt": "less_than",
        "lte": "less_equal",
        "in": "in_array",
        "nin": "not_in_array",
        "contains": "contains",
        "matches": "regex_match",
        "exists": "exists",
        "empty": "array_empty",
    }

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

        # Frontend format support: single condition with field, operator, value
        if not conditions and config.get("operator"):
            field = config.get("field", "")
            operator = config.get("operator", "gt")
            compare_value = config.get("value", "")
            # Map frontend operator to backend operator
            operator = self.OPERATOR_MAP.get(operator, operator)
            conditions = [{"field": field, "operator": operator, "value": compare_value, "type": "auto"}]
        # Legacy support: single condition with compare_value
        elif not conditions:
            operator = config.get("operator", "exists")
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

        # Build matched_conditions list
        matched_conditions = [
            i for i, r in enumerate(condition_results) if r
        ]

        outputs = {
            "result": result,  # Frontend expects this
            "passed": result,  # Backward compat
            "true_output": value if result else None,
            "false_output": None if result else value,
            "matched_conditions": matched_conditions,
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
        {"name": "passed_count", "type": "number", "description": "Count of items that passed"},
        {"name": "failed_count", "type": "number", "description": "Count of items that failed"},
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
            "passed_count": len(passed),
            "failed_count": len(rejected),
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
            # Frontend uses max_iterations, backend also accepts max_items for backward compat
            "max_iterations": {"type": "number", "default": 0, "description": "Maximum items to process (0 = all)"},
            "max_items": {"type": "number", "description": "Alias for max_iterations (backward compat)"},
            "start_index": {"type": "number", "default": 0, "description": "Start from this index"},
            "parallel": {"type": "boolean", "default": False, "description": "Process items in parallel"},
            "batch_size": {"type": "number", "default": 10, "description": "Items per batch for parallel processing"},
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
        # Support both max_iterations (frontend) and max_items (legacy)
        max_items = config.get("max_iterations") or config.get("max_items", 0)
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
            "unique": {"type": "boolean", "default": False, "description": "Remove duplicate values"},
            "unique_key": {"type": "string", "description": "Field to use for uniqueness check (for objects)"},
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
        unique = config.get("unique", False)
        unique_key = config.get("unique_key")

        # Get or initialize results array from context
        results_key = f"_collect_{context.execution_id}"
        results = context.variables.get(results_key, [])
        seen_keys_key = f"_collect_seen_{context.execution_id}"
        seen_keys = context.variables.get(seen_keys_key, set())

        # Add item
        if not (filter_nulls and item is None):
            should_add = True

            # Check uniqueness if enabled
            if unique:
                if unique_key and isinstance(item, dict):
                    # Use specific field for uniqueness
                    key_value = item.get(unique_key)
                    if key_value in seen_keys:
                        should_add = False
                    else:
                        seen_keys.add(key_value)
                else:
                    # Use item itself for uniqueness (for primitives)
                    try:
                        # Try to use item as hashable key
                        item_key = item if not isinstance(item, (dict, list)) else str(item)
                        if item_key in seen_keys:
                            should_add = False
                        else:
                            seen_keys.add(item_key)
                    except TypeError:
                        # Unhashable, skip uniqueness check
                        pass

            if should_add:
                if flatten and isinstance(item, list):
                    results.extend(item)
                else:
                    results.append(item)

        # Store back in context
        context.variables[results_key] = results
        context.variables[seen_keys_key] = seen_keys

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "results": results,
                "count": len(results),
            },
            duration_ms=round(duration, 2),
            metrics={"collected_count": len(results), "unique_enabled": unique},
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


class APIResponseBlock(BaseBlock):
    """
    API Response Block - SOTA

    Formats output as proper REST API response with:
    - Pagination support (offset/limit, cursor-based)
    - Structured metadata (execution time, workflow info, timestamps)
    - Consistent success/error format
    - Schema validation option
    - Multiple response formats (standard, HAL, JSON:API)
    """

    block_type = "api_response"
    display_name = "API Response"
    description = "Format as REST API response with pagination and metadata"

    input_ports = [
        {"name": "data", "type": "any", "required": True, "description": "Data to include in response"},
        {"name": "total_count", "type": "number", "required": False, "description": "Total count for pagination"},
        {"name": "errors", "type": "array", "required": False, "description": "Error messages if any"},
    ]
    output_ports = [
        {"name": "response", "type": "object", "description": "Formatted API response"},
        {"name": "status_code", "type": "number", "description": "HTTP status code"},
        {"name": "headers", "type": "object", "description": "Suggested response headers"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["standard", "hal", "jsonapi"],
                "default": "standard",
                "description": "Response format style",
            },
            "pagination_enabled": {"type": "boolean", "default": True},
            "pagination_style": {
                "type": "string",
                "enum": ["offset", "cursor", "page"],
                "default": "offset",
            },
            "page": {"type": "number", "default": 1},
            "per_page": {"type": "number", "default": 50},
            "include_metadata": {"type": "boolean", "default": True},
            "include_timing": {"type": "boolean", "default": True},
            "include_links": {"type": "boolean", "default": False},
            "resource_name": {"type": "string", "default": "items"},
            "base_url": {"type": "string", "default": "/api/v1"},
            "wrap_single": {"type": "boolean", "default": True, "description": "Wrap single item in array"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Format data as API response."""
        import datetime

        start_time = time.time()

        data = inputs.get("data")
        total_count = inputs.get("total_count")
        errors = inputs.get("errors", [])

        response_format = config.get("format", "standard")
        pagination_enabled = config.get("pagination_enabled", True)
        pagination_style = config.get("pagination_style", "offset")
        page = config.get("page", 1)
        per_page = config.get("per_page", 50)
        include_metadata = config.get("include_metadata", True)
        include_timing = config.get("include_timing", True)
        include_links = config.get("include_links", False)
        resource_name = config.get("resource_name", "items")
        base_url = config.get("base_url", "/api/v1")
        wrap_single = config.get("wrap_single", True)

        # Normalize data to list if needed
        if data is None:
            items = []
        elif isinstance(data, list):
            items = data
        elif wrap_single:
            items = [data]
        else:
            items = data

        # Calculate pagination
        if isinstance(items, list):
            actual_total = total_count if total_count is not None else len(items)
            total_pages = (actual_total + per_page - 1) // per_page if per_page > 0 else 1
            has_next = page < total_pages
            has_prev = page > 1

            # Apply pagination if needed (for offset style)
            if pagination_enabled and pagination_style == "offset":
                offset = (page - 1) * per_page
                paginated_items = items[offset:offset + per_page] if len(items) > per_page else items
            else:
                paginated_items = items
        else:
            paginated_items = items
            actual_total = 1
            total_pages = 1
            has_next = False
            has_prev = False

        # Determine status
        has_errors = bool(errors)
        status_code = 400 if has_errors else 200

        # Build response based on format
        if response_format == "standard":
            response = self._build_standard_response(
                paginated_items, errors, actual_total, page, per_page,
                has_next, has_prev, total_pages, pagination_enabled,
                include_metadata, include_timing, context, start_time, resource_name
            )
        elif response_format == "hal":
            response = self._build_hal_response(
                paginated_items, actual_total, page, per_page,
                has_next, has_prev, base_url, resource_name,
                include_metadata, include_timing, context, start_time
            )
        elif response_format == "jsonapi":
            response = self._build_jsonapi_response(
                paginated_items, errors, actual_total, page, per_page,
                has_next, has_prev, base_url, resource_name,
                include_metadata, include_timing, context, start_time
            )
        else:
            response = {"data": paginated_items}

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "X-Total-Count": str(actual_total),
            "X-Page": str(page),
            "X-Per-Page": str(per_page),
        }

        if include_links and pagination_enabled:
            links = []
            if has_prev:
                links.append(f'<{base_url}/{resource_name}?page={page - 1}&per_page={per_page}>; rel="prev"')
            if has_next:
                links.append(f'<{base_url}/{resource_name}?page={page + 1}&per_page={per_page}>; rel="next"')
            links.append(f'<{base_url}/{resource_name}?page=1&per_page={per_page}>; rel="first"')
            links.append(f'<{base_url}/{resource_name}?page={total_pages}&per_page={per_page}>; rel="last"')
            headers["Link"] = ", ".join(links)

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "response": response,
                "status_code": status_code,
                "headers": headers,
            },
            duration_ms=round(duration, 2),
            metrics={
                "items_count": len(paginated_items) if isinstance(paginated_items, list) else 1,
                "total_count": actual_total,
                "page": page,
                "format": response_format,
            },
        )

    def _build_standard_response(
        self, items, errors, total, page, per_page,
        has_next, has_prev, total_pages, pagination_enabled,
        include_metadata, include_timing, context, start_time, resource_name
    ):
        """Build standard API response format."""
        import datetime

        response = {
            "success": not bool(errors),
            "data": items,
        }

        if errors:
            response["errors"] = errors

        if pagination_enabled and isinstance(items, list):
            response["pagination"] = {
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
                "count": len(items),
            }

        if include_metadata:
            response["meta"] = {
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if include_timing:
                response["meta"]["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return response

    def _build_hal_response(
        self, items, total, page, per_page,
        has_next, has_prev, base_url, resource_name,
        include_metadata, include_timing, context, start_time
    ):
        """Build HAL (Hypertext Application Language) response format."""
        import datetime

        response = {
            "_embedded": {
                resource_name: items,
            },
            "_links": {
                "self": {"href": f"{base_url}/{resource_name}?page={page}&per_page={per_page}"},
                "first": {"href": f"{base_url}/{resource_name}?page=1&per_page={per_page}"},
            },
            "page": page,
            "per_page": per_page,
            "total": total,
        }

        if has_next:
            response["_links"]["next"] = {"href": f"{base_url}/{resource_name}?page={page + 1}&per_page={per_page}"}
        if has_prev:
            response["_links"]["prev"] = {"href": f"{base_url}/{resource_name}?page={page - 1}&per_page={per_page}"}

        if include_metadata:
            response["_meta"] = {
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if include_timing:
                response["_meta"]["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return response

    def _build_jsonapi_response(
        self, items, errors, total, page, per_page,
        has_next, has_prev, base_url, resource_name,
        include_metadata, include_timing, context, start_time
    ):
        """Build JSON:API specification response format."""
        import datetime

        # Transform items to JSON:API format
        jsonapi_items = []
        for i, item in enumerate(items if isinstance(items, list) else [items]):
            item_id = item.get("id", i) if isinstance(item, dict) else i
            jsonapi_items.append({
                "type": resource_name,
                "id": str(item_id),
                "attributes": item if isinstance(item, dict) else {"value": item},
            })

        response = {
            "data": jsonapi_items,
            "links": {
                "self": f"{base_url}/{resource_name}?page[number]={page}&page[size]={per_page}",
                "first": f"{base_url}/{resource_name}?page[number]=1&page[size]={per_page}",
            },
        }

        if errors:
            response["errors"] = [{"detail": str(e)} for e in errors]

        if has_next:
            response["links"]["next"] = f"{base_url}/{resource_name}?page[number]={page + 1}&page[size]={per_page}"
        if has_prev:
            response["links"]["prev"] = f"{base_url}/{resource_name}?page[number]={page - 1}&page[size]={per_page}"

        if include_metadata:
            response["meta"] = {
                "total": total,
                "workflow_id": context.workflow_id,
                "execution_id": context.execution_id,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            if include_timing:
                response["meta"]["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return response


class WebhookBlock(BaseBlock):
    """
    Webhook Block - SOTA

    Push results to external APIs with:
    - Multiple HTTP methods (POST, PUT, PATCH)
    - Authentication (API Key, Bearer Token, Basic Auth, OAuth2)
    - Retry logic with exponential backoff
    - Batching support for large datasets
    - Request/response transformation
    - Async fire-and-forget option
    """

    block_type = "webhook"
    display_name = "Webhook"
    description = "Push results to external API endpoint"

    input_ports = [
        {"name": "data", "type": "any", "required": True, "description": "Data to send"},
        {"name": "url_override", "type": "string", "required": False, "description": "Override configured URL"},
    ]
    output_ports = [
        {"name": "response", "type": "object", "description": "API response"},
        {"name": "status_code", "type": "number", "description": "HTTP status code"},
        {"name": "success", "type": "boolean", "description": "Whether request succeeded"},
        {"name": "request_id", "type": "string", "description": "Request ID for tracking"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Webhook URL"},
            "method": {
                "type": "string",
                "enum": ["POST", "PUT", "PATCH"],
                "default": "POST",
            },
            "auth_type": {
                "type": "string",
                "enum": ["none", "api_key", "bearer", "basic", "oauth2"],
                "default": "none",
            },
            "auth_header": {"type": "string", "default": "Authorization"},
            "auth_value": {"type": "string", "description": "API key, token, or credentials"},
            "auth_prefix": {"type": "string", "default": "Bearer", "description": "Prefix for auth header"},
            "headers": {"type": "object", "default": {}, "description": "Additional headers"},
            "timeout": {"type": "number", "default": 30, "description": "Request timeout in seconds"},
            "retry_enabled": {"type": "boolean", "default": True},
            "retry_max": {"type": "number", "default": 3},
            "retry_delay": {"type": "number", "default": 1, "description": "Initial retry delay in seconds"},
            "batch_enabled": {"type": "boolean", "default": False},
            "batch_size": {"type": "number", "default": 100},
            "batch_field": {"type": "string", "default": "items", "description": "Field name for batch items"},
            "include_metadata": {"type": "boolean", "default": True},
            "wrap_data": {"type": "boolean", "default": True, "description": "Wrap data in object"},
            "data_field": {"type": "string", "default": "data", "description": "Field name for data"},
            "fire_and_forget": {"type": "boolean", "default": False, "description": "Don't wait for response"},
        },
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Send data to webhook endpoint."""
        import asyncio
        import uuid
        import datetime

        try:
            import httpx
            has_httpx = True
        except ImportError:
            has_httpx = False

        start_time = time.time()

        data = inputs.get("data")
        url = inputs.get("url_override") or config.get("url", "")

        if not url:
            return BlockResult(
                outputs={
                    "response": {"error": "No webhook URL configured"},
                    "status_code": 400,
                    "success": False,
                    "request_id": None,
                },
                duration_ms=round((time.time() - start_time) * 1000, 2),
                metrics={"error": "missing_url"},
            )

        method = config.get("method", "POST")
        auth_type = config.get("auth_type", "none")
        auth_header = config.get("auth_header", "Authorization")
        auth_value = config.get("auth_value", "")
        auth_prefix = config.get("auth_prefix", "Bearer")
        extra_headers = config.get("headers", {})
        timeout = config.get("timeout", 30)
        retry_enabled = config.get("retry_enabled", True)
        retry_max = config.get("retry_max", 3)
        retry_delay = config.get("retry_delay", 1)
        batch_enabled = config.get("batch_enabled", False)
        batch_size = config.get("batch_size", 100)
        batch_field = config.get("batch_field", "items")
        include_metadata = config.get("include_metadata", True)
        wrap_data = config.get("wrap_data", True)
        data_field = config.get("data_field", "data")
        fire_and_forget = config.get("fire_and_forget", False)

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": request_id,
            "X-Workflow-ID": context.workflow_id,
            **extra_headers,
        }

        # Add authentication
        if auth_type == "api_key":
            headers[auth_header] = auth_value
        elif auth_type == "bearer":
            headers[auth_header] = f"{auth_prefix} {auth_value}"
        elif auth_type == "basic":
            import base64
            encoded = base64.b64encode(auth_value.encode()).decode()
            headers[auth_header] = f"Basic {encoded}"

        # Build request body
        if wrap_data:
            body = {data_field: data}
            if include_metadata:
                body["metadata"] = {
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id,
                    "request_id": request_id,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
        else:
            body = data

        # Handle batching
        if batch_enabled and isinstance(data, list) and len(data) > batch_size:
            batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            responses = []
            all_success = True

            for batch_idx, batch in enumerate(batches):
                batch_body = {batch_field: batch, "batch_index": batch_idx, "total_batches": len(batches)}
                if include_metadata:
                    batch_body["metadata"] = {
                        "workflow_id": context.workflow_id,
                        "execution_id": context.execution_id,
                        "request_id": f"{request_id}-batch-{batch_idx}",
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    }

                result = await self._send_request(
                    url, method, batch_body, headers, timeout,
                    retry_enabled, retry_max, retry_delay, has_httpx
                )
                responses.append(result)
                if not result.get("success"):
                    all_success = False

            duration = (time.time() - start_time) * 1000
            return BlockResult(
                outputs={
                    "response": {"batches": responses, "total_batches": len(batches)},
                    "status_code": 200 if all_success else 207,  # 207 Multi-Status
                    "success": all_success,
                    "request_id": request_id,
                },
                duration_ms=round(duration, 2),
                metrics={
                    "batches_sent": len(batches),
                    "items_sent": len(data),
                    "all_success": all_success,
                },
            )

        # Fire and forget mode
        if fire_and_forget:
            asyncio.create_task(self._send_request(
                url, method, body, headers, timeout,
                retry_enabled, retry_max, retry_delay, has_httpx
            ))
            duration = (time.time() - start_time) * 1000
            return BlockResult(
                outputs={
                    "response": {"status": "sent_async"},
                    "status_code": 202,  # Accepted
                    "success": True,
                    "request_id": request_id,
                },
                duration_ms=round(duration, 2),
                metrics={"async": True},
            )

        # Normal request
        result = await self._send_request(
            url, method, body, headers, timeout,
            retry_enabled, retry_max, retry_delay, has_httpx
        )

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "response": result.get("body", {}),
                "status_code": result.get("status_code", 500),
                "success": result.get("success", False),
                "request_id": request_id,
            },
            duration_ms=round(duration, 2),
            metrics={
                "status_code": result.get("status_code"),
                "retries": result.get("retries", 0),
                "success": result.get("success"),
            },
        )

    async def _send_request(
        self, url, method, body, headers, timeout,
        retry_enabled, retry_max, retry_delay, has_httpx
    ):
        """Send HTTP request with retry logic."""
        import asyncio
        import json

        retries = 0
        last_error = None

        while retries <= (retry_max if retry_enabled else 0):
            try:
                if has_httpx:
                    import httpx
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        if method == "POST":
                            response = await client.post(url, json=body, headers=headers)
                        elif method == "PUT":
                            response = await client.put(url, json=body, headers=headers)
                        elif method == "PATCH":
                            response = await client.patch(url, json=body, headers=headers)
                        else:
                            response = await client.post(url, json=body, headers=headers)

                        try:
                            response_body = response.json()
                        except Exception:
                            response_body = {"text": response.text}

                        return {
                            "success": 200 <= response.status_code < 300,
                            "status_code": response.status_code,
                            "body": response_body,
                            "retries": retries,
                        }
                else:
                    # Fallback to aiohttp or urllib
                    import urllib.request
                    import urllib.error

                    req = urllib.request.Request(
                        url,
                        data=json.dumps(body).encode("utf-8"),
                        headers=headers,
                        method=method,
                    )

                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        response_body = json.loads(response.read().decode("utf-8"))
                        return {
                            "success": True,
                            "status_code": response.status,
                            "body": response_body,
                            "retries": retries,
                        }

            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries <= retry_max and retry_enabled:
                    # Exponential backoff
                    await asyncio.sleep(retry_delay * (2 ** (retries - 1)))

        return {
            "success": False,
            "status_code": 500,
            "body": {"error": last_error},
            "retries": retries,
        }


class AggregationBlock(BaseBlock):
    """
    Aggregation Block - SOTA

    Pre-output processing for complex nested data:
    - Flatten nested arrays (e.g., detections -> matches)
    - Group by field
    - Calculate statistics (count, sum, avg, min, max)
    - Top N per group
    - Pivot tables
    - Custom aggregation expressions
    """

    block_type = "aggregation"
    display_name = "Aggregation"
    description = "Aggregate, flatten, and summarize data"

    input_ports = [
        {"name": "data", "type": "any", "required": True, "description": "Data to aggregate"},
    ]
    output_ports = [
        {"name": "result", "type": "any", "description": "Aggregated result"},
        {"name": "summary", "type": "object", "description": "Summary statistics"},
        {"name": "flat", "type": "array", "description": "Flattened records"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["flatten", "group", "stats", "top_n", "pivot", "dedupe"],
                "default": "flatten",
            },
            "flatten_config": {
                "type": "object",
                "properties": {
                    "parent_field": {"type": "string", "description": "Field containing parent data"},
                    "child_field": {"type": "string", "description": "Field containing nested array"},
                    "parent_prefix": {"type": "string", "default": "parent_"},
                    "child_prefix": {"type": "string", "default": ""},
                    "include_parent_fields": {"type": "array", "items": {"type": "string"}},
                    "include_child_fields": {"type": "array", "items": {"type": "string"}},
                    "add_index": {"type": "boolean", "default": True},
                },
            },
            "group_config": {
                "type": "object",
                "properties": {
                    "by": {"type": "string", "description": "Field to group by"},
                    "agg_field": {"type": "string", "description": "Field to aggregate"},
                    "agg_func": {
                        "type": "string",
                        "enum": ["count", "sum", "avg", "min", "max", "first", "last", "collect"],
                        "default": "count",
                    },
                    "sort_by": {"type": "string", "enum": ["key", "value"], "default": "value"},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                },
            },
            "stats_config": {
                "type": "object",
                "properties": {
                    "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to calculate stats for"},
                    "include_distribution": {"type": "boolean", "default": False},
                    "percentiles": {"type": "array", "items": {"type": "number"}, "default": [25, 50, 75, 90, 95, 99]},
                },
            },
            "top_n_config": {
                "type": "object",
                "properties": {
                    "n": {"type": "number", "default": 10},
                    "by": {"type": "string", "description": "Field to sort by"},
                    "order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"},
                    "group_by": {"type": "string", "description": "Get top N per group"},
                },
            },
            "pivot_config": {
                "type": "object",
                "properties": {
                    "index": {"type": "string", "description": "Row index field"},
                    "columns": {"type": "string", "description": "Column header field"},
                    "values": {"type": "string", "description": "Values field"},
                    "agg_func": {"type": "string", "enum": ["count", "sum", "avg", "first"], "default": "count"},
                },
            },
            "dedupe_config": {
                "type": "object",
                "properties": {
                    "by": {"type": "string", "description": "Field to deduplicate by"},
                    "keep": {"type": "string", "enum": ["first", "last", "highest", "lowest"], "default": "first"},
                    "score_field": {"type": "string", "description": "Field for highest/lowest comparison"},
                },
            },
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

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Perform aggregation operation."""
        start_time = time.time()

        data = inputs.get("data", [])
        operation = config.get("operation", "flatten")

        if not isinstance(data, list):
            data = [data] if data else []

        result = data
        summary = {}
        flat = []

        if operation == "flatten":
            flat = self._flatten(data, config.get("flatten_config", {}))
            result = flat
            summary = {"original_count": len(data), "flattened_count": len(flat)}

        elif operation == "group":
            result, summary = self._group(data, config.get("group_config", {}))

        elif operation == "stats":
            summary = self._calculate_stats(data, config.get("stats_config", {}))
            result = data

        elif operation == "top_n":
            result = self._top_n(data, config.get("top_n_config", {}))
            summary = {"original_count": len(data), "result_count": len(result)}

        elif operation == "pivot":
            result = self._pivot(data, config.get("pivot_config", {}))
            summary = {"rows": len(result)}

        elif operation == "dedupe":
            result = self._dedupe(data, config.get("dedupe_config", {}))
            summary = {"original_count": len(data), "deduped_count": len(result), "removed": len(data) - len(result)}

        duration = (time.time() - start_time) * 1000

        return BlockResult(
            outputs={
                "result": result,
                "summary": summary,
                "flat": flat if flat else result,
            },
            duration_ms=round(duration, 2),
            metrics={
                "operation": operation,
                "input_count": len(data),
                "output_count": len(result) if isinstance(result, list) else 1,
            },
        )

    def _flatten(self, data: list, config: dict) -> list:
        """Flatten nested arrays into flat records."""
        child_field = config.get("child_field", "matches")
        parent_prefix = config.get("parent_prefix", "parent_")
        child_prefix = config.get("child_prefix", "")
        include_parent_fields = config.get("include_parent_fields")
        include_child_fields = config.get("include_child_fields")
        add_index = config.get("add_index", True)

        flat_records = []
        parent_idx = 0

        for item in data:
            if not isinstance(item, dict):
                flat_records.append({"value": item})
                continue

            # Get nested array
            children = self._get_field_value(item, child_field)
            if not isinstance(children, list):
                children = [children] if children else [None]

            # Get parent fields
            parent_data = {}
            if include_parent_fields:
                for field in include_parent_fields:
                    val = self._get_field_value(item, field)
                    parent_data[f"{parent_prefix}{field}"] = val
            else:
                # Include all non-nested fields
                for key, val in item.items():
                    if key != child_field and not isinstance(val, (list, dict)):
                        parent_data[f"{parent_prefix}{key}"] = val
                    elif key != child_field and isinstance(val, dict):
                        # Flatten nested dict
                        for sub_key, sub_val in val.items():
                            if not isinstance(sub_val, (list, dict)):
                                parent_data[f"{parent_prefix}{key}_{sub_key}"] = sub_val

            # Create flat record for each child
            for child_idx, child in enumerate(children):
                record = {}

                if add_index:
                    record["_parent_index"] = parent_idx
                    record["_child_index"] = child_idx

                # Add parent fields
                record.update(parent_data)

                # Add child fields
                if isinstance(child, dict):
                    if include_child_fields:
                        for field in include_child_fields:
                            val = self._get_field_value(child, field)
                            record[f"{child_prefix}{field}"] = val
                    else:
                        for key, val in child.items():
                            if not isinstance(val, (list, dict)):
                                record[f"{child_prefix}{key}"] = val
                elif child is not None:
                    record[f"{child_prefix}value"] = child

                flat_records.append(record)

            parent_idx += 1

        return flat_records

    def _group(self, data: list, config: dict) -> tuple:
        """Group data by field and aggregate."""
        group_by = config.get("by", "class_name")
        agg_field = config.get("agg_field")
        agg_func = config.get("agg_func", "count")
        sort_by = config.get("sort_by", "value")
        sort_order = config.get("sort_order", "desc")

        groups = {}

        for item in data:
            key = str(self._get_field_value(item, group_by) or "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        result = []
        for key, items in groups.items():
            entry = {"key": key, "count": len(items)}

            if agg_field:
                values = []
                for item in items:
                    val = self._get_field_value(item, agg_field)
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if values:
                    if agg_func == "sum":
                        entry["value"] = sum(values)
                    elif agg_func == "avg":
                        entry["value"] = sum(values) / len(values)
                    elif agg_func == "min":
                        entry["value"] = min(values)
                    elif agg_func == "max":
                        entry["value"] = max(values)
                    elif agg_func == "first":
                        entry["value"] = values[0]
                    elif agg_func == "last":
                        entry["value"] = values[-1]
                    elif agg_func == "collect":
                        entry["values"] = values
                    else:
                        entry["value"] = len(values)
                else:
                    entry["value"] = 0
            else:
                entry["value"] = len(items)

            entry["items"] = items
            result.append(entry)

        # Sort
        sort_key = "value" if sort_by == "value" else "key"
        result.sort(key=lambda x: x.get(sort_key, 0), reverse=(sort_order == "desc"))

        summary = {
            "total_groups": len(result),
            "total_items": len(data),
            "top_group": result[0]["key"] if result else None,
            "top_value": result[0]["value"] if result else None,
        }

        return result, summary

    def _calculate_stats(self, data: list, config: dict) -> dict:
        """Calculate statistics for numeric fields."""
        fields = config.get("fields", ["confidence", "score", "similarity"])
        include_distribution = config.get("include_distribution", False)
        percentiles = config.get("percentiles", [25, 50, 75, 90, 95, 99])

        stats = {
            "count": len(data),
        }

        for field in fields:
            values = []
            for item in data:
                val = self._get_field_value(item, field)
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass

            if values:
                values.sort()
                field_stats = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                }

                if include_distribution and len(values) >= 2:
                    for p in percentiles:
                        idx = int(len(values) * p / 100)
                        field_stats[f"p{p}"] = values[min(idx, len(values) - 1)]

                    # Standard deviation
                    avg = field_stats["avg"]
                    variance = sum((v - avg) ** 2 for v in values) / len(values)
                    field_stats["std"] = variance ** 0.5

                stats[field] = field_stats

        return stats

    def _top_n(self, data: list, config: dict) -> list:
        """Get top N items, optionally per group."""
        n = config.get("n", 10)
        by_field = config.get("by", "confidence")
        order = config.get("order", "desc")
        group_by = config.get("group_by")

        if group_by:
            # Top N per group
            groups = {}
            for item in data:
                key = str(self._get_field_value(item, group_by) or "unknown")
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)

            result = []
            for key, items in groups.items():
                sorted_items = sorted(
                    items,
                    key=lambda x: self._get_field_value(x, by_field) or 0,
                    reverse=(order == "desc"),
                )
                result.extend(sorted_items[:n])

            return result
        else:
            # Global top N
            sorted_data = sorted(
                data,
                key=lambda x: self._get_field_value(x, by_field) or 0,
                reverse=(order == "desc"),
            )
            return sorted_data[:n]

    def _pivot(self, data: list, config: dict) -> list:
        """Create pivot table."""
        index_field = config.get("index", "row")
        columns_field = config.get("columns", "column")
        values_field = config.get("values", "value")
        agg_func = config.get("agg_func", "count")

        # Group by index and column
        pivot_data = {}
        all_columns = set()

        for item in data:
            idx = str(self._get_field_value(item, index_field) or "unknown")
            col = str(self._get_field_value(item, columns_field) or "unknown")
            val = self._get_field_value(item, values_field)

            all_columns.add(col)

            if idx not in pivot_data:
                pivot_data[idx] = {}
            if col not in pivot_data[idx]:
                pivot_data[idx][col] = []

            if val is not None:
                try:
                    pivot_data[idx][col].append(float(val))
                except (ValueError, TypeError):
                    pivot_data[idx][col].append(1)

        # Aggregate
        result = []
        for idx, cols in pivot_data.items():
            row = {"index": idx}
            for col in all_columns:
                values = cols.get(col, [])
                if agg_func == "count":
                    row[col] = len(values)
                elif agg_func == "sum":
                    row[col] = sum(values) if values else 0
                elif agg_func == "avg":
                    row[col] = sum(values) / len(values) if values else 0
                elif agg_func == "first":
                    row[col] = values[0] if values else None
            result.append(row)

        return result

    def _dedupe(self, data: list, config: dict) -> list:
        """Remove duplicate items."""
        by_field = config.get("by", "id")
        keep = config.get("keep", "first")
        score_field = config.get("score_field", "confidence")

        seen = {}

        for item in data:
            key = str(self._get_field_value(item, by_field) or id(item))

            if key not in seen:
                seen[key] = item
            else:
                if keep == "last":
                    seen[key] = item
                elif keep in ("highest", "lowest"):
                    current_score = self._get_field_value(seen[key], score_field) or 0
                    new_score = self._get_field_value(item, score_field) or 0
                    try:
                        if keep == "highest" and float(new_score) > float(current_score):
                            seen[key] = item
                        elif keep == "lowest" and float(new_score) < float(current_score):
                            seen[key] = item
                    except (ValueError, TypeError):
                        pass

        return list(seen.values())
