"""
Workflow Engine - Main Execution Engine

Handles workflow execution: parsing, topological sorting, and block execution.
Supports iteration (ForEach → Collect patterns) for batch processing.
"""

import time
import logging
import asyncio
from typing import Any, Optional
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field

from .base import BaseBlock, BlockResult, ExecutionContext

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Exception raised during workflow execution."""

    def __init__(self, message: str, node_id: Optional[str] = None):
        self.message = message
        self.node_id = node_id
        super().__init__(message)


@dataclass
class IterationState:
    """Tracks iteration state for ForEach→Collect loops."""
    foreach_node_id: str
    collect_node_id: str
    loop_body_nodes: list[str]  # Node IDs in the loop body
    items: list[Any] = field(default_factory=list)
    current_index: int = 0
    collected_results: list[Any] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    iteration_mode: str = "sequential"  # sequential, parallel, batch
    batch_size: int = 1
    max_concurrency: int = 5
    on_error: str = "continue"  # continue, stop, collect_errors

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def is_complete(self) -> bool:
        return self.current_index >= self.total_items

    @property
    def is_first(self) -> bool:
        return self.current_index == 0

    @property
    def is_last(self) -> bool:
        return self.current_index == self.total_items - 1


class WorkflowEngine:
    """
    Main workflow execution engine.

    Handles:
    - Parsing workflow definitions
    - Topological sorting of nodes
    - Sequential execution of blocks
    - ForEach/Collect iteration loops
    - Reference resolution between nodes
    - Error handling and metrics collection
    """

    def __init__(self):
        self._blocks: dict[str, BaseBlock] = {}
        self._register_default_blocks()

    def _register_default_blocks(self):
        """Register all available block types."""
        # Import blocks here to avoid circular imports
        from .blocks import get_all_blocks

        for block_type, block_class in get_all_blocks().items():
            self._blocks[block_type] = block_class()

    def register_block(self, block_type: str, block: BaseBlock):
        """Register a custom block type."""
        self._blocks[block_type] = block

    def get_available_blocks(self) -> dict[str, dict]:
        """Get metadata for all available blocks."""
        return {
            block_type: {
                "type": block.block_type,
                "name": block.display_name,
                "description": block.description,
                "inputs": block.input_ports,
                "outputs": block.output_ports,
                "config_schema": block.config_schema,
            }
            for block_type, block in self._blocks.items()
        }

    async def execute(
        self,
        workflow: dict,
        inputs: dict,
        workflow_id: str,
        execution_id: str,
    ) -> dict:
        """
        Execute a workflow with given inputs.

        Args:
            workflow: Workflow definition (from wf_workflows.definition)
            inputs: Input data (image_url, image_base64, parameters)
            workflow_id: UUID of the workflow
            execution_id: UUID of this execution

        Returns:
            Dict with outputs, metrics, timing, and any errors
        """
        start_time = time.time()

        # Parse workflow
        nodes = workflow.get("nodes", [])
        edges = workflow.get("edges", [])
        outputs_config = workflow.get("outputs", [])
        param_definitions = workflow.get("parameters", [])

        # Resolve workflow parameters from definitions and runtime inputs
        parameters = self._resolve_parameters(param_definitions, inputs.get("parameters", {}))

        # Initialize context
        context = ExecutionContext(
            inputs=inputs,
            nodes={},
            workflow_id=workflow_id,
            execution_id=execution_id,
            parameters=parameters,
        )

        if not nodes:
            return {
                "outputs": {},
                "metrics": {},
                "duration_ms": 0,
                "error": None,
            }

        # Build execution order
        try:
            execution_order = self._topological_sort(nodes, edges)
        except Exception as e:
            return {
                "outputs": {},
                "metrics": {},
                "duration_ms": (time.time() - start_time) * 1000,
                "error": f"Failed to build execution order: {str(e)}",
                "error_node_id": None,
            }

        # Build input mappings from edges (React Flow style)
        edge_inputs = self._build_edge_inputs(edges)

        # Detect iteration loops (ForEach → Collect patterns)
        iteration_loops = self._detect_iteration_loops(nodes, edges, execution_order)

        # Execute nodes in order with iteration support
        metrics = {}
        error_info = None
        active_iterations: dict[str, IterationState] = {}  # foreach_node_id -> state

        i = 0
        while i < len(execution_order):
            node = execution_order[i]
            node_id = node["id"]
            node_type = node["type"]
            node_config = node.get("config", {})
            # Merge edge-based inputs with explicit node inputs (explicit takes precedence)
            node_inputs_config = {**edge_inputs.get(node_id, {}), **node.get("inputs", {})}

            # Get block handler
            block = self._blocks.get(node_type)
            if not block:
                error_info = {
                    "error": f"Unknown block type: {node_type}",
                    "error_node_id": node_id,
                }
                break

            # Resolve input references
            try:
                resolved_inputs = self._resolve_inputs(node_inputs_config, context)
            except Exception as e:
                error_info = {
                    "error": f"Failed to resolve inputs: {str(e)}",
                    "error_node_id": node_id,
                }
                break

            # Validate inputs
            validation_errors = block.validate_inputs(resolved_inputs)
            if validation_errors:
                error_info = {
                    "error": f"Input validation failed: {', '.join(validation_errors)}",
                    "error_node_id": node_id,
                }
                break

            # Resolve config values (handles parameter references in config)
            resolved_config = self._resolve_config(node_config, context)

            # Handle ForEach block - start iteration
            if node_type == "foreach" and node_id in iteration_loops:
                loop_info = iteration_loops[node_id]

                # Check if we're starting a new iteration or continuing
                if node_id not in active_iterations:
                    # Get items to iterate over
                    items = resolved_inputs.get("items", [])
                    if not isinstance(items, list):
                        items = [items] if items else []

                    # Apply config limits
                    start_idx = resolved_config.get("start_index", 0)
                    max_items = resolved_config.get("max_items", 0)
                    if start_idx > 0:
                        items = items[start_idx:]
                    if max_items > 0:
                        items = items[:max_items]

                    # Initialize iteration state
                    active_iterations[node_id] = IterationState(
                        foreach_node_id=node_id,
                        collect_node_id=loop_info["collect_node_id"],
                        loop_body_nodes=loop_info["loop_body"],
                        items=items,
                        iteration_mode=resolved_config.get("iteration_mode", "sequential"),
                        batch_size=resolved_config.get("batch_size", 1),
                        max_concurrency=resolved_config.get("max_concurrency", 5),
                        on_error=resolved_config.get("on_error", "continue"),
                    )

                    if not items:
                        # No items to iterate, output empty
                        context.nodes[node_id] = {
                            "item": None,
                            "index": 0,
                            "total": 0,
                            "is_first": True,
                            "is_last": True,
                            "context": resolved_inputs.get("context"),
                        }
                        metrics[node_id] = {
                            "type": node_type,
                            "duration_ms": 0,
                            "success": True,
                            "total_items": 0,
                            "skipped": True,
                        }
                        i += 1
                        continue

                iter_state = active_iterations[node_id]

                # Set iteration context for blocks to access
                context.variables["_iteration_index"] = iter_state.current_index
                context.variables["_iteration_total"] = iter_state.total_items
                context.variables["_iteration_foreach"] = node_id

                # Output current item
                current_item = iter_state.items[iter_state.current_index] if iter_state.current_index < iter_state.total_items else None
                context.nodes[node_id] = {
                    "item": current_item,
                    "index": iter_state.current_index,
                    "total": iter_state.total_items,
                    "is_first": iter_state.is_first,
                    "is_last": iter_state.is_last,
                    "context": resolved_inputs.get("context"),
                }

                metrics[node_id] = metrics.get(node_id, {
                    "type": node_type,
                    "duration_ms": 0,
                    "success": True,
                    "total_items": iter_state.total_items,
                    "iterations_completed": 0,
                })

                logger.info(f"ForEach iteration {iter_state.current_index + 1}/{iter_state.total_items}")
                i += 1
                continue

            # Handle Collect block - accumulate results and check if loop should continue
            if node_type == "collect":
                # Find if this collect is part of an active iteration
                parent_foreach = None
                for foreach_id, iter_state in active_iterations.items():
                    if iter_state.collect_node_id == node_id:
                        parent_foreach = foreach_id
                        break

                if parent_foreach:
                    iter_state = active_iterations[parent_foreach]

                    # Get the item to collect from this iteration
                    item_to_collect = resolved_inputs.get("item")
                    filter_nulls = resolved_config.get("filter_nulls", True)
                    flatten = resolved_config.get("flatten", False)

                    # Add to collected results
                    if not (filter_nulls and item_to_collect is None):
                        if flatten and isinstance(item_to_collect, list):
                            iter_state.collected_results.extend(item_to_collect)
                        else:
                            iter_state.collected_results.append(item_to_collect)

                    # Check if iteration is complete
                    iter_state.current_index += 1

                    if not iter_state.is_complete:
                        # More items to process - loop back to foreach
                        foreach_index = next(
                            (idx for idx, n in enumerate(execution_order) if n["id"] == parent_foreach),
                            None
                        )
                        if foreach_index is not None:
                            i = foreach_index
                            continue
                    else:
                        # Iteration complete - output final collected results
                        context.nodes[node_id] = {
                            "results": iter_state.collected_results,
                            "count": len(iter_state.collected_results),
                        }
                        metrics[node_id] = {
                            "type": node_type,
                            "duration_ms": 0,
                            "success": True,
                            "collected_count": len(iter_state.collected_results),
                            "iterations": iter_state.total_items,
                        }

                        # Update foreach metrics
                        if parent_foreach in metrics:
                            metrics[parent_foreach]["iterations_completed"] = iter_state.total_items

                        # Clean up iteration state
                        del active_iterations[parent_foreach]

                        # Clear iteration context
                        context.variables.pop("_iteration_index", None)
                        context.variables.pop("_iteration_total", None)
                        context.variables.pop("_iteration_foreach", None)

                        logger.info(f"Iteration complete. Collected {len(iter_state.collected_results)} items.")
                        i += 1
                        continue

            # Execute block normally
            try:
                logger.info(f"Executing block: {node_id} ({node_type})")
                result = await block.execute(resolved_inputs, resolved_config, context)

                # Store result in context
                context.nodes[node_id] = result.outputs

                # Record metrics (aggregate for loop body nodes)
                if node_id in metrics:
                    # Already have metrics, add to duration
                    metrics[node_id]["duration_ms"] += result.duration_ms
                    metrics[node_id]["executions"] = metrics[node_id].get("executions", 1) + 1
                else:
                    metrics[node_id] = {
                        "type": node_type,
                        "duration_ms": result.duration_ms,
                        "success": result.success,
                        **result.metrics,
                    }

                if not result.success:
                    # Check if we're in iteration and should continue on error
                    in_iteration = any(node_id in state.loop_body_nodes for state in active_iterations.values())
                    if in_iteration:
                        parent_state = next(
                            (state for state in active_iterations.values() if node_id in state.loop_body_nodes),
                            None
                        )
                        if parent_state and parent_state.on_error == "continue":
                            parent_state.errors.append({
                                "index": parent_state.current_index,
                                "node_id": node_id,
                                "error": result.error,
                            })
                            i += 1
                            continue
                        elif parent_state and parent_state.on_error == "collect_errors":
                            parent_state.errors.append({
                                "index": parent_state.current_index,
                                "node_id": node_id,
                                "error": result.error,
                            })
                            i += 1
                            continue

                    error_info = {
                        "error": result.error,
                        "error_node_id": node_id,
                    }
                    break

            except Exception as e:
                logger.exception(f"Block execution failed: {node_id}")

                # Check if we're in iteration and should continue on error
                in_iteration = any(node_id in state.loop_body_nodes for state in active_iterations.values())
                if in_iteration:
                    parent_state = next(
                        (state for state in active_iterations.values() if node_id in state.loop_body_nodes),
                        None
                    )
                    if parent_state and parent_state.on_error in ("continue", "collect_errors"):
                        parent_state.errors.append({
                            "index": parent_state.current_index,
                            "node_id": node_id,
                            "error": str(e),
                        })
                        i += 1
                        continue

                error_info = {
                    "error": str(e),
                    "error_node_id": node_id,
                }
                break

            i += 1

        # Collect outputs
        outputs = {}
        if not error_info:
            for output_config in outputs_config:
                output_name = output_config.get("name")
                output_source = output_config.get("source")
                if output_name and output_source:
                    outputs[output_name] = context.resolve_ref(output_source)

        total_duration = (time.time() - start_time) * 1000

        result = {
            "outputs": outputs,
            "metrics": metrics,
            "duration_ms": round(total_duration, 2),
        }

        if error_info:
            result["error"] = error_info["error"]
            result["error_node_id"] = error_info["error_node_id"]
        else:
            result["error"] = None
            result["error_node_id"] = None

        return result

    def _detect_iteration_loops(
        self,
        nodes: list[dict],
        edges: list[dict],
        execution_order: list[dict],
    ) -> dict[str, dict]:
        """
        Detect ForEach → Collect iteration patterns in the workflow.

        Returns:
            Dict mapping foreach_node_id to {collect_node_id, loop_body: [node_ids]}
        """
        loops = {}

        # Find all foreach and collect nodes
        foreach_nodes = [n for n in nodes if n.get("type") == "foreach"]
        collect_nodes = [n for n in nodes if n.get("type") == "collect"]

        if not foreach_nodes or not collect_nodes:
            return loops

        # Build adjacency list
        adjacency = defaultdict(list)
        reverse_adjacency = defaultdict(list)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)
                reverse_adjacency[target].append(source)

        # For each foreach, find the downstream collect that collects its outputs
        for foreach_node in foreach_nodes:
            foreach_id = foreach_node["id"]

            # BFS to find reachable collect nodes
            visited = set()
            queue = [foreach_id]
            reachable_nodes = []

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                if current != foreach_id:
                    reachable_nodes.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Find collect nodes that are reachable
            for collect_node in collect_nodes:
                collect_id = collect_node["id"]
                if collect_id in reachable_nodes:
                    # Found a foreach → collect pattern
                    # Loop body is all nodes between foreach and collect in execution order
                    foreach_idx = next(
                        (i for i, n in enumerate(execution_order) if n["id"] == foreach_id),
                        -1
                    )
                    collect_idx = next(
                        (i for i, n in enumerate(execution_order) if n["id"] == collect_id),
                        -1
                    )

                    if foreach_idx >= 0 and collect_idx > foreach_idx:
                        loop_body = [
                            execution_order[j]["id"]
                            for j in range(foreach_idx + 1, collect_idx)
                        ]
                        loops[foreach_id] = {
                            "collect_node_id": collect_id,
                            "loop_body": loop_body,
                        }
                        break  # One collect per foreach

        return loops

    def _topological_sort(
        self,
        nodes: list[dict],
        edges: list[dict],
    ) -> list[dict]:
        """
        Sort nodes in topological order (dependencies first).

        Uses Kahn's algorithm.
        """
        # Build adjacency list and in-degree count
        node_map = {n["id"]: n for n in nodes}
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)

        for node in nodes:
            in_degree[node["id"]] = 0

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)
                in_degree[target] += 1

        # Find all nodes with no incoming edges
        queue = [n["id"] for n in nodes if in_degree[n["id"]] == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            if node_id in node_map:
                result.append(node_map[node_id])

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(nodes):
            raise WorkflowExecutionError("Workflow contains a cycle")

        return result

    def _build_edge_inputs(self, edges: list[dict]) -> dict[str, dict[str, str]]:
        """
        Build input mappings from edges (React Flow style).

        Converts edges like:
            {source: "input_1", target: "detect_1", sourceHandle: "image", targetHandle: "image"}
        To input mappings like:
            {"detect_1": {"image": "$nodes.input_1.image"}}
        """
        edge_inputs: dict[str, dict[str, str]] = {}

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            source_handle = edge.get("sourceHandle") or edge.get("source_handle")
            target_handle = edge.get("targetHandle") or edge.get("target_handle")

            if not source or not target:
                continue

            # Default handles if not specified
            if not source_handle:
                source_handle = "output"
            if not target_handle:
                target_handle = "input"

            # Build reference string ($nodes.node_id.output_port format)
            ref = f"$nodes.{source}.{source_handle}"

            # Add to target node's inputs
            if target not in edge_inputs:
                edge_inputs[target] = {}
            edge_inputs[target][target_handle] = ref

        return edge_inputs

    def _resolve_inputs(
        self,
        inputs_config: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Resolve all input references to actual values."""
        resolved = {}
        for key, value in inputs_config.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved

    def _resolve_value(self, value: Any, context: ExecutionContext) -> Any:
        """Recursively resolve a value (handles nested dicts/lists)."""
        if isinstance(value, str):
            return context.resolve_ref(value)
        elif isinstance(value, dict):
            return {k: self._resolve_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_value(item, context) for item in value]
        return value

    def _resolve_parameters(
        self,
        definitions: list[dict],
        runtime_values: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve workflow parameters from definitions and runtime values.

        Parameter definitions format:
        [
            {
                "name": "confidence_threshold",
                "type": "number",
                "default": 0.5,
                "description": "Minimum confidence for detections"
            },
            ...
        ]

        Returns dict of resolved parameter values with defaults applied.
        """
        parameters = {}

        for param_def in definitions:
            name = param_def.get("name")
            if not name:
                continue

            # Use runtime value if provided, otherwise use default
            if name in runtime_values:
                value = runtime_values[name]
            else:
                value = param_def.get("default")

            # Type coercion
            param_type = param_def.get("type", "string")
            if value is not None:
                try:
                    if param_type == "number":
                        value = float(value)
                    elif param_type == "integer":
                        value = int(value)
                    elif param_type == "boolean":
                        if isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes")
                        else:
                            value = bool(value)
                    elif param_type == "string":
                        value = str(value)
                    # array and object types are passed through as-is
                except (ValueError, TypeError):
                    pass  # Keep original value if coercion fails

            parameters[name] = value

        return parameters

    def _resolve_config(
        self,
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """
        Resolve parameter references in config values.

        Handles:
        - "{{ params.confidence }}" -> actual value
        - "$params.confidence" -> actual value
        - Nested dicts and lists
        """
        return self._resolve_value(config, context)


# Singleton instance
_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get the singleton workflow engine instance."""
    global _engine
    if _engine is None:
        _engine = WorkflowEngine()
    return _engine
