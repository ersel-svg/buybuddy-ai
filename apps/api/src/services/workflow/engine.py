"""
Workflow Engine - Main Execution Engine

Handles workflow execution: parsing, topological sorting, and block execution.
"""

import time
import logging
from typing import Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

from .base import BaseBlock, BlockResult, ExecutionContext

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Exception raised during workflow execution."""

    def __init__(self, message: str, node_id: Optional[str] = None):
        self.message = message
        self.node_id = node_id
        super().__init__(message)


class WorkflowEngine:
    """
    Main workflow execution engine.

    Handles:
    - Parsing workflow definitions
    - Topological sorting of nodes
    - Sequential execution of blocks
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

        # Execute nodes in order
        metrics = {}
        error_info = None

        for node in execution_order:
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

            # Execute block
            try:
                logger.info(f"Executing block: {node_id} ({node_type})")
                result = await block.execute(resolved_inputs, resolved_config, context)

                # Store result in context
                context.nodes[node_id] = result.outputs

                # Record metrics
                metrics[node_id] = {
                    "type": node_type,
                    "duration_ms": result.duration_ms,
                    "success": result.success,
                    **result.metrics,
                }

                if not result.success:
                    error_info = {
                        "error": result.error,
                        "error_node_id": node_id,
                    }
                    break

            except Exception as e:
                logger.exception(f"Block execution failed: {node_id}")
                error_info = {
                    "error": str(e),
                    "error_node_id": node_id,
                }
                break

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
