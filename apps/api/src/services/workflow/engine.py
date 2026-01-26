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

# WebSocket broadcast support (lazy import to avoid circular dependency)
_ws_broadcast_enabled = True

logger = logging.getLogger(__name__)


async def _broadcast_node_event(
    execution_id: str,
    event_type: str,
    node_id: str,
    node_type: str,
    **kwargs
):
    """
    Broadcast a node execution event via WebSocket.

    Lazily imports websocket module to avoid circular imports.
    Silently fails if broadcasting is not available.
    """
    if not _ws_broadcast_enabled:
        return

    try:
        from api.v1.workflows.websocket import (
            broadcast_node_start,
            broadcast_node_complete,
            broadcast_node_error,
            broadcast_progress,
        )

        if event_type == "node_start":
            await broadcast_node_start(
                execution_id=execution_id,
                node_id=node_id,
                node_type=node_type,
                node_label=kwargs.get("node_label"),
            )
        elif event_type == "node_complete":
            await broadcast_node_complete(
                execution_id=execution_id,
                node_id=node_id,
                duration_ms=kwargs.get("duration_ms", 0),
                outputs_summary=kwargs.get("outputs_summary"),
            )
        elif event_type == "node_error":
            await broadcast_node_error(
                execution_id=execution_id,
                node_id=node_id,
                error=kwargs.get("error", "Unknown error"),
            )
        elif event_type == "progress":
            await broadcast_progress(
                execution_id=execution_id,
                percent=kwargs.get("percent", 0),
                current_node=node_id,
                completed_nodes=kwargs.get("completed_nodes", []),
                total_nodes=kwargs.get("total_nodes", 0),
            )
    except ImportError:
        # WebSocket module not available
        pass
    except Exception as e:
        # Don't fail execution due to broadcast errors
        logger.debug(f"WebSocket broadcast failed: {e}")


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

        # Build conditional edge mappings for branching
        conditional_edges = self._build_conditional_edges(edges, nodes)

        # Detect iteration loops (ForEach → Collect patterns)
        iteration_loops = self._detect_iteration_loops(nodes, edges, execution_order)

        # Execute nodes in order with iteration support
        metrics = {}
        error_info = None
        active_iterations: dict[str, IterationState] = {}  # foreach_node_id -> state

        i = 0
        skipped_nodes: set[str] = set()  # Track nodes skipped due to conditional branching

        while i < len(execution_order):
            node = execution_order[i]
            node_id = node["id"]
            node_type = node["type"]
            node_config = node.get("config", {})

            # Check if this node should be skipped due to conditional branching
            # Recalculate skipped nodes after each condition evaluation
            if conditional_edges:
                skipped_nodes = self._get_skipped_nodes(conditional_edges, context, nodes, edges)

            if node_id in skipped_nodes:
                logger.info(f"Skipping node {node_id} ({node_type}) due to conditional branching")
                metrics[node_id] = {
                    "type": node_type,
                    "duration_ms": 0,
                    "success": True,
                    "skipped": True,
                    "skip_reason": "conditional_branch",
                }
                # Set empty outputs so downstream references don't fail
                context.nodes[node_id] = {"_skipped": True}
                i += 1
                continue

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

                    # Check if parallel/batch execution is requested
                    if active_iterations[node_id].iteration_mode in ("parallel", "batch"):
                        iter_state = active_iterations[node_id]
                        logger.info(f"ForEach parallel execution: {iter_state.total_items} items, mode={iter_state.iteration_mode}, concurrency={iter_state.max_concurrency}")

                        # Get loop body nodes
                        loop_body_node_objs = [
                            n for n in execution_order
                            if n["id"] in iter_state.loop_body_nodes
                        ]

                        # Execute all items in parallel
                        start_parallel = time.time()
                        collected_results, body_metrics, errors = await self._execute_parallel_iteration(
                            iter_state=iter_state,
                            loop_body_nodes=loop_body_node_objs,
                            edge_inputs=edge_inputs,
                            context=context,
                            execution_id=execution_id,
                        )
                        parallel_duration = (time.time() - start_parallel) * 1000

                        # Store results in context for collect node
                        context.nodes[node_id] = {
                            "item": None,  # No single item in parallel mode
                            "index": iter_state.total_items - 1,
                            "total": iter_state.total_items,
                            "is_first": False,
                            "is_last": True,
                            "context": resolved_inputs.get("context"),
                        }

                        # Record metrics
                        metrics[node_id] = {
                            "type": node_type,
                            "duration_ms": round(parallel_duration, 2),
                            "success": True,
                            "total_items": iter_state.total_items,
                            "iterations_completed": iter_state.total_items,
                            "iteration_mode": iter_state.iteration_mode,
                            "errors_count": len(errors),
                        }

                        # Add body node metrics
                        for body_node_id, body_node_metrics in body_metrics.items():
                            metrics[body_node_id] = {
                                **body_node_metrics,
                                "success": True,
                                "parallel": True,
                            }

                        # Store errors in iteration state for collect
                        iter_state.collected_results = collected_results
                        iter_state.errors = errors
                        iter_state.current_index = iter_state.total_items  # Mark as complete

                        # Skip to collect node
                        collect_idx = next(
                            (idx for idx, n in enumerate(execution_order) if n["id"] == iter_state.collect_node_id),
                            None
                        )
                        if collect_idx is not None:
                            i = collect_idx
                            continue
                        else:
                            # No collect node, just continue
                            del active_iterations[node_id]
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

                    # Check if parallel execution already collected results
                    if iter_state.iteration_mode in ("parallel", "batch") and iter_state.is_complete:
                        # Parallel execution already populated collected_results
                        filter_nulls = resolved_config.get("filter_nulls", True)
                        flatten = resolved_config.get("flatten", False)
                        unique = resolved_config.get("unique", False)
                        unique_key = resolved_config.get("unique_key")

                        final_results = iter_state.collected_results

                        # Apply post-processing
                        if filter_nulls:
                            final_results = [r for r in final_results if r is not None]

                        if flatten:
                            flattened = []
                            for r in final_results:
                                if isinstance(r, list):
                                    flattened.extend(r)
                                else:
                                    flattened.append(r)
                            final_results = flattened

                        if unique:
                            if unique_key and all(isinstance(r, dict) for r in final_results):
                                seen_keys = set()
                                unique_results = []
                                for r in final_results:
                                    key_val = r.get(unique_key)
                                    if key_val not in seen_keys:
                                        seen_keys.add(key_val)
                                        unique_results.append(r)
                                final_results = unique_results
                            else:
                                # Simple deduplication for primitives
                                seen = set()
                                unique_results = []
                                for r in final_results:
                                    r_hash = str(r)
                                    if r_hash not in seen:
                                        seen.add(r_hash)
                                        unique_results.append(r)
                                final_results = unique_results

                        context.nodes[node_id] = {
                            "results": final_results,
                            "count": len(final_results),
                            "errors": iter_state.errors if iter_state.errors else None,
                        }
                        metrics[node_id] = {
                            "type": node_type,
                            "duration_ms": 0,
                            "success": True,
                            "collected_count": len(final_results),
                            "iterations": iter_state.total_items,
                            "parallel": True,
                            "errors_count": len(iter_state.errors),
                        }

                        # Clean up
                        del active_iterations[parent_foreach]
                        context.variables.pop("_iteration_index", None)
                        context.variables.pop("_iteration_total", None)
                        context.variables.pop("_iteration_foreach", None)

                        logger.info(f"Parallel iteration complete. Collected {len(final_results)} items with {len(iter_state.errors)} errors.")
                        i += 1
                        continue

                    # Sequential execution - collect one item at a time
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

                # Broadcast node start via WebSocket
                completed_nodes = [n["id"] for n in execution_order[:i] if n["id"] in context.nodes]
                await _broadcast_node_event(
                    execution_id=execution_id,
                    event_type="node_start",
                    node_id=node_id,
                    node_type=node_type,
                    node_label=node.get("data", {}).get("label", node_type),
                )
                await _broadcast_node_event(
                    execution_id=execution_id,
                    event_type="progress",
                    node_id=node_id,
                    node_type=node_type,
                    percent=int((i / len(execution_order)) * 100),
                    completed_nodes=completed_nodes,
                    total_nodes=len(execution_order),
                )

                result = await block.execute(resolved_inputs, resolved_config, context)

                # Store result in context
                context.nodes[node_id] = result.outputs

                # Broadcast node complete via WebSocket
                outputs_summary = {}
                if result.outputs:
                    for k, v in result.outputs.items():
                        if isinstance(v, (list, tuple)):
                            outputs_summary[k] = f"[{len(v)} items]"
                        elif isinstance(v, dict):
                            outputs_summary[k] = f"{{...{len(v)} keys}}"
                        elif isinstance(v, str) and len(v) > 100:
                            outputs_summary[k] = f"{v[:50]}..."
                        else:
                            outputs_summary[k] = str(v)[:100] if v is not None else None

                await _broadcast_node_event(
                    execution_id=execution_id,
                    event_type="node_complete",
                    node_id=node_id,
                    node_type=node_type,
                    duration_ms=result.duration_ms,
                    outputs_summary=outputs_summary,
                )

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

                    # Broadcast node error via WebSocket
                    await _broadcast_node_event(
                        execution_id=execution_id,
                        event_type="node_error",
                        node_id=node_id,
                        node_type=node_type,
                        error=result.error,
                    )
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

                # Broadcast node error via WebSocket
                await _broadcast_node_event(
                    execution_id=execution_id,
                    event_type="node_error",
                    node_id=node_id,
                    node_type=node_type,
                    error=str(e),
                )
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

        # Broadcast execution complete via WebSocket
        try:
            from api.v1.workflows.websocket import broadcast_complete
            await broadcast_complete(
                execution_id=execution_id,
                status="completed" if not error_info else "failed",
                duration_ms=int(total_duration),
                outputs=outputs if not error_info else None,
                error=error_info.get("error") if error_info else None,
            )
        except Exception:
            pass  # Don't fail execution due to broadcast errors

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

    def _build_conditional_edges(self, edges: list[dict], nodes: list[dict]) -> dict[str, dict[str, list[str]]]:
        """
        Build conditional edge mappings for branching.

        Returns:
            Dict mapping condition_node_id -> {"true": [target_ids], "false": [target_ids]}

        Supports both sourceHandle-based ("true_output"/"false_output") and edge condition property.
        """
        # Find condition nodes
        condition_node_ids = {n["id"] for n in nodes if n.get("type") == "condition"}

        conditional_edges: dict[str, dict[str, list[str]]] = {}

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            source_handle = edge.get("sourceHandle") or edge.get("source_handle", "")
            edge_condition = edge.get("condition")  # Explicit condition property

            if not source or not target:
                continue

            # Check if source is a condition node
            if source in condition_node_ids:
                if source not in conditional_edges:
                    conditional_edges[source] = {"true": [], "false": [], "any": []}

                # Determine branch based on sourceHandle or condition property
                if edge_condition in ("true", "false"):
                    conditional_edges[source][edge_condition].append(target)
                elif source_handle in ("true_output", "result") and source_handle != "false_output":
                    conditional_edges[source]["true"].append(target)
                elif source_handle == "false_output":
                    conditional_edges[source]["false"].append(target)
                else:
                    # No explicit condition - runs regardless of result
                    conditional_edges[source]["any"].append(target)

        return conditional_edges

    def _get_skipped_nodes(
        self,
        conditional_edges: dict[str, dict[str, list[str]]],
        context: ExecutionContext,
        nodes: list[dict],
        edges: list[dict],
    ) -> set[str]:
        """
        Determine which nodes should be skipped due to conditional branching.

        Returns set of node IDs that should NOT be executed.
        """
        skipped = set()

        # Build node adjacency for downstream propagation
        adjacency = defaultdict(list)
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                adjacency[source].append(target)

        for cond_node_id, branches in conditional_edges.items():
            # Check if condition has been evaluated
            if cond_node_id not in context.nodes:
                continue

            cond_result = context.nodes[cond_node_id].get("result", False)

            # Determine which branch to skip
            if cond_result:
                # True branch active, skip false branch
                skip_targets = branches.get("false", [])
            else:
                # False branch active, skip true branch
                skip_targets = branches.get("true", [])

            # Add skipped targets and their downstream nodes
            for target in skip_targets:
                skipped.add(target)
                # Propagate skip to downstream nodes (unless they have other inputs)
                self._propagate_skip(target, skipped, adjacency, conditional_edges, context)

        return skipped

    def _propagate_skip(
        self,
        node_id: str,
        skipped: set[str],
        adjacency: dict[str, list[str]],
        conditional_edges: dict[str, dict[str, list[str]]],
        context: ExecutionContext,
    ):
        """Propagate skip status to downstream nodes that only depend on skipped nodes."""
        for downstream in adjacency.get(node_id, []):
            if downstream in skipped:
                continue

            # Don't propagate past condition nodes - they'll make their own decisions
            if downstream in conditional_edges:
                continue

            # Add to skipped and propagate
            skipped.add(downstream)
            self._propagate_skip(downstream, skipped, adjacency, conditional_edges, context)

    async def _execute_parallel_iteration(
        self,
        iter_state: IterationState,
        loop_body_nodes: list[dict],
        edge_inputs: dict[str, dict[str, str]],
        context: ExecutionContext,
        execution_id: str,
    ) -> tuple[list[Any], dict[str, dict], list[dict]]:
        """
        Execute loop body for all items in parallel.

        Returns:
            Tuple of (collected_results, metrics_by_node, errors)
        """
        items = iter_state.items
        max_concurrency = iter_state.max_concurrency
        batch_size = iter_state.batch_size
        on_error = iter_state.on_error

        collected_results = []
        all_metrics: dict[str, dict] = {}
        errors: list[dict] = []

        async def execute_single_item(item: Any, index: int) -> tuple[Any, dict, Optional[dict]]:
            """Execute loop body for a single item."""
            item_context = ExecutionContext(
                inputs=context.inputs,
                nodes=dict(context.nodes),  # Copy to avoid conflicts
                workflow_id=context.workflow_id,
                execution_id=execution_id,
                parameters=context.parameters,
            )

            # Set iteration variables for this item
            item_context.variables["_iteration_index"] = index
            item_context.variables["_iteration_total"] = len(items)
            item_context.variables["_iteration_foreach"] = iter_state.foreach_node_id

            # Set ForEach output for this iteration
            item_context.nodes[iter_state.foreach_node_id] = {
                "item": item,
                "index": index,
                "total": len(items),
                "is_first": index == 0,
                "is_last": index == len(items) - 1,
            }

            item_metrics = {}
            result_item = None
            error_info = None

            # Execute each node in the loop body
            for node in loop_body_nodes:
                node_id = node["id"]
                node_type = node["type"]
                node_config = node.get("config", {})
                node_inputs_config = {**edge_inputs.get(node_id, {}), **node.get("inputs", {})}

                block = self._blocks.get(node_type)
                if not block:
                    error_info = {"index": index, "node_id": node_id, "error": f"Unknown block type: {node_type}"}
                    break

                try:
                    resolved_inputs = self._resolve_inputs(node_inputs_config, item_context)
                    resolved_config = self._resolve_config(node_config, item_context)

                    result = await block.execute(resolved_inputs, resolved_config, item_context)
                    item_context.nodes[node_id] = result.outputs

                    # Accumulate metrics
                    if node_id not in item_metrics:
                        item_metrics[node_id] = {
                            "type": node_type,
                            "duration_ms": 0,
                            "executions": 0,
                        }
                    item_metrics[node_id]["duration_ms"] += result.duration_ms
                    item_metrics[node_id]["executions"] += 1

                    if not result.success:
                        error_info = {"index": index, "node_id": node_id, "error": result.error}
                        if on_error == "stop":
                            break

                except Exception as e:
                    error_info = {"index": index, "node_id": node_id, "error": str(e)}
                    if on_error == "stop":
                        break

            # Get the final output (typically from the last node before collect)
            if loop_body_nodes:
                last_node_id = loop_body_nodes[-1]["id"]
                if last_node_id in item_context.nodes:
                    result_item = item_context.nodes[last_node_id]

            return result_item, item_metrics, error_info

        # Execute based on mode
        if iter_state.iteration_mode == "parallel":
            # Parallel with concurrency limit using semaphore
            semaphore = asyncio.Semaphore(max_concurrency)

            async def limited_execute(item: Any, index: int):
                async with semaphore:
                    return await execute_single_item(item, index)

            results = await asyncio.gather(
                *[limited_execute(item, idx) for idx, item in enumerate(items)],
                return_exceptions=True,
            )

            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    errors.append({"index": idx, "node_id": None, "error": str(res)})
                else:
                    result_item, item_metrics, error_info = res
                    if result_item is not None:
                        collected_results.append(result_item)
                    if error_info:
                        errors.append(error_info)

                    # Merge metrics
                    for node_id, node_metrics in item_metrics.items():
                        if node_id not in all_metrics:
                            all_metrics[node_id] = {
                                "type": node_metrics["type"],
                                "duration_ms": 0,
                                "executions": 0,
                            }
                        all_metrics[node_id]["duration_ms"] += node_metrics["duration_ms"]
                        all_metrics[node_id]["executions"] += node_metrics["executions"]

        elif iter_state.iteration_mode == "batch":
            # Process in batches
            for batch_start in range(0, len(items), batch_size):
                batch_items = items[batch_start:batch_start + batch_size]

                # Execute batch in parallel
                batch_results = await asyncio.gather(
                    *[execute_single_item(item, batch_start + idx) for idx, item in enumerate(batch_items)],
                    return_exceptions=True,
                )

                for idx, res in enumerate(batch_results):
                    if isinstance(res, Exception):
                        errors.append({"index": batch_start + idx, "node_id": None, "error": str(res)})
                    else:
                        result_item, item_metrics, error_info = res
                        if result_item is not None:
                            collected_results.append(result_item)
                        if error_info:
                            errors.append(error_info)

                        # Merge metrics
                        for node_id, node_metrics in item_metrics.items():
                            if node_id not in all_metrics:
                                all_metrics[node_id] = {
                                    "type": node_metrics["type"],
                                    "duration_ms": 0,
                                    "executions": 0,
                                }
                            all_metrics[node_id]["duration_ms"] += node_metrics["duration_ms"]
                            all_metrics[node_id]["executions"] += node_metrics["executions"]

        return collected_results, all_metrics, errors

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
