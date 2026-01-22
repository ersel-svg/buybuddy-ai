"""
Workflow Engine Module

Provides workflow execution capabilities for CV pipelines.

Usage:
    from services.workflow import get_workflow_engine

    engine = get_workflow_engine()
    result = await engine.execute(workflow_definition, inputs, workflow_id, execution_id)
"""

from .engine import WorkflowEngine, get_workflow_engine, WorkflowExecutionError
from .base import BaseBlock, BlockResult, ExecutionContext
from .blocks import get_all_blocks, get_block, get_block_metadata, BLOCK_CATEGORIES

__all__ = [
    # Engine
    "WorkflowEngine",
    "get_workflow_engine",
    "WorkflowExecutionError",
    # Base classes
    "BaseBlock",
    "BlockResult",
    "ExecutionContext",
    # Block registry
    "get_all_blocks",
    "get_block",
    "get_block_metadata",
    "BLOCK_CATEGORIES",
]
