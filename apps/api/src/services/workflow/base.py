"""
Workflow Engine - Base Classes

Defines the base interfaces for workflow blocks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BlockResult:
    """Result from executing a block."""
    outputs: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ExecutionContext:
    """Context passed to blocks during execution."""
    inputs: dict[str, Any]  # Original workflow inputs
    nodes: dict[str, dict[str, Any]]  # Results from executed nodes: {node_id: outputs}
    workflow_id: str
    execution_id: str

    def resolve_ref(self, ref: str) -> Any:
        """
        Resolve a reference string to actual value.

        Reference formats:
        - "$inputs.image" -> context.inputs["image"]
        - "$nodes.detect.detections" -> context.nodes["detect"]["detections"]
        - "literal" -> "literal" (returned as-is)
        """
        if not isinstance(ref, str):
            return ref

        if not ref.startswith("$"):
            return ref

        parts = ref[1:].split(".")

        if parts[0] == "inputs":
            value = self.inputs
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value

        if parts[0] == "nodes":
            if len(parts) < 2:
                return None
            node_id = parts[1]
            value = self.nodes.get(node_id, {})
            for part in parts[2:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return value

        return ref


class BaseBlock(ABC):
    """Base class for all workflow blocks."""

    block_type: str = "base"
    display_name: str = "Base Block"
    description: str = "Base block class"

    # Input/output definitions for UI
    input_ports: list[dict] = []  # [{name, type, required}]
    output_ports: list[dict] = []  # [{name, type}]
    config_schema: dict = {}  # JSON Schema for config

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """
        Execute the block with given inputs and config.

        Args:
            inputs: Resolved input values (after reference resolution)
            config: Block configuration from workflow definition
            context: Execution context with access to all workflow data

        Returns:
            BlockResult with outputs, timing, and optional metrics
        """
        pass

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """
        Validate that required inputs are present.

        Returns list of error messages (empty if valid).
        """
        errors = []
        for port in self.input_ports:
            if port.get("required", False) and port["name"] not in inputs:
                errors.append(f"Missing required input: {port['name']}")
        return errors

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """
        Validate block configuration.

        Returns list of error messages (empty if valid).
        """
        # Override in subclasses for specific validation
        return []


class InputBlock(BaseBlock):
    """Base class for input blocks (no inputs, only outputs)."""

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        # Input blocks get their data from context.inputs
        return BlockResult(outputs=context.inputs)


class ModelBlock(BaseBlock):
    """Base class for model inference blocks."""

    model_type: str = ""  # detection, classification, embedding, segmentation

    def __init__(self):
        self._model_cache: dict[str, Any] = {}

    async def load_model(self, model_id: str, model_source: str) -> Any:
        """
        Load a model (pretrained or trained).

        Implements caching to avoid reloading.
        """
        cache_key = f"{model_source}:{model_id}"
        if cache_key not in self._model_cache:
            if model_source == "pretrained":
                model = await self._load_pretrained(model_id)
            else:
                model = await self._load_trained(model_id)
            self._model_cache[cache_key] = model
        return self._model_cache[cache_key]

    async def _load_pretrained(self, model_id: str) -> Any:
        """Load a pretrained model. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _load_pretrained")

    async def _load_trained(self, model_id: str) -> Any:
        """Load a trained model from database. Override in subclasses."""
        raise NotImplementedError("Subclass must implement _load_trained")


class TransformBlock(BaseBlock):
    """Base class for image/data transformation blocks."""
    pass


class LogicBlock(BaseBlock):
    """Base class for logic/control flow blocks."""
    pass


class OutputBlock(BaseBlock):
    """Base class for output/visualization blocks."""
    pass
