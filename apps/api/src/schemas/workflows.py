"""
Workflow Module - Pydantic Schemas

Schemas for workflow definitions, executions, and models.
"""

from datetime import datetime
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


# ===========================================
# Workflow Definition Schemas
# ===========================================

class WorkflowViewport(BaseModel):
    """React Flow viewport configuration."""
    x: float = 0
    y: float = 0
    zoom: float = 1


class WorkflowNode(BaseModel):
    """A single node in the workflow graph."""
    id: str
    type: str
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    """A connection between two nodes."""
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class WorkflowDefinition(BaseModel):
    """Complete workflow definition stored as JSON."""
    version: str = "1.0"
    nodes: list[WorkflowNode] = Field(default_factory=list)
    edges: list[WorkflowEdge] = Field(default_factory=list)
    viewport: WorkflowViewport = Field(default_factory=WorkflowViewport)
    outputs: list[dict] = Field(default_factory=list)


class WorkflowBase(BaseModel):
    """Base workflow fields."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class WorkflowCreate(WorkflowBase):
    """Schema for creating a new workflow."""
    definition: Optional[WorkflowDefinition] = None


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    definition: Optional[WorkflowDefinition] = None
    status: Optional[Literal["draft", "active", "archived"]] = None


class WorkflowResponse(WorkflowBase):
    """Schema for workflow response."""
    id: str
    definition: dict  # Raw JSON, not validated
    status: str
    run_count: int
    last_run_at: Optional[datetime] = None
    avg_duration_ms: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WorkflowListResponse(BaseModel):
    """Schema for workflow list response."""
    workflows: list[WorkflowResponse]
    total: int


# ===========================================
# Execution Schemas
# ===========================================

class ExecutionInput(BaseModel):
    """Input for running a workflow."""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    parameters: dict = Field(default_factory=dict)


class WorkflowRunRequest(BaseModel):
    """Request to run a workflow."""
    input: Optional[ExecutionInput] = None
    inputs: Optional[dict] = None  # Alternative format from frontend

    def get_execution_input(self) -> ExecutionInput:
        """Get normalized execution input."""
        if self.input:
            return self.input
        if self.inputs:
            return ExecutionInput(
                image_url=self.inputs.get("image_url"),
                image_base64=self.inputs.get("image_base64"),
                parameters=self.inputs.get("parameters", {}),
            )
        return ExecutionInput()


class NodeMetrics(BaseModel):
    """Metrics for a single node execution."""
    node_id: str
    duration_ms: int
    output_count: Optional[int] = None
    error: Optional[str] = None


class ExecutionResponse(BaseModel):
    """Schema for execution response."""
    id: str
    workflow_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_data: Optional[dict] = None
    output_data: Optional[dict] = None
    node_metrics: Optional[dict] = None
    error_message: Optional[str] = None
    error_node_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ExecutionListResponse(BaseModel):
    """Schema for execution list response."""
    executions: list[ExecutionResponse]
    total: int


# ===========================================
# Model Schemas (for unified model picker)
# ===========================================

class PretrainedModelResponse(BaseModel):
    """Schema for pretrained model."""
    id: str
    name: str
    description: Optional[str] = None
    model_type: str
    source: str
    model_path: str
    classes: Optional[list[str]] = None
    class_count: Optional[int] = None
    default_config: dict = Field(default_factory=dict)
    embedding_dim: Optional[int] = None
    input_size: Optional[int] = None
    is_active: bool = True

    class Config:
        from_attributes = True


class TrainedModelResponse(BaseModel):
    """Schema for trained model."""
    id: str
    name: str
    description: Optional[str] = None
    model_type: str
    provider: str  # rf-detr, rt-detr, vit, convnext, dinov2, clip, etc.
    checkpoint_url: Optional[str] = None
    classes: Optional[list] = None
    class_count: Optional[int] = None
    accuracy: Optional[float] = None
    map: Optional[float] = None
    recall_at_1: Optional[float] = None
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True


class ModelCategory(BaseModel):
    """A category of models (pretrained + trained)."""
    pretrained: list[PretrainedModelResponse] = Field(default_factory=list)
    trained: list[TrainedModelResponse] = Field(default_factory=list)


class UnifiedModelsResponse(BaseModel):
    """Unified response with all available models."""
    detection: ModelCategory = Field(default_factory=ModelCategory)
    classification: ModelCategory = Field(default_factory=ModelCategory)
    embedding: ModelCategory = Field(default_factory=ModelCategory)
    segmentation: ModelCategory = Field(default_factory=ModelCategory)


# ===========================================
# Stats Schemas
# ===========================================

class WorkflowStatsResponse(BaseModel):
    """Statistics for the workflow module."""
    total_workflows: int
    active_workflows: int
    draft_workflows: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_execution_time_ms: Optional[float] = None
    recent_workflows: list[WorkflowResponse] = Field(default_factory=list)
