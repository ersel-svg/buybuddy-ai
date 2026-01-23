"""
Workflows - Executions Router

Endpoints for running workflows and viewing execution history.
"""

import logging
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query

from services.supabase import supabase_service
from services.workflow import get_workflow_engine
from schemas.workflows import (
    WorkflowRunRequest,
    ExecutionResponse,
    ExecutionListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/executions", response_model=ExecutionListResponse)
async def list_all_executions(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List all executions across all workflows.
    """
    query = supabase_service.client.table("wf_executions").select(
        "*, workflow:wf_workflows(name)", count="exact"
    )

    if workflow_id:
        query = query.eq("workflow_id", workflow_id)
    if status:
        query = query.eq("status", status)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    executions = []
    for ex in result.data or []:
        workflow = ex.pop("workflow", None)
        if workflow:
            ex["workflow_name"] = workflow.get("name")
        executions.append(ex)

    return ExecutionListResponse(
        executions=executions,
        total=result.count or 0,
    )


@router.post("/{workflow_id}/run", response_model=ExecutionResponse)
async def run_workflow(workflow_id: str, data: WorkflowRunRequest):
    """
    Run a workflow with the provided input.

    Accepts either an image URL or base64 encoded image.
    Returns the execution record with results.

    The workflow engine executes all nodes in topological order,
    passing outputs between connected nodes.
    """
    # Verify workflow exists and is not archived
    workflow = supabase_service.client.table("wf_workflows").select(
        "id, name, status, definition"
    ).eq("id", workflow_id).single().execute()

    if not workflow.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.data.get("status") == "archived":
        raise HTTPException(status_code=400, detail="Cannot run archived workflow")

    # Validate input
    if not data.input.image_url and not data.input.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Either image_url or image_base64 must be provided"
        )

    # Create execution record
    now = datetime.now(timezone.utc)
    execution_data = {
        "workflow_id": workflow_id,
        "status": "pending",
        "input_data": data.input.model_dump(),
        "created_at": now.isoformat(),
    }

    result = supabase_service.client.table("wf_executions").insert(execution_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create execution")

    execution = result.data[0]
    execution_id = execution["id"]

    # Update status to running
    supabase_service.client.table("wf_executions").update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", execution_id).execute()

    try:
        # Execute the workflow
        engine = get_workflow_engine()
        workflow_definition = workflow.data.get("definition", {})

        engine_result = await engine.execute(
            workflow=workflow_definition,
            inputs=data.input.model_dump(),
            workflow_id=workflow_id,
            execution_id=execution_id,
        )

        # Determine status based on result
        completed_at = datetime.now(timezone.utc)
        status = "completed" if engine_result.get("error") is None else "failed"

        update_data = {
            "status": status,
            "completed_at": completed_at.isoformat(),
            "duration_ms": int(engine_result.get("duration_ms", 0)),
            "output_data": engine_result.get("outputs", {}),
            "node_metrics": engine_result.get("metrics", {}),
        }

        if engine_result.get("error"):
            update_data["error_message"] = engine_result["error"]
            update_data["error_node_id"] = engine_result.get("error_node_id")

        updated = supabase_service.client.table("wf_executions").update(
            update_data
        ).eq("id", execution_id).execute()

        return updated.data[0] if updated.data else execution

    except Exception as e:
        logger.exception(f"Workflow execution failed: {workflow_id}")

        # Update execution with error
        error_update = {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": str(e),
        }

        supabase_service.client.table("wf_executions").update(
            error_update
        ).eq("id", execution_id).execute()

        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


@router.get("/{workflow_id}/executions", response_model=ExecutionListResponse)
async def list_workflow_executions(
    workflow_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List executions for a specific workflow.

    Returns paginated list of executions sorted by creation date (newest first).
    """
    # Verify workflow exists
    workflow = supabase_service.client.table("wf_workflows").select("id").eq("id", workflow_id).single().execute()
    if not workflow.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    query = supabase_service.client.table("wf_executions").select(
        "*", count="exact"
    ).eq("workflow_id", workflow_id)

    if status:
        query = query.eq("status", status)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    return ExecutionListResponse(
        executions=result.data or [],
        total=result.count or 0,
    )


@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(execution_id: str):
    """Get a specific execution by ID."""
    result = supabase_service.client.table("wf_executions").select("*").eq("id", execution_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Execution not found")

    return result.data


@router.delete("/executions/{execution_id}")
async def delete_execution(execution_id: str):
    """Delete a specific execution."""
    # Verify exists
    existing = supabase_service.client.table("wf_executions").select("id").eq("id", execution_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Execution not found")

    supabase_service.client.table("wf_executions").delete().eq("id", execution_id).execute()

    return {"status": "deleted", "id": execution_id}


@router.post("/{workflow_id}/clear-executions")
async def clear_workflow_executions(workflow_id: str):
    """
    Delete all executions for a workflow.

    Use with caution - this permanently removes execution history.
    """
    # Verify workflow exists
    workflow = supabase_service.client.table("wf_workflows").select("id, name").eq("id", workflow_id).single().execute()
    if not workflow.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Count executions before deleting
    count_result = supabase_service.client.table("wf_executions").select(
        "id", count="exact"
    ).eq("workflow_id", workflow_id).execute()

    deleted_count = count_result.count or 0

    # Delete all executions
    supabase_service.client.table("wf_executions").delete().eq("workflow_id", workflow_id).execute()

    # Reset workflow stats
    supabase_service.client.table("wf_workflows").update({
        "run_count": 0,
        "last_run_at": None,
        "avg_duration_ms": None,
    }).eq("id", workflow_id).execute()

    return {
        "status": "cleared",
        "workflow_id": workflow_id,
        "deleted_count": deleted_count,
    }
