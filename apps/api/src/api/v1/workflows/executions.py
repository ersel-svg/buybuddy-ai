"""
Workflows - Executions Router

Endpoints for running workflows and viewing execution history.

Supports three execution modes:
- sync: Wait for completion (default)
- async: Return immediately, poll for result
- background: Fire-and-forget with optional webhook
"""

import asyncio
import logging
from typing import Optional, Union
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Request, Response

from services.supabase import supabase_service
from services.workflow import get_workflow_engine
from schemas.workflows import (
    WorkflowRunRequest,
    WorkflowRunResponse,
    ExecutionResponse,
    ExecutionListResponse,
    ExecutionMode,
)
from middleware.rate_limit import limiter

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


@router.post("/{workflow_id}/run", response_model=Union[WorkflowRunResponse, ExecutionResponse])
@router.post("/{workflow_id}/execute", response_model=Union[WorkflowRunResponse, ExecutionResponse], include_in_schema=False)
@limiter.limit("30/minute")  # GPU-intensive endpoint, stricter limit
async def run_workflow(request: Request, response: Response, workflow_id: str, data: WorkflowRunRequest):
    """
    Run a workflow with the provided input.

    Supports three execution modes:
    - **sync** (default): Wait for completion, return full result
    - **async**: Return immediately with execution_id, poll /executions/{id} for result
    - **background**: Fire-and-forget with optional webhook callback

    Query params for async polling:
    - GET /executions/{execution_id} - Check status and get result

    Webhook payload (for background mode):
    ```json
    {
        "event": "workflow.completed" | "workflow.failed",
        "execution_id": "uuid",
        "status": "completed" | "failed",
        "output_data": {...},
        "duration_ms": 1234
    }
    ```
    """
    logger.info(f"=== Workflow execution request received ===")
    logger.info(f"Workflow ID: {workflow_id}, Mode: {data.mode.value}")
    logger.info(f"Request data: input={data.input is not None}, inputs={data.inputs is not None}")
    if data.inputs:
        logger.info(f"Inputs keys: {list(data.inputs.keys())}")

    # Verify workflow exists and is not archived
    workflow = supabase_service.client.table("wf_workflows").select(
        "id, name, status, definition"
    ).eq("id", workflow_id).single().execute()

    if not workflow.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if workflow.data.get("status") == "archived":
        raise HTTPException(status_code=400, detail="Cannot run archived workflow")

    # Normalize input from either format
    execution_input = data.get_execution_input()
    logger.info(f"Execution input: image_url={execution_input.image_url is not None}, image_base64={execution_input.image_base64 is not None}")

    # If no image provided, use a placeholder test image for dry-run
    # This allows testing the workflow structure without real input
    if not execution_input.image_url and not execution_input.image_base64:
        # Generate a simple test image placeholder
        import base64
        from io import BytesIO
        try:
            from PIL import Image
            img = Image.new("RGB", (640, 480), color=(100, 100, 200))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            execution_input.image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            logger.info(f"Using test image for workflow {workflow_id} (no input provided)")
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Either image_url or image_base64 must be provided"
            )

    # Create execution record with mode and options
    now = datetime.now(timezone.utc)
    execution_data = {
        "workflow_id": workflow_id,
        "status": "pending",
        "input_data": execution_input.model_dump(),
        "created_at": now.isoformat(),
        # Async execution fields
        "execution_mode": data.mode.value,
        "priority": data.priority,
        "callback_url": data.callback_url,
        "timeout_seconds": data.timeout_seconds,
        "max_retries": data.max_retries if data.retry_on_failure else 0,
        "retry_count": 0,
    }

    result = supabase_service.client.table("wf_executions").insert(execution_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create execution")

    execution = result.data[0]
    execution_id = execution["id"]

    # Handle based on execution mode
    if data.mode == ExecutionMode.SYNC:
        # Synchronous execution - wait for result
        return await _execute_sync(
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_def=workflow.data.get("definition", {}),
            inputs=execution_input.model_dump(),
            timeout=data.timeout_seconds,
        )

    elif data.mode == ExecutionMode.ASYNC:
        # Async execution - return immediately, worker will process
        # Estimate wait time based on queue depth
        estimated_wait = await _estimate_wait_time(workflow_id)

        return WorkflowRunResponse(
            execution_id=execution_id,
            status="pending",
            mode=data.mode,
            status_url=f"/api/v1/workflows/executions/{execution_id}",
            estimated_wait_seconds=estimated_wait,
        )

    else:  # BACKGROUND
        # Fire and forget - worker will process, webhook on completion
        return WorkflowRunResponse(
            execution_id=execution_id,
            status="pending",
            mode=data.mode,
            status_url=f"/api/v1/workflows/executions/{execution_id}",
        )


async def _execute_sync(
    workflow_id: str,
    execution_id: str,
    workflow_def: dict,
    inputs: dict,
    timeout: int,
) -> ExecutionResponse:
    """Execute workflow synchronously with timeout."""

    # Update status to running
    supabase_service.client.table("wf_executions").update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", execution_id).execute()

    try:
        engine = get_workflow_engine()

        # Execute with timeout
        engine_result = await asyncio.wait_for(
            engine.execute(
                workflow=workflow_def,
                inputs=inputs,
                workflow_id=workflow_id,
                execution_id=execution_id,
            ),
            timeout=timeout,
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

        return updated.data[0] if updated.data else {"id": execution_id, **update_data}

    except asyncio.TimeoutError:
        error_msg = f"Execution timeout after {timeout}s"
        supabase_service.client.table("wf_executions").update({
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_message": error_msg,
        }).eq("id", execution_id).execute()

        raise HTTPException(status_code=408, detail=error_msg)

    except Exception as e:
        logger.exception(f"Workflow execution failed: {workflow_id}")

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


async def _estimate_wait_time(workflow_id: str) -> int:
    """Estimate wait time based on queue depth and avg duration."""
    # Count pending jobs ahead in queue
    pending = supabase_service.client.table("wf_executions").select(
        "id", count="exact"
    ).eq("status", "pending").in_(
        "execution_mode", ["async", "background"]
    ).execute()

    # Get avg duration for this workflow
    stats = supabase_service.client.table("wf_workflows").select(
        "avg_duration_ms"
    ).eq("id", workflow_id).single().execute()

    queue_depth = pending.count or 0
    avg_ms = stats.data.get("avg_duration_ms", 5000) if stats.data else 5000

    # Estimate: queue_depth * avg_duration / concurrent_workers (assume 5)
    return max(1, int((queue_depth * avg_ms / 1000) / 5))


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


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel a running or pending execution."""
    # Get current execution
    result = supabase_service.client.table("wf_executions").select("*").eq("id", execution_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Execution not found")

    current_status = result.data.get("status")
    if current_status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel execution with status: {current_status}"
        )

    # Update to cancelled
    updated = supabase_service.client.table("wf_executions").update({
        "status": "cancelled",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", execution_id).execute()

    return {"status": "cancelled", "id": execution_id}


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
