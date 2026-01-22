"""
Workflows - CRUD Router

Endpoints for managing workflow definitions.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from services.supabase import supabase_service
from schemas.workflows import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowListResponse,
)

router = APIRouter()


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by status: draft, active, archived"),
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List all workflows with optional filtering.

    Returns paginated list of workflows sorted by last updated.
    """
    query = supabase_service.client.table("wf_workflows").select("*", count="exact")

    if status:
        query = query.eq("status", status)

    if search:
        query = query.ilike("name", f"%{search}%")

    query = query.order("updated_at", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    return WorkflowListResponse(
        workflows=result.data or [],
        total=result.count or 0,
    )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get a single workflow by ID."""
    result = supabase_service.client.table("wf_workflows").select("*").eq("id", workflow_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return result.data


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(data: WorkflowCreate):
    """
    Create a new workflow.

    If no definition is provided, creates an empty workflow with default structure.
    """
    workflow_data = {
        "name": data.name,
        "description": data.description,
        "status": "draft",
    }

    if data.definition:
        workflow_data["definition"] = data.definition.model_dump()
    else:
        # Default empty workflow structure
        workflow_data["definition"] = {
            "version": "1.0",
            "nodes": [],
            "edges": [],
            "viewport": {"x": 0, "y": 0, "zoom": 1},
            "outputs": [],
        }

    result = supabase_service.client.table("wf_workflows").insert(workflow_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create workflow")

    return result.data[0]


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: str, data: WorkflowUpdate):
    """
    Update a workflow.

    Only provided fields are updated. Definition is replaced entirely if provided.
    """
    # Verify workflow exists
    existing = supabase_service.client.table("wf_workflows").select("id, status").eq("id", workflow_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    update_data = {}

    if data.name is not None:
        update_data["name"] = data.name

    if data.description is not None:
        update_data["description"] = data.description

    if data.status is not None:
        update_data["status"] = data.status

    if data.definition is not None:
        update_data["definition"] = data.definition.model_dump()

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("wf_workflows").update(update_data).eq("id", workflow_id).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to update workflow")

    return result.data[0]


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow and all its executions.

    This is a permanent action and cannot be undone.
    """
    # Verify exists
    existing = supabase_service.client.table("wf_workflows").select("id, name").eq("id", workflow_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Delete workflow (executions will cascade delete)
    supabase_service.client.table("wf_workflows").delete().eq("id", workflow_id).execute()

    return {
        "status": "deleted",
        "id": workflow_id,
        "name": existing.data.get("name"),
    }


@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse)
async def duplicate_workflow(workflow_id: str, new_name: Optional[str] = None):
    """
    Create a copy of an existing workflow.

    The copy starts as a draft with run_count reset to 0.
    """
    # Get source workflow
    source = supabase_service.client.table("wf_workflows").select("*").eq("id", workflow_id).single().execute()
    if not source.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Create copy
    copy_data = {
        "name": new_name or f"{source.data['name']} (Copy)",
        "description": source.data.get("description"),
        "definition": source.data.get("definition"),
        "status": "draft",
    }

    result = supabase_service.client.table("wf_workflows").insert(copy_data).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to duplicate workflow")

    return result.data[0]


@router.patch("/{workflow_id}/status")
async def update_workflow_status(workflow_id: str, status: str):
    """
    Update only the status of a workflow.

    Valid statuses: draft, active, archived
    """
    valid_statuses = ["draft", "active", "archived"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    result = supabase_service.client.table("wf_workflows").update(
        {"status": status}
    ).eq("id", workflow_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {"status": status, "workflow_id": workflow_id}
