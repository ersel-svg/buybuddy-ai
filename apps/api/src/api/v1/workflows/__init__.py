"""
Workflows API Router

This module contains all endpoints for the workflow system:
- Workflows: CRUD operations for workflow definitions
- Executions: Run workflows and view execution history
- Models: Unified model picker for all model types
- Blocks: Available workflow blocks metadata
"""

from fastapi import APIRouter

from services.supabase import supabase_service
from services.workflow import get_block_metadata, BLOCK_CATEGORIES

# Import sub-routers
from .workflows import router as workflows_router
from .executions import router as executions_router
from .models import router as models_router

router = APIRouter()

# Include sub-routers
router.include_router(workflows_router, tags=["Workflows"])
router.include_router(executions_router, tags=["Workflow Executions"])
router.include_router(models_router, prefix="/models", tags=["Workflow Models"])


# Health check endpoint for workflows module
@router.get("/health")
async def workflows_health_check() -> dict[str, str]:
    """Workflows module health check."""
    return {
        "status": "healthy",
        "module": "workflows",
        "version": "0.1.0",
    }


# Blocks metadata endpoint for frontend block palette
@router.get("/blocks")
async def get_available_blocks() -> dict:
    """
    Get all available workflow blocks with their metadata.

    Returns block definitions including input/output ports and config schemas.
    Used by the frontend to build the block palette and validate connections.
    """
    blocks = get_block_metadata()

    return {
        "blocks": blocks,
        "categories": BLOCK_CATEGORIES,
    }


# Stats endpoint for workflows dashboard
@router.get("/stats")
async def get_workflows_stats() -> dict:
    """Get Workflows dashboard statistics."""
    try:
        # Get workflow counts by status
        workflows_result = supabase_service.client.table("wf_workflows").select(
            "id, status, run_count", count="exact"
        ).execute()

        total_workflows = workflows_result.count or 0
        status_counts = {"draft": 0, "active": 0, "archived": 0}
        total_runs = 0

        for wf in workflows_result.data or []:
            status = wf.get("status", "draft")
            if status in status_counts:
                status_counts[status] += 1
            total_runs += wf.get("run_count", 0) or 0

        # Get execution stats
        executions_result = supabase_service.client.table("wf_executions").select(
            "id, status, duration_ms", count="exact"
        ).execute()

        total_executions = executions_result.count or 0
        exec_status_counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
        durations = []

        for ex in executions_result.data or []:
            status = ex.get("status", "pending")
            if status in exec_status_counts:
                exec_status_counts[status] += 1
            if ex.get("duration_ms"):
                durations.append(ex["duration_ms"])

        avg_duration = sum(durations) / len(durations) if durations else None

        # Get recent workflows
        recent_result = supabase_service.client.table("wf_workflows").select("*").order(
            "updated_at", desc=True
        ).limit(5).execute()

        # Get model counts
        pretrained_count = supabase_service.client.table("wf_pretrained_models").select(
            "id", count="exact"
        ).eq("is_active", True).execute()

        od_models_count = supabase_service.client.table("od_trained_models").select(
            "id", count="exact"
        ).or_("is_active.eq.true,is_default.eq.true").execute()

        cls_models_count = supabase_service.client.table("cls_trained_models").select(
            "id", count="exact"
        ).or_("is_active.eq.true,is_default.eq.true").execute()

        emb_models_count = supabase_service.client.table("trained_models").select(
            "id", count="exact"
        ).or_("is_active.eq.true,is_default.eq.true").execute()

        return {
            "total_workflows": total_workflows,
            "workflows_by_status": status_counts,
            "total_executions": total_executions,
            "executions_by_status": exec_status_counts,
            "avg_execution_time_ms": round(avg_duration, 2) if avg_duration else None,
            "total_runs": total_runs,
            "recent_workflows": recent_result.data or [],
            "model_counts": {
                "pretrained": pretrained_count.count or 0,
                "detection_trained": od_models_count.count or 0,
                "classification_trained": cls_models_count.count or 0,
                "embedding_trained": emb_models_count.count or 0,
            },
        }

    except Exception as e:
        print(f"Workflows Stats error: {e}")
        return {
            "total_workflows": 0,
            "workflows_by_status": {"draft": 0, "active": 0, "archived": 0},
            "total_executions": 0,
            "executions_by_status": {"pending": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0},
            "avg_execution_time_ms": None,
            "total_runs": 0,
            "recent_workflows": [],
            "model_counts": {
                "pretrained": 0,
                "detection_trained": 0,
                "classification_trained": 0,
                "embedding_trained": 0,
            },
        }
