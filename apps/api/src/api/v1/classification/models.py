"""
Classification - Models Router

Endpoints for managing trained classification models.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

from services.supabase import supabase_service
from schemas.classification import (
    CLSTrainedModelUpdate,
    CLSTrainedModelResponse,
)

router = APIRouter()


@router.get("", response_model=list[CLSTrainedModelResponse])
async def list_models(
    model_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    training_run_id: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
):
    """List trained classification models."""
    offset = (page - 1) * limit

    query = supabase_service.client.table("cls_trained_models").select("*", count="exact")

    if model_type:
        query = query.eq("model_type", model_type)
    if is_active is not None:
        query = query.eq("is_active", is_active)
    if training_run_id:
        query = query.eq("training_run_id", training_run_id)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)

    result = query.execute()

    return result.data or []


@router.get("/default/{model_type}", response_model=CLSTrainedModelResponse)
async def get_default_model(model_type: str):
    """Get the default model for a given model type."""
    # Try to get default
    result = supabase_service.client.table("cls_trained_models").select("*").eq("model_type", model_type).eq("is_default", True).single().execute()

    if result.data:
        return result.data

    # Fallback to best active model
    fallback = supabase_service.client.table("cls_trained_models").select("*").eq("model_type", model_type).eq("is_active", True).order("accuracy", desc=True).limit(1).execute()

    if fallback.data:
        return fallback.data[0]

    raise HTTPException(status_code=404, detail=f"No model found for type: {model_type}")


@router.get("/{model_id}", response_model=CLSTrainedModelResponse)
async def get_model(model_id: str):
    """Get a single model by ID."""
    result = supabase_service.client.table("cls_trained_models").select("*").eq("id", model_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return result.data


@router.patch("/{model_id}", response_model=CLSTrainedModelResponse)
async def update_model(model_id: str, data: CLSTrainedModelUpdate):
    """Update model metadata."""
    update_data = data.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = supabase_service.client.table("cls_trained_models").update(update_data).eq("id", model_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return result.data[0]


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    # Get model to check if default
    model = supabase_service.client.table("cls_trained_models").select("is_default, checkpoint_url").eq("id", model_id).single().execute()

    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.data.get("is_default"):
        raise HTTPException(status_code=400, detail="Cannot delete the default model")

    # TODO: Delete checkpoint from storage
    # if model.data.get("checkpoint_url"):
    #     delete_from_storage(model.data["checkpoint_url"])

    supabase_service.client.table("cls_trained_models").delete().eq("id", model_id).execute()

    return {"success": True, "message": "Model deleted"}


@router.post("/{model_id}/activate")
async def activate_model(model_id: str):
    """Activate a model for use."""
    result = supabase_service.client.table("cls_trained_models").update({"is_active": True}).eq("id", model_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"success": True, "message": "Model activated"}


@router.post("/{model_id}/deactivate")
async def deactivate_model(model_id: str):
    """Deactivate a model."""
    # Check if model is default
    model = supabase_service.client.table("cls_trained_models").select("is_default").eq("id", model_id).single().execute()

    if model.data and model.data.get("is_default"):
        raise HTTPException(status_code=400, detail="Cannot deactivate the default model")

    result = supabase_service.client.table("cls_trained_models").update({"is_active": False}).eq("id", model_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"success": True, "message": "Model deactivated"}


@router.post("/{model_id}/set-default")
async def set_default_model(model_id: str):
    """Set a model as the default for its type."""
    # Get model to check type and active status
    model = supabase_service.client.table("cls_trained_models").select("model_type, is_active").eq("id", model_id).single().execute()

    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.data.get("is_active"):
        raise HTTPException(status_code=400, detail="Model must be active to be set as default")

    # The trigger will handle unsetting other defaults
    result = supabase_service.client.table("cls_trained_models").update({"is_default": True}).eq("id", model_id).execute()

    return {"success": True, "message": "Model set as default"}


@router.get("/{model_id}/download")
async def get_model_download_url(model_id: str):
    """Get download URL for model checkpoint."""
    result = supabase_service.client.table("cls_trained_models").select("checkpoint_url, onnx_url, torchscript_url, name").eq("id", model_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "name": result.data.get("name"),
        "checkpoint_url": result.data.get("checkpoint_url"),
        "onnx_url": result.data.get("onnx_url"),
        "torchscript_url": result.data.get("torchscript_url"),
    }


@router.post("/{model_id}/export-onnx")
async def export_to_onnx(model_id: str):
    """Export model to ONNX format."""
    # Check if already exported
    model = supabase_service.client.table("cls_trained_models").select("onnx_url, checkpoint_url").eq("id", model_id).single().execute()

    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.data.get("onnx_url"):
        return {"status": "already_exported", "onnx_url": model.data["onnx_url"]}

    # TODO: Submit ONNX export job to RunPod
    # For now, return pending status

    return {"status": "pending", "message": "ONNX export job submitted"}


@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get detailed metrics for a model."""
    result = supabase_service.client.table("cls_trained_models").select(
        "accuracy, f1_score, top5_accuracy, precision_macro, recall_macro, confusion_matrix, per_class_metrics, class_names"
    ).eq("id", model_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "overall": {
            "accuracy": result.data.get("accuracy"),
            "f1_score": result.data.get("f1_score"),
            "top5_accuracy": result.data.get("top5_accuracy"),
            "precision_macro": result.data.get("precision_macro"),
            "recall_macro": result.data.get("recall_macro"),
        },
        "confusion_matrix": result.data.get("confusion_matrix"),
        "per_class_metrics": result.data.get("per_class_metrics"),
        "class_names": result.data.get("class_names"),
    }
