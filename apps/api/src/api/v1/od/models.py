"""
Object Detection - Model Registry Router

Endpoints for managing trained OD models.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.supabase import supabase_service

router = APIRouter()


@router.get("")
async def list_models(
    is_active: Optional[bool] = None,
    model_type: Optional[str] = None,
    limit: int = 50,
):
    """List trained models with optional filters."""
    query = supabase_service.client.table("od_trained_models").select("*")

    if is_active is not None:
        query = query.eq("is_active", is_active)
    if model_type:
        query = query.eq("model_type", model_type)

    query = query.order("created_at", desc=True).limit(limit)
    result = query.execute()

    return result.data or []


@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get a single trained model by ID."""
    result = supabase_service.client.table("od_trained_models").select("*").eq("id", model_id).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Model not found")

    model = result.data

    # Get training run info if available
    if model.get("training_run_id"):
        training = supabase_service.client.table("od_training_runs").select(
            "name, dataset_id, config"
        ).eq("id", model["training_run_id"]).single().execute()

        if training.data:
            model["training_run_name"] = training.data.get("name")
            model["dataset_id"] = training.data.get("dataset_id")
            model["class_names"] = training.data.get("config", {}).get("class_names", [])

    return model


@router.patch("/{model_id}")
async def update_model(
    model_id: str,
    data: dict,
):
    """Update a trained model's metadata."""
    # Verify model exists
    existing = supabase_service.client.table("od_trained_models").select("id").eq("id", model_id).single().execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Model not found")

    # Only allow certain fields to be updated
    allowed_fields = {"name", "description", "is_active"}
    update_data = {k: v for k, v in data.items() if k in allowed_fields}

    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    result = supabase_service.client.table("od_trained_models").update(update_data).eq("id", model_id).execute()

    return {"id": model_id, "updated": True}


@router.post("/{model_id}/set-default")
async def set_default_model(model_id: str):
    """Set a model as the default for inference."""
    # Verify model exists
    model = supabase_service.client.table("od_trained_models").select("*").eq("id", model_id).single().execute()
    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if model is active
    if not model.data.get("is_active"):
        raise HTTPException(status_code=400, detail="Model must be active to be set as default")

    # Clear existing default for same model type
    model_type = model.data.get("model_type")
    supabase_service.client.table("od_trained_models").update({
        "is_default": False
    }).eq("model_type", model_type).eq("is_default", True).execute()

    # Set this model as default
    supabase_service.client.table("od_trained_models").update({
        "is_default": True
    }).eq("id", model_id).execute()

    return {"id": model_id, "is_default": True}


@router.delete("/{model_id}")
async def delete_model(model_id: str, force: bool = False):
    """Delete a trained model."""
    # Verify model exists
    model = supabase_service.client.table("od_trained_models").select("*").eq("id", model_id).single().execute()
    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if it's the default model
    if model.data.get("is_default") and not force:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the default model. Set another model as default first or use force=true"
        )

    # Delete from storage if URL exists
    checkpoint_url = model.data.get("checkpoint_url")
    if checkpoint_url:
        try:
            # Extract path from URL
            bucket = "od-models"
            # URL format: https://...supabase.co/storage/v1/object/public/od-models/path/to/file
            if f"/storage/v1/object/public/{bucket}/" in checkpoint_url:
                file_path = checkpoint_url.split(f"/storage/v1/object/public/{bucket}/")[1]
                supabase_service.client.storage.from_(bucket).remove([file_path])
        except Exception as e:
            print(f"Failed to delete model file: {e}")
            # Continue with deletion even if file removal fails

    # Delete from database
    supabase_service.client.table("od_trained_models").delete().eq("id", model_id).execute()

    return {"status": "deleted", "id": model_id}


@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """Download a trained model checkpoint."""
    # Verify model exists
    model = supabase_service.client.table("od_trained_models").select("checkpoint_url, name, model_type").eq("id", model_id).single().execute()
    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    checkpoint_url = model.data.get("checkpoint_url")
    if not checkpoint_url:
        raise HTTPException(status_code=404, detail="Model checkpoint not available")

    # Generate a signed URL for download
    bucket = "od-models"
    try:
        # Extract path from URL
        if f"/storage/v1/object/public/{bucket}/" in checkpoint_url:
            file_path = checkpoint_url.split(f"/storage/v1/object/public/{bucket}/")[1]
            signed_url = supabase_service.client.storage.from_(bucket).create_signed_url(
                file_path,
                expires_in=3600  # 1 hour
            )
            return {
                "download_url": signed_url.get("signedURL"),
                "filename": f"{model.data['name']}_{model.data['model_type']}.pt",
                "expires_in": 3600,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {str(e)}")


@router.get("/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get detailed metrics for a trained model."""
    # Verify model exists and get training run info
    model = supabase_service.client.table("od_trained_models").select(
        "*, training_run:od_training_runs(*)"
    ).eq("id", model_id).single().execute()

    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    training_run = model.data.get("training_run")
    if not training_run:
        return {
            "model_id": model_id,
            "map": model.data.get("map"),
            "map_50": model.data.get("map_50"),
            "training_metrics": None,
        }

    return {
        "model_id": model_id,
        "map": model.data.get("map"),
        "map_50": model.data.get("map_50"),
        "class_count": model.data.get("class_count"),
        "training_run_id": training_run.get("id"),
        "training_run_name": training_run.get("name"),
        "total_epochs": training_run.get("total_epochs"),
        "best_epoch": training_run.get("best_epoch"),
        "metrics_history": training_run.get("metrics_history"),
        "config": training_run.get("config"),
    }


@router.post("/{model_id}/activate")
async def activate_model(model_id: str):
    """Activate a model for use in inference."""
    # Verify model exists
    model = supabase_service.client.table("od_trained_models").select("*").eq("id", model_id).single().execute()
    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.data.get("is_active"):
        return {"id": model_id, "is_active": True, "message": "Model is already active"}

    # Activate the model
    supabase_service.client.table("od_trained_models").update({
        "is_active": True
    }).eq("id", model_id).execute()

    return {"id": model_id, "is_active": True}


@router.post("/{model_id}/deactivate")
async def deactivate_model(model_id: str):
    """Deactivate a model."""
    # Verify model exists
    model = supabase_service.client.table("od_trained_models").select("*").eq("id", model_id).single().execute()
    if not model.data:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.data.get("is_default"):
        raise HTTPException(status_code=400, detail="Cannot deactivate the default model")

    if not model.data.get("is_active"):
        return {"id": model_id, "is_active": False, "message": "Model is already inactive"}

    # Deactivate the model
    supabase_service.client.table("od_trained_models").update({
        "is_active": False
    }).eq("id", model_id).execute()

    return {"id": model_id, "is_active": False}


@router.get("/default/{model_type}")
async def get_default_model(model_type: str):
    """Get the default model for a specific model type."""
    result = supabase_service.client.table("od_trained_models").select("*").eq(
        "model_type", model_type
    ).eq("is_default", True).single().execute()

    if not result.data:
        # Try to get any active model of this type
        fallback = supabase_service.client.table("od_trained_models").select("*").eq(
            "model_type", model_type
        ).eq("is_active", True).order("map", desc=True).limit(1).execute()

        if fallback.data and len(fallback.data) > 0:
            return fallback.data[0]

        raise HTTPException(status_code=404, detail=f"No default model found for type: {model_type}")

    return result.data
