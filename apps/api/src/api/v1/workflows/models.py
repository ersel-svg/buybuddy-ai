"""
Workflows - Unified Models Router

Endpoint for getting all available models (pretrained + trained) for workflow blocks.
"""

from typing import Optional
from fastapi import APIRouter, Query

from services.supabase import supabase_service
from schemas.workflows import (
    UnifiedModelsResponse,
    ModelCategory,
    PretrainedModelResponse,
    TrainedModelResponse,
)

router = APIRouter()


@router.get("", response_model=UnifiedModelsResponse)
async def get_workflow_models(
    model_type: Optional[str] = Query(
        None,
        description="Filter by model type: detection, classification, embedding, segmentation"
    ),
    include_inactive: bool = Query(False, description="Include inactive models"),
):
    """
    Get all available models for workflow blocks.

    Returns pretrained models from wf_pretrained_models table and
    trained models from od_trained_models, cls_trained_models, and trained_models tables.

    Models are grouped by type: detection, classification, embedding, segmentation.
    """
    response = UnifiedModelsResponse()

    # Fetch pretrained models
    pretrained_query = supabase_service.client.table("wf_pretrained_models").select("*")
    if model_type:
        pretrained_query = pretrained_query.eq("model_type", model_type)
    if not include_inactive:
        pretrained_query = pretrained_query.eq("is_active", True)

    pretrained_result = pretrained_query.order("name").execute()

    # Group pretrained by type
    pretrained_by_type = {
        "detection": [],
        "classification": [],
        "embedding": [],
        "segmentation": [],
    }

    for model in pretrained_result.data or []:
        mt = model.get("model_type")
        if mt in pretrained_by_type:
            pretrained_by_type[mt].append(PretrainedModelResponse(
                id=model["id"],
                name=model["name"],
                description=model.get("description"),
                model_type=mt,
                source=model["source"],
                model_path=model["model_path"],
                classes=model.get("classes"),
                class_count=model.get("class_count"),
                default_config=model.get("default_config", {}),
                embedding_dim=model.get("embedding_dim"),
                input_size=model.get("input_size"),
                is_active=model.get("is_active", True),
            ))

    # Fetch trained OD models
    if not model_type or model_type == "detection":
        od_query = supabase_service.client.table("od_trained_models").select("*")
        if not include_inactive:
            od_query = od_query.or_("is_active.eq.true,is_default.eq.true")
        od_result = od_query.order("created_at", desc=True).execute()

        for model in od_result.data or []:
            pretrained_by_type["detection"].append(TrainedModelResponse(
                id=model["id"],
                name=model["name"],
                description=model.get("description"),
                model_type="detection",
                provider=model.get("model_type", "unknown"),  # rf-detr, rt-detr, yolo-nas
                checkpoint_url=model.get("checkpoint_url"),
                classes=model.get("class_mapping"),
                class_count=model.get("class_count"),
                map=model.get("map"),
                is_active=model.get("is_active", False),
                created_at=model.get("created_at"),
            ))

    # Fetch trained classification models
    if not model_type or model_type == "classification":
        cls_query = supabase_service.client.table("cls_trained_models").select("*")
        if not include_inactive:
            cls_query = cls_query.or_("is_active.eq.true,is_default.eq.true")
        cls_result = cls_query.order("created_at", desc=True).execute()

        for model in cls_result.data or []:
            pretrained_by_type["classification"].append(TrainedModelResponse(
                id=model["id"],
                name=model["name"],
                description=model.get("description"),
                model_type="classification",
                provider=model.get("model_type", "unknown"),  # vit, convnext, efficientnet, swin
                checkpoint_url=model.get("checkpoint_url"),
                classes=model.get("class_mapping"),
                class_count=model.get("class_count"),
                accuracy=model.get("test_accuracy"),
                is_active=model.get("is_active", False),
                created_at=model.get("created_at"),
            ))

    # Fetch trained embedding models
    if not model_type or model_type == "embedding":
        emb_query = supabase_service.client.table("trained_models").select(
            "*, training_run:training_runs(num_classes, base_model_type), "
            "checkpoint:training_checkpoints(checkpoint_url), "
            "embedding_model:embedding_models(model_family, embedding_dim, config)"
        )
        if not include_inactive:
            emb_query = emb_query.or_("is_active.eq.true,is_default.eq.true")
        emb_result = emb_query.order("created_at", desc=True).execute()

        for model in emb_result.data or []:
            training_run = model.get("training_run", {}) or {}
            checkpoint = model.get("checkpoint", {}) or {}
            embedding_model = model.get("embedding_model", {}) or {}

            test_metrics = model.get("test_metrics", {}) or {}
            recall_at_1 = test_metrics.get("recall_at_1") if isinstance(test_metrics, dict) else None

            pretrained_by_type["embedding"].append(TrainedModelResponse(
                id=model["id"],
                name=model["name"],
                description=model.get("description"),
                model_type="embedding",
                provider=embedding_model.get("model_family", training_run.get("base_model_type", "unknown")),
                checkpoint_url=checkpoint.get("checkpoint_url"),
                class_count=training_run.get("num_classes"),
                recall_at_1=recall_at_1,
                is_active=model.get("is_active", False),
                created_at=model.get("created_at"),
            ))

    # Build response
    response.detection = ModelCategory(
        pretrained=[m for m in pretrained_by_type["detection"] if isinstance(m, PretrainedModelResponse)],
        trained=[m for m in pretrained_by_type["detection"] if isinstance(m, TrainedModelResponse)],
    )
    response.classification = ModelCategory(
        pretrained=[m for m in pretrained_by_type["classification"] if isinstance(m, PretrainedModelResponse)],
        trained=[m for m in pretrained_by_type["classification"] if isinstance(m, TrainedModelResponse)],
    )
    response.embedding = ModelCategory(
        pretrained=[m for m in pretrained_by_type["embedding"] if isinstance(m, PretrainedModelResponse)],
        trained=[m for m in pretrained_by_type["embedding"] if isinstance(m, TrainedModelResponse)],
    )
    response.segmentation = ModelCategory(
        pretrained=[m for m in pretrained_by_type["segmentation"] if isinstance(m, PretrainedModelResponse)],
        trained=[m for m in pretrained_by_type["segmentation"] if isinstance(m, TrainedModelResponse)],
    )

    return response


@router.get("/pretrained")
async def list_pretrained_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
):
    """List only pretrained models."""
    query = supabase_service.client.table("wf_pretrained_models").select("*").eq("is_active", True)

    if model_type:
        query = query.eq("model_type", model_type)

    result = query.order("name").execute()
    return result.data or []


@router.get("/pretrained/{model_id}")
async def get_pretrained_model(model_id: str):
    """Get a specific pretrained model by ID."""
    result = supabase_service.client.table("wf_pretrained_models").select("*").eq("id", model_id).single().execute()

    if not result.data:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Model not found")

    return result.data
