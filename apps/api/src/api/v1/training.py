"""
Training API router for managing training runs, checkpoints, and trained models.

Supports:
- Multi-model training (DINOv2, DINOv3, CLIP, custom)
- Product ID-based data splitting (no leakage)
- Checkpoint management
- Model evaluation on test set
- Model deployment for embedding extraction
"""

from io import BytesIO
from typing import Optional, Literal
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.supabase import SupabaseService, supabase_service
from services.runpod import RunpodService, runpod_service
from auth.dependencies import get_current_user
from schemas.data_loading import DataLoadingConfig

# Router with authentication required for all endpoints
router = APIRouter(dependencies=[Depends(get_current_user)])


# ===========================================
# Schemas
# ===========================================

class LabelConfig(BaseModel):
    """Label field configuration for training."""
    label_field: Literal[
        "product_id",  # Default: each product is a class
        "category",    # Train category classifier
        "brand_name",  # Train brand classifier
        "custom"       # Custom mapping provided
    ] = "product_id"

    # For custom mapping: {value: class_id} or grouping
    custom_mapping: Optional[dict[str, str]] = None

    # Minimum samples per class (classes with fewer samples are excluded)
    min_samples_per_class: int = Field(2, ge=1)


class SplitConfig(BaseModel):
    """Data split configuration."""
    train_ratio: float = Field(0.70, ge=0.5, le=0.9)
    val_ratio: float = Field(0.15, ge=0.05, le=0.3)
    test_ratio: float = Field(0.15, ge=0.05, le=0.3)
    seed: int = Field(42, ge=1)

    # Stratify by this field (keeps distribution across splits)
    stratify_by: Optional[Literal["brand_name", "category"]] = "brand_name"


class TrainingConfigOverrides(BaseModel):
    """Training config overrides (optional)."""
    epochs: Optional[int] = Field(None, ge=1, le=100)
    batch_size: Optional[int] = Field(None, ge=4, le=128)
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=1e-2)
    weight_decay: Optional[float] = Field(None, ge=0, le=0.5)
    warmup_epochs: Optional[int] = Field(None, ge=0, le=10)
    llrd_factor: Optional[float] = Field(None, ge=0.5, le=1.0)
    arcface_margin: Optional[float] = Field(None, ge=0.1, le=0.8)
    arcface_scale: Optional[float] = Field(None, ge=16, le=128)
    augmentation_strength: Optional[Literal["none", "light", "moderate", "strong"]] = None
    use_gem_pooling: Optional[bool] = None
    use_arcface: Optional[bool] = None
    use_llrd: Optional[bool] = None
    mixed_precision: Optional[bool] = None
    gradient_accumulation_steps: Optional[int] = Field(None, ge=1, le=16)

    # Data loading configuration
    data_loading: Optional[DataLoadingConfig] = Field(
        None,
        description="Image preloading and dataloader configuration"
    )

    # Scheduler parameters
    scheduler_eta_min: Optional[float] = Field(
        None,
        ge=1e-8,
        le=1e-4,
        description="Scheduler minimum learning rate (eta_min)"
    )

    # Head learning rate multiplier
    head_lr_multiplier: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Learning rate multiplier for head layers"
    )

    # Validation batch multiplier
    val_batch_multiplier: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Validation batch size multiplier"
    )


class SOTALossConfig(BaseModel):
    """SOTA loss configuration."""
    arcface_weight: float = Field(1.0, ge=0, le=2)
    triplet_weight: float = Field(0.5, ge=0, le=2)
    domain_weight: float = Field(0.1, ge=0, le=1)
    arcface_margin: float = Field(0.5, ge=0.1, le=0.8)
    arcface_scale: float = Field(64.0, ge=16, le=128)
    triplet_margin: float = Field(0.3, ge=0.1, le=0.5)
    # Circle Loss (CVPR 2020)
    circle_weight: float = Field(0.3, ge=0, le=2, description="Weight for Circle Loss")
    circle_margin: float = Field(0.25, ge=0.1, le=0.5, description="Relaxation margin for Circle Loss")
    circle_gamma: float = Field(256.0, ge=64, le=512, description="Scale factor for Circle Loss")


class SOTASamplingConfig(BaseModel):
    """SOTA batch sampling configuration."""
    products_per_batch: int = Field(8, ge=2, le=32)
    samples_per_product: int = Field(4, ge=2, le=16)
    synthetic_ratio: float = Field(0.5, ge=0, le=1)


class SOTACurriculumConfig(BaseModel):
    """SOTA curriculum learning configuration."""
    warmup_epochs: int = Field(2, ge=0, le=10)
    easy_epochs: int = Field(5, ge=0, le=20)
    hard_epochs: int = Field(10, ge=0, le=50)
    finetune_epochs: int = Field(3, ge=0, le=10)


class SOTATTAConfig(BaseModel):
    """SOTA Test-Time Augmentation configuration."""
    enabled: bool = Field(False, description="Enable TTA during evaluation")
    mode: Literal["light", "standard", "full"] = Field(
        "light",
        description="TTA mode: light (2 views), standard (4 views), full (5+ views)"
    )


class SOTAConfig(BaseModel):
    """
    SOTA Training configuration.

    All features can be toggled on/off via config flags:
    - Combined loss (ArcFace + Triplet + Circle + Domain Adversarial)
    - P-K batch sampling with domain balancing
    - Curriculum learning
    - Domain adaptation
    - Early stopping
    - Test-Time Augmentation (TTA)
    """
    enabled: bool = Field(True, description="Enable SOTA training features")
    use_combined_loss: bool = Field(True, description="Use combined loss (ArcFace + Triplet + Circle + Domain)")
    use_pk_sampling: bool = Field(True, description="Use P-K batch sampling with domain balancing")
    use_curriculum: bool = Field(False, description="Enable curriculum learning")
    use_domain_adaptation: bool = Field(True, description="Enable domain adversarial training")
    use_early_stopping: bool = Field(True, description="Stop training when no improvement")
    early_stopping_patience: int = Field(5, ge=2, le=20, description="Epochs to wait before stopping")

    # Detailed configs
    loss: SOTALossConfig = Field(default_factory=SOTALossConfig)
    sampling: SOTASamplingConfig = Field(default_factory=SOTASamplingConfig)
    curriculum: SOTACurriculumConfig = Field(default_factory=SOTACurriculumConfig)
    tta: SOTATTAConfig = Field(default_factory=SOTATTAConfig)

    # Triplet mining integration
    triplet_mining_run_id: Optional[str] = Field(None, description="Pre-mined triplets from triplet mining run")


class ImageConfig(BaseModel):
    """Image selection configuration for training."""
    # Image types to include
    image_types: list[Literal["synthetic", "real", "augmented"]] = Field(
        default=["synthetic", "real", "augmented"],
        description="Which image types to include in training"
    )

    # Frame selection for synthetic images
    frame_selection: Literal["first", "key_frames", "interval", "all"] = Field(
        default="key_frames",
        description="How to select frames: first (1 frame), key_frames (4 angles), interval, all"
    )
    frame_interval: int = Field(5, ge=1, description="For interval selection, pick every N frames")
    max_frames_per_type: int = Field(10, ge=1, le=50, description="Maximum frames per image type per product")

    # Include matched cutouts as additional training data
    include_matched_cutouts: bool = Field(
        default=True,
        description="Include cutouts matched to products as 'real' domain training data"
    )


class CreateTrainingRunRequest(BaseModel):
    """Request to create a new training run."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # Base model
    base_model_type: Literal[
        "dinov2-small", "dinov2-base", "dinov2-large",
        "dinov3-small", "dinov3-base", "dinov3-large",
        "clip-vit-l-14"
    ] = "dinov3-base"

    # Data source
    data_source: Literal["all_products", "matched_products", "dataset", "selected"] = "matched_products"
    dataset_id: Optional[str] = None
    product_ids: Optional[list[str]] = None  # For 'selected' data source

    # Image configuration (which image types and how many frames)
    image_config: ImageConfig = Field(default_factory=ImageConfig)

    # Label configuration (what to train the model to classify)
    label_config: LabelConfig = Field(default_factory=LabelConfig)

    # Split config
    split_config: SplitConfig = Field(default_factory=SplitConfig)

    # Training config
    use_preset: bool = True  # Use model-specific preset
    config_overrides: Optional[TrainingConfigOverrides] = None
    saved_config_id: Optional[str] = None  # Use saved config

    # SOTA Training config (advanced features)
    sota_config: Optional[SOTAConfig] = None

    # Hard negative pairs (product pairs that look similar but are different)
    hard_negative_pairs: Optional[list[tuple[str, str]]] = None


class RegisterModelRequest(BaseModel):
    """Request to register a checkpoint as a trained model."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class EvaluateModelRequest(BaseModel):
    """Request to evaluate a trained model."""
    checkpoint_id: Optional[str] = None  # If None, uses best checkpoint
    metrics: list[str] = ["recall@1", "recall@5", "recall@10", "mAP"]
    include_cross_domain: bool = True
    include_hard_cases: bool = True


class SaveConfigRequest(BaseModel):
    """Request to save a training config."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    base_model_type: str
    config: dict


# ===========================================
# Response schemas
# ===========================================

class TrainingRunResponse(BaseModel):
    """Training run response."""
    id: str
    name: str
    status: str
    base_model_type: str
    current_epoch: int
    total_epochs: int
    train_product_count: Optional[int]
    val_product_count: Optional[int]
    test_product_count: Optional[int]
    best_val_loss: Optional[float]
    best_val_recall_at_1: Optional[float]
    created_at: datetime


# ===========================================
# Dependency
# ===========================================

def get_supabase() -> SupabaseService:
    """Get Supabase service instance."""
    return supabase_service


def get_runpod() -> RunpodService:
    """Get RunPod service instance."""
    return runpod_service


# ===========================================
# Training Runs Endpoints
# ===========================================

@router.get("/runs")
async def list_training_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    db: SupabaseService = Depends(get_supabase),
):
    """List all training runs."""
    return await db.get_training_runs(status=status, limit=limit)


@router.get("/runs/{run_id}")
async def get_training_run(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get training run details."""
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run


@router.post("/runs")
async def create_training_run(
    request: CreateTrainingRunRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Create and start a new training run.

    Steps:
    1. Fetch products based on data_source
    2. Split into train/val/test by product_id
    3. Build training config (preset + overrides)
    4. Create run record
    5. Dispatch to RunPod training worker
    """
    # Validate ratios
    total_ratio = request.split_config.train_ratio + request.split_config.val_ratio + request.split_config.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Train/val/test ratios must sum to 1.0, got {total_ratio}",
        )

    # Get products based on data source
    if request.data_source == "all_products":
        products = await db.get_products_for_training()
    elif request.data_source == "matched_products":
        products = await db.get_matched_products_for_training()
    elif request.data_source == "dataset":
        if not request.dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required for 'dataset' data source")
        products = await db.get_dataset_products(request.dataset_id)
    elif request.data_source == "selected":
        if not request.product_ids:
            raise HTTPException(status_code=400, detail="product_ids required for 'selected' data source")
        products = await db.get_products_by_ids(request.product_ids)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown data source: {request.data_source}")

    if len(products) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough products for training: {len(products)} (minimum 10)",
        )

    # Determine label field and extract class labels
    label_field = request.label_config.label_field
    min_samples = request.label_config.min_samples_per_class

    # Build class labels for each product
    from collections import Counter
    label_to_products: dict[str, list[str]] = {}

    for p in products:
        if label_field == "product_id":
            label = p["id"]
        elif label_field == "category":
            label = p.get("category") or "unknown"
        elif label_field == "brand_name":
            label = p.get("brand_name") or "unknown"
        elif label_field == "custom":
            # Use custom mapping if provided
            mapping = request.label_config.custom_mapping or {}
            raw_value = str(p.get("id", ""))
            label = mapping.get(raw_value, raw_value)
        else:
            label = p["id"]

        if label not in label_to_products:
            label_to_products[label] = []
        label_to_products[label].append(p["id"])

    # Filter classes with minimum samples
    valid_labels = {
        label: pids for label, pids in label_to_products.items()
        if len(pids) >= min_samples
    }

    if len(valid_labels) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough classes with minimum {min_samples} samples. Found {len(valid_labels)} classes.",
        )

    # Split by label (class) to prevent data leakage
    from random import Random
    rng = Random(request.split_config.seed)

    # Stratified split: split labels, keeping distribution
    all_labels = list(valid_labels.keys())
    rng.shuffle(all_labels)

    n = len(all_labels)
    train_end = int(n * request.split_config.train_ratio)
    val_end = train_end + int(n * request.split_config.val_ratio)

    train_labels = all_labels[:train_end]
    val_labels = all_labels[train_end:val_end]
    test_labels = all_labels[val_end:]

    # Expand to product_ids (all products belonging to train labels go to train set, etc.)
    train_product_ids = [pid for label in train_labels for pid in valid_labels[label]]
    val_product_ids = [pid for label in val_labels for pid in valid_labels[label]]
    test_product_ids = [pid for label in test_labels for pid in valid_labels[label]]

    # Build label mapping (for worker)
    label_mapping = {label: idx for idx, label in enumerate(sorted(valid_labels.keys()))}
    num_classes = len(label_mapping)

    # Build identifier mapping (product_id -> product identifiers for inference)
    # This is used after training to map predictions back to product info
    all_product_ids = set(train_product_ids + val_product_ids + test_product_ids)
    products_by_id = {p["id"]: p for p in products if p["id"] in all_product_ids}

    identifier_mapping = {}
    for product_id in all_product_ids:
        p = products_by_id.get(product_id, {})
        identifier_mapping[product_id] = {
            "barcode": p.get("barcode"),
            "product_name": p.get("product_name"),
            "brand_name": p.get("brand_name"),
            "category": p.get("category"),
            "identifiers": p.get("identifiers"),  # Additional identifiers (UPC, EAN, etc.)
        }

    # Build training images for each product
    # Uses frames_path directly for efficiency (skips slow product_images queries)
    image_cfg = request.image_config
    training_images: dict[str, list[dict]] = {}  # product_id -> list of images
    image_stats = {"synthetic": 0, "real": 0, "augmented": 0, "cutout": 0}

    # Helper function to generate frame URLs from frames_path
    def generate_frame_urls(frames_path: str, frame_count: int, frame_selection: str, frame_interval: int, max_frames: int) -> list[dict]:
        """Generate frame URLs from frames_path."""
        if not frames_path or frame_count <= 0:
            return []

        # Determine which frame indices to use
        if frame_selection == "first":
            indices = [0]
        elif frame_selection == "all":
            indices = list(range(min(frame_count, max_frames)))
        elif frame_selection == "key_frames":
            # Pick 4 frames at 0°, 90°, 180°, 270° (roughly)
            step = max(1, frame_count // 4)
            indices = [0] + [i * step for i in range(1, 4) if i * step < frame_count]
            indices = indices[:max_frames]
        elif frame_selection == "interval":
            indices = list(range(0, frame_count, frame_interval))[:max_frames]
        else:
            indices = [0]

        # Generate URLs
        base_url = frames_path.rstrip("/")
        frames = []
        for idx in indices:
            frames.append({
                "url": f"{base_url}/frame_{idx:04d}.png",
                "image_type": "synthetic",
                "frame_index": idx,
                "domain": "synthetic",
            })
        return frames

    # OPTIMIZED: Use frames_path directly for all products (avoids N queries)
    # This is faster than querying product_images table for each product
    if "synthetic" in image_cfg.image_types:
        for product_id in all_product_ids:
            product_data = products_by_id.get(product_id, {})
            frames_path = product_data.get("frames_path")
            frame_count = product_data.get("frame_count", 0)

            if frames_path and frame_count > 0:
                frames = generate_frame_urls(
                    frames_path=frames_path,
                    frame_count=frame_count,
                    frame_selection=image_cfg.frame_selection,
                    frame_interval=image_cfg.frame_interval,
                    max_frames=image_cfg.max_frames_per_type,
                )
                if frames:
                    training_images[product_id] = frames
                    image_stats["synthetic"] += len(frames)

    # Include matched cutouts as real-domain training data (batch query)
    if image_cfg.include_matched_cutouts:
        all_cutouts = await db.get_matched_cutouts_for_products(list(all_product_ids))
        for cutout in all_cutouts:
            product_id = cutout.get("matched_product_id")
            if product_id and product_id in all_product_ids:
                if product_id not in training_images:
                    training_images[product_id] = []
                training_images[product_id].append({
                    "url": cutout["image_url"],
                    "image_type": "cutout",
                    "frame_index": 0,
                    "domain": "real",
                    "cutout_id": cutout["id"],
                })
                image_stats["cutout"] += 1

    # Check if we have enough images
    total_images = sum(len(imgs) for imgs in training_images.values())
    if total_images < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough training images: {total_images} (minimum 100). "
                   f"Stats: {image_stats}. Try including more image types.",
        )

    # Build training config
    config = await _build_training_config(
        base_model_type=request.base_model_type,
        use_preset=request.use_preset,
        overrides=request.config_overrides,
        saved_config_id=request.saved_config_id,
        db=db,
    )

    # Add hard negative pairs to config if provided
    if request.hard_negative_pairs:
        config["hard_negative_pairs"] = request.hard_negative_pairs

    # Add SOTA config if provided
    if request.sota_config:
        config["sota_config"] = request.sota_config.model_dump()

    # Add split product IDs to config for worker
    config["train_product_ids"] = train_product_ids
    config["val_product_ids"] = val_product_ids
    config["test_product_ids"] = test_product_ids

    # Create run record
    run_data = {
        "name": request.name,
        "description": request.description,
        "base_model_type": request.base_model_type,
        "data_source": request.data_source,
        "dataset_id": request.dataset_id,
        "image_config": image_cfg.model_dump(),  # Store image configuration
        "image_stats": image_stats,  # Store image type counts
        "label_config": request.label_config.model_dump(),  # Store label configuration
        "split_config": request.split_config.model_dump(),
        "train_product_ids": train_product_ids,
        "val_product_ids": val_product_ids,
        "test_product_ids": test_product_ids,
        "train_product_count": len(train_product_ids),
        "val_product_count": len(val_product_ids),
        "test_product_count": len(test_product_ids),
        "total_images": total_images,
        "num_classes": num_classes,  # Number of unique labels (classes)
        "label_mapping": label_mapping,  # Map label -> class index
        "identifier_mapping": identifier_mapping,  # Map product_id -> identifiers for inference
        "training_config": config,
        "sota_config": request.sota_config.model_dump() if request.sota_config else None,
        "total_epochs": config.get("epochs", 10),
        "status": "pending",
    }

    run = await db.create_training_run(run_data)

    # Dispatch to RunPod (async)
    try:
        runpod_job = await runpod_service.start_training_job(
            training_run_id=run["id"],
            model_type=request.base_model_type,
            config=config,
            training_images=training_images,  # Pass images with URLs to worker
        )
        await db.update_training_run(run["id"], {
            "runpod_job_id": runpod_job.get("id"),
            "status": "preparing",
        })
    except Exception as e:
        await db.update_training_run(run["id"], {
            "status": "failed",
            "error_message": str(e),
        })
        raise HTTPException(status_code=500, detail=f"Failed to start training job: {e}")

    return run


@router.post("/runs/{run_id}/cancel")
async def cancel_training_run(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Cancel a running training job."""
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run["status"] not in ["pending", "preparing", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status: {run['status']}",
        )

    # Cancel RunPod job if exists
    if run.get("runpod_job_id"):
        try:
            from services.runpod import EndpointType
            await runpod_service.cancel_job(
                endpoint_type=EndpointType.TRAINING,
                job_id=run["runpod_job_id"],
            )
        except Exception:
            pass  # Best effort

    await db.update_training_run(run_id, {"status": "cancelled"})
    return {"status": "cancelled"}


@router.post("/runs/{run_id}/resume")
async def resume_training_run(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
    runpod: RunpodService = Depends(get_runpod),
):
    """
    Resume a failed or cancelled training run from the latest checkpoint.

    Creates a new training run with the same configuration, starting from
    the last saved checkpoint.
    """
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run["status"] not in ["failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Can only resume failed or cancelled runs. Current status: {run['status']}",
        )

    # Get the latest checkpoint for this run
    checkpoints = await db.get_training_checkpoints(run_id)
    if not checkpoints:
        raise HTTPException(
            status_code=400,
            detail="No checkpoints found for this run. Cannot resume.",
        )

    # Sort by epoch descending and get the latest
    checkpoints.sort(key=lambda x: x.get("epoch", 0), reverse=True)
    latest_checkpoint = checkpoints[0]

    # Calculate remaining epochs
    completed_epochs = latest_checkpoint.get("epoch", 0)
    total_epochs = run.get("total_epochs", 10)
    remaining_epochs = total_epochs - completed_epochs

    if remaining_epochs <= 0:
        raise HTTPException(
            status_code=400,
            detail="Training was already completed. No epochs remaining.",
        )

    # Create a new run with the same config but starting from checkpoint
    new_run_data = {
        "name": f"{run['name']} (resumed)",
        "description": f"Resumed from epoch {completed_epochs}. Original: {run.get('description', '')}",
        "base_model_type": run["base_model_type"],
        "data_source": run["data_source"],
        "dataset_id": run.get("dataset_id"),
        "label_config": run.get("label_config"),
        "split_config": run["split_config"],
        "train_product_ids": run["train_product_ids"],
        "val_product_ids": run["val_product_ids"],
        "test_product_ids": run["test_product_ids"],
        "train_product_count": run["train_product_count"],
        "val_product_count": run["val_product_count"],
        "test_product_count": run["test_product_count"],
        "num_classes": run["num_classes"],
        "label_mapping": run.get("label_mapping"),
        "identifier_mapping": run.get("identifier_mapping"),  # Preserve for inference
        "training_config": run["training_config"],
        "total_epochs": total_epochs,
        "current_epoch": completed_epochs,
        "status": "pending",
        # Link to original run
        "resumed_from_run_id": run_id,
        "resumed_from_checkpoint_id": latest_checkpoint["id"],
    }

    new_run = await db.create_training_run(new_run_data)

    # Dispatch to RunPod with checkpoint URL
    try:
        job_id = await runpod.start_training_job(
            training_run_id=new_run["id"],
            model_type=run["base_model_type"],
            config=run["training_config"],
            checkpoint_url=latest_checkpoint["checkpoint_url"],  # Resume from checkpoint
            start_epoch=completed_epochs,
        )

        await db.update_training_run(new_run["id"], {
            "status": "preparing",
            "runpod_job_id": job_id,
        })
    except Exception as e:
        await db.update_training_run(new_run["id"], {
            "status": "failed",
            "error_message": f"Failed to dispatch to RunPod: {str(e)}",
        })
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

    return {
        "new_run_id": new_run["id"],
        "resumed_from_epoch": completed_epochs,
        "remaining_epochs": remaining_epochs,
        "checkpoint_url": latest_checkpoint["checkpoint_url"],
    }


@router.delete("/runs/{run_id}")
async def delete_training_run(
    run_id: str,
    force: bool = False,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Delete a training run and its checkpoints.

    If the run has registered trained_models, deletion will be blocked unless force=true.
    """
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run["status"] in ["running", "preparing"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running job. Cancel it first.",
        )

    result = await db.delete_training_run(run_id, force=force)

    if not result.get("success"):
        raise HTTPException(
            status_code=409,
            detail={
                "message": result.get("error"),
                "linked_models": result.get("linked_models", []),
            },
        )

    return {"deleted": True}


@router.get("/runs/{run_id}/identifier-mapping")
async def get_identifier_mapping(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get identifier mapping for a training run.

    Returns a JSON mapping of product_id -> product identifiers.
    Useful for inference to map predictions back to product metadata.

    Example output:
    {
        "product_abc123": {
            "barcode": "012345678901",
            "product_name": "Coca-Cola Classic 330ml",
            "brand_name": "Coca-Cola",
            "category": "Beverages",
            "identifiers": {"upc": "012345678901", "ean": "5449000000996"}
        },
        ...
    }
    """
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    identifier_mapping = run.get("identifier_mapping")
    if not identifier_mapping:
        raise HTTPException(
            status_code=404,
            detail="Identifier mapping not available for this run (older runs may not have this data)",
        )

    return {
        "run_id": run_id,
        "run_name": run.get("name"),
        "label_field": run.get("label_config", {}).get("label_field", "product_id"),
        "num_products": len(identifier_mapping),
        "identifier_mapping": identifier_mapping,
    }


# ===========================================
# Checkpoints Endpoints
# ===========================================

@router.get("/runs/{run_id}/checkpoints")
async def list_checkpoints(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """List checkpoints for a training run."""
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return await db.get_training_checkpoints(run_id)


@router.get("/runs/{run_id}/metrics-history")
async def get_metrics_history(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get training metrics history for a run (per-epoch metrics for charts)."""
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return await db.get_training_metrics_history(run_id)


@router.get("/runs/{run_id}/progress")
async def get_training_progress(
    run_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get current training progress summary."""
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Get latest metrics
    metrics_history = await db.get_training_metrics_history(run_id)
    latest_metrics = metrics_history[-1] if metrics_history else None

    return {
        "current_epoch": run.get("current_epoch", 0),
        "total_epochs": run.get("total_epochs", 0),
        "progress": run.get("progress", 0.0),
        "status": run.get("status", "pending"),
        "latest_train_loss": latest_metrics.get("train_loss") if latest_metrics else None,
        "latest_val_loss": latest_metrics.get("val_loss") if latest_metrics else None,
        "best_val_loss": run.get("best_val_loss"),
        "best_recall_at_1": run.get("best_val_recall_at_1"),
        "message": run.get("message"),
        "metrics": run.get("metrics"),
    }


@router.delete("/checkpoints/{checkpoint_id}")
async def delete_checkpoint(
    checkpoint_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a checkpoint."""
    checkpoint = await db.get_training_checkpoint(checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Check if linked to a trained model
    linked_model = await db.get_trained_model_by_checkpoint(checkpoint_id)
    if linked_model:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete checkpoint linked to a registered model",
        )

    await db.delete_training_checkpoint(checkpoint_id)
    return {"deleted": True}


@router.post("/checkpoints/{checkpoint_id}/register")
async def register_checkpoint(
    checkpoint_id: str,
    request: RegisterModelRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Register a checkpoint as a trained model."""
    checkpoint = await db.get_training_checkpoint(checkpoint_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    model_data = {
        "training_run_id": checkpoint["training_run_id"],
        "checkpoint_id": checkpoint_id,
        "name": request.name,
        "description": request.description,
    }

    model = await db.create_trained_model(model_data)
    return model


# ===========================================
# Trained Models Endpoints
# ===========================================

@router.get("/models")
async def list_trained_models(
    db: SupabaseService = Depends(get_supabase),
):
    """List all trained models."""
    return await db.get_trained_models()


@router.get("/models/{model_id}")
async def get_trained_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get trained model details."""
    model = await db.get_trained_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id}")
async def delete_trained_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a trained model."""
    model = await db.get_trained_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.get("is_default"):
        raise HTTPException(status_code=400, detail="Cannot delete default models")

    if model.get("is_active"):
        raise HTTPException(status_code=400, detail="Cannot delete active model. Deactivate first.")

    await db.delete_trained_model(model_id)
    return {"deleted": True}


@router.post("/models/{model_id}/deploy")
async def deploy_trained_model(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Deploy a trained model for embedding extraction.

    Creates an embedding_model entry and sets it as active.
    """
    model = await db.get_trained_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.get("test_evaluated"):
        raise HTTPException(
            status_code=400,
            detail="Model must be evaluated on test set before deployment",
        )

    # Get checkpoint URL
    checkpoint = await db.get_training_checkpoint(model["checkpoint_id"])
    if not checkpoint:
        raise HTTPException(status_code=400, detail="Checkpoint not found")

    # Get training run for base model info
    run = await db.get_training_run(model["training_run_id"])

    # Create embedding model entry
    embedding_model_data = {
        "name": model["name"],
        "model_type": "custom",
        "model_family": run["base_model_type"].split("-")[0],  # dinov2, dinov3, clip
        "checkpoint_url": checkpoint["checkpoint_url"],
        "embedding_dim": _get_embedding_dim(run["base_model_type"]),
        "is_pretrained": False,
        "base_model_id": run["base_model_type"],
        "config": run.get("training_config", {}),
    }

    embedding_model = await db.create_embedding_model(embedding_model_data)

    # Update trained model with embedding_model_id
    await db.update_trained_model(model_id, {
        "embedding_model_id": embedding_model["id"],
        "is_active": True,
    })

    # Activate for matching
    await db.activate_embedding_model(embedding_model["id"])

    return {
        "deployed": True,
        "embedding_model_id": embedding_model["id"],
    }


# ===========================================
# Evaluation Endpoints
# ===========================================

@router.post("/runs/{run_id}/evaluate")
async def evaluate_run(
    run_id: str,
    request: EvaluateModelRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """
    Run evaluation on test set.

    Uses the held-out test set that was NEVER used during training.
    """
    run = await db.get_training_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training must be completed first")

    # Get checkpoint to evaluate
    if request.checkpoint_id:
        checkpoint = await db.get_training_checkpoint(request.checkpoint_id)
    else:
        # Use best checkpoint
        checkpoints = await db.get_training_checkpoints(run_id)
        checkpoint = next((c for c in checkpoints if c.get("is_best")), None)

    if not checkpoint:
        raise HTTPException(status_code=400, detail="No checkpoint found for evaluation")

    # Get test product IDs from the training run
    test_product_ids = run.get("test_product_ids", [])
    if not test_product_ids:
        raise HTTPException(status_code=400, detail="No test products found for evaluation")

    # Dispatch evaluation job to RunPod
    try:
        job = await runpod_service.start_evaluation_job(
            training_run_id=run_id,
            checkpoint_id=checkpoint["id"],
            checkpoint_url=checkpoint["checkpoint_url"],
            model_type=run["base_model_type"],
            test_product_ids=test_product_ids,
        )

        # Update checkpoint with evaluation status
        await db.update_training_checkpoint(checkpoint["id"], {
            "evaluation_status": "running",
            "evaluation_job_id": job.get("id"),
        })

        return {
            "status": "evaluation_started",
            "job_id": job.get("id"),
            "checkpoint_id": checkpoint["id"],
            "test_product_count": len(test_product_ids),
            "metrics_requested": request.metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")


@router.get("/models/{model_id}/evaluation")
async def get_model_evaluation(
    model_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Get evaluation results for a trained model."""
    model = await db.get_trained_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    evaluations = await db.get_model_evaluations(model_id)
    if not evaluations:
        raise HTTPException(status_code=404, detail="No evaluation results found")

    return evaluations[0]  # Return latest evaluation


@router.post("/models/compare")
async def compare_models(
    model_ids: list[str],
    db: SupabaseService = Depends(get_supabase),
):
    """Compare multiple trained models."""
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 models required for comparison")

    if len(model_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 models can be compared")

    comparison = []
    for model_id in model_ids:
        model = await db.get_trained_model(model_id)
        if not model:
            continue

        comparison.append({
            "id": model["id"],
            "name": model["name"],
            "test_metrics": model.get("test_metrics"),
            "cross_domain_metrics": model.get("cross_domain_metrics"),
        })

    return comparison


# ===========================================
# Products for Training Endpoint
# ===========================================

@router.get("/products")
async def get_products_for_training(
    data_source: Literal["all_products", "matched_products", "dataset"] = Query("all_products"),
    dataset_id: Optional[str] = Query(None),
    min_frames: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get products available for training.

    Returns a list of products with their frame counts for training preview.
    """
    # Get products based on data source
    if data_source == "all_products":
        products = await db.get_products_for_training(limit=limit)
    elif data_source == "matched_products":
        products = await db.get_matched_products_for_training(limit=limit)
    elif data_source == "dataset":
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required for 'dataset' data source")
        products = await db.get_products_for_training(dataset_id=dataset_id, limit=limit)
    else:
        products = await db.get_products_for_training(limit=limit)

    # Filter by minimum frames if specified
    if min_frames > 0:
        products = [p for p in products if p.get("frame_count", 0) >= min_frames]

    # Format response
    result = []
    for p in products:
        result.append({
            "id": p.get("id"),
            "barcode": p.get("barcode"),
            "short_code": p.get("short_code"),
            "upc": p.get("upc"),
            "brand_name": p.get("brand_name"),
            "frames_path": p.get("frames_path"),
            "frame_count": p.get("frame_count", 0),
        })

    return {
        "products": result,
        "total": len(result),
    }


# ===========================================
# Label Field Stats Endpoint
# ===========================================

@router.get("/label-stats")
async def get_label_field_stats(
    data_source: Literal["all_products", "matched_products", "dataset"] = Query("all_products"),
    dataset_id: Optional[str] = Query(None),
    db: SupabaseService = Depends(get_supabase),
):
    """
    Get statistics for available label fields.

    Returns class counts for each label field option to help users
    choose the right label configuration.
    """
    # Get products based on data source
    if data_source == "all_products":
        products = await db.get_products_for_training(limit=50000)
    elif data_source == "matched_products":
        products = await db.get_matched_products_for_training(limit=50000)
    elif data_source == "dataset":
        if not dataset_id:
            raise HTTPException(status_code=400, detail="dataset_id required")
        products = await db.get_products_for_training(dataset_id=dataset_id, limit=50000)
    else:
        products = await db.get_products_for_training(limit=50000)

    from collections import Counter

    # Calculate stats for each label field
    stats = {}

    # Product ID (default - each product is a class)
    stats["product_id"] = {
        "label": "Product ID",
        "description": "Each product becomes its own class",
        "total_products": len(products),
        "total_classes": len(products),
        "min_samples_per_class": 1,
        "max_samples_per_class": 1,
        "avg_samples_per_class": 1.0,
    }

    # Define all groupable fields with their labels
    groupable_fields = {
        "category": {"label": "Category", "description": "Train a category classifier"},
        "brand_name": {"label": "Brand", "description": "Train a brand classifier"},
        "container_type": {"label": "Container Type", "description": "Group by container (bottle, can, box, etc.)"},
        "sub_brand": {"label": "Sub-Brand", "description": "Group by sub-brand variants"},
        "manufacturer_country": {"label": "Country", "description": "Group by manufacturer country"},
        "variant_flavor": {"label": "Flavor/Variant", "description": "Group by flavor or variant"},
        "net_quantity": {"label": "Size/Quantity", "description": "Group by product size"},
    }

    # Calculate stats for each field
    for field_name, field_info in groupable_fields.items():
        field_counts = Counter(p.get(field_name) or "unknown" for p in products)
        # Filter out "unknown" if there are other values
        if len(field_counts) > 1 and "unknown" in field_counts:
            non_unknown_count = sum(v for k, v in field_counts.items() if k != "unknown")
            unknown_count = field_counts["unknown"]
        else:
            non_unknown_count = len(products)
            unknown_count = 0

        stats[field_name] = {
            "label": field_info["label"],
            "description": field_info["description"],
            "total_products": len(products),
            "total_classes": len([k for k in field_counts if k != "unknown"]),
            "min_samples_per_class": min(field_counts.values()) if field_counts else 0,
            "max_samples_per_class": max(field_counts.values()) if field_counts else 0,
            "avg_samples_per_class": len(products) / len(field_counts) if field_counts else 0,
            "top_classes": [(k, v) for k, v in field_counts.most_common(10) if k != "unknown"],
            "unknown_count": unknown_count,
            "coverage_percent": round((non_unknown_count / len(products)) * 100, 1) if products else 0,
        }

    # Also check for custom fields if available
    custom_field_keys = set()
    for p in products:
        if p.get("custom_fields"):
            custom_field_keys.update(p["custom_fields"].keys())

    for custom_key in custom_field_keys:
        field_counts = Counter(
            p.get("custom_fields", {}).get(custom_key) or "unknown"
            for p in products
        )
        stats[f"custom:{custom_key}"] = {
            "label": f"Custom: {custom_key}",
            "description": f"Custom field '{custom_key}'",
            "total_products": len(products),
            "total_classes": len([k for k in field_counts if k != "unknown"]),
            "min_samples_per_class": min(field_counts.values()) if field_counts else 0,
            "max_samples_per_class": max(field_counts.values()) if field_counts else 0,
            "avg_samples_per_class": len(products) / len(field_counts) if field_counts else 0,
            "top_classes": [(k, v) for k, v in field_counts.most_common(10) if k != "unknown"],
            "is_custom": True,
        }

    return {
        "data_source": data_source,
        "total_products": len(products),
        "label_fields": stats,
    }


# ===========================================
# Config Endpoints
# ===========================================

@router.get("/configs/presets")
async def list_presets():
    """Get all model-specific presets."""
    # Import from bb-models
    try:
        from bb_models.configs.presets import MODEL_PRESETS
        return MODEL_PRESETS
    except ImportError:
        return {}


@router.get("/configs/presets/{model_type}")
async def get_preset(model_type: str):
    """Get preset for a specific model."""
    try:
        from bb_models.configs.presets import get_preset
        preset = get_preset(model_type)
        if not preset:
            raise HTTPException(status_code=404, detail=f"No preset for model: {model_type}")
        return preset
    except ImportError:
        raise HTTPException(status_code=500, detail="bb_models package not available")


@router.get("/configs/saved")
async def list_saved_configs(
    db: SupabaseService = Depends(get_supabase),
):
    """List user-saved training configs."""
    return await db.get_training_configs()


@router.post("/configs")
async def save_config(
    request: SaveConfigRequest,
    db: SupabaseService = Depends(get_supabase),
):
    """Save a custom training config."""
    config_data = {
        "name": request.name,
        "description": request.description,
        "base_model_type": request.base_model_type,
        "config": request.config,
    }
    return await db.create_training_config(config_data)


@router.delete("/configs/{config_id}")
async def delete_config(
    config_id: str,
    db: SupabaseService = Depends(get_supabase),
):
    """Delete a saved config."""
    config = await db.get_training_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")

    if config.get("is_default"):
        raise HTTPException(status_code=400, detail="Cannot delete default configs")

    await db.delete_training_config(config_id)
    return {"deleted": True}


# ===========================================
# Helper functions
# ===========================================

async def _build_training_config(
    base_model_type: str,
    use_preset: bool,
    overrides: Optional[TrainingConfigOverrides],
    saved_config_id: Optional[str],
    db: SupabaseService,
) -> dict:
    """Build training config from preset + overrides."""
    # Start with preset
    if use_preset:
        try:
            from bb_models.configs.presets import get_preset
            config = get_preset(base_model_type) or {}
        except ImportError:
            config = {}
    else:
        config = {}

    # Or use saved config
    if saved_config_id:
        saved = await db.get_training_config(saved_config_id)
        if saved:
            config = saved.get("config", config)

    # Apply overrides
    if overrides:
        override_dict = overrides.model_dump(exclude_none=True)
        for key, value in override_dict.items():
            config[key] = value

    # Ensure required fields
    config.setdefault("epochs", 10)
    config.setdefault("batch_size", 32)
    config.setdefault("learning_rate", 1e-4)

    return config


def _get_embedding_dim(model_type: str) -> int:
    """Get embedding dimension for a model type."""
    dims = {
        "dinov2-small": 384,
        "dinov2-base": 768,
        "dinov2-large": 1024,
        "dinov3-small": 384,
        "dinov3-base": 768,
        "dinov3-large": 1024,
        "clip-vit-l-14": 1024,
    }
    return dims.get(model_type, 768)
