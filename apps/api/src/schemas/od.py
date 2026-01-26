"""
Object Detection Module - Pydantic Schemas

Schemas for OD images, classes, datasets, annotations, training, and models.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from schemas.data_loading import DataLoadingConfig


# ===========================================
# Image Schemas
# ===========================================

class ODImageBase(BaseModel):
    filename: str
    original_filename: Optional[str] = None
    width: int
    height: int
    file_size_bytes: Optional[int] = None
    source: str = "upload"
    folder: Optional[str] = None
    tags: Optional[list[str]] = None


class ODImageCreate(ODImageBase):
    image_url: str
    thumbnail_url: Optional[str] = None
    buybuddy_image_id: Optional[str] = None
    buybuddy_evaluation_id: Optional[str] = None


class ODImageUpdate(BaseModel):
    folder: Optional[str] = None
    tags: Optional[list[str]] = None
    status: Optional[str] = None


class ODImageResponse(ODImageBase):
    id: str
    image_url: str
    thumbnail_url: Optional[str] = None
    status: str
    buybuddy_image_id: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ODImagesResponse(BaseModel):
    images: list[ODImageResponse]
    total: int
    page: int
    limit: int


# ===========================================
# Class Schemas
# ===========================================

class ODClassBase(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    color: str = "#3B82F6"
    category: Optional[str] = None


class ODClassCreate(ODClassBase):
    dataset_id: Optional[str] = None  # Required for dataset-specific classes, None for templates
    aliases: Optional[list[str]] = None


class ODClassUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None


class ODClassResponse(ODClassBase):
    id: str
    dataset_id: Optional[str] = None  # None means template class
    aliases: Optional[list[str]] = None
    annotation_count: int = 0
    is_active: bool = True
    is_system: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ODClassMergeRequest(BaseModel):
    source_class_ids: list[str]
    target_class_id: str


# ===========================================
# Dataset Schemas
# ===========================================

class ODDatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    annotation_type: str = "bbox"


class ODDatasetCreate(ODDatasetBase):
    pass


class ODDatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ODDatasetResponse(ODDatasetBase):
    id: str
    image_count: int = 0
    annotated_image_count: int = 0
    annotation_count: int = 0
    class_count: int = 0
    version: int = 1
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ODDatasetImageResponse(BaseModel):
    id: str
    dataset_id: str
    image_id: str
    status: str
    annotation_count: int = 0
    split: Optional[str] = None
    added_at: datetime
    last_annotated_at: Optional[datetime] = None  # When this image was last annotated
    image: ODImageResponse

    class Config:
        from_attributes = True


class ODDatasetWithImagesResponse(ODDatasetResponse):
    images: list[ODDatasetImageResponse] = []


class ODAddImagesRequest(BaseModel):
    image_ids: list[str]


# ===========================================
# Annotation Schemas
# ===========================================

class BBox(BaseModel):
    x: float = Field(..., ge=0, le=1, description="Top-left X (0-1)")
    y: float = Field(..., ge=0, le=1, description="Top-left Y (0-1)")
    width: float = Field(..., gt=0, le=1, description="Width (0-1)")
    height: float = Field(..., gt=0, le=1, description="Height (0-1)")


class Point(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)


class ODAnnotationBase(BaseModel):
    class_id: str
    bbox: BBox
    polygon: Optional[list[Point]] = None
    is_ai_generated: bool = False
    confidence: Optional[float] = Field(None, ge=0, le=1)
    ai_model: Optional[str] = None
    attributes: Optional[dict] = None


class ODAnnotationCreate(ODAnnotationBase):
    pass


class ODAnnotationUpdate(BaseModel):
    class_id: Optional[str] = None
    bbox: Optional[BBox] = None
    polygon: Optional[list[Point]] = None
    is_reviewed: Optional[bool] = None
    attributes: Optional[dict] = None


class ODAnnotationResponse(BaseModel):
    id: str
    dataset_id: str
    image_id: str
    class_id: str
    class_name: Optional[str] = None
    class_color: Optional[str] = None
    bbox: BBox
    polygon: Optional[list[Point]] = None
    is_ai_generated: bool = False
    confidence: Optional[float] = None
    ai_model: Optional[str] = None
    is_reviewed: bool = False
    attributes: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ODBulkAnnotationsCreate(BaseModel):
    annotations: list[ODAnnotationCreate]


class ODBulkAnnotationsResponse(BaseModel):
    created: int
    annotation_ids: list[str]


# ===========================================
# Stats Schemas
# ===========================================

class ODStatsResponse(BaseModel):
    total_images: int
    total_datasets: int
    total_annotations: int
    total_classes: int
    total_models: int
    images_by_status: dict[str, int]
    recent_datasets: Optional[list[ODDatasetResponse]] = None


# ===========================================
# Training Schemas (Phase 8)
# ===========================================

class ODTrainingConfigBase(BaseModel):
    """Training configuration with SOTA features for OD models."""

    model_config = {"extra": "allow"}  # Allow extra fields from wizard

    # Basic parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0001
    image_size: int = 640
    accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")

    # Optimizer
    optimizer: str = Field(default="adamw", description="Optimizer: adamw, sgd, adam")

    # SOTA: Augmentation preset (IMPORTANT for user)
    augmentation_preset: str = Field(
        default="sota",
        description="Augmentation preset: sota-v2, sota, heavy, medium, light, none"
    )
    augmentation_config: Optional[dict] = Field(default=None, description="Custom augmentation config")

    # SOTA: EMA (Exponential Moving Average)
    use_ema: bool = Field(default=True, description="Enable EMA for stable training")
    ema_decay: float = Field(default=0.9999, description="EMA decay rate")
    ema_warmup_steps: int = Field(default=2000, description="EMA warmup steps")

    # SOTA: LLRD (Layer-wise Learning Rate Decay)
    llrd_decay: float = Field(
        default=0.9,
        description="Layer-wise LR decay (1.0 = no decay, 0.9 = recommended)"
    )
    head_lr_factor: float = Field(
        default=10.0,
        description="Detection head LR multiplier"
    )

    # SOTA: Scheduler
    scheduler: str = Field(default="cosine", description="Scheduler: cosine, step, linear")
    warmup_epochs: int = Field(default=3, description="Linear warmup epochs")
    min_lr_ratio: float = Field(default=0.01, description="Min LR ratio for scheduler")

    # SOTA: Mixed Precision
    mixed_precision: bool = Field(default=True, description="Enable FP16 training")

    # SOTA: Regularization
    weight_decay: float = Field(default=0.0001, description="Weight decay")
    gradient_clip: float = Field(default=1.0, description="Gradient clipping max norm")
    label_smoothing: float = Field(default=0.0, description="Label smoothing factor")

    # SOTA: Multi-scale training
    multi_scale: bool = Field(default=False, description="Enable multi-scale training")
    multi_scale_range: Optional[list[float]] = Field(default=None, description="Multi-scale range [min, max]")

    # Model options
    pretrained: bool = Field(default=True, description="Use pretrained weights")
    freeze_backbone: bool = Field(default=False, description="Freeze backbone during training")
    freeze_epochs: int = Field(default=0, description="Epochs to keep backbone frozen")

    # Early stopping
    patience: int = Field(default=20, description="Early stopping patience (epochs)")

    # Checkpointing
    save_freq: int = Field(default=5, description="Save checkpoint every N epochs")

    # Preprocessing (from wizard)
    resize_strategy: Optional[str] = Field(default=None, description="Resize strategy: letterbox, stretch, crop")
    tiling: Optional[dict] = Field(default=None, description="Tiling configuration")

    # Offline augmentation (from wizard)
    offline_augmentation: Optional[dict] = Field(default=None, description="Offline augmentation config")

    # Data split (from wizard)
    train_split: float = Field(default=0.8, description="Training split ratio")
    val_split: float = Field(default=0.15, description="Validation split ratio")
    test_split: float = Field(default=0.05, description="Test split ratio")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Data loading configuration
    data_loading: Optional[DataLoadingConfig] = Field(
        default=None,
        description="Image preloading and dataloader configuration"
    )


class ODTrainingRunCreate(BaseModel):
    model_config = {"extra": "allow"}  # Allow extra fields from wizard

    name: str
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    dataset_id: str
    dataset_version_id: Optional[str] = None
    model_type: str = Field(
        default="rt-detr",
        description="Model type: rt-detr (Apache 2.0), d-fine (Apache 2.0)"
    )
    model_size: str = Field(
        default="l",
        description="Model size: s, m, l (and x for d-fine)"
    )
    config: Optional[ODTrainingConfigBase] = None


class ODTrainingRunResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: str
    model_type: str
    model_size: str
    status: str
    current_epoch: int = 0
    total_epochs: int
    best_map: Optional[float] = None
    best_epoch: Optional[int] = None
    config: Optional[dict] = None  # Training configuration (JSONB)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# Model Schemas (Phase 9)
# ===========================================

class ODTrainedModelResponse(BaseModel):
    id: str
    training_run_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    model_type: str
    checkpoint_url: str
    map: Optional[float] = None
    map_50: Optional[float] = None
    class_count: int
    is_active: bool = False
    is_default: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# Import Schemas
# ===========================================

class ImportURLsRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=100)
    folder: Optional[str] = None
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None  # Optional: add to dataset directly


class ImportPreviewResponse(BaseModel):
    format_detected: str
    total_images: int
    total_annotations: int
    classes_found: list[str]
    sample_images: list[dict]
    errors: list[str] = []


class ClassMappingItem(BaseModel):
    source_name: str
    target_class_id: Optional[str] = None
    create_new: bool = False
    skip: bool = False
    color: Optional[str] = None


class ImportAnnotatedRequest(BaseModel):
    dataset_id: str
    class_mapping: list[ClassMappingItem]
    skip_duplicates: bool = True
    merge_annotations: bool = False


class ImportResultResponse(BaseModel):
    success: bool
    images_imported: int = 0
    annotations_imported: int = 0
    images_skipped: int = 0
    duplicates_found: int = 0
    errors: list[str] = []


class DuplicateCheckRequest(BaseModel):
    file_hash: Optional[str] = None
    phash: Optional[str] = None


class DuplicateImageInfo(BaseModel):
    id: str
    filename: str
    image_url: str
    similarity: float


class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    similar_images: list[DuplicateImageInfo] = []


class DuplicateGroup(BaseModel):
    images: list[dict]
    max_similarity: float


class DuplicateGroupsResponse(BaseModel):
    groups: list[DuplicateGroup]
    total_groups: int


class BulkOperationRequest(BaseModel):
    image_ids: list[str]


class BulkTagRequest(BaseModel):
    image_ids: list[str]
    tags: list[str]
    operation: str = "add"  # "add", "remove", "replace"


class BulkMoveRequest(BaseModel):
    image_ids: list[str]
    folder: Optional[str] = None


class BulkOperationResponse(BaseModel):
    success: bool
    affected_count: int
    errors: list[str] = []


# ===========================================
# AI Annotation Schemas (Phase 6)
# ===========================================

class AIModelType(str):
    """Supported AI models for annotation."""
    GROUNDING_DINO = "grounding_dino"
    SAM3 = "sam3"
    SAM2 = "sam2"
    FLORENCE2 = "florence2"


class AIPredictRequest(BaseModel):
    """Request for single image AI prediction."""
    image_id: str
    model: str = Field(
        default="grounding_dino",
        description="AI model to use: grounding_dino, sam3, florence2, or rf:{model_id} for Roboflow models"
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description="Text prompt for detection (e.g., 'shelf . product . price tag'). Required for open-vocab models, ignored for Roboflow models."
    )
    box_threshold: float = Field(default=0.3, ge=0, le=1)
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    use_nms: bool = Field(
        default=False,
        description="Apply Non-Maximum Suppression to filter overlapping boxes"
    )
    nms_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="NMS IoU threshold (lower = more aggressive filtering)"
    )
    filter_classes: Optional[list[str]] = Field(
        default=None,
        description="Filter predictions to only include these classes. None = return all classes. Example: ['void', 'object']"
    )


class AISegmentRequest(BaseModel):
    """Request for interactive SAM segmentation."""
    image_id: str
    model: str = Field(
        default="sam2",
        description="Segmentation model: sam2 or sam3"
    )
    prompt_type: str = Field(
        ...,
        description="Prompt type: 'point' or 'box'"
    )
    point: Optional[tuple[float, float]] = Field(
        None,
        description="Point coordinates (x, y) normalized 0-1"
    )
    box: Optional[list[float]] = Field(
        None,
        description="Box coordinates [x, y, width, height] normalized 0-1"
    )
    label: int = Field(
        default=1,
        description="Point label: 1=foreground, 0=background"
    )
    text_prompt: Optional[str] = Field(
        None,
        description="Optional text prompt for SAM3"
    )


class AIBatchAnnotateRequest(BaseModel):
    """Request for bulk AI annotation job."""
    dataset_id: str
    image_ids: Optional[list[str]] = Field(
        None,
        description="Specific images to annotate. None = all unannotated"
    )
    model: str = Field(
        default="grounding_dino",
        description="AI model to use: grounding_dino, sam3, florence2, or rf:{model_id} for Roboflow models"
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description="Detection prompt. Required for open-vocab models (grounding_dino, sam3, florence2), ignored for Roboflow models."
    )
    box_threshold: float = Field(default=0.3, ge=0, le=1)
    text_threshold: float = Field(default=0.25, ge=0, le=1)
    auto_accept: bool = Field(
        default=False,
        description="If true, save as confirmed annotations"
    )
    class_mapping: Optional[dict[str, str]] = Field(
        None,
        description="Map detected labels to class IDs: {'detected_label': 'class_id'} or {'detected_label': '__new__:classname'} to create new class"
    )
    filter_classes: Optional[list[str]] = Field(
        default=None,
        description="Filter predictions to only include these classes. Useful for Roboflow models with fixed class sets. None = return all classes."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Limit the number of images to process. None = process all matching images. Useful for testing or cost control.",
        ge=1,
        le=10000
    )


class AIPrediction(BaseModel):
    """Single AI prediction result."""
    bbox: BBox
    label: str
    confidence: float = Field(ge=0, le=1)
    mask: Optional[str] = Field(None, description="Base64 encoded mask PNG")


class AIPredictResponse(BaseModel):
    """Response for single image prediction."""
    predictions: list[AIPrediction]
    model: str
    processing_time_ms: Optional[int] = None
    nms_applied: Optional[bool] = None


class AISegmentResponse(BaseModel):
    """Response for interactive segmentation."""
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    mask: Optional[str] = Field(None, description="Base64 encoded mask PNG")
    processing_time_ms: Optional[int] = None


class AIBatchJobResponse(BaseModel):
    """Response for batch annotation job creation."""
    job_id: str
    status: str
    total_images: int
    message: str


class AIJobStatusResponse(BaseModel):
    """Response for batch job status."""
    job_id: str
    status: str
    progress: int = 0
    total_images: int = 0
    predictions_generated: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AIWebhookPayload(BaseModel):
    """Webhook payload from RunPod."""
    id: str  # RunPod job ID
    status: str
    output: Optional[dict] = None
    error: Optional[str] = None


# ===========================================
# Export Schemas (Phase 7)
# ===========================================

class ExportConfig(BaseModel):
    """Configuration for dataset export."""
    train_split: float = Field(default=0.8, ge=0, le=1)
    val_split: float = Field(default=0.15, ge=0, le=1)
    test_split: float = Field(default=0.05, ge=0, le=1)
    image_size: Optional[int] = Field(None, description="Resize images to this size")
    include_unannotated: bool = False


class ExportRequest(BaseModel):
    """Request to export a dataset."""
    format: str = Field(..., description="Export format: yolo, coco")
    include_images: bool = Field(default=True, description="Include image files in export")
    version_id: Optional[str] = Field(None, description="Export specific version")
    split: Optional[str] = Field(None, description="Export specific split only")
    config: Optional[ExportConfig] = None


class ExportJobResponse(BaseModel):
    """Response for export job creation."""
    job_id: str
    status: str
    download_url: Optional[str] = None
    progress: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ===========================================
# Dataset Version Schemas (Phase 7)
# ===========================================

class DatasetVersionCreate(BaseModel):
    """Request to create a dataset version."""
    name: Optional[str] = None
    description: Optional[str] = None
    train_split: float = Field(default=0.8, ge=0, le=1)
    val_split: float = Field(default=0.15, ge=0, le=1)
    test_split: float = Field(default=0.05, ge=0, le=1)


class DatasetVersionResponse(BaseModel):
    """Response for dataset version."""
    id: str
    dataset_id: str
    version_number: int
    name: Optional[str] = None
    description: Optional[str] = None
    image_count: int = 0
    annotation_count: int = 0
    class_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    class_mapping: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True
