"""
Classification Module - Pydantic Schemas

Schemas for classification images, classes, datasets, labels, training, and models.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ===========================================
# Image Schemas
# ===========================================

class CLSImageBase(BaseModel):
    filename: str
    original_filename: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size_bytes: Optional[int] = None
    source: str = "upload"
    folder: Optional[str] = None
    tags: Optional[list[str]] = None


class CLSImageCreate(CLSImageBase):
    image_url: str
    storage_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    file_hash: Optional[str] = None
    phash: Optional[str] = None


class CLSImageUpdate(BaseModel):
    folder: Optional[str] = None
    tags: Optional[list[str]] = None
    status: Optional[str] = None


class CLSImageResponse(CLSImageBase):
    id: str
    image_url: str
    storage_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: str = "pending"
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    file_hash: Optional[str] = None
    phash: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CLSImagesResponse(BaseModel):
    images: list[CLSImageResponse]
    total: int
    page: int
    limit: int


# ===========================================
# Class Schemas
# ===========================================

class CLSClassBase(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    color: str = "#3B82F6"


class CLSClassCreate(CLSClassBase):
    dataset_id: str
    parent_class_id: Optional[str] = None


class CLSClassUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    is_active: Optional[bool] = None


class CLSClassResponse(CLSClassBase):
    id: str
    parent_class_id: Optional[str] = None
    image_count: int = 0
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CLSClassMergeRequest(BaseModel):
    source_class_ids: list[str]
    target_class_id: str


class CLSClassBulkCreate(BaseModel):
    classes: list[CLSClassCreate]


# ===========================================
# Dataset Schemas
# ===========================================

class CLSDatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    task_type: Literal["single_label", "multi_label"] = "single_label"


class CLSDatasetCreate(CLSDatasetBase):
    split_ratios: Optional[dict] = None  # {train: 0.8, val: 0.1, test: 0.1}
    preprocessing: Optional[dict] = None  # {image_size: 224, normalize: true}


class CLSDatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    split_ratios: Optional[dict] = None
    preprocessing: Optional[dict] = None


class CLSDatasetResponse(CLSDatasetBase):
    id: str
    image_count: int = 0
    labeled_image_count: int = 0
    class_count: int = 0
    split_ratios: dict = {"train": 0.8, "val": 0.1, "test": 0.1}
    preprocessing: dict = {"image_size": 224, "normalize": True}
    version: int = 1
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CLSDatasetImageResponse(BaseModel):
    id: str
    dataset_id: str
    image_id: str
    status: str = "pending"
    split: Optional[str] = None
    added_at: datetime
    labeled_at: Optional[datetime] = None
    image: CLSImageResponse
    labels: Optional[list["CLSLabelResponse"]] = None

    class Config:
        from_attributes = True


class CLSDatasetWithImagesResponse(CLSDatasetResponse):
    images: list[CLSDatasetImageResponse] = []


class CLSAddImagesRequest(BaseModel):
    image_ids: list[str]


class CLSRemoveImagesRequest(BaseModel):
    image_ids: list[str]


# ===========================================
# Label Schemas
# ===========================================

class CLSLabelBase(BaseModel):
    class_id: str
    confidence: Optional[float] = Field(None, ge=0, le=1)
    is_ai_generated: bool = False
    ai_model: Optional[str] = None


class CLSLabelCreate(CLSLabelBase):
    pass


class CLSLabelingSubmit(BaseModel):
    """Schema for labeling workflow submission."""
    action: str = Field(..., description="Action: label, skip, or review")
    class_id: Optional[str] = Field(None, description="Class ID for label action")
    class_ids: Optional[list[str]] = Field(None, description="Multiple class IDs for multi-label")
    confidence: Optional[float] = Field(None, ge=0, le=1)


class CLSLabelUpdate(BaseModel):
    class_id: Optional[str] = None
    is_reviewed: Optional[bool] = None


class CLSLabelResponse(CLSLabelBase):
    id: str
    dataset_id: str
    image_id: str
    is_reviewed: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    # Joined class info
    class_name: Optional[str] = None
    class_color: Optional[str] = None

    class Config:
        from_attributes = True


class CLSBulkLabelRequest(BaseModel):
    """Bulk set labels for multiple images."""
    image_ids: list[str]
    class_id: str


class CLSBulkClearLabelsRequest(BaseModel):
    """Clear labels for multiple images."""
    image_ids: list[str]


# ===========================================
# Dataset Version Schemas
# ===========================================

class CLSDatasetVersionCreate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CLSDatasetVersionResponse(BaseModel):
    id: str
    dataset_id: str
    version_number: int
    name: Optional[str] = None
    description: Optional[str] = None
    image_count: int
    labeled_image_count: int
    class_count: int
    class_mapping: dict
    class_names: list[str]
    split_counts: dict
    created_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# Training Schemas
# ===========================================

class CLSTrainingConfig(BaseModel):
    """Training configuration."""
    # Model
    model_type: Literal["vit", "convnext", "efficientnet", "swin", "dinov2", "clip"]
    model_size: str
    pretrained: bool = True
    freeze_backbone_epochs: int = 0

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.05

    # SOTA Features
    use_ema: bool = True
    ema_decay: float = 0.9999
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # LR Schedule
    lr_scheduler: Literal["cosine", "step", "plateau", "one_cycle"] = "cosine"
    warmup_epochs: int = 5
    llrd_decay: float = 0.9

    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    drop_path_rate: float = 0.1

    # Augmentation
    augmentation_preset: Literal["sota", "heavy", "medium", "light", "none"] = "sota"
    augmentation_overrides: Optional[dict] = None
    image_size: int = 224

    # Class Imbalance
    class_weights: Literal["balanced", "sqrt", "none"] = "balanced"
    focal_loss_gamma: float = 0.0

    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_f1"


class CLSTrainingRunCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: str
    dataset_version_id: Optional[str] = None
    config: CLSTrainingConfig


class CLSTrainingRunUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CLSTrainingRunResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: str
    dataset_version_id: Optional[str] = None
    task_type: str
    num_classes: int
    model_type: str
    model_size: str
    config: dict
    status: str = "pending"
    current_epoch: int = 0
    total_epochs: int
    best_accuracy: Optional[float] = None
    best_f1: Optional[float] = None
    best_top5_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    metrics_history: Optional[list] = None
    runpod_job_id: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# Model Schemas
# ===========================================

class CLSTrainedModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class CLSTrainedModelResponse(BaseModel):
    id: str
    training_run_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    model_type: str
    model_size: Optional[str] = None
    task_type: str
    checkpoint_url: str
    onnx_url: Optional[str] = None
    torchscript_url: Optional[str] = None
    num_classes: int
    class_names: list[str]
    class_mapping: dict
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    top5_accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    confusion_matrix: Optional[list] = None
    per_class_metrics: Optional[dict] = None
    is_active: bool = True
    is_default: bool = False
    file_size_bytes: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# Import Schemas
# ===========================================

class ImportURLsRequest(BaseModel):
    urls: list[str]
    folder: Optional[str] = None
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None


class ImportFromProductsRequest(BaseModel):
    """Import images from Products module."""
    product_ids: Optional[list[str]] = None  # Specific products

    # Label strategy
    label_source: Literal["category", "brand", "product_name", "manual"] = "category"

    # Image types
    image_types: list[Literal["synthetic", "real", "augmented"]] = ["synthetic", "real"]
    max_frames_per_product: int = 5

    # Options
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None

    # Filters (if product_ids not provided)
    status: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None


class ImportFromCutoutsRequest(BaseModel):
    """Import images from Cutouts module."""
    cutout_ids: Optional[list[str]] = None

    # Label strategy
    label_source: Literal["matched_product_category", "matched_product_brand", "manual"] = "matched_product_category"

    # Options
    only_matched: bool = True
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None


class ImportFromODRequest(BaseModel):
    """Import images from Object Detection module."""
    od_image_ids: Optional[list[str]] = None

    # Options
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None


class ImportLabeledDatasetRequest(BaseModel):
    """Import labeled dataset from ZIP (folder structure)."""
    dataset_id: str
    class_mapping: list["ClassMappingItem"]
    skip_duplicates: bool = True


class ClassMappingItem(BaseModel):
    source_name: str  # Folder name in ZIP
    target_class_id: Optional[str] = None
    create_new: bool = False
    color: Optional[str] = None
    skip: bool = False


class ImportPreviewResponse(BaseModel):
    format: str
    class_names: list[str]
    image_count: int
    annotation_count: Optional[int] = None
    suggested_mapping: list[dict]


class ImportResultResponse(BaseModel):
    success: bool
    images_imported: int
    images_skipped: int
    duplicates_found: int
    labels_created: int = 0
    classes_created: int = 0
    errors: list[str] = []


# ===========================================
# Bulk Operation Schemas
# ===========================================

class BulkOperationRequest(BaseModel):
    image_ids: list[str]


class BulkTagRequest(BaseModel):
    image_ids: list[str]
    action: Literal["add", "remove", "replace"]
    tags: list[str]


class BulkMoveRequest(BaseModel):
    image_ids: list[str]
    folder: str


class BulkAddToDatasetRequest(BaseModel):
    image_ids: list[str]
    dataset_id: str


class BulkOperationResponse(BaseModel):
    success: bool
    affected_count: int
    errors: list[str] = []


# ===========================================
# Duplicate Detection Schemas
# ===========================================

class DuplicateCheckRequest(BaseModel):
    file_hash: Optional[str] = None
    phash: Optional[str] = None


class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    exact_match: Optional[CLSImageResponse] = None
    similar_matches: list[CLSImageResponse] = []


class DuplicateGroupResponse(BaseModel):
    group_id: str
    images: list[CLSImageResponse]
    similarity: float


class DuplicateGroupsResponse(BaseModel):
    groups: list[DuplicateGroupResponse]
    total_duplicates: int


# ===========================================
# Labeling Queue Schemas
# ===========================================

class LabelingQueueRequest(BaseModel):
    mode: Literal["all", "unlabeled", "review", "random", "low_confidence"] = "unlabeled"
    split: Optional[Literal["train", "val", "test"]] = None
    class_id: Optional[str] = None
    limit: int = 100


class LabelingQueueResponse(BaseModel):
    image_ids: list[str]
    total: int
    mode: str


class LabelingImageResponse(BaseModel):
    image: CLSImageResponse
    current_labels: list[CLSLabelResponse]
    dataset_image_status: str
    position: int
    total: int
    prev_image_id: Optional[str] = None
    next_image_id: Optional[str] = None


class LabelingProgressResponse(BaseModel):
    total: int
    labeled: int
    pending: int
    review: int
    completed: int
    skipped: int
    progress_pct: float


# ===========================================
# Split Schemas
# ===========================================

class AutoSplitRequest(BaseModel):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratified: bool = True
    seed: Optional[int] = None


class ManualSplitRequest(BaseModel):
    train_image_ids: list[str]
    val_image_ids: list[str]
    test_image_ids: list[str]


class SplitStatsResponse(BaseModel):
    train_count: int
    val_count: int
    test_count: int
    unassigned_count: int
    class_distribution: dict  # {class_name: {train: n, val: n, test: n}}


# ===========================================
# Prediction Schemas
# ===========================================

class PredictionRequest(BaseModel):
    model_id: str


class PredictionResponse(BaseModel):
    predictions: list[dict]  # [{class_name: str, confidence: float}]
    top_class: str
    top_confidence: float


class BatchPredictionRequest(BaseModel):
    model_id: str
    image_ids: Optional[list[str]] = None
    image_urls: Optional[list[str]] = None


class BatchPredictionResponse(BaseModel):
    results: list[dict]  # [{image_id: str, predictions: [...]}]
    processed_count: int
    failed_count: int


# Forward references
CLSDatasetImageResponse.model_rebuild()
ImportLabeledDatasetRequest.model_rebuild()
