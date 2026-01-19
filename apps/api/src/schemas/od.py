"""
Object Detection Module - Pydantic Schemas

Schemas for OD images, classes, datasets, annotations, training, and models.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


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
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0001
    image_size: int = 640


class ODTrainingRunCreate(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_id: str
    dataset_version_id: Optional[str] = None
    model_type: str = "rf-detr"
    model_size: str = "medium"
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
