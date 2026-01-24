"""
Data Loading Configuration Schemas

Shared configuration schemas for image preloading and DataLoader settings
across all training types (OD, CLS, Embedding).
"""

from typing import Optional
from pydantic import BaseModel, Field


class PreloadConfig(BaseModel):
    """Image preloading configuration for training datasets."""

    enabled: bool = Field(
        default=True,
        description="Enable image preloading before training"
    )
    batched: bool = Field(
        default=False,
        description="Use batched preloading with gc.collect() for memory efficiency"
    )
    batch_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Batch size for batched preloading"
    )
    max_workers: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Maximum parallel download workers"
    )
    http_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="HTTP timeout in seconds for image downloads"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed downloads"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay in seconds between retry attempts"
    )


class DataLoaderConfig(BaseModel):
    """PyTorch DataLoader configuration."""

    num_workers: int = Field(
        default=4,
        ge=0,
        le=16,
        description="Number of DataLoader workers (0 = main process)"
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster GPU transfer"
    )
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of batches to prefetch per worker"
    )


class DataLoadingConfig(BaseModel):
    """Complete data loading configuration combining preload and dataloader settings."""

    preload: Optional[PreloadConfig] = Field(
        default=None,
        description="Image preloading configuration"
    )
    dataloader: Optional[DataLoaderConfig] = Field(
        default=None,
        description="PyTorch DataLoader configuration"
    )
