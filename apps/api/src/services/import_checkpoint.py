"""
Import Checkpoint Service

Manages persistent checkpoints for large import jobs (Roboflow, etc.)
to survive API restarts without losing progress.
"""

import os
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ImportCheckpoint:
    """Checkpoint state for resumable imports."""

    job_id: str
    stage: str  # "downloading", "parsing", "uploading", "db_insert", "completed"

    # Download state
    download_complete: bool = False
    zip_file_path: Optional[str] = None
    zip_file_size: int = 0
    zip_file_hash: Optional[str] = None

    # Parsing state
    total_images_in_zip: int = 0
    parsed_annotations: bool = False

    # Upload state (to storage)
    images_to_upload: int = 0
    uploaded_images: list[str] = field(default_factory=list)  # Original filenames
    uploaded_storage_map: dict[str, str] = field(
        default_factory=dict
    )  # orig -> storage name
    upload_failures: list[str] = field(default_factory=list)  # Failed uploads

    # DB state
    inserted_image_ids: list[str] = field(default_factory=list)
    dataset_mappings_inserted: bool = False
    annotations_inserted: bool = False

    # Metadata
    last_updated: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ImportCheckpoint":
        # Filter to only known fields
        known_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def can_resume(self) -> bool:
        """Check if this checkpoint can be resumed."""
        return self.download_complete and self.zip_file_path is not None


@dataclass
class StreamingCheckpoint:
    """Checkpoint for streaming imports (image-level granularity).

    Unlike ZIP imports, streaming imports process images one by one,
    so we track progress at the individual image level.
    """

    job_id: str
    total_images: int = 0

    # Progress tracking (per-image)
    processed_ids: set[str] = field(default_factory=set)  # Roboflow image IDs
    failed_ids: dict[str, str] = field(default_factory=dict)  # {rf_id: error}

    # Mapping data (needed for resume to avoid duplicates)
    storage_map: dict[str, str] = field(default_factory=dict)  # {rf_id: storage_filename}
    db_image_ids: dict[str, str] = field(default_factory=dict)  # {rf_id: db_image_id}

    # Metadata
    last_updated: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "job_id": self.job_id,
            "total_images": self.total_images,
            "processed_ids": list(self.processed_ids),  # set -> list for JSON
            "failed_ids": self.failed_ids,
            "storage_map": self.storage_map,
            "db_image_ids": self.db_image_ids,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StreamingCheckpoint":
        """Create from dict (e.g., loaded from DB)."""
        cp = cls(
            job_id=data.get("job_id", ""),
            total_images=data.get("total_images", 0),
            failed_ids=data.get("failed_ids", {}),
            storage_map=data.get("storage_map", {}),
            db_image_ids=data.get("db_image_ids", {}),
            last_updated=data.get("last_updated", ""),
        )
        cp.processed_ids = set(data.get("processed_ids", []))  # list -> set
        return cp

    def can_resume(self) -> bool:
        """Streaming can always resume as long as we have total_images."""
        return self.total_images > 0


class ImportCheckpointService:
    """Manages import checkpoints with persistent storage."""

    def __init__(self):
        self.base_dir = Path(settings.roboflow_import_dir)
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """Create base import directory if it doesn't exist."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create import directory: {e}")

    def get_job_dir(self, job_id: str) -> Path:
        """Get the directory for a specific job."""
        return self.base_dir / job_id

    def get_zip_path(self, job_id: str) -> Path:
        """Get the expected ZIP file path for a job."""
        return self.get_job_dir(job_id) / f"{job_id}.zip"

    def get_checkpoint_path(self, job_id: str) -> Path:
        """Get the checkpoint file path for a job."""
        return self.get_job_dir(job_id) / "checkpoint.json"

    def create_job_dir(self, job_id: str) -> Path:
        """Create directory for a job's import files."""
        job_dir = self.get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_checkpoint(self, checkpoint: ImportCheckpoint) -> None:
        """Save checkpoint to disk and update job record in database."""
        from services.supabase import supabase_service

        checkpoint.last_updated = datetime.utcnow().isoformat()

        # Save to file (for quick recovery)
        try:
            checkpoint_path = self.get_checkpoint_path(checkpoint.job_id)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint to file: {e}")

        # Update job record in database (authoritative state)
        try:
            supabase_service.client.table("jobs").update(
                {
                    "result": {
                        "stage": checkpoint.stage,
                        "checkpoint": checkpoint.to_dict(),
                        "message": f"Stage: {checkpoint.stage}",
                        "can_resume": checkpoint.can_resume(),
                    }
                }
            ).eq("id", checkpoint.job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update job checkpoint in DB: {e}")

    def load_checkpoint(self, job_id: str) -> Optional[ImportCheckpoint]:
        """Load checkpoint from disk or database."""
        from services.supabase import supabase_service

        # Try file first (faster)
        checkpoint_path = self.get_checkpoint_path(job_id)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                return ImportCheckpoint.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from file: {e}")

        # Fallback to database
        try:
            result = (
                supabase_service.client.table("jobs")
                .select("result")
                .eq("id", job_id)
                .single()
                .execute()
            )

            if result.data and result.data.get("result", {}).get("checkpoint"):
                return ImportCheckpoint.from_dict(result.data["result"]["checkpoint"])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from DB: {e}")

        return None

    def save_streaming_checkpoint(self, checkpoint: StreamingCheckpoint) -> None:
        """Save streaming checkpoint to DB (no file needed - no large download).

        IMPORTANT: This merges with existing result data to preserve progress stats
        that are updated by update_job() in the streaming import.
        """
        from services.supabase import supabase_service

        checkpoint.last_updated = datetime.utcnow().isoformat()
        print(f"[CHECKPOINT] Saving streaming checkpoint: {len(checkpoint.processed_ids)}/{checkpoint.total_images} images")

        try:
            # First, get existing result to preserve other fields (images_processed, etc.)
            existing_result = {}
            try:
                existing = supabase_service.client.table("jobs").select("result").eq("id", checkpoint.job_id).single().execute()
                if existing.data and existing.data.get("result"):
                    existing_result = existing.data["result"]
            except Exception:
                pass

            # Merge checkpoint data with existing result
            merged_result = {
                **existing_result,  # Preserve existing fields (images_processed, images_total, etc.)
                "stage": "streaming",
                "checkpoint": checkpoint.to_dict(),
                "can_resume": True,
                "processed_count": len(checkpoint.processed_ids),
                "failed_count": len(checkpoint.failed_ids),
            }

            supabase_service.client.table("jobs").update({
                "result": merged_result
            }).eq("id", checkpoint.job_id).execute()
        except Exception as e:
            logger.warning(f"Failed to save streaming checkpoint to DB: {e}")

    def load_streaming_checkpoint(self, job_id: str) -> Optional[StreamingCheckpoint]:
        """Load streaming checkpoint from DB."""
        from services.supabase import supabase_service

        try:
            result = (
                supabase_service.client.table("jobs")
                .select("result")
                .eq("id", job_id)
                .single()
                .execute()
            )

            if result.data and result.data.get("result", {}).get("checkpoint"):
                return StreamingCheckpoint.from_dict(result.data["result"]["checkpoint"])
        except Exception as e:
            logger.warning(f"Failed to load streaming checkpoint from DB: {e}")

        return None

    def cleanup_job(self, job_id: str) -> bool:
        """Clean up all files for a job. Returns True if successful."""
        job_dir = self.get_job_dir(job_id)
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up import directory for job {job_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to clean up job {job_id}: {e}")
                return False
        return True

    def cleanup_old_imports(self, max_age_hours: int = None) -> int:
        """Clean up import directories for old completed/failed jobs."""
        from services.supabase import supabase_service

        if max_age_hours is None:
            max_age_hours = settings.roboflow_import_max_age_hours

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned = 0

        try:
            # Get old completed/failed/cancelled jobs
            result = (
                supabase_service.client.table("jobs")
                .select("id")
                .eq("type", "roboflow_import")
                .in_("status", ["completed", "failed", "cancelled"])
                .lt("updated_at", cutoff.isoformat())
                .execute()
            )

            for job in result.data or []:
                if self.cleanup_job(job["id"]):
                    cleaned += 1

        except Exception as e:
            logger.warning(f"Failed to cleanup old imports: {e}")

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old import directories")

        return cleaned

    def get_interrupted_jobs(self) -> list[dict[str, Any]]:
        """Get jobs that were interrupted (running status but no active task)."""
        from services.supabase import supabase_service

        try:
            result = (
                supabase_service.client.table("jobs")
                .select("*")
                .eq("type", "roboflow_import")
                .eq("status", "running")
                .execute()
            )

            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get interrupted jobs: {e}")
            return []

    def verify_zip_integrity(
        self, job_id: str, expected_hash: str = None
    ) -> bool:
        """Verify ZIP file exists and optionally check hash."""
        zip_path = self.get_zip_path(job_id)

        if not zip_path.exists():
            return False

        # Check file is not empty
        if zip_path.stat().st_size == 0:
            return False

        if expected_hash:
            actual_hash = self._calculate_file_hash(zip_path)
            return actual_hash == expected_hash

        return True

    def _calculate_file_hash(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate hash of a file."""
        if algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha256":
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def get_disk_usage(self) -> dict[str, Any]:
        """Get disk usage statistics for import directory."""
        total_size = 0
        job_count = 0

        if not self.base_dir.exists():
            return {"total_bytes": 0, "total_mb": 0, "job_count": 0}

        for job_dir in self.base_dir.iterdir():
            if job_dir.is_dir():
                job_count += 1
                for file in job_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "total_bytes": total_size,
            "total_mb": round(total_size / (1024 * 1024), 2),
            "job_count": job_count,
        }


# Singleton instance
checkpoint_service = ImportCheckpointService()
