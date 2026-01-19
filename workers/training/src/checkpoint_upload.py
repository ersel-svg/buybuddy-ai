"""
Checkpoint upload and metrics history utilities.

Handles:
- Uploading checkpoints to Supabase Storage (with compression)
- Saving per-epoch metrics to training_metrics_history
- Updating training progress in training_runs
"""

import os
import io
import gzip
import time
from pathlib import Path
from typing import Optional, Any
from supabase import Client


# Maximum file size for Supabase Storage (50MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


def upload_checkpoint_to_storage(
    client: Client,
    checkpoint_path: str,
    training_run_id: str,
    epoch: int,
    is_best: bool = False,
) -> Optional[str]:
    """
    Upload a checkpoint file to Supabase Storage with compression.

    Args:
        client: Supabase client
        checkpoint_path: Local path to checkpoint file
        training_run_id: Training run ID
        epoch: Epoch number
        is_best: Whether this is the best checkpoint

    Returns:
        Public URL of uploaded checkpoint, or None if failed
    """
    try:
        # Read original file
        with open(checkpoint_path, "rb") as f:
            file_data = f.read()

        original_size = len(file_data)

        # Compress with gzip if file is larger than threshold
        if original_size > MAX_UPLOAD_SIZE:
            print(f"Compressing checkpoint ({original_size / 1024 / 1024:.1f}MB)...")
            compressed_buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=compressed_buffer, mode='wb', compresslevel=6) as gz:
                gz.write(file_data)
            file_data = compressed_buffer.getvalue()
            compressed_size = len(file_data)
            print(f"Compressed: {original_size / 1024 / 1024:.1f}MB -> {compressed_size / 1024 / 1024:.1f}MB ({100 * compressed_size / original_size:.0f}%)")

            # Check if still too large
            if compressed_size > MAX_UPLOAD_SIZE:
                print(f"Warning: Compressed checkpoint ({compressed_size / 1024 / 1024:.1f}MB) still exceeds 50MB limit")
                print("Checkpoint will be saved locally only, not uploaded to storage")
                return None

            # Use .pth.gz extension for compressed files
            suffix = "best" if is_best else f"epoch_{epoch:03d}"
            storage_path = f"training/{training_run_id}/checkpoint_{suffix}.pth.gz"
            content_type = "application/gzip"
        else:
            suffix = "best" if is_best else f"epoch_{epoch:03d}"
            storage_path = f"training/{training_run_id}/checkpoint_{suffix}.pth"
            content_type = "application/octet-stream"

        # Upload to storage bucket "checkpoints"
        bucket = client.storage.from_("checkpoints")

        # Try to remove existing file (ignore errors)
        try:
            bucket.remove([storage_path])
        except:
            pass

        # Upload
        result = bucket.upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": content_type}
        )

        # Get public URL
        public_url = bucket.get_public_url(storage_path)

        print(f"Uploaded checkpoint to: {public_url}")
        return public_url

    except Exception as e:
        print(f"Failed to upload checkpoint: {e}")
        return None


def save_metrics_history(
    client: Client,
    training_run_id: str,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    learning_rate: float,
    epoch_duration: float,
    curriculum_phase: Optional[str] = None,
) -> bool:
    """
    Save epoch metrics to training_metrics_history table.

    Args:
        client: Supabase client
        training_run_id: Training run ID
        epoch: Epoch number (0-indexed)
        train_metrics: Training metrics dict
        val_metrics: Validation metrics dict
        learning_rate: Current learning rate
        epoch_duration: Epoch duration in seconds
        curriculum_phase: Current curriculum phase if applicable

    Returns:
        True if successful, False otherwise
    """
    try:
        data = {
            "training_run_id": training_run_id,
            "epoch": epoch + 1,  # 1-indexed for display
            "train_loss": train_metrics.get("loss"),
            "arcface_loss": train_metrics.get("arcface_loss"),
            "triplet_loss": train_metrics.get("triplet_loss"),
            "domain_loss": train_metrics.get("domain_loss"),
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_recall_at_1": val_metrics.get("recall@1"),
            "val_recall_at_5": val_metrics.get("recall@5"),
            "val_recall_at_10": val_metrics.get("recall@10"),
            "learning_rate": learning_rate,
            "epoch_duration_seconds": epoch_duration,
            "curriculum_phase": curriculum_phase,
        }

        # Upsert (update if exists, insert if not)
        client.table("training_metrics_history").upsert(
            data,
            on_conflict="training_run_id,epoch"
        ).execute()

        return True

    except Exception as e:
        print(f"Failed to save metrics history: {e}")
        return False


def update_training_progress(
    client: Client,
    training_run_id: str,
    epoch: int,
    total_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
    is_best: bool = False,
    checkpoint_url: Optional[str] = None,
    message: Optional[str] = None,
) -> bool:
    """
    Update training progress in training_runs table.

    Args:
        client: Supabase client
        training_run_id: Training run ID
        epoch: Current epoch (0-indexed)
        total_epochs: Total epochs
        train_metrics: Training metrics
        val_metrics: Validation metrics
        is_best: Whether this epoch had best metrics
        checkpoint_url: URL of uploaded checkpoint
        message: Status message

    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate progress
        progress = (epoch + 1) / total_epochs

        # Build update payload
        payload: dict[str, Any] = {
            "current_epoch": epoch + 1,  # 1-indexed
            "progress": progress,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {
                "train_loss": train_metrics.get("loss"),
                "val_loss": val_metrics.get("loss"),
                "val_accuracy": val_metrics.get("accuracy"),
                "val_recall_at_1": val_metrics.get("recall@1"),
                "val_recall_at_5": val_metrics.get("recall@5"),
            },
        }

        if message:
            payload["message"] = message

        if is_best:
            payload["best_val_loss"] = val_metrics.get("loss")
            payload["best_val_recall_at_1"] = val_metrics.get("recall@1")
            payload["best_val_recall_at_5"] = val_metrics.get("recall@5")
            payload["best_epoch"] = epoch + 1

        if checkpoint_url:
            payload["checkpoint_url"] = checkpoint_url

        client.table("training_runs").update(payload).eq("id", training_run_id).execute()

        return True

    except Exception as e:
        print(f"Failed to update training progress: {e}")
        return False


def save_checkpoint_record(
    client: Client,
    training_run_id: str,
    epoch: int,
    checkpoint_url: str,
    train_loss: float,
    val_metrics: dict,
    is_best: bool = False,
    is_final: bool = False,
    file_size_bytes: Optional[int] = None,
) -> Optional[str]:
    """
    Save checkpoint record to training_checkpoints table.

    Args:
        client: Supabase client
        training_run_id: Training run ID
        epoch: Epoch number (0-indexed)
        checkpoint_url: URL of checkpoint in storage
        train_loss: Training loss
        val_metrics: Validation metrics
        is_best: Whether this is the best checkpoint
        is_final: Whether this is the final checkpoint
        file_size_bytes: Size of checkpoint file

    Returns:
        Checkpoint ID if successful, None otherwise
    """
    try:
        data = {
            "training_run_id": training_run_id,
            "epoch": epoch + 1,  # 1-indexed
            "checkpoint_url": checkpoint_url,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            "val_recall_at_1": val_metrics.get("recall@1"),
            "val_recall_at_5": val_metrics.get("recall@5"),
            "val_recall_at_10": val_metrics.get("recall@10"),
            "is_best": is_best,
            "is_final": is_final,
        }

        if file_size_bytes:
            data["file_size_bytes"] = file_size_bytes

        result = client.table("training_checkpoints").insert(data).execute()

        if result.data:
            return result.data[0]["id"]
        return None

    except Exception as e:
        print(f"Failed to save checkpoint record: {e}")
        return None
