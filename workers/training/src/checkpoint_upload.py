"""
Checkpoint upload and metrics history utilities.

Handles:
- Uploading checkpoints to Supabase Storage (model weights only, compressed)
- Saving per-epoch metrics to training_metrics_history
- Updating training progress in training_runs
"""

import os
import io
import gzip
import time
import tempfile
import httpx
from pathlib import Path
from typing import Optional, Any
from supabase import Client

# Lazy import torch to avoid import errors when not needed
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# Maximum file size for Supabase Storage
# Supabase Pro with custom limit: 250MB
# DINOv2-base in FP16 is ~167MB
MAX_UPLOAD_SIZE = 250 * 1024 * 1024  # 250MB limit (configured in Supabase dashboard)

# Upload timeout in seconds
# RunPod -> Supabase can be slow (~150 KB/s = 18 min for 168MB)
# Set to 30 minutes to be safe
UPLOAD_TIMEOUT = 1800


def upload_checkpoint_to_storage(
    client: Client,
    checkpoint_path: str,
    training_run_id: str,
    epoch: int,
    is_best: bool = False,
) -> Optional[str]:
    """
    Upload a checkpoint file to Supabase Storage.

    Extracts only model weights (no optimizer state) to reduce size,
    then compresses with gzip if needed.

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
        torch = _get_torch()

        # Load full checkpoint
        print(f"Loading checkpoint for upload...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract only model weights (skip optimizer, scheduler, etc.)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Standard checkpoint format - extract only model weights
            # Convert to float16 to reduce size (safe for inference)
            model_state = checkpoint["model_state_dict"]
            model_state_fp16 = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in model_state.items()
            }
            lightweight_checkpoint = {
                "model_state_dict": model_state_fp16,
                "epoch": checkpoint.get("epoch", epoch),
                "val_loss": checkpoint.get("val_loss"),
                "val_recall_at_1": checkpoint.get("val_recall_at_1"),
                "dtype": "float16",  # Mark as fp16 for loading
            }
            print(f"Extracted model weights (fp16, removed optimizer state)")
        else:
            # Already just weights or unknown format - use as is
            lightweight_checkpoint = checkpoint

        # Save lightweight checkpoint to temp file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(lightweight_checkpoint, tmp.name)
            tmp_path = tmp.name

        # Get file size without loading into RAM
        file_size = os.path.getsize(tmp_path)
        print(f"Lightweight checkpoint size: {file_size / 1024 / 1024:.1f}MB")

        suffix = "best" if is_best else f"epoch_{epoch:03d}"

        # Check if file exceeds limit
        if file_size > MAX_UPLOAD_SIZE:
            limit_mb = MAX_UPLOAD_SIZE / 1024 / 1024
            print(f"Warning: Checkpoint ({file_size / 1024 / 1024:.1f}MB) exceeds {limit_mb:.0f}MB limit")
            print("Checkpoint saved locally only, not uploaded to storage")
            os.unlink(tmp_path)  # Clean up temp file
            return None

        storage_path = f"training/{training_run_id}/checkpoint_{suffix}.pth"
        content_type = "application/octet-stream"
        bucket_name = "checkpoints"

        # Get Supabase URL and key from client
        supabase_url = client.supabase_url.rstrip('/')
        supabase_key = client.supabase_key

        print(f"[DEBUG] Storage path: {storage_path}")
        print(f"[DEBUG] Supabase URL: {supabase_url}")
        print(f"[DEBUG] File size: {file_size / 1024 / 1024:.1f}MB")

        # Build upload URL - Supabase Storage REST API
        upload_url = f"{supabase_url}/storage/v1/object/{bucket_name}/{storage_path}"
        print(f"[DEBUG] Upload URL: {upload_url}")

        headers = {
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": content_type,
            "x-upsert": "true",  # Overwrite if exists
        }

        # Upload with retries and timeout (streaming - memory efficient)
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                print(f"[UPLOAD] Attempt {attempt + 1}/{max_retries} - Starting streaming upload...")
                print(f"[UPLOAD] Timeout set to {UPLOAD_TIMEOUT}s")

                start_time = time.time()

                # Stream upload - file is read in chunks, not loaded entirely into RAM
                with open(tmp_path, "rb") as f:
                    with httpx.Client(timeout=httpx.Timeout(UPLOAD_TIMEOUT, connect=30.0)) as http_client:
                        response = http_client.post(
                            upload_url,
                            content=f,  # Stream file directly
                            headers=headers,
                        )

                elapsed = time.time() - start_time
                print(f"[UPLOAD] Request completed in {elapsed:.1f}s")
                print(f"[UPLOAD] Response status: {response.status_code}")

                if response.status_code in (200, 201):
                    # Success - build public URL
                    public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
                    print(f"[UPLOAD] SUCCESS! Checkpoint uploaded to: {public_url}")
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    return public_url
                else:
                    # Server returned an error
                    error_body = response.text[:500]
                    print(f"[UPLOAD] Server error: {response.status_code}")
                    print(f"[UPLOAD] Response body: {error_body}")
                    last_error = f"HTTP {response.status_code}: {error_body}"

            except httpx.TimeoutException as e:
                elapsed = time.time() - start_time
                print(f"[UPLOAD] TIMEOUT after {elapsed:.1f}s: {e}")
                last_error = f"Timeout after {elapsed:.1f}s"

            except httpx.RequestError as e:
                print(f"[UPLOAD] Network error: {type(e).__name__}: {e}")
                last_error = f"Network error: {e}"

            except Exception as e:
                print(f"[UPLOAD] Unexpected error: {type(e).__name__}: {e}")
                last_error = f"Unexpected: {e}"

            # Retry logic
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5s, 10s, 15s
                print(f"[UPLOAD] Retrying in {wait_time}s...")
                time.sleep(wait_time)

        print(f"[UPLOAD] FAILED - All {max_retries} attempts failed. Last error: {last_error}")
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None

    except Exception as e:
        print(f"Failed to upload checkpoint: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file if it exists
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
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
            "training_type": "embedding",  # Unified table requires training_type
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

        # Upsert with new unique constraint (training_run_id, training_type, epoch)
        client.table("training_metrics_history").upsert(
            data,
            on_conflict="training_run_id,training_type,epoch"
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
