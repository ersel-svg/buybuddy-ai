"""
RunPod Handler for Model Training.

Handles training job requests with:
- Multi-model support via bb-models
- Multi-image-type support (synthetic, real, augmented)
- UPC-based data splitting
- Checkpoint management
- Progress reporting
"""

import os
import json
import time
import signal
import traceback
from pathlib import Path
from typing import Optional

import runpod
import torch
import httpx
from supabase import create_client, Client


# Global Supabase client (lazily initialized)
_supabase_client: Optional[Client] = None

# ============================================
# Graceful Shutdown Tracking
# ============================================

_shutdown_requested = False
_current_training_run_id: Optional[str] = None


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


def _update_training_run_status_on_shutdown(status: str, message: str = None):
    """Update training run status in Supabase during shutdown."""
    global _current_training_run_id

    if not _current_training_run_id:
        return

    client = get_supabase_client()
    if not client:
        print(f"[Shutdown] No Supabase client, cannot update status")
        return

    try:
        current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "status": status,
            "completed_at": current_time,
            "updated_at": current_time,
        }
        if message:
            payload["message"] = message[:1000]  # Truncate for DB

        client.table("training_runs").update(payload).eq("id", _current_training_run_id).execute()
        print(f"[Shutdown] Training run status updated to: {status}")
    except Exception as e:
        print(f"[Shutdown] Failed to update status: {e}")


def _signal_handler(signum, frame):
    """Handle shutdown signals (SIGTERM/SIGINT) gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[Shutdown] Received {signal_name}. Initiating graceful shutdown...")
    _update_training_run_status_on_shutdown("cancelled", f"Training interrupted by {signal_name}")


def _register_signal_handlers(training_run_id: str):
    """Register signal handlers for graceful shutdown."""
    global _current_training_run_id, _shutdown_requested
    _current_training_run_id = training_run_id
    _shutdown_requested = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(f"[Shutdown] Signal handlers registered for training run: {training_run_id}")


def get_supabase_client() -> Optional[Client]:
    """Get or create Supabase client singleton."""
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        return None

    _supabase_client = create_client(supabase_url, supabase_key)
    return _supabase_client


# =============================================
# Load environment variables from .env file
# =============================================
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_paths = [
        Path("/workspace/.env"),
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            for line in env_path.read_text().strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break

# Load .env on module import
load_env_file()

from trainer import UnifiedTrainer
from dataset import ProductDataset
from splitter import UPCStratifiedSplitter
from evaluator import ModelEvaluator, DomainAwareEvaluator
from checkpoint_upload import (
    upload_checkpoint_to_storage,
    save_metrics_history,
    update_training_progress,
    save_checkpoint_record,
)

# Import bb-models
from bb_models import get_model_config, is_model_supported, list_available_models
from bb_models.configs.presets import get_preset, list_presets


def validate_job_input(job_input: dict) -> tuple[bool, str]:
    """Validate job input parameters."""
    # Accept both training_run_id and training_job_id for backwards compatibility
    has_id = "training_run_id" in job_input or "training_job_id" in job_input
    if not has_id:
        return False, "Missing required field: training_run_id"

    if "model_type" not in job_input:
        return False, "Missing required field: model_type"

    model_type = job_input["model_type"]
    if not is_model_supported(model_type):
        return False, f"Unsupported model type: {model_type}. Available: {list_available_models()}"

    return True, ""


def get_job_id(job_input: dict) -> str:
    """Get job ID from input, supporting both field names."""
    return job_input.get("training_run_id") or job_input.get("training_job_id")


def get_training_config(job_input: dict) -> dict:
    """
    Build training configuration from job input.

    Priority: user overrides > model preset > defaults
    """
    model_type = job_input["model_type"]

    # Start with model preset
    preset = get_preset(model_type)
    if preset is None:
        preset = get_preset("dinov2-base")  # Fallback to default

    config = preset.copy()

    # Apply user overrides
    user_config = job_input.get("config", {})
    for key, value in user_config.items():
        if value is not None:
            config[key] = value

    # Ensure required fields
    config.setdefault("epochs", 10)
    config.setdefault("batch_size", 32)
    config.setdefault("learning_rate", 1e-4)
    config.setdefault("use_arcface", True)
    config.setdefault("use_gem_pooling", True)
    config.setdefault("use_llrd", True)
    config.setdefault("augmentation_strength", "moderate")
    config.setdefault("mixed_precision", True)
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("warmup_epochs", 1)
    config.setdefault("save_every_n_epochs", 1)
    config.setdefault("eval_every_n_epochs", 1)

    return config


def report_progress(
    job_id: str,
    status: str,
    progress: float,
    metrics: Optional[dict] = None,
    message: Optional[str] = None,
    checkpoint_url: Optional[str] = None,
):
    """Report training progress to API using Supabase SDK."""
    client = get_supabase_client()

    if client is None:
        print(f"[Progress] {status}: {progress:.1%} - {message}")
        return

    try:
        current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {
            "status": status,
            "progress": progress,
            "updated_at": current_time,
        }

        # Set started_at when training begins
        if status == "running" and progress == 0.0:
            payload["started_at"] = current_time

        # Set completed_at when training finishes
        if status in ("completed", "failed", "cancelled"):
            payload["completed_at"] = current_time

        if metrics:
            payload["metrics"] = metrics
        if message:
            payload["message"] = message
        if checkpoint_url:
            payload["checkpoint_url"] = checkpoint_url

        print(f"[Progress] Updating DB: status={status}, progress={progress:.1%}")
        result = client.table("training_runs").update(payload).eq("id", job_id).execute()

        if not result.data:
            print(f"[Progress] WARNING: No rows updated for job_id={job_id}")
        else:
            print(f"[Progress] DB updated successfully")

    except Exception as e:
        print(f"[Progress] ERROR: Failed to report progress: {e}")
        import traceback
        traceback.print_exc()


def handle_evaluation(job_input: dict) -> dict:
    """
    Handle evaluation-only job.

    Loads a checkpoint and evaluates it on provided test products.
    """
    training_run_id = job_input.get("training_run_id")
    checkpoint_id = job_input.get("checkpoint_id")
    checkpoint_url = job_input.get("checkpoint_url")
    model_type = job_input.get("model_type")
    test_product_ids = job_input.get("test_product_ids", [])

    supabase_url = job_input.get("supabase_url") or os.environ.get("SUPABASE_URL")
    supabase_key = job_input.get("supabase_key") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    # Set HuggingFace token for DINOv3 models
    hf_token = job_input.get("hf_token") or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print(f"=" * 60)
    print(f"Starting Evaluation Job")
    print(f"Training Run: {training_run_id}")
    print(f"Checkpoint: {checkpoint_id}")
    print(f"Model Type: {model_type}")
    print(f"Test Products: {len(test_product_ids)}")
    print(f"=" * 60)

    try:
        # Get Supabase client
        client = get_supabase_client()
        if client is None:
            # Fallback: create a new client with provided credentials
            from supabase import create_client
            client = create_client(supabase_url, supabase_key)

        # Fetch test products using Supabase SDK
        try:
            response = client.rpc("get_products_by_ids", {"product_ids": test_product_ids}).execute()
            test_products = response.data
        except Exception:
            # Fallback: query products directly
            response = client.table("products").select("*").in_("id", test_product_ids).execute()
            test_products = response.data

        if not test_products:
            return {"error": "Failed to fetch products or no products found"}

        print(f"Fetched {len(test_products)} test products")

        # Create test dataset
        test_dataset = ProductDataset(
            products=test_products,
            model_type=model_type,
            augmentation_strength="none",
            is_training=False,
        )

        # Load model from checkpoint
        print(f"Loading model from checkpoint...")

        # Download checkpoint if URL provided
        if checkpoint_url:
            import tempfile
            checkpoint_response = httpx.get(checkpoint_url, timeout=300)
            checkpoint_response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                f.write(checkpoint_response.content)
                checkpoint_path = f.name

            checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        else:
            return {"error": "No checkpoint URL provided"}

        # Initialize model
        from bb_models import get_backbone
        from bb_models.heads.gem_pooling import GeMPooling

        backbone = get_backbone(model_type, load_pretrained=False)
        model = torch.nn.Sequential(backbone, GeMPooling())

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Run evaluation
        print("Running evaluation...")

        evaluator = ModelEvaluator(
            model=model,
            model_type=model_type,
            device=device,
        )

        test_metrics = evaluator.evaluate(test_dataset)

        print(f"\nTest Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        # Cross-domain evaluation
        domain_evaluator = DomainAwareEvaluator(
            model=model,
            device=device,
        )

        cross_domain_metrics = domain_evaluator.evaluate_cross_domain(test_dataset)

        # Update checkpoint in database with evaluation results
        update_payload = {
            "evaluation_status": "completed",
            "evaluation_metrics": {
                **test_metrics,
                "cross_domain": {
                    k: v for k, v in cross_domain_metrics.items()
                    if isinstance(v, (int, float, str))
                },
            },
        }

        client.table("training_checkpoints").update(update_payload).eq("id", checkpoint_id).execute()

        result = {
            "status": "completed",
            "checkpoint_id": checkpoint_id,
            "training_run_id": training_run_id,
            "test_metrics": test_metrics,
            "cross_domain_metrics": {
                k: v for k, v in cross_domain_metrics.items()
                if k not in ("per_category", "hard_examples", "confused_pairs")
            },
            "hard_examples": cross_domain_metrics.get("hard_examples", [])[:20],
            "test_product_count": len(test_products),
        }

        print(f"\n{'=' * 60}")
        print(f"Evaluation Completed!")
        print(f"Recall@1: {test_metrics.get('recall@1', 'N/A')}")
        print(f"Recall@5: {test_metrics.get('recall@5', 'N/A')}")
        print(f"{'=' * 60}")

        return result

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        print(f"\nEvaluation Failed: {error_msg}")
        print(traceback_str)

        # Update checkpoint status
        if checkpoint_id:
            try:
                eval_client = get_supabase_client()
                if eval_client is None and supabase_url and supabase_key:
                    from supabase import create_client
                    eval_client = create_client(supabase_url, supabase_key)
                if eval_client:
                    eval_client.table("training_checkpoints").update(
                        {"evaluation_status": "failed", "evaluation_error": error_msg}
                    ).eq("id", checkpoint_id).execute()
            except:
                pass

        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback_str,
        }


def handler(job):
    """
    Main RunPod handler for training and evaluation jobs.

    Expected input for training:
    {
        "training_run_id": "uuid",
        "model_type": "dinov2-base",
        "checkpoint_url": "https://...",  # Optional: resume from checkpoint
        "config": {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 1e-4,
            ...
        },
        "supabase_url": "https://...",
        "supabase_key": "...",
    }

    Expected input for evaluation:
    {
        "mode": "evaluate",
        "training_run_id": "uuid",
        "checkpoint_id": "uuid",
        "checkpoint_url": "https://...",
        "model_type": "dinov2-base",
        "test_product_ids": [...],
        "supabase_url": "https://...",
        "supabase_key": "...",
    }
    """
    job_input = job.get("input", {})

    # Check if this is an evaluation job
    if job_input.get("mode") == "evaluate":
        return handle_evaluation(job_input)

    # Otherwise, handle as training job

    # Validate input
    valid, error = validate_job_input(job_input)
    if not valid:
        return {"error": error}

    training_job_id = get_job_id(job_input)
    model_type = job_input["model_type"]
    checkpoint_url = job_input.get("checkpoint_url")

    # Register signal handlers for graceful shutdown
    _register_signal_handlers(training_job_id)

    print(f"=" * 60)
    print(f"Starting Training Job: {training_job_id}")
    print(f"Model Type: {model_type}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"=" * 60)

    try:
        # Report starting
        report_progress(training_job_id, "running", 0.0, message="Initializing training...")

        # Get training config
        config = get_training_config(job_input)
        print(f"\nTraining Config:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        # Setup data - support both nested and flat structure
        data_config = job_input.get("data", {})
        supabase_url = (
            job_input.get("supabase_url") or
            data_config.get("supabase_url") or
            os.environ.get("SUPABASE_URL")
        )
        supabase_key = (
            job_input.get("supabase_key") or
            data_config.get("supabase_key") or
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )

        if not supabase_url or not supabase_key:
            return {"error": "Missing Supabase credentials"}

        # Set env vars for other modules
        os.environ["SUPABASE_URL"] = supabase_url
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = supabase_key

        # Set HuggingFace token for DINOv3 models
        hf_token = job_input.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Fetch training data
        report_progress(training_job_id, "running", 0.05, message="Fetching training data...")

        # Check if API provided training_images (new format with URLs)
        training_images = job_input.get("training_images")

        if training_images:
            # NEW FORMAT: API provides images with URLs
            print("Using NEW format: training_images provided by API")

            # Get product_ids for each split from config
            train_product_ids = set(data_config.get("train_product_ids", []) or config.get("train_product_ids", []))
            val_product_ids = set(data_config.get("val_product_ids", []) or config.get("val_product_ids", []))
            test_product_ids = set(data_config.get("test_product_ids", []) or config.get("test_product_ids", []))

            # If splits not provided, use all products for training
            if not train_product_ids:
                all_product_ids = list(training_images.keys())
                n = len(all_product_ids)
                train_end = int(n * 0.70)
                val_end = int(n * 0.85)
                train_product_ids = set(all_product_ids[:train_end])
                val_product_ids = set(all_product_ids[train_end:val_end])
                test_product_ids = set(all_product_ids[val_end:])

            # Filter training_images by split
            train_images = {pid: imgs for pid, imgs in training_images.items() if pid in train_product_ids}
            val_images = {pid: imgs for pid, imgs in training_images.items() if pid in val_product_ids}
            test_images = {pid: imgs for pid, imgs in training_images.items() if pid in test_product_ids}

            # Fetch product metadata for the datasets using Supabase SDK
            handler_client = get_supabase_client()
            if handler_client is None:
                from supabase import create_client
                handler_client = create_client(supabase_url, supabase_key)

            all_product_ids = list(training_images.keys())
            products_data = []

            # Fetch in batches of 100
            for i in range(0, len(all_product_ids), 100):
                batch_ids = all_product_ids[i:i + 100]
                try:
                    response = handler_client.table("products").select(
                        "id,barcode,brand_name,category,product_name"
                    ).in_("id", batch_ids).execute()
                    products_data.extend(response.data)
                except Exception as e:
                    print(f"Warning: Failed to fetch batch {i}: {e}")

            # Split product metadata
            train_data = [p for p in products_data if p["id"] in train_product_ids]
            val_data = [p for p in products_data if p["id"] in val_product_ids]
            test_data = [p for p in products_data if p["id"] in test_product_ids]

            print(f"\nData Split (NEW format):")
            print(f"  Train: {len(train_data)} products, {sum(len(imgs) for imgs in train_images.values())} images")
            print(f"  Val: {len(val_data)} products, {sum(len(imgs) for imgs in val_images.values())} images")
            print(f"  Test: {len(test_data)} products, {sum(len(imgs) for imgs in test_images.values())} images")

            if len(train_data) < 10:
                return {"error": f"Not enough training data: {len(train_data)} products"}

            # Create datasets with training_images
            report_progress(training_job_id, "running", 0.10, message="Creating datasets...")

            train_dataset = ProductDataset(
                products=train_data,
                model_type=model_type,
                augmentation_strength=config.get("augmentation_strength", "moderate"),
                is_training=True,
                training_images=train_images,
            )

            val_dataset = ProductDataset(
                products=val_data,
                model_type=model_type,
                augmentation_strength="none",
                is_training=False,
                training_images=val_images,
            )

            test_dataset = ProductDataset(
                products=test_data,
                model_type=model_type,
                augmentation_strength="none",
                is_training=False,
                training_images=test_images,
            )
        else:
            # LEGACY FORMAT: Fetch from Supabase using UPCStratifiedSplitter
            print("Using LEGACY format: fetching from Supabase")

            splitter = UPCStratifiedSplitter(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
            )

            product_ids = data_config.get("product_ids") or job_input.get("product_ids")
            train_data, val_data, test_data = splitter.split(
                product_ids=product_ids,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            print(f"\nData Split (LEGACY format):")
            print(f"  Train: {len(train_data)} products")
            print(f"  Val: {len(val_data)} products")
            print(f"  Test: {len(test_data)} products")

            if len(train_data) < 10:
                return {"error": f"Not enough training data: {len(train_data)} products"}

            # Create datasets (legacy format, no training_images)
            report_progress(training_job_id, "running", 0.10, message="Creating datasets...")

            train_dataset = ProductDataset(
                products=train_data,
                model_type=model_type,
                augmentation_strength=config.get("augmentation_strength", "moderate"),
                is_training=True,
            )

            val_dataset = ProductDataset(
                products=val_data,
                model_type=model_type,
                augmentation_strength="none",
                is_training=False,
            )

            test_dataset = ProductDataset(
                products=test_data,
                model_type=model_type,
                augmentation_strength="none",
                is_training=False,
            )

        # Initialize trainer (UnifiedTrainer handles all feature toggles via config)
        report_progress(training_job_id, "running", 0.15, message="Initializing model...")

        trainer = UnifiedTrainer(
            model_type=model_type,
            config=config,
            checkpoint_url=checkpoint_url,
            job_id=training_job_id,
        )

        # Training callback for progress
        def progress_callback(epoch, batch, total_batches, metrics):
            # Calculate overall progress (15% to 85% for training)
            epoch_progress = batch / total_batches
            overall_progress = 0.15 + (epoch + epoch_progress) / config["epochs"] * 0.70

            report_progress(
                training_job_id,
                "running",
                overall_progress,
                metrics=metrics,
                message=f"Epoch {epoch + 1}/{config['epochs']}, Batch {batch}/{total_batches}",
            )

        # Epoch callback for metrics history and checkpoint uploads
        total_epochs = config["epochs"]
        save_interval = config.get("save_every_n_epochs", 1)

        def epoch_callback(epoch, train_metrics, val_metrics, epoch_time, is_best, curriculum_phase, learning_rate):
            handler_client = get_supabase_client()
            if handler_client is None:
                print(f"[Epoch {epoch + 1}] No Supabase client, skipping metrics save")
                return

            # Save metrics history
            save_metrics_history(
                client=handler_client,
                training_run_id=training_job_id,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=learning_rate,
                epoch_duration=epoch_time,
                curriculum_phase=curriculum_phase,
            )

            # Update training progress
            update_training_progress(
                client=handler_client,
                training_run_id=training_job_id,
                epoch=epoch,
                total_epochs=total_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
                message=f"Epoch {epoch + 1}/{total_epochs} completed",
            )

            # Upload checkpoint if best or at save interval
            if is_best or (epoch + 1) % save_interval == 0:
                checkpoint_path = trainer.checkpoint_manager.get_last_checkpoint()
                print(f"[Epoch {epoch + 1}] Checkpoint path: {checkpoint_path}")

                if not checkpoint_path:
                    print(f"[Epoch {epoch + 1}] WARNING: No checkpoint path returned from checkpoint_manager")
                    # List checkpoints in manager for debugging
                    checkpoints_list = trainer.checkpoint_manager.list_checkpoints()
                    print(f"[Epoch {epoch + 1}] Checkpoint manager has {len(checkpoints_list)} checkpoints")
                elif not checkpoint_path.exists():
                    print(f"[Epoch {epoch + 1}] WARNING: Checkpoint path does not exist: {checkpoint_path}")
                else:
                    print(f"[Epoch {epoch + 1}] Uploading checkpoint to storage...")
                    checkpoint_url = upload_checkpoint_to_storage(
                        client=handler_client,
                        checkpoint_path=str(checkpoint_path),
                        training_run_id=training_job_id,
                        epoch=epoch,
                        is_best=is_best,
                    )

                    if checkpoint_url:
                        file_size = checkpoint_path.stat().st_size
                        print(f"[Epoch {epoch + 1}] Checkpoint uploaded: {checkpoint_url} ({file_size / 1024 / 1024:.1f}MB)")

                        record_id = save_checkpoint_record(
                            client=handler_client,
                            training_run_id=training_job_id,
                            epoch=epoch,
                            checkpoint_url=checkpoint_url,
                            train_loss=train_metrics.get("loss", 0),
                            val_metrics=val_metrics,
                            is_best=is_best,
                            is_final=(epoch + 1 == total_epochs),
                            file_size_bytes=file_size,
                        )

                        if record_id:
                            print(f"[Epoch {epoch + 1}] Checkpoint record saved: {record_id}")
                        else:
                            print(f"[Epoch {epoch + 1}] WARNING: Failed to save checkpoint record to database")
                    else:
                        print(f"[Epoch {epoch + 1}] WARNING: Checkpoint upload failed - no URL returned")

        # Train
        report_progress(training_job_id, "running", 0.15, message="Starting training...")

        training_result = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            progress_callback=progress_callback,
            epoch_callback=epoch_callback,
            shutdown_checker=is_shutdown_requested,
        )

        # Evaluate on test set
        report_progress(training_job_id, "running", 0.90, message="Evaluating on test set...")

        evaluator = ModelEvaluator(
            model=trainer.model,
            model_type=model_type,
            device=trainer.device,
        )

        test_metrics = evaluator.evaluate(test_dataset)

        print(f"\nTest Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Cross-domain evaluation if we have mixed data
        report_progress(training_job_id, "running", 0.92, message="Running cross-domain evaluation...")

        domain_evaluator = DomainAwareEvaluator(
            model=trainer.model,
            device=trainer.device,
        )

        cross_domain_metrics = domain_evaluator.evaluate_cross_domain(test_dataset)

        print(f"\nCross-Domain Metrics:")
        for key, value in cross_domain_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif key in ("per_category", "hard_examples", "confused_pairs"):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")

        # Upload final checkpoint
        report_progress(training_job_id, "running", 0.95, message="Uploading checkpoint...")

        checkpoint_info = trainer.save_final_checkpoint()
        print(f"Final checkpoint info: {checkpoint_info}")

        # Save final checkpoint record to database (fallback if per-epoch saves failed)
        if checkpoint_info.get("uploaded") and checkpoint_info.get("url"):
            final_client = get_supabase_client()
            if final_client:
                # Get best val metrics from training result
                final_train_loss = training_result.get("final_metrics", {}).get("train_loss", 0)
                final_val_metrics = {
                    "loss": training_result.get("best_val_loss"),
                    "recall@1": training_result.get("best_recall_at_1"),
                    "recall@5": training_result.get("final_metrics", {}).get("val_recall_at_5"),
                }

                # Check if we already have this checkpoint saved
                existing = final_client.table("training_checkpoints").select("id").eq(
                    "training_run_id", training_job_id
                ).eq("is_final", True).execute()

                if not existing.data:
                    print(f"Saving final checkpoint record to database...")
                    record_id = save_checkpoint_record(
                        client=final_client,
                        training_run_id=training_job_id,
                        epoch=training_result.get("best_epoch", config["epochs"]) - 1,
                        checkpoint_url=checkpoint_info["url"],
                        train_loss=final_train_loss,
                        val_metrics=final_val_metrics,
                        is_best=True,
                        is_final=True,
                    )
                    if record_id:
                        print(f"Final checkpoint record saved: {record_id}")
                    else:
                        print("WARNING: Failed to save final checkpoint record")
                else:
                    print(f"Final checkpoint already exists in database")

        # Prepare result
        result = {
            "status": "completed",
            "training_job_id": training_job_id,
            "model_type": model_type,
            "epochs_trained": training_result["epochs_trained"],
            "best_epoch": training_result["best_epoch"],
            "training_metrics": training_result["final_metrics"],
            "test_metrics": test_metrics,
            "cross_domain_metrics": {
                k: v for k, v in cross_domain_metrics.items()
                if k not in ("per_category", "hard_examples", "confused_pairs")
            },
            "hard_examples": cross_domain_metrics.get("hard_examples", []),
            "confused_pairs": cross_domain_metrics.get("confused_pairs", []),
            "checkpoint": checkpoint_info,
            "data_stats": {
                "train_products": len(train_data),
                "val_products": len(val_data),
                "test_products": len(test_data),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "train_image_types": train_dataset.get_image_type_distribution(),
                "train_domain_distribution": train_dataset.get_domain_distribution(),
                "data_format": "new" if train_dataset.use_new_format else "legacy",
            },
            "sota_enabled": config.get("sota_config", {}).get("enabled", False),
            # Include preprocessing config for deployment
            "preprocessing_config": train_dataset.preprocessing_config,
        }

        # Include recall metrics if available from SOTA trainer
        if "best_recall_at_1" in training_result:
            result["best_recall_at_1"] = training_result["best_recall_at_1"]
        if "metric_history" in training_result:
            result["metric_history"] = training_result["metric_history"]

        # Report completion with checkpoint URL
        all_metrics = {
            **training_result["final_metrics"],
            **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
            **{f"cross_{k}": v for k, v in cross_domain_metrics.items()
               if isinstance(v, (int, float)) and k not in ("per_category", "hard_examples", "confused_pairs")},
        }
        report_progress(
            training_job_id,
            "completed",
            1.0,
            metrics=all_metrics,
            message="Training completed successfully",
            checkpoint_url=checkpoint_info.get("url"),
        )

        print(f"\n{'=' * 60}")
        print(f"Training Completed!")
        print(f"Best Epoch: {training_result['best_epoch']}")
        print(f"Checkpoint: {checkpoint_info.get('url', 'local')}")
        print(f"{'=' * 60}")

        return result

    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()

        print(f"\nTraining Failed: {error_msg}")
        print(traceback_str)

        report_progress(
            training_job_id,
            "failed",
            0.0,
            message=f"Training failed: {error_msg}",
        )

        return {
            "status": "failed",
            "error": error_msg,
            "traceback": traceback_str,
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
