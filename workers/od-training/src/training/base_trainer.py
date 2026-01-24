"""
SOTA Base Trainer for Object Detection.

Integrates all SOTA training features:
- EMA (Exponential Moving Average)
- LLRD (Layer-wise Learning Rate Decay)
- Warmup + Cosine LR Scheduler
- Mixed Precision (FP16) Training
- Gradient Clipping
- Multi-scale Training Support

This class should be inherited by model-specific trainers
(RT-DETR, D-FINE, etc.)

Usage:
    class RTDETRTrainer(SOTABaseTrainer):
        def setup_model(self):
            self.model = load_rtdetr_model(...)

        def compute_loss(self, outputs, targets):
            return rtdetr_loss(outputs, targets)
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from .ema import ModelEMA
from .optimizer import build_llrd_optimizer, build_simple_optimizer
from .scheduler import build_scheduler

# Import evaluation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation import COCOEvaluator


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_type: str = "rt-detr"
    model_size: str = "l"
    pretrained: bool = True
    num_classes: int = 80

    # Training
    epochs: int = 100
    batch_size: int = 16
    accumulation_steps: int = 1  # Gradient accumulation

    # Optimizer
    optimizer: str = "adamw"
    base_lr: float = 0.0001
    weight_decay: float = 0.0001
    llrd_decay: float = 0.9  # Layer-wise LR decay
    head_lr_factor: float = 10.0  # Head LR multiplier

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.01

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000

    # Mixed Precision
    use_amp: bool = True

    # Regularization
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0

    # Multi-scale Training
    multi_scale: bool = False
    img_size: int = 640
    multi_scale_range: Tuple[float, float] = (0.5, 1.5)

    # Checkpointing
    save_freq: int = 5
    patience: int = 20  # Early stopping
    device: str = "cuda"

    # Augmentation
    augmentation_preset: str = "sota"
    augmentation_overrides: Optional[Dict[str, Any]] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    train_img_dir: str = ""
    train_ann_file: str = ""
    val_img_dir: str = ""
    val_ann_file: str = ""
    num_classes: int = 80
    class_names: Optional[List[str]] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: str = "./output"
    checkpoint_dir: str = "./output/checkpoints"
    log_dir: str = "./output/logs"


@dataclass
class TrainingResult:
    """Training result."""
    best_checkpoint: str
    best_metrics: Dict[str, float]
    best_epoch: int
    total_epochs: int
    final_metrics: Dict[str, float]
    training_time: float


class SOTABaseTrainer(ABC):
    """
    SOTA Base Trainer with all modern training techniques.

    Subclasses must implement:
    - setup_model(): Initialize the model
    - compute_loss(outputs, targets): Compute training loss
    - forward_pass(images): Model forward pass
    - postprocess(outputs): Convert outputs to predictions for evaluation
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        dataset_config: DatasetConfig,
        output_config: OutputConfig,
        progress_callback: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.output_config = output_config
        self.progress_callback = progress_callback

        # Device setup
        self.device = torch.device(
            training_config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Create directories
        Path(output_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(output_config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components (will be set in setup methods)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.evaluator = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_map = 0.0
        self.best_epoch = 0
        self.best_checkpoint_path = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_map": [],
            "val_map_50": [],
            "lr": [],
        }

    def _get_optimal_num_workers(self) -> int:
        """
        Auto-detect optimal number of DataLoader workers based on system resources.

        Considers:
        - Available RAM (each worker uses ~1-2GB for URL-based loading)
        - CPU cores
        - GPU memory (larger models need more headroom)

        Returns:
            Optimal number of workers (1-8)
        """
        import os
        import multiprocessing

        # Default fallback
        default_workers = 2

        try:
            # Get CPU count
            cpu_count = multiprocessing.cpu_count()

            # Get available RAM (in GB)
            available_ram_gb = None
            try:
                import psutil
                mem = psutil.virtual_memory()
                available_ram_gb = mem.available / (1024 ** 3)
            except ImportError:
                # Fallback: read from /proc/meminfo on Linux
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if 'MemAvailable' in line:
                                available_ram_gb = int(line.split()[1]) / (1024 ** 2)
                                break
                except:
                    pass

            # Get GPU memory (in GB)
            gpu_memory_gb = None
            try:
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except:
                pass

            # Calculate optimal workers
            # Rule: Each worker uses ~1.5-2GB RAM for URL loading + image processing
            ram_per_worker = 2.0  # GB

            if available_ram_gb is not None:
                # Reserve 8GB for main process + model + GPU transfers
                usable_ram = max(0, available_ram_gb - 8)
                workers_by_ram = int(usable_ram / ram_per_worker)
            else:
                workers_by_ram = default_workers

            # Don't use more workers than CPU cores - 2 (leave headroom)
            workers_by_cpu = max(1, cpu_count - 2)

            # Final decision: minimum of RAM-based and CPU-based limits
            optimal = min(workers_by_ram, workers_by_cpu)

            # Clamp to reasonable range (max 4 to prevent OOM with large datasets)
            optimal = max(1, min(4, optimal))

            print(f"[AUTO-CONFIG] CPU cores: {cpu_count}, "
                  f"Available RAM: {available_ram_gb:.1f}GB, "
                  f"GPU Memory: {gpu_memory_gb:.1f}GB" if gpu_memory_gb else "")
            print(f"[AUTO-CONFIG] Workers by RAM: {workers_by_ram}, "
                  f"Workers by CPU: {workers_by_cpu}, "
                  f"Selected: {optimal}")

            return optimal

        except Exception as e:
            print(f"[WARNING] Auto-config failed: {e}, using default {default_workers} workers")
            return default_workers

    @abstractmethod
    def setup_model(self):
        """
        Initialize the model.

        Must set self.model to a nn.Module.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            outputs: Model outputs
            targets: List of target dicts

        Returns:
            Total loss (scalar tensor)
        """
        pass

    @abstractmethod
    def forward_pass(
        self,
        images: torch.Tensor,
    ) -> Any:
        """
        Model forward pass.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Model outputs (model-specific format)
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        outputs: Any,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert model outputs to predictions for evaluation.

        Args:
            outputs: Model outputs
            targets: Targets (for image size info)

        Returns:
            List of prediction dicts with 'boxes', 'scores', 'labels'
        """
        pass

    def setup_data(self, url_dataset_data: Optional[Dict[str, Any]] = None):
        """
        Setup data loaders with augmentation pipeline.

        Args:
            url_dataset_data: If provided, use URL-based dataset loading.
                Expected format:
                {
                    "train_images": [...],
                    "val_images": [...],
                    "class_mapping": {...},
                    "class_names": [...],
                    "num_classes": int,
                }
                If None, use file-based loading (legacy).
        """
        from data import ODDataset, create_dataloader
        from augmentations import AugmentationPipeline

        # Training augmentation pipeline
        train_pipeline = AugmentationPipeline.from_preset(
            preset_name=self.training_config.augmentation_preset,
            img_size=self.training_config.img_size,
            is_training=True,
            overrides=self.training_config.augmentation_overrides,
        )

        # Validation pipeline (no augmentation)
        val_pipeline = AugmentationPipeline.from_preset(
            preset_name="none",
            img_size=self.training_config.img_size,
            is_training=False,
        )

        # Create datasets
        if url_dataset_data is not None:
            # NEW: URL-based dataset loading
            from data.url_dataset import URLODDataset, create_url_dataloader

            print("[INFO] Using URL-based dataset loading")

            # Get preload config from url_dataset_data or use defaults
            # Config can be passed from API: url_dataset_data["preload_config"]
            preload_config = url_dataset_data.get("preload_config", {})

            # Determine preload behavior from config (default: enabled with batching)
            preload_enabled = preload_config.get("enabled", True)

            print(f"[INFO] Preload config: enabled={preload_enabled}, "
                  f"batched={preload_config.get('batched', True)}, "
                  f"batch_size={preload_config.get('batch_size', 500)}, "
                  f"max_workers={preload_config.get('max_workers', 16)}")

            train_dataset = URLODDataset(
                image_data=url_dataset_data["train_images"],
                class_mapping=url_dataset_data["class_mapping"],
                transform=train_pipeline,
                img_size=self.training_config.img_size,
                preload=preload_enabled,  # Configurable via preload_config
                use_memory_cache=False,  # IMPORTANT: Prevents OOM - file cache only
                preload_config=preload_config,  # NEW: Pass full preload config
            )

            val_dataset = URLODDataset(
                image_data=url_dataset_data["val_images"],
                class_mapping=url_dataset_data["class_mapping"],
                transform=val_pipeline,
                img_size=self.training_config.img_size,
                preload=preload_enabled,  # Configurable via preload_config
                use_memory_cache=False,  # IMPORTANT: Prevents OOM - file cache only
                preload_config=preload_config,  # NEW: Pass full preload config
            )

            # Auto-detect optimal num_workers based on available resources
            num_workers = self._get_optimal_num_workers()
            print(f"[INFO] Using {num_workers} DataLoader workers (auto-detected)")

            # Create data loaders
            self.train_loader = create_url_dataloader(
                train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )

            self.val_loader = create_url_dataloader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )

            # Update class info
            if url_dataset_data.get("num_classes"):
                self.dataset_config.num_classes = url_dataset_data["num_classes"]
            if url_dataset_data.get("class_names"):
                self.dataset_config.class_names = url_dataset_data["class_names"]

        else:
            # LEGACY: File-based dataset loading
            print("[INFO] Using file-based dataset loading")

            train_dataset = ODDataset(
                img_dir=self.dataset_config.train_img_dir,
                ann_file=self.dataset_config.train_ann_file,
                transform=train_pipeline,
                img_size=self.training_config.img_size,
                class_names=self.dataset_config.class_names,
            )

            val_dataset = ODDataset(
                img_dir=self.dataset_config.val_img_dir,
                ann_file=self.dataset_config.val_ann_file,
                transform=val_pipeline,
                img_size=self.training_config.img_size,
                class_names=self.dataset_config.class_names,
            )

            # Create data loaders
            self.train_loader = create_dataloader(
                train_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )

            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=4,
                drop_last=False,
            )

            # Update num_classes from dataset if not set
            if self.dataset_config.num_classes == 0:
                self.dataset_config.num_classes = train_dataset.get_num_classes()

        print(f"Train dataset: {len(train_dataset)} images")
        print(f"Val dataset: {len(val_dataset)} images")

    def setup_optimizer(self):
        """Setup optimizer with LLRD."""
        if self.training_config.llrd_decay < 1.0:
            # Use layer-wise learning rate decay
            self.optimizer = build_llrd_optimizer(
                model=self.model,
                base_lr=self.training_config.base_lr,
                weight_decay=self.training_config.weight_decay,
                llrd_decay=self.training_config.llrd_decay,
                head_lr_factor=self.training_config.head_lr_factor,
                model_type=self.training_config.model_type,
                optimizer_type=self.training_config.optimizer,
            )
        else:
            # Standard optimizer
            self.optimizer = build_simple_optimizer(
                model=self.model,
                optimizer_type=self.training_config.optimizer,
                lr=self.training_config.base_lr,
                weight_decay=self.training_config.weight_decay,
            )

        print(f"Optimizer: {self.training_config.optimizer}")
        print(f"Base LR: {self.training_config.base_lr}")
        if self.training_config.llrd_decay < 1.0:
            print(f"LLRD decay: {self.training_config.llrd_decay}")

    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Ensure at least 1 step per epoch to avoid division by zero
        steps_per_epoch = max(1, len(self.train_loader) // self.training_config.accumulation_steps)

        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            warmup_epochs=self.training_config.warmup_epochs,
            total_epochs=self.training_config.epochs,
            steps_per_epoch=steps_per_epoch,
            scheduler_type=self.training_config.scheduler,
            min_lr_ratio=self.training_config.min_lr_ratio,
        )

        print(f"Scheduler: {self.training_config.scheduler}")
        print(f"Warmup epochs: {self.training_config.warmup_epochs}")

    def setup_ema(self):
        """Setup Exponential Moving Average."""
        if self.training_config.use_ema:
            self.ema = ModelEMA(
                model=self.model,
                decay=self.training_config.ema_decay,
                warmup_steps=self.training_config.ema_warmup_steps,
                device=self.device,
            )
            print(f"EMA enabled (decay={self.training_config.ema_decay})")
        else:
            self.ema = None

    def setup_amp(self):
        """Setup Automatic Mixed Precision."""
        if self.training_config.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            print("AMP (FP16) enabled")
        else:
            self.scaler = None

    def setup_evaluator(self):
        """Setup COCO evaluator."""
        self.evaluator = COCOEvaluator(
            num_classes=self.dataset_config.num_classes,
        )

    def setup(self, url_dataset_data: Optional[Dict[str, Any]] = None):
        """
        Setup all components.

        Args:
            url_dataset_data: If provided, use URL-based dataset loading.
        """
        print("\n" + "=" * 50)
        print("Setting up training...")
        print("=" * 50)

        self.setup_model()
        self.model.to(self.device)
        print(f"Model: {self.training_config.model_type}-{self.training_config.model_size}")

        self.setup_data(url_dataset_data=url_dataset_data)
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_ema()
        self.setup_amp()
        self.setup_evaluator()

        # Initialize optimizer step count to avoid LR scheduler warning
        # This is needed because our scheduler.step() is per-batch, not per-epoch
        if hasattr(self.optimizer, '_step_count'):
            self.optimizer._step_count = 1

        print("=" * 50 + "\n")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Multi-scale training
            if self.training_config.multi_scale:
                images = self._apply_multi_scale(images)

            # Forward pass with AMP
            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.forward_pass(images)
                    loss = self.compute_loss(outputs, targets)
                    loss = loss / self.training_config.accumulation_steps

                # NaN loss detection
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(
                        f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. "
                        "Training is unstable. Try: lower learning rate, gradient clipping, or check data."
                    )

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.training_config.accumulation_steps == 0:
                    # Gradient clipping
                    if self.training_config.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.gradient_clip,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # Update EMA
                    if self.ema is not None:
                        self.ema.update(self.model, self.global_step)

                    # Update scheduler (step-wise)
                    self.scheduler.step()
                    self.global_step += 1
            else:
                outputs = self.forward_pass(images)
                loss = self.compute_loss(outputs, targets)
                loss = loss / self.training_config.accumulation_steps

                # NaN loss detection
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(
                        f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. "
                        "Training is unstable. Try: lower learning rate, gradient clipping, or check data."
                    )

                loss.backward()

                if (batch_idx + 1) % self.training_config.accumulation_steps == 0:
                    if self.training_config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.training_config.gradient_clip,
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.ema is not None:
                        self.ema.update(self.model, self.global_step)

                    self.scheduler.step()
                    self.global_step += 1

            total_loss += loss.item() * self.training_config.accumulation_steps
            num_batches += 1

            # Log progress - every 5 batches or if total batches <= 10
            log_interval = min(5, max(1, len(self.train_loader) // 2))
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Batch {batch_idx}/{len(self.train_loader)} - "
                      f"Loss: {loss.item() * self.training_config.accumulation_steps:.4f} - "
                      f"LR: {current_lr:.6f}")

        avg_loss = total_loss / max(1, num_batches)
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.history["train_loss"].append(avg_loss)
        self.history["lr"].append(current_lr)

        return {
            "loss": avg_loss,
            "lr": current_lr,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model using EMA weights if available."""
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        self.evaluator.reset()

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            # Forward pass
            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.forward_pass(images)
            else:
                outputs = self.forward_pass(images)

            # Convert to predictions
            predictions = self.postprocess(outputs, targets)

            # Update evaluator
            self.evaluator.update(predictions, targets)

        # Compute metrics
        metrics = self.evaluator.compute()

        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.model)

        # Store history
        self.history["val_map"].append(metrics["mAP"])
        self.history["val_map_50"].append(metrics["mAP_50"])

        return {
            "map": metrics["mAP"],
            "map_50": metrics["mAP_50"],
            "map_75": metrics["mAP_75"],
            "ar": metrics.get("AR", 0),
        }

    def _apply_multi_scale(self, images: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale training."""
        import torch.nn.functional as F
        import random

        scale = random.uniform(*self.training_config.multi_scale_range)
        new_size = int(self.training_config.img_size * scale)
        new_size = (new_size // 32) * 32  # Ensure divisible by 32

        if new_size != images.shape[-1]:
            images = F.interpolate(
                images,
                size=(new_size, new_size),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_map": self.best_map,
            "config": {
                "training": asdict(self.training_config),
                "dataset": asdict(self.dataset_config),
            },
            "history": self.history,
        }

        # Add EMA state
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        # Add scaler state
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt",
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best (FP16 inference-only for fast upload & inference)
        if is_best:
            best_path = os.path.join(
                self.output_config.checkpoint_dir,
                "best_model.pt",
            )

            # Create FP16 inference-only checkpoint (much smaller, faster upload)
            # Use EMA weights if available (usually better for inference)
            if self.ema is not None:
                # EMA stores shadow model, get its state_dict directly
                model_weights = self.ema.shadow.state_dict()
            else:
                model_weights = self.model.state_dict()

            # Convert to FP16 for smaller file size (~50% reduction)
            # Only convert tensors with float32 dtype, skip non-tensors
            def to_fp16(v):
                if torch.is_tensor(v) and v.dtype == torch.float32:
                    return v.half()
                return v

            inference_checkpoint = {
                "model_state_dict": {k: to_fp16(v) for k, v in model_weights.items()},
                "epoch": epoch,
                "metrics": metrics,
                "best_map": self.best_map,
                "config": {
                    "training": asdict(self.training_config),
                    "dataset": asdict(self.dataset_config),
                },
                "precision": "fp16",
                "inference_only": True,
            }
            torch.save(inference_checkpoint, best_path)
            self.best_checkpoint_path = best_path

            # Log size reduction
            file_size_mb = os.path.getsize(best_path) / (1024 * 1024)
            print(f"  New best model! mAP: {metrics.get('map', 0):.4f} (FP16: {file_size_mb:.1f} MB)")

        # Save latest
        latest_path = os.path.join(
            self.output_config.checkpoint_dir,
            "latest.pt",
        )
        torch.save(checkpoint, latest_path)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_map = checkpoint["best_map"]
        self.history = checkpoint.get("history", self.history)

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, url_dataset_data: Optional[Dict[str, Any]] = None) -> TrainingResult:
        """
        Main training loop.

        Args:
            url_dataset_data: If provided, use URL-based dataset loading.
        """
        start_time = time.time()

        # Setup all components
        self.setup(url_dataset_data=url_dataset_data)

        print(f"\nStarting training for {self.training_config.epochs} epochs")
        print("=" * 50)

        epochs_without_improvement = 0

        for epoch in range(1, self.training_config.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.training_config.epochs} ===")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, LR: {train_metrics['lr']:.6f}")

            # Validate
            val_metrics = self.validate()
            current_map = val_metrics["map"]
            print(f"Val - mAP: {current_map:.4f}, mAP@50: {val_metrics['map_50']:.4f}, "
                  f"mAP@75: {val_metrics['map_75']:.4f}")

            # Check improvement
            is_best = current_map > self.best_map or self.best_checkpoint_path is None
            if is_best:
                self.best_map = current_map
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            all_metrics = {**train_metrics, **val_metrics}
            if is_best or epoch % self.training_config.save_freq == 0:
                self.save_checkpoint(epoch, all_metrics, is_best)

            # Progress callback
            if self.progress_callback:
                self.progress_callback(epoch, all_metrics)

            # Early stopping
            if epochs_without_improvement >= self.training_config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        training_time = time.time() - start_time

        # Save training history
        history_path = os.path.join(self.output_config.log_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 50)
        print(f"Training complete!")
        print(f"Best mAP: {self.best_map:.4f} at epoch {self.best_epoch}")
        print(f"Total time: {training_time / 3600:.2f} hours")
        print("=" * 50)

        return TrainingResult(
            best_checkpoint=self.best_checkpoint_path,
            best_metrics={"map": self.best_map},
            best_epoch=self.best_epoch,
            total_epochs=epoch,
            final_metrics=all_metrics,
            training_time=training_time,
        )
