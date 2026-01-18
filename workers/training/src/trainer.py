"""
Model Trainer for fine-tuning embedding models.

Features:
- Multi-model support via bb-models
- ArcFace loss for metric learning
- GeM pooling
- Layer-wise learning rate decay
- Mixed precision training
- Gradient accumulation
- Checkpoint management
- SOTA: Multi-loss training (ArcFace + Triplet + Domain)
- SOTA: P-K batch sampling with domain balancing
- SOTA: Curriculum learning
- SOTA: Early stopping with multiple metrics
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# bb-models imports
from bb_models import get_backbone, get_model_config
from bb_models.heads.gem_pooling import GeMPooling, AdaptiveGeMPooling
from bb_models.heads.arcface import ArcFaceHead, EnhancedArcFaceLoss
from bb_models.heads.projection import MLPProjectionHead
from bb_models.utils.llrd import get_llrd_optimizer_params
from bb_models.utils.checkpoint import CheckpointManager

# SOTA imports
from losses import CombinedProductLoss, OnlineHardTripletLoss
from samplers import PKDomainSampler, CurriculumSampler
from early_stopping import EarlyStopping, CurriculumScheduler, MetricTracker


@dataclass
class TrainingResult:
    """Training result container."""
    epochs_trained: int
    best_epoch: int
    best_val_loss: float
    final_metrics: dict


class EmbeddingModel(nn.Module):
    """
    Complete embedding model with backbone, pooling, and heads.

    Architecture:
    Backbone -> GeM Pooling -> Projection -> (ArcFace for training)
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int,
        embedding_dim: int = 512,
        use_gem_pooling: bool = True,
        use_arcface: bool = True,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
        checkpoint_url: Optional[str] = None,
    ):
        super().__init__()

        self.model_type = model_type
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_arcface = use_arcface

        # Get model config
        config = get_model_config(model_type)
        backbone_dim = config.embedding_dim

        # Backbone
        self.backbone = get_backbone(
            model_id=model_type,
            checkpoint_url=checkpoint_url,
            load_pretrained=True,
        )

        # Pooling
        if use_gem_pooling:
            self.pooling = AdaptiveGeMPooling(channels=backbone_dim)
        else:
            self.pooling = nn.Identity()

        # Projection head
        self.projection = MLPProjectionHead(
            in_features=backbone_dim,
            hidden_features=backbone_dim,
            out_features=embedding_dim,
            dropout=0.1,
            normalize=True,
        )

        # ArcFace head for training
        if use_arcface:
            self.arcface = ArcFaceHead(
                in_features=embedding_dim,
                num_classes=num_classes,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        else:
            self.arcface = None
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]
            labels: Class labels for ArcFace [B]

        Returns:
            Dictionary with embeddings and logits
        """
        # Backbone features
        features = self.backbone(x)

        # Pool if needed based on feature dimensions
        if features.dim() == 4:
            # [B, C, H, W] -> apply spatial pooling (e.g., for CNN backbones)
            pooled = self.pooling(features)
        elif features.dim() == 3:
            # [B, N, D] -> mean pool over sequence (for ViT patch tokens)
            pooled = features.mean(dim=1)
        else:
            # [B, D] -> already pooled (DINOv2 CLS token), skip pooling
            pooled = features

        # Projection to embedding
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        result = {"embeddings": embeddings}

        # Classification logits
        if self.training and labels is not None:
            if self.use_arcface:
                logits = self.arcface(embeddings, labels)
            else:
                logits = self.classifier(embeddings)
            result["logits"] = logits

        return result

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embedding for inference."""
        self.eval()
        with torch.no_grad():
            result = self.forward(x)
            return result["embeddings"]


class ModelTrainer:
    """
    Trainer for fine-tuning embedding models.

    Implements:
    - Mixed precision training
    - Gradient accumulation
    - LLRD (Layer-wise Learning Rate Decay)
    - Warmup + Cosine annealing
    - Checkpoint management
    - Early stopping
    """

    def __init__(
        self,
        model_type: str,
        config: dict,
        checkpoint_url: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        self.model_type = model_type
        self.config = config
        self.job_id = job_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.checkpoint_manager = None

        # Resume from checkpoint if provided
        self.checkpoint_url = checkpoint_url
        self.start_epoch = 0

    def _build_model(self, num_classes: int) -> EmbeddingModel:
        """Build the embedding model."""
        model = EmbeddingModel(
            model_type=self.model_type,
            num_classes=num_classes,
            embedding_dim=self.config.get("embedding_dim", 512),
            use_gem_pooling=self.config.get("use_gem_pooling", True),
            use_arcface=self.config.get("use_arcface", True),
            arcface_margin=self.config.get("arcface_margin", 0.5),
            arcface_scale=self.config.get("arcface_scale", 64.0),
            checkpoint_url=self.checkpoint_url,
        )
        return model.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with LLRD if enabled."""
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)

        if self.config.get("use_llrd", True):
            # Get layer-wise params from backbone
            backbone_params = get_llrd_optimizer_params(
                self.model.backbone,
                base_lr=lr,
                llrd_decay=self.config.get("llrd_decay", 0.9),
                weight_decay=weight_decay,
            )

            # Add head params with higher LR
            head_params = [
                {
                    "params": self.model.pooling.parameters(),
                    "lr": lr * 10,
                    "weight_decay": weight_decay,
                },
                {
                    "params": self.model.projection.parameters(),
                    "lr": lr * 10,
                    "weight_decay": weight_decay,
                },
            ]

            if self.model.arcface is not None:
                head_params.append({
                    "params": self.model.arcface.parameters(),
                    "lr": lr * 10,
                    "weight_decay": weight_decay,
                })

            all_params = backbone_params + head_params
        else:
            all_params = [
                {"params": self.model.parameters(), "lr": lr, "weight_decay": weight_decay}
            ]

        optimizer = torch.optim.AdamW(all_params)
        return optimizer

    def _build_scheduler(self, steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler with warmup."""
        epochs = self.config.get("epochs", 10)
        warmup_epochs = self.config.get("warmup_epochs", 1)

        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine annealing
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps - warmup_steps,
            T_mult=1,
            eta_min=1e-6,
        )

        # Combine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return scheduler

    def _build_checkpoint_manager(self) -> CheckpointManager:
        """Build checkpoint manager."""
        output_dir = Path(f"/tmp/checkpoints/{self.job_id or 'training'}")
        output_dir.mkdir(parents=True, exist_ok=True)

        return CheckpointManager(
            output_dir=output_dir,
            model_id=f"{self.model_type}_{self.job_id or 'model'}",
            keep_last_n=3,
            keep_best=True,
        )

    def train(
        self,
        train_dataset,
        val_dataset,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run training loop.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            progress_callback: Callback(epoch, batch, total_batches, metrics)

        Returns:
            Training result dictionary
        """
        # Get number of classes from dataset
        num_classes = train_dataset.num_classes
        print(f"Number of classes: {num_classes}")

        # Build model
        self.model = self._build_model(num_classes)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Build optimizer and scheduler
        batch_size = self.config.get("batch_size", 32)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(len(train_loader))
        self.checkpoint_manager = self._build_checkpoint_manager()

        # Mixed precision scaler
        use_amp = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if use_amp else None

        # Loss function
        criterion = EnhancedArcFaceLoss(
            embedding_dim=self.config.get("embedding_dim", 512),
            num_classes=num_classes,
            margin=self.config.get("arcface_margin", 0.5),
            scale=self.config.get("arcface_scale", 64.0),
            label_smoothing=self.config.get("label_smoothing", 0.1),
        ).to(self.device) if self.config.get("use_arcface", True) else nn.CrossEntropyLoss()

        # Training state
        epochs = self.config.get("epochs", 10)
        best_val_loss = float("inf")
        best_epoch = 0
        patience = self.config.get("early_stopping_patience", 5)
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {batch_size * grad_accum}")
        print(f"  Mixed precision: {use_amp}")

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(
                train_loader=train_loader,
                criterion=criterion,
                epoch=epoch,
                grad_accum=grad_accum,
                use_amp=use_amp,
                progress_callback=progress_callback,
            )

            # Validate
            val_metrics = self._validate(val_loader, criterion)

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Check for improvement
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch + 1
                patience_counter = 0

                # Save best checkpoint
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                )
            else:
                patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % self.config.get("save_every_n_epochs", 1) == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "epochs_trained": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_metrics": {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            },
        }

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion,
        epoch: int,
        grad_accum: int,
        use_amp: bool,
        progress_callback: Optional[Callable],
    ) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = self.model(images, labels)
                embeddings = outputs["embeddings"]
                logits = outputs.get("logits")

                if logits is not None:
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(embeddings, labels)

                loss = loss / grad_accum

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * grad_accum

            # Progress callback
            if progress_callback and batch_idx % 10 == 0:
                progress_callback(
                    epoch,
                    batch_idx,
                    num_batches,
                    {"train_loss": total_loss / (batch_idx + 1)},
                )

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, criterion) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images, labels)
            embeddings = outputs["embeddings"]
            logits = outputs.get("logits")

            if logits is not None:
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                loss = criterion(embeddings, labels)
                preds = embeddings.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total,
        }

    def save_final_checkpoint(self) -> dict:
        """Save and upload final checkpoint."""
        if self.checkpoint_manager is None:
            return {"error": "No checkpoint manager"}

        # Get best checkpoint info
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()

        if best_checkpoint and os.environ.get("AWS_ACCESS_KEY_ID"):
            # Upload to S3
            url = self.checkpoint_manager.upload_to_s3(best_checkpoint)
            return {
                "path": str(best_checkpoint),
                "url": url,
                "uploaded": True,
            }

        return {
            "path": str(best_checkpoint) if best_checkpoint else None,
            "uploaded": False,
        }


class SOTAModelTrainer(ModelTrainer):
    """
    SOTA Trainer with advanced features.

    Features beyond ModelTrainer:
    - Multi-loss training (ArcFace + Triplet + Domain Adversarial)
    - P-K batch sampling with domain balancing
    - Curriculum learning
    - Enhanced early stopping
    - Recall@K metrics
    """

    def __init__(
        self,
        model_type: str,
        config: dict,
        checkpoint_url: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        super().__init__(model_type, config, checkpoint_url, job_id)

        # SOTA-specific config
        self.sota_config = config.get("sota_config", {})
        self.use_combined_loss = self.sota_config.get("use_combined_loss", True)
        self.use_pk_sampling = self.sota_config.get("use_pk_sampling", True)
        self.use_curriculum = self.sota_config.get("use_curriculum", False)
        self.use_domain_adaptation = self.sota_config.get("use_domain_adaptation", True)

        # Curriculum state
        self.curriculum_phase = "warmup"
        self.curriculum_scheduler = None

        # Enhanced metrics
        self.metric_tracker = MetricTracker(
            metrics=["val_loss", "val_recall_1", "val_recall_5"],
            modes={"val_loss": "min", "val_recall_1": "max", "val_recall_5": "max"}
        )

    def _build_loss_function(self, num_classes: int):
        """Build loss function based on config."""
        if self.use_combined_loss:
            loss_config = self.sota_config.get("loss", {})
            return CombinedProductLoss(
                num_classes=num_classes,
                embedding_dim=self.config.get("embedding_dim", 512),
                arcface_weight=loss_config.get("arcface_weight", 1.0),
                triplet_weight=loss_config.get("triplet_weight", 0.5),
                domain_weight=loss_config.get("domain_weight", 0.1),
                arcface_margin=loss_config.get("arcface_margin", 0.5),
                arcface_scale=loss_config.get("arcface_scale", 64.0),
                triplet_margin=loss_config.get("triplet_margin", 0.3),
                use_domain_adaptation=self.use_domain_adaptation,
            ).to(self.device)
        else:
            # Fall back to standard ArcFace
            return EnhancedArcFaceLoss(
                embedding_dim=self.config.get("embedding_dim", 512),
                num_classes=num_classes,
                margin=self.config.get("arcface_margin", 0.5),
                scale=self.config.get("arcface_scale", 64.0),
                label_smoothing=self.config.get("label_smoothing", 0.1),
            ).to(self.device)

    def _build_train_loader(self, train_dataset, batch_size: int) -> DataLoader:
        """Build training data loader with P-K sampling."""
        if self.use_pk_sampling:
            # Extract labels and domains from dataset
            labels = []
            domains = []
            for i in range(len(train_dataset)):
                item = train_dataset.get_item_info(i)
                labels.append(item["label"])
                domains.append(1 if item.get("domain", "synthetic") == "real" else 0)

            sampling_config = self.sota_config.get("sampling", {})
            sampler = PKDomainSampler(
                labels=labels,
                domains=domains,
                products_per_batch=sampling_config.get("products_per_batch", 8),
                samples_per_product=sampling_config.get("samples_per_product", 4),
                synthetic_ratio=sampling_config.get("synthetic_ratio", 0.5),
                drop_last=True,
            )

            return DataLoader(
                train_dataset,
                batch_sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )
        else:
            # Standard DataLoader
            return DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )

    def _setup_curriculum(self, num_epochs: int):
        """Setup curriculum learning scheduler."""
        if self.use_curriculum:
            curriculum_config = self.sota_config.get("curriculum", {})
            self.curriculum_scheduler = CurriculumScheduler(
                warmup_epochs=curriculum_config.get("warmup_epochs", 2),
                easy_epochs=curriculum_config.get("easy_epochs", 5),
                hard_epochs=curriculum_config.get("hard_epochs", 10),
                finetune_epochs=curriculum_config.get("finetune_epochs", 3),
            )

    def train(
        self,
        train_dataset,
        val_dataset,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run SOTA training loop.
        """
        # Get number of classes from dataset
        num_classes = train_dataset.num_classes
        print(f"Number of classes: {num_classes}")

        # Build model
        self.model = self._build_model(num_classes)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Build optimizer
        batch_size = self.config.get("batch_size", 32)
        grad_accum = self.config.get("gradient_accumulation_steps", 1)

        # Build data loaders
        train_loader = self._build_train_loader(train_dataset, batch_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(len(train_loader))
        self.checkpoint_manager = self._build_checkpoint_manager()

        # Mixed precision
        use_amp = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if use_amp else None

        # Build loss function
        criterion = self._build_loss_function(num_classes)

        # Setup curriculum
        epochs = self.config.get("epochs", 10)
        self._setup_curriculum(epochs)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.get("early_stopping_patience", 5),
            min_delta=self.config.get("early_stopping_min_delta", 1e-4),
            mode="min",
        )

        # Training state
        best_val_loss = float("inf")
        best_recall_at_1 = 0.0
        best_epoch = 0

        print(f"\n{'='*60}")
        print(f"SOTA Training Configuration")
        print(f"{'='*60}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {batch_size * grad_accum}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Combined loss: {self.use_combined_loss}")
        print(f"  P-K sampling: {self.use_pk_sampling}")
        print(f"  Curriculum learning: {self.use_curriculum}")
        print(f"  Domain adaptation: {self.use_domain_adaptation}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # Update curriculum phase
            if self.curriculum_scheduler:
                self.curriculum_phase = self.curriculum_scheduler.get_phase(epoch)
                print(f"Curriculum phase: {self.curriculum_phase}")

            # Train one epoch
            train_metrics = self._train_epoch_sota(
                train_loader=train_loader,
                criterion=criterion,
                epoch=epoch,
                grad_accum=grad_accum,
                use_amp=use_amp,
                progress_callback=progress_callback,
            )

            # Validate with recall metrics
            val_metrics = self._validate_sota(val_loader, criterion)

            epoch_time = time.time() - epoch_start

            # Log metrics
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2%} | "
                f"R@1: {val_metrics['recall@1']:.2%} | "
                f"R@5: {val_metrics['recall@5']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )

            if self.use_combined_loss:
                print(
                    f"  Loss Components: "
                    f"ArcFace={train_metrics.get('arcface_loss', 0):.4f} | "
                    f"Triplet={train_metrics.get('triplet_loss', 0):.4f} | "
                    f"Domain={train_metrics.get('domain_loss', 0):.4f}"
                )

            # Track metrics
            self.metric_tracker.update({
                "epoch": epoch + 1,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Check for improvement
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_recall_at_1 = val_metrics["recall@1"]
                best_epoch = epoch + 1

                # Save best checkpoint
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    val_recall_at_1=val_metrics["recall@1"],
                    val_recall_at_5=val_metrics["recall@5"],
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                )

            # Regular checkpoint
            if (epoch + 1) % self.config.get("save_every_n_epochs", 1) == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    val_loss=val_metrics["loss"],
                    val_recall_at_1=val_metrics["recall@1"],
                    val_recall_at_5=val_metrics["recall@5"],
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                )

            # Early stopping check
            if early_stopping(val_metrics["loss"], epoch):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return {
            "epochs_trained": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_recall_at_1": best_recall_at_1,
            "final_metrics": {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "recall@1": val_metrics["recall@1"],
                "recall@5": val_metrics["recall@5"],
            },
            "metric_history": self.metric_tracker.get_history(),
        }

    def _train_epoch_sota(
        self,
        train_loader: DataLoader,
        criterion,
        epoch: int,
        grad_accum: int,
        use_amp: bool,
        progress_callback: Optional[Callable],
    ) -> dict:
        """Train one epoch with SOTA features."""
        self.model.train()

        total_loss = 0.0
        arcface_loss_total = 0.0
        triplet_loss_total = 0.0
        domain_loss_total = 0.0
        num_batches = len(train_loader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            domains = batch.get("domain")
            if domains is not None:
                domains = domains.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = self.model(images, labels)
                embeddings = outputs["embeddings"]

                if isinstance(criterion, CombinedProductLoss):
                    # Combined loss with components
                    losses = criterion(embeddings, labels, domains)
                    loss = losses["total"]
                    arcface_loss_total += losses.get("arcface", torch.tensor(0.0)).item()
                    triplet_loss_total += losses.get("triplet", torch.tensor(0.0)).item()
                    domain_loss_total += losses.get("domain", torch.tensor(0.0)).item()
                else:
                    # Standard loss
                    logits = outputs.get("logits")
                    if logits is not None:
                        loss = criterion(logits, labels)
                    else:
                        loss = criterion(embeddings, labels)

                loss = loss / grad_accum

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += loss.item() * grad_accum

            # Progress callback
            if progress_callback and batch_idx % 10 == 0:
                progress_callback(
                    epoch,
                    batch_idx,
                    num_batches,
                    {"train_loss": total_loss / (batch_idx + 1)},
                )

        metrics = {"loss": total_loss / num_batches}
        if self.use_combined_loss:
            metrics["arcface_loss"] = arcface_loss_total / num_batches
            metrics["triplet_loss"] = triplet_loss_total / num_batches
            metrics["domain_loss"] = domain_loss_total / num_batches

        return metrics

    @torch.no_grad()
    def _validate_sota(self, val_loader: DataLoader, criterion) -> dict:
        """Validate with recall@k metrics."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []

        for batch in val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images, labels)
            embeddings = outputs["embeddings"]
            logits = outputs.get("logits")

            # Collect embeddings for recall computation
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            if logits is not None:
                if isinstance(criterion, CombinedProductLoss):
                    losses = criterion(embeddings, labels)
                    loss = losses["total"]
                else:
                    loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
            else:
                loss = torch.tensor(0.0)
                preds = embeddings.argmax(dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Compute recall@k
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        recall_at_1 = self._compute_recall_at_k(all_embeddings, all_labels, k=1)
        recall_at_5 = self._compute_recall_at_k(all_embeddings, all_labels, k=5)
        recall_at_10 = self._compute_recall_at_k(all_embeddings, all_labels, k=10)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total if total > 0 else 0,
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
        }

    def _compute_recall_at_k(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        k: int,
    ) -> float:
        """Compute recall@k metric."""
        n_samples = embeddings.size(0)

        # Clamp k to max possible value (n_samples - 1 because we exclude self)
        k = min(k, n_samples - 1)
        if k <= 0:
            return 0.0

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Mask self-similarity
        sim_matrix.fill_diagonal_(-float('inf'))

        # Get top-k indices
        _, top_k_indices = sim_matrix.topk(k, dim=1)

        # Check if correct label in top-k
        correct = 0
        for i in range(len(labels)):
            query_label = labels[i]
            retrieved_labels = labels[top_k_indices[i]]
            if query_label in retrieved_labels:
                correct += 1

        return correct / len(labels) if len(labels) > 0 else 0.0
