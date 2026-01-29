"""
Workflow Model Loader Service

Centralized service for loading models from pretrained and trained sources.
Handles:
- od_trained_models (RT-DETR, D-FINE, YOLO-NAS)
- cls_trained_models (ViT, ConvNeXt, EfficientNet, Swin)
- trained_models (Fine-tuned embeddings)
- wf_pretrained_models (YOLO, DINOv2, CLIP)
"""

import os
import logging
import tempfile
from typing import Any, Optional
from functools import lru_cache
from dataclasses import dataclass

import httpx

from services.supabase import supabase_service

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: dict[str, Any] = {}


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    id: str
    name: str
    model_type: str
    source: str  # pretrained or trained
    checkpoint_url: Optional[str] = None
    class_mapping: Optional[dict] = None
    embedding_dim: Optional[int] = None
    config: Optional[dict] = None


class ModelLoader:
    """
    Singleton model loader with caching.

    Provides unified interface for loading models from various sources:
    - Pretrained: wf_pretrained_models (YOLO, DINOv2, CLIP, etc.)
    - Trained Detection: od_trained_models (RT-DETR, D-FINE, YOLO-NAS)
    - Trained Classification: cls_trained_models (ViT, ConvNeXt, etc.)
    - Trained Embedding: trained_models (fine-tuned DINOv2, CLIP)
    """

    _instance: Optional["ModelLoader"] = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._download_cache: dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def clear_cache(self):
        """Clear all cached models."""
        global _model_cache
        _model_cache.clear()
        self._download_cache.clear()
        logger.info("Model cache cleared")

    # =========================================================================
    # Detection Models
    # =========================================================================

    async def get_detection_model_info(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> Optional[ModelInfo]:
        """Get detection model info from database."""
        try:
            if model_source == "pretrained":
                result = supabase_service.client.table("wf_pretrained_models").select(
                    "*"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                # Use model_id as model_type for worker (yolo11n, yolov8n, etc.)
                # Worker expects model_type to be the actual model identifier
                # Handle None classes gracefully (e.g., Grounding DINO has dynamic classes)
                classes = result.data.get("classes")
                class_mapping = {i: c for i, c in enumerate(classes)} if classes else None

                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=result.data["id"],  # Use model ID for worker compatibility
                    source="pretrained",
                    checkpoint_url=result.data.get("model_path"),
                    class_mapping=class_mapping,
                    config=result.data.get("default_config", {}),
                )
            else:
                # Try od_trained_models first
                try:
                    result = supabase_service.client.table("od_trained_models").select(
                        "*"
                    ).eq("id", model_id).single().execute()

                    if result.data:
                        return ModelInfo(
                            id=result.data["id"],
                            name=result.data["name"],
                            model_type=result.data.get("model_type", "yolo"),
                            source="trained",
                            checkpoint_url=result.data.get("checkpoint_url"),
                            class_mapping=result.data.get("class_mapping", {}),
                            config={
                                "model_size": result.data.get("model_size"),
                                "map": result.data.get("map"),
                                "map_50": result.data.get("map_50"),
                            },
                        )
                except Exception:
                    pass  # Model not in od_trained_models, try next table

                # Try od_roboflow_models (e.g., Slot Detection)
                try:
                    result = supabase_service.client.table("od_roboflow_models").select(
                        "*"
                    ).eq("id", model_id).single().execute()

                    if result.data:
                        # Convert classes array to class_mapping dict
                        classes = result.data.get("classes", [])
                        class_mapping = {i: c for i, c in enumerate(classes)} if classes else {}

                        return ModelInfo(
                            id=result.data["id"],
                            name=result.data.get("display_name") or result.data["name"],
                            model_type=result.data.get("architecture", "yolov8"),
                            source="trained",
                            checkpoint_url=result.data.get("checkpoint_url"),
                            class_mapping=class_mapping,
                            config={
                                "architecture": result.data.get("architecture"),
                                "map": result.data.get("map"),
                                "map_50": result.data.get("map_50"),
                            },
                        )
                except Exception:
                    pass  # Model not in od_roboflow_models either

                return None
        except Exception as e:
            logger.error(f"Failed to get detection model info: {e}")
            return None

    async def load_detection_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> tuple[Any, ModelInfo]:
        """
        Load detection model.

        Returns: (model, model_info)
        """
        cache_key = f"detection:{model_source}:{model_id}"

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        model_info = await self.get_detection_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Detection model not found: {model_id}")

        try:
            from ultralytics import YOLO

            if model_source == "pretrained":
                # Load pretrained YOLO model
                model_path = model_info.checkpoint_url or "yolo11n.pt"
                model = YOLO(model_path)
            else:
                # Load trained model from checkpoint
                if not model_info.checkpoint_url:
                    raise ValueError("Trained model has no checkpoint URL")

                # Download checkpoint
                local_path = await self._download_checkpoint(
                    model_info.checkpoint_url,
                    f"detection_{model_id}"
                )

                # Load based on model type
                model_type = model_info.model_type.lower()

                if "detr" in model_type or "d-fine" in model_type:
                    # RT-DETR or D-FINE models
                    # These use Ultralytics format or custom loaders
                    logger.info(f"Loading {model_type} model from {local_path}")
                    model = YOLO(local_path)
                else:
                    # Standard YOLO format
                    model = YOLO(local_path)

            _model_cache[cache_key] = (model, model_info)
            return model, model_info

        except ImportError:
            raise RuntimeError("ultralytics package not installed. Run: pip install ultralytics")

    # =========================================================================
    # Classification Models
    # =========================================================================

    async def get_classification_model_info(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> Optional[ModelInfo]:
        """Get classification model info from database."""
        try:
            if model_source == "pretrained":
                result = supabase_service.client.table("wf_pretrained_models").select(
                    "*"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                # Use model_id as model_type for worker (vit-base, convnext-base, etc.)
                classes = result.data.get("classes") or []
                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=result.data["id"],  # Use model ID for worker compatibility
                    source="pretrained",
                    checkpoint_url=result.data.get("model_path"),
                    class_mapping={i: c for i, c in enumerate(classes)},
                    config=result.data.get("default_config", {}),
                )
            else:
                result = supabase_service.client.table("cls_trained_models").select(
                    "*"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=result.data.get("model_type", "vit"),
                    source="trained",
                    checkpoint_url=result.data.get("checkpoint_url"),
                    class_mapping=result.data.get("class_mapping", {}),
                    config={
                        "model_size": result.data.get("model_size"),
                        "accuracy": result.data.get("test_accuracy"),
                        "num_classes": result.data.get("class_count"),
                    },
                )
        except Exception as e:
            logger.error(f"Failed to get classification model info: {e}")
            return None

    async def load_classification_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> tuple[Any, Any, ModelInfo]:
        """
        Load classification model and processor.

        Returns: (model, processor, model_info)
        """
        cache_key = f"classification:{model_source}:{model_id}"

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        model_info = await self.get_classification_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Classification model not found: {model_id}")

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            # Map model types to HuggingFace model names
            base_models = {
                "vit": "google/vit-base-patch16-224",
                "vit-tiny": "WinKawaks/vit-tiny-patch16-224",
                "vit-small": "WinKawaks/vit-small-patch16-224",
                "vit-base": "google/vit-base-patch16-224",
                "vit-large": "google/vit-large-patch16-224",
                "convnext": "facebook/convnext-base-224",
                "convnext-tiny": "facebook/convnext-tiny-224",
                "convnext-small": "facebook/convnext-small-224",
                "convnext-base": "facebook/convnext-base-224",
                "convnext-large": "facebook/convnext-large-224",
                "swin": "microsoft/swin-base-patch4-window7-224",
                "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
                "swin-small": "microsoft/swin-small-patch4-window7-224",
                "swin-base": "microsoft/swin-base-patch4-window7-224",
                "efficientnet": "google/efficientnet-b0",
                "efficientnet-b0": "google/efficientnet-b0",
                "efficientnet-b1": "google/efficientnet-b1",
                "efficientnet-b2": "google/efficientnet-b2",
                "efficientnet-b3": "google/efficientnet-b3",
            }

            if model_source == "pretrained":
                model_path = model_info.checkpoint_url or "google/vit-base-patch16-224"
                processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModelForImageClassification.from_pretrained(model_path)
            else:
                # For trained models, load base architecture then weights
                model_type = model_info.model_type.lower()
                base_model = base_models.get(model_type, "google/vit-base-patch16-224")

                processor = AutoImageProcessor.from_pretrained(base_model)

                # Load with custom number of classes
                num_classes = model_info.config.get("num_classes") if model_info.config else None
                if num_classes:
                    model = AutoModelForImageClassification.from_pretrained(
                        base_model,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    model = AutoModelForImageClassification.from_pretrained(base_model)

                # Load trained weights if checkpoint exists
                if model_info.checkpoint_url:
                    local_path = await self._download_checkpoint(
                        model_info.checkpoint_url,
                        f"classification_{model_id}"
                    )
                    state_dict = torch.load(local_path, map_location="cpu")
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded trained weights from {local_path}")

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            _model_cache[cache_key] = (model, processor, model_info)
            return model, processor, model_info

        except ImportError:
            raise RuntimeError("transformers package not installed")

    # =========================================================================
    # Embedding Models
    # =========================================================================

    async def get_embedding_model_info(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> Optional[ModelInfo]:
        """Get embedding model info from database."""
        try:
            if model_source == "pretrained":
                result = supabase_service.client.table("wf_pretrained_models").select(
                    "*"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                # Use model_id as model_type for worker (dinov2-base, clip-vit-b-32, etc.)
                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=result.data["id"],  # Use model ID for worker compatibility
                    source="pretrained",
                    checkpoint_url=result.data.get("model_path"),
                    embedding_dim=result.data.get("embedding_dim", 768),
                    config=result.data.get("default_config", {}),
                )
            else:
                # trained_models table links to training_checkpoints for checkpoint_url
                result = supabase_service.client.table("trained_models").select(
                    "*, embedding_model:embedding_models(*), checkpoint:training_checkpoints(checkpoint_url)"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                embedding_model = result.data.get("embedding_model", {}) or {}
                checkpoint = result.data.get("checkpoint", {}) or {}

                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=embedding_model.get("model_family", "dinov2"),
                    source="trained",
                    checkpoint_url=checkpoint.get("checkpoint_url"),  # From training_checkpoints
                    embedding_dim=embedding_model.get("embedding_dim", 768),
                    config={
                        "model_family": embedding_model.get("model_family"),
                        "recall_at_1": result.data.get("test_metrics", {}).get("recall_at_1"),
                    },
                )
        except Exception as e:
            logger.error(f"Failed to get embedding model info: {e}")
            return None

    async def load_embedding_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> tuple[Any, Any, ModelInfo]:
        """
        Load embedding model and processor.

        Returns: (model, processor, model_info)
        """
        cache_key = f"embedding:{model_source}:{model_id}"

        if cache_key in _model_cache:
            return _model_cache[cache_key]

        model_info = await self.get_embedding_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Embedding model not found: {model_id}")

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            # Map model types to HuggingFace model names
            base_models = {
                "dinov2": "facebook/dinov2-base",
                "dinov2-small": "facebook/dinov2-small",
                "dinov2-base": "facebook/dinov2-base",
                "dinov2-large": "facebook/dinov2-large",
                "dinov2-giant": "facebook/dinov2-giant",
                "clip": "openai/clip-vit-base-patch32",
                "clip-vit-b-32": "openai/clip-vit-base-patch32",
                "clip-vit-b-16": "openai/clip-vit-base-patch16",
                "clip-vit-l-14": "openai/clip-vit-large-patch14",
                "siglip": "google/siglip-base-patch16-224",
            }

            model_type = model_info.model_type.lower()
            model_path = model_info.checkpoint_url or base_models.get(model_type, "facebook/dinov2-base")

            # Handle CLIP separately
            if "clip" in model_type.lower():
                from transformers import CLIPProcessor, CLIPModel
                processor = CLIPProcessor.from_pretrained(model_path)
                model = CLIPModel.from_pretrained(model_path)
            else:
                processor = AutoImageProcessor.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            _model_cache[cache_key] = (model, processor, model_info)
            return model, processor, model_info

        except ImportError:
            raise RuntimeError("transformers package not installed")

    # =========================================================================
    # Segmentation Models (SAM, SAM2)
    # =========================================================================

    async def get_segmentation_model_info(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> Optional[ModelInfo]:
        """Get segmentation model info from database."""
        try:
            if model_source == "pretrained":
                result = supabase_service.client.table("wf_pretrained_models").select(
                    "*"
                ).eq("id", model_id).single().execute()

                if not result.data:
                    return None

                return ModelInfo(
                    id=result.data["id"],
                    name=result.data["name"],
                    model_type=result.data["id"],  # Use model ID (sam2-base, sam2-large, etc.)
                    source="pretrained",
                    checkpoint_url=result.data.get("model_path"),
                    config=result.data.get("default_config", {}),
                )
            else:
                # No trained segmentation models for now
                return None
        except Exception as e:
            logger.error(f"Failed to get segmentation model info: {e}")
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _download_checkpoint(self, url: str, prefix: str) -> str:
        """Download checkpoint from URL to local temp file."""
        if url in self._download_cache:
            if os.path.exists(self._download_cache[url]):
                return self._download_cache[url]

        logger.info(f"Downloading checkpoint from {url}")

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Determine file extension
            ext = ".pt"
            if url.endswith(".onnx"):
                ext = ".onnx"
            elif url.endswith(".pth"):
                ext = ".pth"

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                prefix=f"{prefix}_",
                suffix=ext,
                delete=False
            ) as f:
                f.write(response.content)
                local_path = f.name

        self._download_cache[url] = local_path
        logger.info(f"Downloaded checkpoint to {local_path}")
        return local_path

    async def list_available_models(
        self,
        model_type: Optional[str] = None,
        source: Optional[str] = None,
        include_inactive: bool = False,
    ) -> list[dict]:
        """
        List all available models across all sources.

        Returns flattened list for UI dropdowns.
        """
        models = []

        # Fetch pretrained models
        if source is None or source == "pretrained":
            query = supabase_service.client.table("wf_pretrained_models").select("*")
            if model_type:
                query = query.eq("model_type", model_type)
            if not include_inactive:
                query = query.eq("is_active", True)

            result = query.order("name").execute()

            for m in result.data or []:
                models.append({
                    "id": m["id"],
                    "name": m["name"],
                    "model_type": m.get("model_type", "unknown"),
                    "category": m.get("model_type", "detection"),
                    "source": "pretrained",
                    "provider": m.get("source", "ultralytics"),
                    "is_active": m.get("is_active", True),
                    "is_default": True,
                    "embedding_dim": m.get("embedding_dim"),
                    "class_count": m.get("class_count"),
                    "created_at": m.get("created_at"),
                })

        # Fetch trained detection models
        if source is None or source == "trained":
            if model_type is None or model_type == "detection":
                query = supabase_service.client.table("od_trained_models").select("*")
                if not include_inactive:
                    query = query.or_("is_active.eq.true,is_default.eq.true")

                result = query.order("created_at", desc=True).execute()

                for m in result.data or []:
                    models.append({
                        "id": m["id"],
                        "name": m["name"],
                        "model_type": m.get("model_type", "rt-detr"),
                        "category": "detection",
                        "source": "trained",
                        "provider": m.get("model_type", "rt-detr"),
                        "is_active": m.get("is_active", False),
                        "is_default": m.get("is_default", False),
                        "metrics": {
                            "map": m.get("map"),
                            "map_50": m.get("map_50"),
                        },
                        "class_count": m.get("class_count"),
                        "created_at": m.get("created_at"),
                    })

                # Fetch Roboflow trained models (e.g., Slot Detection)
                rf_query = supabase_service.client.table("od_roboflow_models").select("*")
                if not include_inactive:
                    rf_query = rf_query.or_("is_active.eq.true,is_default.eq.true")

                rf_result = rf_query.order("created_at", desc=True).execute()

                for m in rf_result.data or []:
                    models.append({
                        "id": m["id"],
                        "name": m.get("display_name") or m["name"],
                        "model_type": m.get("architecture", "yolov8"),
                        "category": "detection",
                        "source": "trained",
                        "provider": m.get("architecture", "yolov8"),
                        "is_active": m.get("is_active", False),
                        "is_default": m.get("is_default", False),
                        "metrics": {
                            "map": m.get("map"),
                            "map_50": m.get("map_50"),
                        },
                        "class_count": m.get("class_count"),
                        "created_at": m.get("created_at"),
                    })

            # Fetch trained classification models
            if model_type is None or model_type == "classification":
                query = supabase_service.client.table("cls_trained_models").select("*")
                if not include_inactive:
                    query = query.or_("is_active.eq.true,is_default.eq.true")

                result = query.order("created_at", desc=True).execute()

                for m in result.data or []:
                    models.append({
                        "id": m["id"],
                        "name": m["name"],
                        "model_type": m.get("model_type", "vit"),
                        "category": "classification",
                        "source": "trained",
                        "provider": m.get("model_type", "vit"),
                        "is_active": m.get("is_active", False),
                        "is_default": m.get("is_default", False),
                        "metrics": {
                            "accuracy": m.get("test_accuracy"),
                        },
                        "class_count": m.get("class_count"),
                        "created_at": m.get("created_at"),
                    })

            # Fetch trained embedding models
            if model_type is None or model_type == "embedding":
                query = supabase_service.client.table("trained_models").select(
                    "*, embedding_model:embedding_models(model_family, embedding_dim)"
                )
                if not include_inactive:
                    query = query.or_("is_active.eq.true,is_default.eq.true")

                result = query.order("created_at", desc=True).execute()

                for m in result.data or []:
                    emb_info = m.get("embedding_model", {}) or {}
                    test_metrics = m.get("test_metrics", {}) or {}

                    models.append({
                        "id": m["id"],
                        "name": m["name"],
                        "model_type": emb_info.get("model_family", "dinov2"),
                        "category": "embedding",
                        "source": "trained",
                        "provider": emb_info.get("model_family", "dinov2"),
                        "is_active": m.get("is_active", False),
                        "is_default": m.get("is_default", False),
                        "embedding_dim": emb_info.get("embedding_dim"),
                        "metrics": {
                            "recall_at_1": test_metrics.get("recall_at_1") if isinstance(test_metrics, dict) else None,
                        },
                        "created_at": m.get("created_at"),
                    })

        return models


# Singleton accessor
def get_model_loader() -> ModelLoader:
    """Get the singleton model loader instance."""
    return ModelLoader.get_instance()
