"""
Custom Trained Model Wrapper.

Supports models trained with the OD training pipeline:
- RT-DETR (Real-Time Detection Transformer)
- D-FINE (Detection-aware FINE-grained matching)

These are "closed-vocabulary" models - they detect fixed classes
defined at training time, without requiring text prompts.

Key features:
- Downloads model checkpoints from Supabase Storage
- Caches locally for faster subsequent runs
- Uses HuggingFace transformers for inference
- **CRITICAL: Reads model_size from checkpoint to use correct architecture**
- Supports FP16 checkpoints (converted to FP32 for inference)
- Outputs normalized bounding boxes [{bbox, label, confidence}, ...]

Checkpoint Format (from OD Training Worker):
{
    "model_state_dict": {...},
    "config": {
        "training": {
            "model_type": "rt-detr",
            "model_size": "l",  # s/m/l -> r18vd/r50vd/r101vd
            ...
        },
        "dataset": {...}
    },
    "precision": "fp16",
    "inference_only": True
}
"""

import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import httpx
import torch
from PIL import Image
from loguru import logger

from .base import BaseModel
from config import config


class TrainedModel(BaseModel):
    """
    Wrapper for custom trained RT-DETR and D-FINE models.

    Downloads model weights from Supabase Storage on first load,
    caches locally for subsequent runs.

    IMPORTANT: This class reads model_size from checkpoint metadata
    to ensure the correct HuggingFace model architecture is used.
    """

    # Model name mapping - MUST match training worker
    RTDETR_MODEL_NAMES = {
        "s": "PekingU/rtdetr_r18vd",
        "small": "PekingU/rtdetr_r18vd",
        "m": "PekingU/rtdetr_r50vd",
        "medium": "PekingU/rtdetr_r50vd",
        "l": "PekingU/rtdetr_r101vd",
        "large": "PekingU/rtdetr_r101vd",
    }

    DFINE_MODEL_NAMES = {
        "s": "ustc-community/dfine-small-coco",
        "small": "ustc-community/dfine-small-coco",
        "m": "ustc-community/dfine-medium-coco",
        "medium": "ustc-community/dfine-medium-coco",
        "l": "ustc-community/dfine-large-coco",
        "large": "ustc-community/dfine-large-coco",
    }

    def __init__(
        self,
        checkpoint_url: str,
        architecture: str,
        classes: List[str],
        class_mapping: Dict[str, Dict[str, Any]],
        device: str = "cuda",
        cache_dir: str = "/tmp/trained_models",
    ):
        """
        Initialize trained model wrapper.

        Args:
            checkpoint_url: Supabase Storage URL for .pt checkpoint
            architecture: Model architecture (rt-detr, d-fine)
            classes: List of class names in index order
            class_mapping: Full class mapping from training {index: {id, name, color}}
            device: Device to run inference on (cuda/cpu)
            cache_dir: Directory to cache downloaded weights
        """
        super().__init__(device)
        self.checkpoint_url = checkpoint_url
        self.architecture = architecture.lower()
        self.classes = classes
        self.class_mapping = class_mapping
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._checkpoint_metadata: Optional[Dict] = None

    def _get_cache_path(self) -> str:
        """Generate local cache path from checkpoint URL."""
        # Use MD5 hash of URL for unique filename
        url_hash = hashlib.md5(self.checkpoint_url.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"trained_{url_hash}.pt")

    def _download_weights(self) -> str:
        """
        Download model weights from Supabase Storage.

        Returns:
            Local path to downloaded weights
        """
        local_path = self._get_cache_path()

        # Skip if already cached
        if os.path.exists(local_path):
            logger.info(f"Using cached trained model: {local_path}")
            return local_path

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Download weights
        logger.info(f"Downloading trained model from {self.checkpoint_url[:60]}...")

        try:
            with httpx.Client(timeout=300) as client:
                response = client.get(self.checkpoint_url)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(response.content)

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"Trained model downloaded: {local_path} ({file_size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to download trained model: {e}")
            # Clean up partial download
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

        return local_path

    def _extract_checkpoint_metadata(self, checkpoint: Dict) -> Tuple[str, str]:
        """
        Extract model_size and actual architecture from checkpoint.

        Returns:
            Tuple of (model_size, actual_architecture)
        """
        # Try to get from checkpoint config
        training_config = checkpoint.get("config", {}).get("training", {})

        model_size = training_config.get("model_size", "m")  # Default to medium
        actual_architecture = training_config.get("model_type", self.architecture)

        # Store metadata for logging
        self._checkpoint_metadata = {
            "model_size": model_size,
            "model_type": actual_architecture,
            "precision": checkpoint.get("precision", "fp32"),
            "inference_only": checkpoint.get("inference_only", False),
            "best_map": checkpoint.get("best_map"),
        }

        logger.info(f"Checkpoint metadata: model_size={model_size}, model_type={actual_architecture}")

        # Warn if architecture mismatch
        if actual_architecture.lower().replace("-", "") != self.architecture.replace("-", ""):
            logger.warning(
                f"Architecture mismatch! Requested: {self.architecture}, "
                f"Checkpoint was trained with: {actual_architecture}. "
                f"Using checkpoint's actual architecture."
            )

        return model_size, actual_architecture.lower()

    def _load_model(self) -> Any:
        """Load the appropriate model based on checkpoint metadata."""
        weights_path = self._download_weights()

        logger.info(f"Loading {self.architecture} trained model from {weights_path}...")

        # Load checkpoint to extract metadata
        checkpoint = torch.load(weights_path, map_location="cpu")  # Load to CPU first

        # Get actual model size and architecture from checkpoint
        model_size, actual_architecture = self._extract_checkpoint_metadata(checkpoint)

        # Route to appropriate loader based on ACTUAL architecture
        if actual_architecture in ["rt-detr", "rtdetr"]:
            return self._load_rtdetr(checkpoint, model_size)
        elif actual_architecture in ["d-fine", "dfine"]:
            return self._load_dfine(checkpoint, model_size)
        else:
            raise ValueError(f"Unsupported trained model architecture: {actual_architecture}")

    def _load_rtdetr(self, checkpoint: Dict, model_size: str) -> Any:
        """
        Load RT-DETR model using HuggingFace transformers.

        CRITICAL: Uses model_size from checkpoint to load correct architecture.
        """
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig

        # Get correct model name based on model_size
        model_name = self.RTDETR_MODEL_NAMES.get(model_size.lower(), self.RTDETR_MODEL_NAMES["m"])
        logger.info(f"Loading RT-DETR architecture: {model_name} (size: {model_size})")

        # Load config and adjust num_labels
        model_config = RTDetrConfig.from_pretrained(model_name)
        model_config.num_labels = len(self.classes)

        # Ensure valid prior_prob
        if not hasattr(model_config, 'prior_prob') or model_config.prior_prob <= 0 or model_config.prior_prob >= 1:
            model_config.prior_prob = 0.01

        # Initialize model FROM PRETRAINED (not random!)
        # This ensures architecture weights are properly initialized
        model = RTDetrForObjectDetection.from_pretrained(
            model_name,
            config=model_config,
            ignore_mismatched_sizes=True,  # Allow num_labels mismatch
        )

        # Load trained weights
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Handle FP16 checkpoint - convert back to FP32 for inference
        if checkpoint.get("precision") == "fp16":
            logger.info("Converting FP16 checkpoint to FP32 for inference...")
            model_state = {
                k: v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v
                for k, v in model_state.items()
            }

        # Load state dict with validation
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
            logger.debug(f"Missing keys: {missing_keys[:10]}...")  # Log first 10
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            logger.debug(f"Unexpected keys: {unexpected_keys[:10]}...")

        model.eval()
        model = model.to(self.device)

        # Load processor for pre/post-processing (same for all RT-DETR variants)
        self._processor = RTDetrImageProcessor.from_pretrained(model_name)

        logger.info(f"RT-DETR trained model loaded on {self.device} (size: {model_size})")

        return model

    def _load_dfine(self, checkpoint: Dict, model_size: str) -> Any:
        """
        Load D-FINE model using HuggingFace transformers.

        CRITICAL: Uses model_size from checkpoint to load correct architecture.
        Falls back to RT-DETR if D-FINE is not available.
        """
        from transformers import RTDetrImageProcessor

        # Get correct model name based on model_size
        model_name = self.DFINE_MODEL_NAMES.get(model_size.lower(), self.DFINE_MODEL_NAMES["l"])
        logger.info(f"Loading D-FINE architecture: {model_name} (size: {model_size})")

        try:
            from transformers import AutoModelForObjectDetection, AutoConfig

            # Load D-FINE config
            model_config = AutoConfig.from_pretrained(model_name)
            model_config.num_labels = len(self.classes)

            # Initialize model FROM PRETRAINED
            model = AutoModelForObjectDetection.from_pretrained(
                model_name,
                config=model_config,
                ignore_mismatched_sizes=True,
            )

            logger.info(f"D-FINE model loaded from {model_name}")

        except Exception as e:
            logger.warning(f"D-FINE not available from HuggingFace: {e}")
            logger.warning("This may indicate training used RT-DETR fallback. Loading RT-DETR instead.")

            # Fallback to RT-DETR with same size
            from transformers import RTDetrForObjectDetection, RTDetrConfig

            fallback_model_name = self.RTDETR_MODEL_NAMES.get(model_size.lower(), self.RTDETR_MODEL_NAMES["l"])
            logger.info(f"Falling back to RT-DETR: {fallback_model_name}")

            model_config = RTDetrConfig.from_pretrained(fallback_model_name)
            model_config.num_labels = len(self.classes)

            if not hasattr(model_config, 'prior_prob') or model_config.prior_prob <= 0 or model_config.prior_prob >= 1:
                model_config.prior_prob = 0.01

            model = RTDetrForObjectDetection.from_pretrained(
                fallback_model_name,
                config=model_config,
                ignore_mismatched_sizes=True,
            )

        # Load trained weights
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Handle FP16 checkpoint - convert back to FP32 for inference
        if checkpoint.get("precision") == "fp16":
            logger.info("Converting FP16 checkpoint to FP32 for inference...")
            model_state = {
                k: v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v
                for k, v in model_state.items()
            }

        # Load state dict with validation
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

        model.eval()
        model = model.to(self.device)

        # Load processor (RT-DETR processor works for both)
        rtdetr_model_name = self.RTDETR_MODEL_NAMES.get(model_size.lower(), self.RTDETR_MODEL_NAMES["m"])
        self._processor = RTDetrImageProcessor.from_pretrained(rtdetr_model_name)

        logger.info(f"D-FINE trained model loaded on {self.device} (size: {model_size})")

        return model

    def predict(
        self,
        image_url: str,
        text_prompt: str = "",  # Ignored for closed-vocab
        box_threshold: float = 0.3,
        **kwargs
    ) -> List[Dict]:
        """
        Run inference on image.

        Returns normalized predictions (same format as other models):
        [
          {
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
            "label": "product",
            "confidence": 0.95
          },
          ...
        ]
        """
        # Lazy load model
        if self._model is None:
            self._model = self._load_model()

        # Download image
        image = self.download_image(image_url)
        img_width, img_height = image.size

        # Preprocess image
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process predictions
        target_sizes = torch.tensor([[img_height, img_width]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]

        # Extract predictions
        predictions = []
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()  # [x1, y1, x2, y2] in pixels

        for score, label_idx, box in zip(scores, labels, boxes):
            if score < box_threshold:
                continue

            x1, y1, x2, y2 = box

            # Normalize coordinates to [0, 1] and clamp to valid range
            norm_x = max(0.0, min(1.0, float(x1 / img_width)))
            norm_y = max(0.0, min(1.0, float(y1 / img_height)))
            norm_width = max(0.0, min(1.0, float((x2 - x1) / img_width)))
            norm_height = max(0.0, min(1.0, float((y2 - y1) / img_height)))

            # Ensure bbox stays within bounds
            if norm_x + norm_width > 1.0:
                norm_width = 1.0 - norm_x
            if norm_y + norm_height > 1.0:
                norm_height = 1.0 - norm_y

            # Map class index to name
            label = self.classes[int(label_idx)] if int(label_idx) < len(self.classes) else f"class_{int(label_idx)}"

            predictions.append({
                "bbox": {
                    "x": norm_x,
                    "y": norm_y,
                    "width": norm_width,
                    "height": norm_height,
                },
                "label": label,
                "confidence": float(score)
            })

        logger.info(f"Trained model detected {len(predictions)} objects")

        return predictions
