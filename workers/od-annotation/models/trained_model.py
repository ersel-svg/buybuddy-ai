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
- Supports FP16 checkpoints (converted to FP32 for inference)
- Outputs normalized bounding boxes [{bbox, label, confidence}, ...]
"""

import os
import hashlib
from typing import Any, Dict, List
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
    """

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

    def _load_model(self) -> Any:
        """Load the appropriate model based on architecture."""
        weights_path = self._download_weights()

        logger.info(f"Loading {self.architecture} trained model from {weights_path}...")

        if self.architecture in ["rt-detr", "rtdetr"]:
            return self._load_rtdetr(weights_path)
        elif self.architecture in ["d-fine", "dfine"]:
            return self._load_dfine(weights_path)
        else:
            raise ValueError(f"Unsupported trained model architecture: {self.architecture}")

    def _load_rtdetr(self, weights_path: str) -> Any:
        """Load RT-DETR model using HuggingFace transformers."""
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Create model config with correct num_labels
        # Use a base model config as template
        config = RTDetrConfig.from_pretrained("PekingU/rtdetr_r50vd")
        config.num_labels = len(self.classes)

        # Initialize model architecture
        model = RTDetrForObjectDetection(config)

        # Load trained weights
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Handle FP16 checkpoint - convert back to FP32 for inference
        if checkpoint.get("precision") == "fp16":
            logger.info("Converting FP16 checkpoint to FP32 for inference...")
            model_state = {
                k: v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v
                for k, v in model_state.items()
            }

        # Load state dict
        model.load_state_dict(model_state, strict=False)
        model.eval()

        # Move to device
        model = model.to(self.device)

        # Load processor for pre/post-processing
        self._processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

        logger.info(f"RT-DETR trained model loaded on {self.device}")

        return model

    def _load_dfine(self, weights_path: str) -> Any:
        """Load D-FINE model using HuggingFace transformers."""
        from transformers import AutoModelForObjectDetection, AutoConfig, RTDetrImageProcessor

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)

        try:
            # Try to load D-FINE config from HuggingFace
            config = AutoConfig.from_pretrained("ustc-community/dfine-large-coco")
            config.num_labels = len(self.classes)

            # Initialize model architecture
            model = AutoModelForObjectDetection.from_config(config)

        except Exception as e:
            logger.warning(f"D-FINE not available from HuggingFace: {e}")
            logger.info("Falling back to RT-DETR architecture for D-FINE checkpoint...")

            # Fallback to RT-DETR (D-FINE uses similar architecture)
            from transformers import RTDetrForObjectDetection, RTDetrConfig

            config = RTDetrConfig.from_pretrained("PekingU/rtdetr_r50vd")
            config.num_labels = len(self.classes)

            model = RTDetrForObjectDetection(config)

        # Load trained weights
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Handle FP16 checkpoint - convert back to FP32 for inference
        if checkpoint.get("precision") == "fp16":
            logger.info("Converting FP16 checkpoint to FP32 for inference...")
            model_state = {
                k: v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v
                for k, v in model_state.items()
            }

        # Load state dict (strict=False to allow architecture differences)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        # Move to device
        model = model.to(self.device)

        # Load processor for pre/post-processing
        self._processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")

        logger.info(f"D-FINE trained model loaded on {self.device}")

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
            outputs = self.model(**inputs)

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
