"""
Roboflow Trained Model Wrapper.

Supports models trained on Roboflow platform:
- YOLOv8 (all variants: n, s, m, l, x)
- RF-DETR / RT-DETR (Roboflow's Detection Transformer)

These are "closed-vocabulary" models - they detect fixed classes
defined at training time, without requiring text prompts.

Key differences from open-vocabulary models (Grounding DINO):
- No text_prompt parameter needed
- Classes are fixed at training time
- Typically faster inference
- Outputs same format: [{bbox, label, confidence}, ...]
"""

import os
import hashlib
from typing import Any, Optional
import httpx
from PIL import Image
from loguru import logger

from .base import BaseModel
from config import config


class RoboflowModel(BaseModel):
    """
    Wrapper for Roboflow-trained YOLO/DETR models.

    Downloads model weights from Supabase Storage on first load,
    caches locally for subsequent runs.
    """

    def __init__(
        self,
        checkpoint_url: str,
        architecture: str,
        classes: list[str],
        device: str = "cuda",
        cache_dir: str = "/tmp/roboflow_models",
    ):
        """
        Initialize Roboflow model wrapper.

        Args:
            checkpoint_url: Supabase Storage URL for .pt checkpoint
            architecture: Model architecture (yolov8m, rf-detr, etc.)
            classes: List of class names this model detects
            device: Device to run inference on (cuda/cpu)
            cache_dir: Directory to cache downloaded weights
        """
        super().__init__(device)
        self.checkpoint_url = checkpoint_url
        self.architecture = architecture
        self.classes = classes
        self.cache_dir = cache_dir
        self._model = None

    def _get_cache_path(self) -> str:
        """Generate local cache path from checkpoint URL."""
        # Use MD5 hash of URL for unique filename
        url_hash = hashlib.md5(self.checkpoint_url.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"roboflow_{url_hash}.pt")

    def _download_weights(self) -> str:
        """
        Download model weights from Supabase Storage.

        Returns:
            Local path to downloaded weights
        """
        local_path = self._get_cache_path()

        # Skip if already cached
        if os.path.exists(local_path):
            logger.info(f"Using cached model: {local_path}")
            return local_path

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Download weights
        logger.info(f"Downloading model from {self.checkpoint_url[:60]}...")

        try:
            with httpx.Client(timeout=300) as client:
                response = client.get(self.checkpoint_url)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    f.write(response.content)

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"Model downloaded: {local_path} ({file_size_mb:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Clean up partial download
            if os.path.exists(local_path):
                os.remove(local_path)
            raise

        return local_path

    def _load_model(self) -> Any:
        """Load the appropriate model based on architecture."""
        weights_path = self._download_weights()

        logger.info(f"Loading {self.architecture} model from {weights_path}...")

        if self.architecture.startswith("yolov8") or self.architecture.startswith("yolov"):
            return self._load_yolo(weights_path)
        elif self.architecture in ["rf-detr", "rt-detr", "rtdetr"]:
            return self._load_rtdetr(weights_path)
        else:
            # Default to YOLO for unknown architectures
            logger.warning(f"Unknown architecture '{self.architecture}', attempting YOLO loader")
            return self._load_yolo(weights_path)

    def _load_yolo(self, weights_path: str) -> Any:
        """Load YOLOv8 model using ultralytics."""
        from ultralytics import YOLO

        model = YOLO(weights_path)

        # Move to device if CUDA available
        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                model.to(self.device)
                logger.info(f"YOLOv8 model loaded on CUDA")
            else:
                logger.warning("CUDA not available, using CPU")
        else:
            logger.info(f"YOLOv8 model loaded on {self.device}")

        return model

    def _load_rtdetr(self, weights_path: str) -> Any:
        """Load RT-DETR model using ultralytics."""
        from ultralytics import RTDETR

        model = RTDETR(weights_path)

        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                model.to(self.device)
                logger.info(f"RT-DETR model loaded on CUDA")
            else:
                logger.warning("CUDA not available, using CPU")
        else:
            logger.info(f"RT-DETR model loaded on {self.device}")

        return model

    def predict(
        self,
        image_url: str,
        text_prompt: str = "",  # Ignored - closed vocabulary
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,  # Ignored
        **kwargs,
    ) -> list[dict]:
        """
        Run inference on an image.

        Args:
            image_url: URL of the image to process
            text_prompt: Ignored (closed-vocabulary model)
            box_threshold: Confidence threshold for detections
            text_threshold: Ignored (closed-vocabulary model)

        Returns:
            List of predictions with bbox, label, confidence
            Same format as Grounding DINO for compatibility
        """
        # Download and prepare image
        image = self.download_image(image_url)
        img_width, img_height = image.size

        logger.debug(f"Running {self.architecture} inference on {img_width}x{img_height} image")

        # Run inference
        results = self.model(
            image,
            conf=box_threshold,
            verbose=False,
        )

        # Process results
        predictions = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_idx = int(boxes.cls[i].cpu().numpy())

                # Map class index to name from our classes list
                if class_idx < len(self.classes):
                    label = self.classes[class_idx]
                elif hasattr(self.model, 'names') and class_idx in self.model.names:
                    # Fallback to model's built-in names
                    label = self.model.names[class_idx]
                else:
                    label = f"class_{class_idx}"

                # Convert to normalized format (same as Grounding DINO)
                predictions.append({
                    "bbox": {
                        "x": float(x1 / img_width),
                        "y": float(y1 / img_height),
                        "width": float((x2 - x1) / img_width),
                        "height": float((y2 - y1) / img_height),
                    },
                    "label": label,
                    "confidence": confidence,
                })

        logger.debug(f"Roboflow model found {len(predictions)} objects")
        return predictions

    def predict_with_nms(
        self,
        image_url: str,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Run inference with NMS.

        Note: YOLO/RT-DETR already applies NMS internally during inference,
        but this method provides an additional pass for consistency with
        other models in the system.
        """
        # YOLO already applies NMS internally, so just run regular predict
        # The nms_threshold here would be for class-agnostic NMS which
        # is handled at the API level
        return self.predict(
            image_url=image_url,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
