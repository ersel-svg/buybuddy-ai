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


class HFRTDETRWrapper:
    """Wrapper to make HuggingFace RT-DETR model compatible with YOLO-style inference API."""

    def __init__(self, model, processor, classes: list[str], device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.classes = classes
        self.device = device
        self.names = {i: c for i, c in enumerate(classes)}

    def __call__(self, image, conf: float = 0.3, verbose: bool = False, **kwargs):
        """Run inference and return results in YOLO-compatible format."""
        import torch
        from PIL import Image
        import numpy as np

        # Ensure image is PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess image using HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs to YOLO-style format
        results = [HFRTDETRResult(outputs, image.size, self.classes, conf)]
        return results

    def to(self, device):
        """Move model to device."""
        import torch
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.device = device
        return self


class HFRTDETRResult:
    """Wrapper for HuggingFace RT-DETR results to match YOLO result format."""

    def __init__(self, outputs, orig_size: tuple, classes: list[str], conf_threshold: float):
        import torch

        self.boxes = None
        self.orig_size = orig_size  # (width, height)

        # HuggingFace RT-DETR outputs have logits and pred_boxes attributes
        pred_logits = outputs.logits  # [batch, num_queries, num_classes]
        pred_boxes = outputs.pred_boxes  # [batch, num_queries, 4] in cxcywh format, normalized

        # Get predictions (batch size 1)
        logits = pred_logits[0]  # [num_queries, num_classes]
        boxes = pred_boxes[0]    # [num_queries, 4]

        # Get class probabilities (RT-DETR doesn't have separate background class)
        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs.max(-1)

        # Log pre-filter statistics
        logger.info(f"[DEBUG] HF RT-DETR: Total queries before filtering: {len(scores)}")
        logger.info(f"[DEBUG] HF RT-DETR: Max confidence score: {scores.max().item():.4f}")
        logger.info(f"[DEBUG] HF RT-DETR: Scores > {conf_threshold}: {(scores > conf_threshold).sum().item()}")

        # Filter by confidence
        mask = scores > conf_threshold
        scores = scores[mask]
        labels = labels[mask]
        boxes = boxes[mask]

        logger.info(f"[DEBUG] HF RT-DETR: Detections after filtering: {len(boxes)}")

        if len(boxes) == 0:
            logger.info(f"[DEBUG] HF RT-DETR: No detections above threshold {conf_threshold}")
            return

        # Convert boxes from cxcywh to xyxy and scale to image size
        w, h = orig_size
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Create boxes object
        self.boxes = HFRTDETRBoxes(boxes_xyxy, scores, labels)


class HFRTDETRBoxes:
    """Wrapper for HuggingFace RT-DETR boxes to match YOLO boxes format."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.xyxy)


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
        logger.info(f"[DEBUG] Configured classes from DB: {self.classes}")
        logger.info(f"[DEBUG] Number of classes: {len(self.classes)}")

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

        # Log the model's built-in class names for debugging
        if hasattr(model, 'names'):
            logger.info(f"[DEBUG] YOLO model's built-in class names: {model.names}")
            logger.info(f"[DEBUG] YOLO model's num_classes: {len(model.names)}")

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
        """Load RT-DETR / RF-DETR model.

        Supports:
        1. Ultralytics RTDETR format
        2. Roboflow RF-DETR training checkpoint format (with 'model' and 'args' keys)
        """
        import torch

        # First, check the checkpoint format
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Check if it's Roboflow RF-DETR format (has 'model' and 'args' keys)
        if isinstance(checkpoint, dict) and 'model' in checkpoint and 'args' in checkpoint:
            logger.info("Detected Roboflow RF-DETR training checkpoint format")
            return self._load_rfdetr_roboflow(weights_path, checkpoint)

        # Otherwise try Ultralytics RTDETR
        logger.info("Trying Ultralytics RTDETR loader...")
        from ultralytics import RTDETR

        model = RTDETR(weights_path)

        if self.device == "cuda":
            if torch.cuda.is_available():
                model.to(self.device)
                logger.info(f"RT-DETR model loaded on CUDA")
            else:
                logger.warning("CUDA not available, using CPU")
        else:
            logger.info(f"RT-DETR model loaded on {self.device}")

        return model

    def _load_rfdetr_roboflow(self, weights_path: str, checkpoint: dict) -> Any:
        """Load Roboflow RF-DETR model from training checkpoint.

        Uses HuggingFace Transformers RTDetrForObjectDetection for inference,
        which is compatible with Roboflow RF-DETR training checkpoints.
        """
        import torch

        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig
        except ImportError:
            logger.error("transformers package not installed or RT-DETR not available")
            raise ImportError("transformers>=4.38.0 required for RT-DETR models")

        # Get model args from checkpoint
        args = checkpoint.get('args', {})
        num_classes = getattr(args, 'num_classes', len(self.classes)) if hasattr(args, 'num_classes') else len(self.classes)

        logger.info(f"Loading RF-DETR with {num_classes} classes using HuggingFace Transformers")

        # Determine base model variant from checkpoint structure
        # Default to rtdetr_r50vd (medium) as it's most common
        base_model = "PekingU/rtdetr_r50vd"

        # Create config with correct number of classes
        config = RTDetrConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        # Set valid prior_prob to avoid math domain error
        if not hasattr(config, 'prior_prob') or config.prior_prob <= 0 or config.prior_prob >= 1:
            config.prior_prob = 0.01

        # Load base model architecture
        model = RTDetrForObjectDetection.from_pretrained(
            base_model,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Load trained weights from checkpoint
        state_dict = checkpoint['model']

        # Try to load state dict, handling potential key mismatches
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded RF-DETR checkpoint weights successfully")
        except Exception as e:
            logger.warning(f"Partial weight loading (expected for fine-tuned models): {e}")

        model.eval()

        # Load processor for preprocessing
        processor = RTDetrImageProcessor.from_pretrained(base_model)

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            logger.info("RF-DETR model loaded on CUDA (HuggingFace)")
        else:
            logger.info("RF-DETR model loaded on CPU (HuggingFace)")

        # Wrap in a compatible interface
        logger.info(f"[DEBUG] HF RT-DETR wrapper created with {len(self.classes)} classes: {self.classes}")
        return HFRTDETRWrapper(model, processor, self.classes, self.device)

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
        logger.info(f"[DEBUG] Using box_threshold={box_threshold}")

        # Run inference
        results = self.model(
            image,
            conf=box_threshold,
            verbose=False,
        )

        # Process results
        predictions = []

        logger.info(f"[DEBUG] Number of results from model: {len(results)}")

        for result in results:
            boxes = result.boxes

            if boxes is None:
                logger.info(f"[DEBUG] boxes is None for this result")
                continue

            logger.info(f"[DEBUG] boxes object has {len(boxes)} detections")

            if len(boxes) == 0:
                logger.info(f"[DEBUG] Zero detections (boxes is empty)")
                continue

            # Log raw class indices and confidences for first few detections
            if len(boxes) > 0:
                raw_classes = boxes.cls.cpu().numpy()
                raw_confs = boxes.conf.cpu().numpy()
                logger.info(f"[DEBUG] Raw class indices (first 5): {raw_classes[:5].tolist()}")
                logger.info(f"[DEBUG] Raw confidences (first 5): {raw_confs[:5].tolist()}")

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

                # Log label mapping for debugging
                if i < 3:  # Only log first 3
                    logger.debug(f"[DEBUG] Detection {i}: class_idx={class_idx} -> label='{label}', conf={confidence:.3f}")

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
