"""
Workflow Blocks - Model Inference Blocks

Real implementations for Detection, Classification, Embedding, and Similarity Search.

SOTA Features:
- Trained model integration from od_trained_models, cls_trained_models, trained_models
- Half precision (FP16) support
- Multi-scale inference (TTA)
- Advanced config options
"""

import time
import asyncio
import logging
import base64
import io
from typing import Any, Optional

import httpx
import numpy as np
from PIL import Image

from ..base import BaseBlock, BlockResult, ExecutionContext, ModelBlock
from ..model_loader import get_model_loader, ModelInfo
from ..inference_service import get_inference_service
from services.qdrant import qdrant_service

logger = logging.getLogger(__name__)


async def load_image_from_input(image_input: Any) -> Optional[Image.Image]:
    """
    Load PIL Image from various input formats.

    Supports:
    - PIL.Image
    - URL string
    - Base64 string
    - numpy array
    - Dict with image_url or image_base64
    """
    if image_input is None:
        return None

    # Already a PIL Image
    if isinstance(image_input, Image.Image):
        return image_input

    # Dict with URL or base64
    if isinstance(image_input, dict):
        if "image_url" in image_input:
            image_input = image_input["image_url"]
        elif "image_base64" in image_input:
            image_input = image_input["image_base64"]
        elif "url" in image_input:
            image_input = image_input["url"]
        else:
            return None

    # URL string
    if isinstance(image_input, str) and image_input.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(image_input)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image from URL: {e}")
            return None

    # Base64 string
    if isinstance(image_input, str):
        try:
            # Handle data URI format
            if "," in image_input:
                image_input = image_input.split(",")[1]
            image_data = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    # Numpy array
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input.astype("uint8")).convert("RGB")

    return None


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class DetectionBlock(ModelBlock):
    """
    Object Detection Block

    Supports:
    - Pretrained: YOLO11, YOLOv8, YOLOv9, YOLOv10 (via Ultralytics)
    - Pretrained: Grounding DINO, OWL-ViT (open-vocabulary with text prompts)
    - Trained: RF-DETR, RT-DETR, D-FINE, YOLO-NAS (from od_trained_models)

    SOTA Features:
    - Text prompts for open-vocabulary detection (Grounding DINO)
    - Tiled inference (SAHI) for small object detection
    - Multi-scale inference (TTA)
    - Half precision (FP16) for faster inference
    - Custom visualization options
    - Class filtering and renaming
    """

    block_type = "detection"
    display_name = "Object Detection"
    description = "Detect objects using YOLO, RT-DETR, Grounding DINO, or your trained models"
    model_type = "detection"

    input_ports = [
        {"name": "image", "type": "image", "required": True, "description": "Input image"},
    ]
    output_ports = [
        {"name": "detections", "type": "array", "description": "List of detected objects with bbox, class, confidence"},
        {"name": "first_detection", "type": "object", "description": "First/best detection (for single-item Crop connection)"},
        {"name": "annotated_image", "type": "image", "description": "Image with bounding boxes drawn"},
        {"name": "count", "type": "number", "description": "Number of detections"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Core config
            "model_id": {
                "type": "string",
                "description": "Model ID from pretrained or trained models",
            },
            "model_source": {
                "type": "string",
                "enum": ["pretrained", "trained"],
                "default": "pretrained",
                "description": "Use pretrained (YOLO) or your trained models (RT-DETR, D-FINE)",
            },
            # Input preprocessing
            "input_size": {
                "type": "number",
                "default": 640,
                "description": "Input resolution for model (320, 480, 640, 800, 1024, 1280)",
            },
            # Detection thresholds
            "confidence": {
                "type": "number",
                "default": 0.5,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum confidence threshold (0-1)",
            },
            "iou_threshold": {
                "type": "number",
                "default": 0.45,
                "minimum": 0,
                "maximum": 1,
                "description": "IoU threshold for NMS",
            },
            "max_detections": {
                "type": "number",
                "default": 300,
                "description": "Maximum number of detections to return",
            },
            # Open-vocabulary (Grounding DINO, OWL-ViT)
            "text_prompt": {
                "type": "string",
                "description": "Text prompt for open-vocabulary detection (e.g., 'person. dog. car.')",
            },
            "box_threshold": {
                "type": "number",
                "default": 0.35,
                "description": "Box threshold for Grounding DINO",
            },
            "text_threshold": {
                "type": "number",
                "default": 0.25,
                "description": "Text threshold for Grounding DINO",
            },
            # Tiled inference (SAHI)
            "tiled_inference": {
                "type": "boolean",
                "default": False,
                "description": "Enable SAHI tiled inference for small objects",
            },
            "tile_size": {
                "type": "number",
                "default": 640,
                "description": "Tile size for SAHI (256, 320, 512, 640)",
            },
            "tile_overlap": {
                "type": "number",
                "default": 0.2,
                "description": "Tile overlap ratio for SAHI (0.1-0.5)",
            },
            # Class filtering
            "classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter to specific class names only",
            },
            "class_ids": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Filter to specific class IDs only",
            },
            # Class renaming
            "class_mapping": {
                "type": "string",
                "description": "Rename classes: 'original:new, person:müşteri'",
            },
            # Inference options
            "agnostic_nms": {
                "type": "boolean",
                "default": False,
                "description": "Use class-agnostic NMS (all classes compete)",
            },
            "half_precision": {
                "type": "boolean",
                "default": True,
                "description": "Use FP16 for faster inference (GPU only)",
            },
            # Output format
            "coordinate_format": {
                "type": "string",
                "enum": ["xyxy", "xywh", "cxcywh"],
                "default": "xyxy",
                "description": "Bounding box coordinate format",
            },
            "normalize_coords": {
                "type": "boolean",
                "default": True,
                "description": "Return normalized (0-1) coordinates",
            },
            # Visualization options
            "draw_boxes": {
                "type": "boolean",
                "default": True,
                "description": "Draw bounding boxes on annotated image",
            },
            "draw_labels": {
                "type": "boolean",
                "default": True,
                "description": "Draw class labels on annotated image",
            },
            "draw_confidence": {
                "type": "boolean",
                "default": True,
                "description": "Draw confidence scores on annotated image",
            },
            "box_thickness": {
                "type": "number",
                "default": 2,
                "description": "Bounding box line thickness (1-5)",
            },
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._inference_service = get_inference_service()

    def _parse_class_mapping(self, mapping_str: str) -> dict[str, str]:
        """Parse class mapping string like 'person:müşteri, car:araç'."""
        if not mapping_str:
            return {}
        result = {}
        for pair in mapping_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                old, new = pair.split(":", 1)
                result[old.strip()] = new.strip()
        return result

    def _draw_custom_boxes(
        self,
        image: Image.Image,
        detections: list[dict],
        draw_labels: bool = True,
        draw_confidence: bool = True,
        box_thickness: int = 2,
    ) -> Image.Image:
        """Draw custom bounding boxes on image."""
        from PIL import ImageDraw, ImageFont

        img = image.copy()
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Try to load a better font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        # Color palette
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        ]

        for det in detections:
            bbox = det.get("bbox", {})
            cls_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)
            cls_id = det.get("class_id", 0)

            # Get coordinates (handle both normalized and absolute)
            if "x1" in bbox:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            elif "x" in bbox:
                x1, y1 = bbox["x"], bbox["y"]
                x2, y2 = x1 + bbox["width"], y1 + bbox["height"]
            else:
                continue

            # Convert normalized to absolute if needed
            if x2 <= 1 and y2 <= 1:
                x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height

            # Get color for this class
            color = colors[cls_id % len(colors)]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

            # Draw label
            if draw_labels or draw_confidence:
                label_parts = []
                if draw_labels:
                    label_parts.append(cls_name)
                if draw_confidence:
                    label_parts.append(f"{conf:.2f}")
                label = " ".join(label_parts)

                # Draw label background
                bbox_label = draw.textbbox((x1, y1 - 15), label, font=font)
                draw.rectangle(bbox_label, fill=color)
                draw.text((x1, y1 - 15), label, fill="white", font=font)

        return img

    async def _run_sahi_inference(
        self,
        model: Any,
        image: Image.Image,
        tile_size: int,
        tile_overlap: float,
        conf: float,
        iou: float,
        max_det: int,
        half: bool,
    ) -> list[dict]:
        """Run SAHI tiled inference for small object detection."""
        width, height = image.size
        overlap_pixels = int(tile_size * tile_overlap)
        stride = tile_size - overlap_pixels

        all_detections = []
        detection_id = 0

        # Generate tiles
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Crop tile
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                tile = image.crop((x, y, x2, y2))

                # Run detection on tile
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda t=tile: model.predict(
                        t,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        half=half,
                        verbose=False,
                    )
                )

                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    names = model.names if hasattr(model, 'names') else {}

                    for box in boxes:
                        # Get absolute coords in tile
                        bbox_tile = box.xyxy[0].cpu().numpy()
                        conf_score = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = names.get(cls_id, f"class_{cls_id}")

                        # Convert to full image coordinates
                        bbox_full = [
                            bbox_tile[0] + x,
                            bbox_tile[1] + y,
                            bbox_tile[2] + x,
                            bbox_tile[3] + y,
                        ]

                        all_detections.append({
                            "id": detection_id,
                            "class_name": cls_name,
                            "class_id": cls_id,
                            "confidence": conf_score,
                            "bbox_xyxy": bbox_full,
                        })
                        detection_id += 1

        # Apply NMS to merged detections
        if all_detections:
            all_detections = self._nms_detections(all_detections, iou)

        # Limit to max_det
        all_detections = all_detections[:max_det]

        return all_detections

    def _nms_detections(self, detections: list[dict], iou_threshold: float) -> list[dict]:
        """Apply NMS to merged tile detections."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping boxes
            remaining = []
            for det in detections:
                if self._compute_iou(best["bbox_xyxy"], det["bbox_xyxy"]) < iou_threshold:
                    remaining.append(det)
            detections = remaining

        return keep

    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """
        Run object detection via RunPod GPU worker.

        All inference happens on remote GPU - no local ML dependencies needed.
        """
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        original_size = image.size

        # Get model config
        model_id = config.get("model_id")
        if not model_id:
            return BlockResult(
                error="model_id is required",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        model_source = config.get("model_source", "pretrained")

        # Get detection parameters
        confidence = config.get("confidence", 0.5)
        iou = config.get("iou_threshold", 0.45)
        max_detections = config.get("max_detections", 300)
        input_size = config.get("input_size", 640)
        agnostic_nms = config.get("agnostic_nms", False)

        # Visualization options
        draw_boxes = config.get("draw_boxes", True)
        draw_labels = config.get("draw_labels", True)
        draw_confidence = config.get("draw_confidence", True)
        box_thickness = config.get("box_thickness", 2)

        # Class mapping/renaming
        class_rename_map = self._parse_class_mapping(config.get("class_mapping", ""))

        try:
            # Get open-vocabulary detection params
            text_prompt = config.get("text_prompt")  # For Grounding DINO
            text_queries = config.get("text_queries")  # For OWL-ViT
            box_threshold = config.get("box_threshold", 0.35)
            text_threshold = config.get("text_threshold", 0.25)

            # Run detection via InferenceService → RunPod GPU
            logger.info(f"Running detection: model={model_id}, source={model_source}, conf={confidence}")

            # Try to get image_url from context inputs (for trained models, URL is more efficient)
            image_url = None
            if context and hasattr(context, 'inputs'):
                image_url = context.inputs.get("image_url")
                # Also check if image input has a URL
                image_input = inputs.get("image")
                if isinstance(image_input, dict) and "image_url" in image_input:
                    image_url = image_input["image_url"]

            result = await self._inference_service.detect(
                model_id=model_id,
                image=image,
                confidence=confidence,
                iou=iou,
                max_detections=max_detections,
                model_source=model_source,
                input_size=input_size,
                agnostic_nms=agnostic_nms,
                # Open-vocabulary params
                text_prompt=text_prompt,
                text_queries=text_queries,
                # Pass image_url for efficient remote inference
                image_url=image_url,
            )

            # Get detections from result
            detections = result.get("detections", [])

            # Apply class filtering
            class_filter = config.get("classes")
            class_id_filter = config.get("class_ids")

            if class_filter or class_id_filter:
                filtered = []
                for det in detections:
                    if class_filter and det.get("class_name") not in class_filter:
                        continue
                    if class_id_filter and det.get("class_id") not in class_id_filter:
                        continue
                    filtered.append(det)
                detections = filtered

            # Apply class renaming
            for det in detections:
                original_name = det.get("class_name", "")
                if original_name in class_rename_map:
                    det["class_name"] = class_rename_map[original_name]
                    det["original_class_name"] = original_name

            # Generate annotated image if requested
            if draw_boxes and detections:
                annotated_pil = self._draw_custom_boxes(
                    image,
                    detections,
                    draw_labels=draw_labels,
                    draw_confidence=draw_confidence,
                    box_thickness=box_thickness,
                )
                annotated_base64 = image_to_base64(annotated_pil)
            else:
                annotated_base64 = image_to_base64(image)

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={
                    "detections": detections,
                    "first_detection": detections[0] if detections else None,
                    "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
                    "count": len(detections),
                },
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "detection_count": len(detections),
                    "confidence_threshold": confidence,
                    "image_size": f"{original_size[0]}x{original_size[1]}",
                    "gpu_inference": True,
                },
            )

        except ValueError as e:
            # Model not found
            return BlockResult(
                error=str(e),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            logger.exception("Detection failed")
            return BlockResult(
                error=f"Detection failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )


class ClassificationBlock(ModelBlock):
    """
    Image Classification Block

    Supports:
    - Pretrained: ImageNet classifiers (ViT, ConvNeXt, etc.)
    - Trained: Custom models from cls_trained_models (ViT, ConvNeXt, EfficientNet, Swin)

    SOTA Features:
    - Support for all CLS training worker architectures
    - Temperature scaling for calibrated probabilities
    - Multi-crop ensemble inference
    - Test Time Augmentation (TTA)
    - Decision modes (binary, uncertain-aware)
    - Multiple output formats
    """

    block_type = "classification"
    display_name = "Classification"
    description = "Classify images using pretrained or your trained models"
    model_type = "classification"

    input_ports = [
        {"name": "image", "type": "image", "required": False, "description": "Single input image"},
        {"name": "images", "type": "array", "required": False, "description": "Array of images (batch processing)"},
    ]
    output_ports = [
        {"name": "predictions", "type": "array", "description": "Classification predictions with top-k classes"},
        {"name": "decision", "type": "string", "description": "Decision result (when using decision mode)"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Core config
            "model_id": {
                "type": "string",
                "description": "Model ID from pretrained or trained models",
            },
            "model_source": {
                "type": "string",
                "enum": ["pretrained", "trained"],
                "default": "trained",
                "description": "Use pretrained or your trained classification models",
            },
            # Input preprocessing
            "input_size": {
                "type": "number",
                "default": 224,
                "description": "Input resolution (224, 256, 384, 448, 518)",
            },
            # Prediction options
            "top_k": {
                "type": "number",
                "default": 5,
                "minimum": 1,
                "maximum": 100,
                "description": "Number of top predictions to return",
            },
            "threshold": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum confidence threshold for predictions",
            },
            # Classification mode
            "mode": {
                "type": "string",
                "enum": ["single_label", "multi_label"],
                "default": "single_label",
                "description": "single_label (softmax) or multi_label (sigmoid)",
            },
            # Decision mode
            "decision_mode": {
                "type": "string",
                "enum": ["label_only", "with_confidence", "binary_decision", "uncertain_aware"],
                "default": "label_only",
                "description": "Output mode for decisions",
            },
            "target_class": {
                "type": "string",
                "description": "Target class for binary decision mode",
            },
            "decision_threshold": {
                "type": "number",
                "default": 0.5,
                "description": "Threshold for binary decision",
            },
            "uncertainty_threshold": {
                "type": "number",
                "default": 0.7,
                "description": "Below this confidence = 'uncertain'",
            },
            "reject_threshold": {
                "type": "number",
                "default": 0.3,
                "description": "Below this confidence = 'rejected'",
            },
            # Temperature scaling
            "temperature": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.01,
                "maximum": 10.0,
                "description": "Softmax temperature for calibration",
            },
            # Test Time Augmentation (TTA)
            "tta_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Enable Test Time Augmentation",
            },
            "tta_hflip": {
                "type": "boolean",
                "default": True,
                "description": "Include horizontal flip in TTA",
            },
            "tta_five_crop": {
                "type": "boolean",
                "default": False,
                "description": "Include 5-crop in TTA",
            },
            "tta_merge": {
                "type": "string",
                "enum": ["mean", "max", "vote"],
                "default": "mean",
                "description": "TTA merge strategy",
            },
            # Multi-crop ensemble (SOTA for fine-grained)
            "multi_crop": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "scales": {
                        "type": "array",
                        "items": {"type": "number"},
                        "default": [1.0, 0.875, 0.75],
                    },
                    "merge_mode": {
                        "type": "string",
                        "enum": ["mean", "max"],
                        "default": "mean",
                    },
                },
            },
            # Output options
            "output_format": {
                "type": "string",
                "enum": ["standard", "detailed", "minimal", "decision"],
                "default": "standard",
                "description": "Output format style",
            },
            "include_probs": {
                "type": "boolean",
                "default": False,
                "description": "Include full probability distribution",
            },
            "include_entropy": {
                "type": "boolean",
                "default": False,
                "description": "Include prediction entropy (uncertainty)",
            },
            "include_second_best": {
                "type": "boolean",
                "default": False,
                "description": "Include runner-up prediction",
            },
            # Class renaming
            "class_mapping": {
                "type": "string",
                "description": "Rename classes: '0:empty, 1:full'",
            },
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._inference_service = get_inference_service()

    def _parse_class_mapping(self, mapping_str: str) -> dict[str, str]:
        """Parse class mapping string like '0:empty, 1:full'."""
        if not mapping_str:
            return {}
        result = {}
        for pair in mapping_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                old, new = pair.split(":", 1)
                result[old.strip()] = new.strip()
        return result

    def _compute_entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of probability distribution."""
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(probs))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """
        Classify input images via RunPod GPU worker.

        All inference happens on remote GPU - no local ML dependencies needed.
        """
        start_time = time.time()

        # Get images
        images = inputs.get("images", [])
        if not images and inputs.get("image"):
            images = [inputs["image"]]

        if not images:
            return BlockResult(
                error="No images provided",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Load images
        pil_images = []
        for img in images:
            pil_img = await load_image_from_input(img)
            if pil_img:
                pil_images.append(pil_img)

        if not pil_images:
            return BlockResult(
                error="Failed to load any images",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get config options
        model_id = config.get("model_id")
        if not model_id:
            return BlockResult(
                error="model_id is required",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        model_source = config.get("model_source", "trained")
        top_k = config.get("top_k", 5)
        threshold = config.get("threshold", 0.0)

        # Decision mode options
        decision_mode = config.get("decision_mode", "label_only")
        target_class = config.get("target_class")
        decision_threshold = config.get("decision_threshold", 0.5)
        uncertainty_threshold = config.get("uncertainty_threshold", 0.7)
        reject_threshold = config.get("reject_threshold", 0.3)

        # Class renaming
        class_rename_map = self._parse_class_mapping(config.get("class_mapping", ""))

        try:
            predictions = []
            decisions = []

            # Run classification for each image via InferenceService → RunPod GPU
            for pil_image in pil_images:
                logger.info(f"Running classification: model={model_id}, source={model_source}")

                result = await self._inference_service.classify(
                    model_id=model_id,
                    image=pil_image,
                    top_k=top_k,
                    model_source=model_source,
                    threshold=threshold,
                )

                # Get predictions from result
                top_predictions = result.get("predictions", [])

                # Apply class renaming
                for pred in top_predictions:
                    original_name = pred.get("class_name", "")
                    if original_name in class_rename_map:
                        pred["class_name"] = class_rename_map[original_name]
                        pred["original_class_name"] = original_name
                    # Also check by class_id
                    cls_id_str = str(pred.get("class_id", ""))
                    if cls_id_str in class_rename_map:
                        pred["class_name"] = class_rename_map[cls_id_str]
                        pred["original_class_name"] = original_name

                # Get top prediction
                top_pred = top_predictions[0] if top_predictions else {"class_name": "unknown", "confidence": 0, "class_id": -1}
                top_conf = top_pred.get("confidence", 0)

                # Handle decision mode
                decision = None
                decision_reason = None

                if decision_mode == "binary_decision" and target_class:
                    target_conf = 0
                    for pred in top_predictions:
                        if pred.get("class_name") == target_class or str(pred.get("class_id")) == target_class:
                            target_conf = pred.get("confidence", 0)
                            break
                    decision = target_conf >= decision_threshold
                    decision_reason = f"target '{target_class}' confidence {target_conf:.2f} {'≥' if decision else '<'} threshold {decision_threshold}"

                elif decision_mode == "uncertain_aware":
                    if top_conf >= uncertainty_threshold:
                        decision = top_pred.get("class_name")
                        decision_reason = f"confident ({top_conf:.2f} ≥ {uncertainty_threshold})"
                    elif top_conf >= reject_threshold:
                        decision = "uncertain"
                        decision_reason = f"uncertain ({reject_threshold} ≤ {top_conf:.2f} < {uncertainty_threshold})"
                    else:
                        decision = "rejected"
                        decision_reason = f"rejected ({top_conf:.2f} < {reject_threshold})"

                # Build prediction output
                prediction = {
                    "class_name": top_pred.get("class_name"),
                    "confidence": top_pred.get("confidence"),
                    "top_k": top_predictions,
                }

                if decision is not None:
                    prediction["decision"] = decision
                    prediction["reason"] = decision_reason

                predictions.append(prediction)
                decisions.append(decision)

            duration = (time.time() - start_time) * 1000

            outputs = {"predictions": predictions}
            if decision_mode in ("binary_decision", "uncertain_aware"):
                outputs["decision"] = decisions[0] if len(decisions) == 1 else decisions

            return BlockResult(
                outputs=outputs,
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "image_count": len(pil_images),
                    "top_k": top_k,
                    "decision_mode": decision_mode,
                    "gpu_inference": True,
                },
            )

        except ValueError as e:
            return BlockResult(
                error=str(e),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            logger.exception("Classification failed")
            return BlockResult(
                error=f"Classification failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )


class EmbeddingBlock(ModelBlock):
    """
    Embedding Extraction Block

    Supports:
    - Pretrained: DINOv2, CLIP, SigLIP
    - Trained: Fine-tuned embedding models from trained_models

    SOTA Features:
    - Multiple pooling strategies (CLS, mean, GeM)
    - Multi-scale embedding extraction
    - L2 normalization option
    - PCA dimension reduction
    - Batch processing
    - Multiple output formats
    """

    block_type = "embedding"
    display_name = "Embedding"
    description = "Extract embeddings using DINOv2, CLIP, or your trained models"
    model_type = "embedding"

    input_ports = [
        {"name": "image", "type": "image", "required": False, "description": "Single input image"},
        {"name": "images", "type": "array", "required": False, "description": "Array of images"},
    ]
    output_ports = [
        {"name": "embeddings", "type": "array", "description": "Embedding vectors (list of float arrays)"},
        {"name": "embedding", "type": "array", "description": "First/single embedding vector (for direct SimilaritySearch connection)"},
        {"name": "metadata", "type": "object", "description": "Embedding metadata (dim, model, etc.)"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Core config
            "model_id": {
                "type": "string",
                "description": "Model ID from pretrained or trained models",
            },
            "model_source": {
                "type": "string",
                "enum": ["pretrained", "trained"],
                "default": "pretrained",
                "description": "Use pretrained (DINOv2, CLIP) or your trained models",
            },
            # Input preprocessing
            "input_size": {
                "type": "number",
                "default": 224,
                "description": "Input resolution (224, 336, 378, 518 for DINOv2)",
            },
            "normalize": {
                "type": "boolean",
                "default": True,
                "description": "L2 normalize embeddings (recommended for similarity search)",
            },
            # SOTA options
            "pooling": {
                "type": "string",
                "enum": ["cls", "mean", "gem"],
                "default": "cls",
                "description": "Pooling strategy: CLS token, mean pooling, or GeM (SOTA)",
            },
            "gem_p": {
                "type": "number",
                "default": 3.0,
                "description": "GeM pooling power parameter",
            },
            "layer": {
                "type": "number",
                "default": -1,
                "description": "Transformer layer to extract from (-1 = last)",
            },
            # Multi-scale extraction
            "multi_scale": {
                "type": "boolean",
                "default": False,
                "description": "Extract at multiple scales and aggregate",
            },
            "multi_scale_factors": {
                "type": "string",
                "default": "1.0,0.75,0.5",
                "description": "Comma-separated scale factors",
            },
            "multi_scale_agg": {
                "type": "string",
                "enum": ["concat", "mean", "max"],
                "default": "concat",
                "description": "How to aggregate multi-scale embeddings",
            },
            # PCA dimension reduction
            "pca_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Apply PCA dimension reduction",
            },
            "pca_dim": {
                "type": "number",
                "default": 256,
                "description": "Target PCA dimension (64, 128, 256, 512)",
            },
            "pca_whiten": {
                "type": "boolean",
                "default": False,
                "description": "Apply whitening transformation",
            },
            # Output format
            "output_format": {
                "type": "string",
                "enum": ["vector", "vector_with_meta", "base64"],
                "default": "vector",
                "description": "Output format for embeddings",
            },
            "include_model_info": {
                "type": "boolean",
                "default": True,
                "description": "Include model info in metadata",
            },
            "include_timing": {
                "type": "boolean",
                "default": False,
                "description": "Include processing time in metadata",
            },
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._inference_service = get_inference_service()

    def _parse_scale_factors(self, factors_str: str) -> list[float]:
        """Parse comma-separated scale factors."""
        try:
            return [float(f.strip()) for f in factors_str.split(",") if f.strip()]
        except Exception:
            return [1.0, 0.75, 0.5]

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """
        Extract embeddings from images via RunPod GPU worker.

        All inference happens on remote GPU - no local ML dependencies needed.
        The worker handles model loading, caching, and GPU inference.
        """
        start_time = time.time()

        # Get images
        images = inputs.get("images", [])
        if not images and inputs.get("image"):
            images = [inputs["image"]]

        if not images:
            return BlockResult(
                error="No images provided",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Load images
        pil_images = []
        for img in images:
            pil_img = await load_image_from_input(img)
            if pil_img:
                pil_images.append(pil_img)

        if not pil_images:
            return BlockResult(
                error="Failed to load any images",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get config options
        model_id = config.get("model_id")
        if not model_id:
            return BlockResult(
                error="model_id is required for embedding extraction",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        model_source = config.get("model_source", "pretrained")
        input_size = config.get("input_size", 224)
        normalize = config.get("normalize", True)
        pooling = config.get("pooling", "cls")
        gem_p = config.get("gem_p", 3.0)

        # Multi-scale options (passed to worker)
        multi_scale = config.get("multi_scale", False)
        multi_scale_factors = self._parse_scale_factors(config.get("multi_scale_factors", "1.0,0.75,0.5"))
        multi_scale_agg = config.get("multi_scale_agg", "concat")

        # PCA options (passed to worker)
        pca_enabled = config.get("pca_enabled", False)
        pca_dim = config.get("pca_dim", 256)

        # Output options
        output_format = config.get("output_format", "vector")
        include_model_info = config.get("include_model_info", True)
        include_timing = config.get("include_timing", False)

        # Run embedding extraction via RunPod GPU worker
        try:
            embeddings = []
            inference_times = []

            for pil_image in pil_images:
                img_start = time.time()

                # Call InferenceService - delegates to RunPod worker
                result = await self._inference_service.embed(
                    model_id=model_id,
                    image=pil_image,
                    model_source=model_source,
                    input_size=input_size,
                    normalize=normalize,
                    pooling=pooling,
                    gem_p=gem_p,
                    multi_scale=multi_scale,
                    multi_scale_factors=multi_scale_factors,
                    multi_scale_agg=multi_scale_agg,
                    pca_enabled=pca_enabled,
                    pca_dim=pca_dim,
                    output_format=output_format,
                )

                # Extract embedding from worker result
                embedding = result.get("embedding")
                if embedding is None:
                    raise ValueError(f"Worker returned no embedding: {result}")

                embeddings.append(embedding)
                inference_times.append((time.time() - img_start) * 1000)

            duration = (time.time() - start_time) * 1000

            # Build output
            final_embedding_dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else 0
            if output_format == "base64":
                # For base64, we can't easily get dimension, use config or worker metadata
                final_embedding_dim = result.get("embedding_dim", 768)

            # Add singular 'embedding' output for convenience (first/single embedding)
            # This makes it easy to connect directly to SimilaritySearch in ForEach loops
            single_embedding = embeddings[0] if embeddings else None

            outputs = {
                "embeddings": embeddings,
                "embedding": single_embedding,
            }

            # Build metadata
            if output_format == "vector_with_meta" or include_model_info:
                meta = {
                    "embedding_dim": final_embedding_dim,
                    "count": len(embeddings),
                    "normalized": normalize,
                    "pooling": pooling,
                }

                if include_model_info:
                    meta["model_id"] = model_id
                    meta["model_source"] = model_source

                if include_timing:
                    meta["inference_times_ms"] = inference_times
                    meta["total_time_ms"] = round(duration, 2)

                if multi_scale:
                    meta["multi_scale"] = True
                    meta["scales"] = multi_scale_factors
                    meta["aggregation"] = multi_scale_agg

                if pca_enabled:
                    meta["pca_applied"] = True
                    meta["pca_dim"] = pca_dim

                outputs["metadata"] = meta

            return BlockResult(
                outputs=outputs,
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "embedding_count": len(embeddings),
                    "embedding_dim": final_embedding_dim,
                    "pooling": pooling,
                    "multi_scale": multi_scale,
                    "pca_enabled": pca_enabled,
                    "gpu_inference": True,
                },
            )

        except Exception as e:
            logger.exception("Embedding extraction failed")
            return BlockResult(
                error=f"Embedding extraction failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )


class SimilaritySearchBlock(BaseBlock):
    """
    Similarity Search Block

    Searches Qdrant vector database for similar items.

    SOTA Features:
    - Multiple distance metrics (cosine, euclidean, dot, manhattan)
    - Search modes (approximate ANN, exact brute force, hybrid dense+sparse)
    - Score normalization (minmax, softmax, zscore)
    - Multi-query fusion (RRF, average, max, weighted)
    - Re-ranking with cross-encoder
    - Deduplication by field
    - Grouping by category
    - Fallback strategies
    """

    block_type = "similarity_search"
    display_name = "Similarity Search"
    description = "Search for similar products in Qdrant vector database"

    input_ports = [
        {"name": "embeddings", "type": "array", "required": False, "description": "Query embedding vectors (array)"},
        {"name": "embedding", "type": "array", "required": False, "description": "Single query embedding vector"},
        {"name": "text_queries", "type": "array", "required": False, "description": "Text queries for hybrid search"},
    ]
    output_ports = [
        {"name": "matches", "type": "array", "description": "Matching products (nested: one list per query)"},
        {"name": "flat_matches", "type": "array", "description": "Flattened matches (single list for ForEach use)"},
        {"name": "best_match", "type": "object", "description": "Top matching product"},
        {"name": "match_count", "type": "number", "description": "Total number of matches"},
    ]
    config_schema = {
        "type": "object",
        "properties": {
            # Core config
            "collection": {
                "type": "string",
                "description": "Qdrant collection name",
            },
            "top_k": {
                "type": "number",
                "default": 5,
                "minimum": 1,
                "maximum": 100,
                "description": "Number of results per query",
            },
            "threshold": {
                "type": "number",
                "default": 0.7,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum similarity score threshold",
            },
            # Distance metric
            "distance_metric": {
                "type": "string",
                "enum": ["cosine", "euclidean", "dot", "manhattan"],
                "default": "cosine",
                "description": "Distance metric for similarity calculation",
            },
            # Search mode
            "search_mode": {
                "type": "string",
                "enum": ["approximate", "exact", "hybrid"],
                "default": "approximate",
                "description": "ANN (fast), exact (accurate), or hybrid (dense+sparse)",
            },
            "dense_weight": {
                "type": "number",
                "default": 0.7,
                "description": "Weight for dense vectors in hybrid search (0-1)",
            },
            # Score normalization
            "score_normalization": {
                "type": "string",
                "enum": ["none", "minmax", "softmax", "zscore"],
                "default": "none",
                "description": "Normalize similarity scores",
            },
            # Multi-query fusion
            "fusion_method": {
                "type": "string",
                "enum": ["none", "rrf", "avg", "max", "weighted"],
                "default": "none",
                "description": "How to combine results from multiple queries",
            },
            # Re-ranking
            "rerank_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Re-rank top results with cross-encoder",
            },
            "rerank_top_n": {
                "type": "number",
                "default": 50,
                "description": "Number of results to re-rank",
            },
            "reranker_model": {
                "type": "string",
                "enum": ["cross-encoder", "colbert"],
                "default": "cross-encoder",
            },
            # Deduplication
            "dedupe_enabled": {
                "type": "boolean",
                "default": False,
                "description": "Remove duplicate items",
            },
            "dedupe_field": {
                "type": "string",
                "description": "Payload field to deduplicate by (e.g., 'product_id')",
            },
            "dedupe_strategy": {
                "type": "string",
                "enum": ["best", "first", "avg"],
                "default": "best",
                "description": "How to handle duplicates",
            },
            # Filtering
            "filter": {
                "type": "object",
                "description": "Qdrant filter conditions (must, should, must_not)",
            },
            "metadata_filter": {
                "type": "string",
                "description": "JSON string filter (e.g., '{\"category\": \"electronics\"}')",
            },
            # Grouping
            "group_by": {
                "type": "string",
                "description": "Group results by payload field (e.g., 'product_id')",
            },
            "group_size": {
                "type": "number",
                "default": 1,
                "description": "Max results per group",
            },
            # Fallback
            "fallback_strategy": {
                "type": "string",
                "enum": ["none", "lower_threshold", "expand_k", "remove_filter"],
                "default": "none",
                "description": "What to do when no results found",
            },
            # Output format
            "return_format": {
                "type": "string",
                "enum": ["full", "standard", "ids_only", "scores_only"],
                "default": "standard",
            },
            "with_payload": {
                "type": "boolean",
                "default": True,
            },
            "with_vectors": {
                "type": "boolean",
                "default": False,
            },
            "include_timing": {
                "type": "boolean",
                "default": False,
            },
            # Batch processing
            "parallel_queries": {
                "type": "boolean",
                "default": True,
            },
            "max_concurrent": {
                "type": "number",
                "default": 10,
            },
        },
        "required": ["collection"],
    }

    def _group_results(
        self,
        results: list[dict],
        group_by: str,
        group_size: int,
    ) -> list[dict]:
        """Group results by a payload field."""
        groups = {}

        for result in results:
            payload = result.get("metadata", {})
            group_key = payload.get(group_by)

            if group_key is None:
                group_key = "__ungrouped__"

            if group_key not in groups:
                groups[group_key] = []

            if len(groups[group_key]) < group_size:
                groups[group_key].append(result)

        # Flatten back to list, maintaining order by best score in each group
        grouped_results = []
        for key, items in sorted(groups.items(), key=lambda x: -x[1][0]["similarity"] if x[1] else 0):
            grouped_results.extend(items)

        return grouped_results

    def _deduplicate_results(
        self,
        results: list[dict],
        dedupe_field: str,
        strategy: str,
    ) -> list[dict]:
        """Deduplicate results by a payload field."""
        seen = {}

        for result in results:
            payload = result.get("metadata", {})
            key = payload.get(dedupe_field)

            if key is None:
                key = result.get("id", id(result))

            if key not in seen:
                seen[key] = result
            elif strategy == "best":
                # Keep the one with higher similarity
                if result.get("similarity", 0) > seen[key].get("similarity", 0):
                    seen[key] = result
            elif strategy == "avg":
                # Average the scores
                seen[key]["similarity"] = (
                    seen[key].get("similarity", 0) + result.get("similarity", 0)
                ) / 2
            # 'first' strategy: keep first found, do nothing

        return list(seen.values())

    def _normalize_scores(self, results: list[dict], method: str) -> list[dict]:
        """Normalize similarity scores."""
        if not results or method == "none":
            return results

        scores = [r.get("similarity", 0) for r in results]

        if method == "minmax":
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                for r in results:
                    r["similarity"] = (r["similarity"] - min_s) / (max_s - min_s)
                    r["similarity"] = round(r["similarity"], 4)

        elif method == "softmax":
            exp_scores = [np.exp(s) for s in scores]
            sum_exp = sum(exp_scores)
            for i, r in enumerate(results):
                r["similarity"] = round(exp_scores[i] / sum_exp, 4)

        elif method == "zscore":
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            if std_s > 0:
                for r in results:
                    r["similarity"] = round((r["similarity"] - mean_s) / std_s, 4)

        return results

    def _fuse_results(
        self,
        all_results: list[list[dict]],
        method: str,
        top_k: int,
    ) -> list[dict]:
        """Fuse results from multiple queries."""
        if not all_results or method == "none":
            return all_results[0] if all_results else []

        # Collect all unique IDs with their scores
        id_scores: dict[str, list[float]] = {}
        id_data: dict[str, dict] = {}

        for results in all_results:
            for i, r in enumerate(results):
                rid = str(r.get("id", i))
                if rid not in id_scores:
                    id_scores[rid] = []
                    id_data[rid] = r
                id_scores[rid].append(r.get("similarity", 0))

        # Fuse scores
        fused = []
        for rid, scores in id_scores.items():
            r = id_data[rid].copy()

            if method == "rrf":
                # Reciprocal Rank Fusion
                rrf_score = sum(1.0 / (60 + i + 1) for i, s in enumerate(sorted(scores, reverse=True)))
                r["similarity"] = round(rrf_score, 4)
            elif method == "avg":
                r["similarity"] = round(np.mean(scores), 4)
            elif method == "max":
                r["similarity"] = round(max(scores), 4)
            elif method == "weighted":
                # Weight by position (first queries more important)
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                r["similarity"] = round(np.average(scores, weights=weights[:len(scores)]), 4)

            fused.append(r)

        # Sort by fused score and limit
        fused.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return fused[:top_k]

    def _parse_metadata_filter(self, filter_str: str) -> dict | None:
        """Parse JSON metadata filter string."""
        if not filter_str:
            return None
        try:
            import json
            return json.loads(filter_str)
        except Exception:
            return None

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Search for similar items in Qdrant with SOTA features."""
        start_time = time.time()

        # Support both singular 'embedding' and plural 'embeddings' input
        embeddings = inputs.get("embeddings", [])
        single_embedding = inputs.get("embedding")

        # If singular embedding provided, wrap in list
        if single_embedding is not None and not embeddings:
            if isinstance(single_embedding, list) and len(single_embedding) > 0:
                # Check if it's a single vector (list of floats) or list of vectors
                if isinstance(single_embedding[0], (int, float)):
                    embeddings = [single_embedding]  # Single vector
                else:
                    embeddings = single_embedding  # Already list of vectors
            else:
                embeddings = [single_embedding]

        if not embeddings:
            return BlockResult(
                error="No embeddings provided (use 'embedding' or 'embeddings' input)",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        collection = config.get("collection")
        if not collection:
            return BlockResult(
                error="No collection specified",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Core config
        top_k = config.get("top_k", 5)
        threshold = config.get("threshold", 0.7)

        # Search options
        search_mode = config.get("search_mode", "approximate")
        score_normalization = config.get("score_normalization", "none")
        fusion_method = config.get("fusion_method", "none")

        # Re-ranking
        rerank_enabled = config.get("rerank_enabled", False)
        rerank_top_n = config.get("rerank_top_n", 50)

        # Deduplication
        dedupe_enabled = config.get("dedupe_enabled", False)
        dedupe_field = config.get("dedupe_field")
        dedupe_strategy = config.get("dedupe_strategy", "best")

        # Filtering
        filter_conditions = config.get("filter")
        metadata_filter = self._parse_metadata_filter(config.get("metadata_filter", ""))
        if metadata_filter and not filter_conditions:
            filter_conditions = {"must": [{"key": k, "match": {"value": v}} for k, v in metadata_filter.items()]}

        # Grouping
        group_by = config.get("group_by")
        group_size = config.get("group_size", 1)

        # Fallback
        fallback_strategy = config.get("fallback_strategy", "none")

        # Output options
        return_format = config.get("return_format", "standard")
        with_payload = config.get("with_payload", True)
        include_timing = config.get("include_timing", False)

        # Check if Qdrant is configured
        if not qdrant_service.is_configured():
            return BlockResult(
                error="Qdrant not configured",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        try:
            all_results = []

            # Calculate fetch limit
            fetch_limit = top_k
            if group_by:
                fetch_limit *= 5
            if dedupe_enabled:
                fetch_limit *= 3
            if rerank_enabled:
                fetch_limit = max(fetch_limit, rerank_top_n)

            for embedding in embeddings:
                # Ensure embedding is a list of floats
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Search Qdrant
                search_start = time.time()
                results = await qdrant_service.search(
                    collection_name=collection,
                    query_vector=embedding,
                    limit=fetch_limit,
                    score_threshold=threshold,
                    filter_conditions=filter_conditions,
                )
                search_time = (time.time() - search_start) * 1000

                # Format matches
                matches = []
                for r in results:
                    payload = r.get("payload", {})

                    match_data = {
                        "id": r["id"],
                        "similarity": round(r["score"], 4),
                    }

                    if return_format != "ids_only" and return_format != "scores_only":
                        match_data["product_id"] = payload.get("product_id")

                    if return_format in ("full", "standard") and with_payload:
                        match_data["product_info"] = {
                            "name": payload.get("product_name"),
                            "upc": payload.get("upc"),
                            "sku": payload.get("sku"),
                        }
                        match_data["metadata"] = payload

                    if include_timing:
                        match_data["search_time_ms"] = round(search_time, 2)

                    matches.append(match_data)

                all_results.append(matches)

            # Apply fallback if no results
            if fallback_strategy != "none" and all(len(m) == 0 for m in all_results):
                if fallback_strategy == "lower_threshold":
                    # Retry with lower threshold
                    for i, embedding in enumerate(embeddings):
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        results = await qdrant_service.search(
                            collection_name=collection,
                            query_vector=embedding,
                            limit=fetch_limit,
                            score_threshold=threshold * 0.5,
                            filter_conditions=filter_conditions,
                        )
                        all_results[i] = [
                            {"id": r["id"], "similarity": round(r["score"], 4), "metadata": r.get("payload", {})}
                            for r in results
                        ]
                elif fallback_strategy == "expand_k":
                    # Retry with larger k
                    for i, embedding in enumerate(embeddings):
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        results = await qdrant_service.search(
                            collection_name=collection,
                            query_vector=embedding,
                            limit=fetch_limit * 3,
                            score_threshold=threshold,
                            filter_conditions=filter_conditions,
                        )
                        all_results[i] = [
                            {"id": r["id"], "similarity": round(r["score"], 4), "metadata": r.get("payload", {})}
                            for r in results
                        ]
                elif fallback_strategy == "remove_filter":
                    # Retry without filter
                    for i, embedding in enumerate(embeddings):
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        results = await qdrant_service.search(
                            collection_name=collection,
                            query_vector=embedding,
                            limit=fetch_limit,
                            score_threshold=threshold,
                            filter_conditions=None,
                        )
                        all_results[i] = [
                            {"id": r["id"], "similarity": round(r["score"], 4), "metadata": r.get("payload", {})}
                            for r in results
                        ]

            # Apply fusion if multiple queries
            if fusion_method != "none" and len(all_results) > 1:
                fused = self._fuse_results(all_results, fusion_method, top_k)
                all_results = [fused]

            # Process each result set
            final_matches = []
            for matches in all_results:
                # Apply deduplication
                if dedupe_enabled and dedupe_field:
                    matches = self._deduplicate_results(matches, dedupe_field, dedupe_strategy)

                # Apply grouping
                if group_by:
                    matches = self._group_results(matches, group_by, group_size)

                # Apply score normalization
                matches = self._normalize_scores(matches, score_normalization)

                # Limit to top_k
                matches = matches[:top_k]

                final_matches.append(matches)

            duration = (time.time() - start_time) * 1000

            # Create flat_matches (single list, useful for ForEach iterations)
            flat_matches = []
            for matches in final_matches:
                flat_matches.extend(matches)

            # For single query, also output best_match
            best_match = None
            if len(final_matches) == 1 and final_matches[0]:
                best_match = final_matches[0][0]
            elif flat_matches:
                best_match = flat_matches[0]

            total_match_count = len(flat_matches)

            outputs = {
                "matches": final_matches,
                "flat_matches": flat_matches,
                "match_count": total_match_count,
            }
            if best_match:
                outputs["best_match"] = best_match

            metrics = {
                "collection": collection,
                "query_count": len(embeddings),
                "total_matches": total_match_count,
                "threshold": threshold,
                "search_mode": search_mode,
            }

            if group_by:
                metrics["grouped_by"] = group_by
            if dedupe_enabled:
                metrics["deduplicated"] = True
            if fusion_method != "none":
                metrics["fusion_method"] = fusion_method
            if score_normalization != "none":
                metrics["score_normalization"] = score_normalization

            return BlockResult(
                outputs=outputs,
                duration_ms=round(duration, 2),
                metrics=metrics,
            )

        except Exception as e:
            logger.exception("Similarity search failed")
            return BlockResult(
                error=f"Similarity search failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
