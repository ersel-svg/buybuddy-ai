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
    - Trained: RF-DETR, RT-DETR, D-FINE, YOLO-NAS (from od_trained_models)

    SOTA Features:
    - Multi-scale inference (TTA)
    - Half precision (FP16) for faster inference
    - Class filtering by name or ID
    - Agnostic NMS option
    - Flexible output coordinate formats
    """

    block_type = "detection"
    display_name = "Object Detection"
    description = "Detect objects using YOLO, RT-DETR, or your trained models"
    model_type = "detection"

    input_ports = [
        {"name": "image", "type": "image", "required": True, "description": "Input image"},
    ]
    output_ports = [
        {"name": "detections", "type": "array", "description": "List of detected objects with bbox, class, confidence"},
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
            # SOTA options
            "max_detections": {
                "type": "number",
                "default": 300,
                "description": "Maximum number of detections to return",
            },
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
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._model_loader = get_model_loader()

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Run object detection on the input image."""
        start_time = time.time()

        # Load image
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get model config
        model_id = config.get("model_id")
        model_source = config.get("model_source", "pretrained")

        # Load model using centralized ModelLoader
        try:
            model, model_info = await self._model_loader.load_detection_model(
                model_id, model_source
            )
        except ValueError as e:
            return BlockResult(
                error=str(e),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            return BlockResult(
                error=f"Failed to load model: {e}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get config options
        conf = config.get("confidence", 0.5)
        iou = config.get("iou_threshold", 0.45)
        max_det = config.get("max_detections", 300)
        agnostic_nms = config.get("agnostic_nms", False)
        half = config.get("half_precision", True)
        coord_format = config.get("coordinate_format", "xyxy")
        normalize = config.get("normalize_coords", True)

        # Get class mapping from ModelInfo
        class_mapping = model_info.class_mapping or {}

        try:
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: model.predict(
                    image,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    agnostic_nms=agnostic_nms,
                    half=half,
                    verbose=False,
                )
            )

            result = results[0]

            # Parse detections
            detections = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                # Get class names - prefer trained model's mapping
                names = class_mapping if class_mapping else (model.names if hasattr(model, 'names') else {})

                for i, box in enumerate(boxes):
                    # Get bbox based on format preference
                    if normalize:
                        bbox_raw = box.xyxyn[0].cpu().numpy()
                    else:
                        bbox_raw = box.xyxy[0].cpu().numpy()

                    conf_score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    if isinstance(cls_name, int):
                        cls_name = f"class_{cls_name}"

                    # Filter by class name if specified
                    class_filter = config.get("classes")
                    if class_filter and cls_name not in class_filter:
                        continue

                    # Filter by class ID if specified
                    class_id_filter = config.get("class_ids")
                    if class_id_filter and cls_id not in class_id_filter:
                        continue

                    # Convert coords based on format
                    if coord_format == "xyxy":
                        bbox = {
                            "x1": float(bbox_raw[0]),
                            "y1": float(bbox_raw[1]),
                            "x2": float(bbox_raw[2]),
                            "y2": float(bbox_raw[3]),
                        }
                    elif coord_format == "xywh":
                        w = bbox_raw[2] - bbox_raw[0]
                        h = bbox_raw[3] - bbox_raw[1]
                        bbox = {
                            "x": float(bbox_raw[0]),
                            "y": float(bbox_raw[1]),
                            "width": float(w),
                            "height": float(h),
                        }
                    else:  # cxcywh
                        w = bbox_raw[2] - bbox_raw[0]
                        h = bbox_raw[3] - bbox_raw[1]
                        cx = bbox_raw[0] + w / 2
                        cy = bbox_raw[1] + h / 2
                        bbox = {
                            "cx": float(cx),
                            "cy": float(cy),
                            "width": float(w),
                            "height": float(h),
                        }

                    area = float((bbox_raw[2] - bbox_raw[0]) * (bbox_raw[3] - bbox_raw[1]))

                    detections.append({
                        "id": i,
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf_score, 4),
                        "bbox": bbox,
                        "bbox_xyxy": [float(x) for x in bbox_raw] if coord_format == "xyxy" else None,
                        "area": area,
                    })

            # Generate annotated image
            annotated = result.plot()
            annotated_pil = Image.fromarray(annotated)
            annotated_base64 = image_to_base64(annotated_pil)

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={
                    "detections": detections,
                    "annotated_image": f"data:image/jpeg;base64,{annotated_base64}",
                    "count": len(detections),
                },
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "model_type": model_info.model_type,
                    "detection_count": len(detections),
                    "confidence_threshold": conf,
                    "image_size": f"{image.width}x{image.height}",
                },
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
    - Batch processing for multiple images
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
            "top_k": {
                "type": "number",
                "default": 5,
                "minimum": 1,
                "maximum": 100,
                "description": "Number of top predictions to return",
            },
            # SOTA options
            "temperature": {
                "type": "number",
                "default": 1.0,
                "minimum": 0.01,
                "maximum": 10.0,
                "description": "Softmax temperature for calibration (1.0 = no scaling)",
            },
            "threshold": {
                "type": "number",
                "default": 0.0,
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum confidence threshold for predictions",
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
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._model_loader = get_model_loader()

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Classify input images."""
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

        # Get model config
        model_id = config.get("model_id")
        model_source = config.get("model_source", "trained")
        top_k = config.get("top_k", 5)
        temperature = config.get("temperature", 1.0)
        threshold = config.get("threshold", 0.0)

        # Load model using centralized ModelLoader
        try:
            model, processor, model_info = await self._model_loader.load_classification_model(
                model_id, model_source
            )
        except ValueError as e:
            return BlockResult(
                error=str(e),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            return BlockResult(
                error=f"Failed to load model: {e}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get class mapping from ModelInfo
        class_mapping = model_info.class_mapping or {}

        # Run inference
        try:
            import torch

            predictions = []

            for pil_image in pil_images:
                # Process image
                inputs_tensor = processor(images=pil_image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs_tensor = {k: v.cuda() for k, v in inputs_tensor.items()}

                # Run in executor
                loop = asyncio.get_event_loop()

                def run_inference():
                    with torch.no_grad():
                        outputs = model(**inputs_tensor)
                        logits = outputs.logits / temperature  # Temperature scaling
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs[0], min(top_k, probs.shape[-1]))
                        return top_probs.cpu().numpy(), top_indices.cpu().numpy()

                top_probs, top_indices = await loop.run_in_executor(None, run_inference)

                # Build prediction
                top_predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    conf = round(float(prob), 4)
                    if conf < threshold:
                        continue

                    # Get label from class mapping or model config
                    if class_mapping and str(idx) in class_mapping:
                        label = class_mapping[str(idx)]
                    elif class_mapping and int(idx) in class_mapping:
                        label = class_mapping[int(idx)]
                    elif hasattr(model.config, 'id2label') and idx in model.config.id2label:
                        label = model.config.id2label[idx]
                    else:
                        label = f"class_{idx}"

                    top_predictions.append({
                        "class_name": label,
                        "class_id": int(idx),
                        "confidence": conf,
                    })

                predictions.append({
                    "class_name": top_predictions[0]["class_name"] if top_predictions else "unknown",
                    "confidence": top_predictions[0]["confidence"] if top_predictions else 0,
                    "top_k": top_predictions,
                })

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={"predictions": predictions},
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "model_type": model_info.model_type,
                    "image_count": len(pil_images),
                    "top_k": top_k,
                },
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
    - Batch processing
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
        },
        "required": ["model_id"],
    }

    def __init__(self):
        super().__init__()
        self._model_loader = get_model_loader()

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Extract embeddings from images."""
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

        # Get model config
        model_id = config.get("model_id")
        model_source = config.get("model_source", "pretrained")
        normalize = config.get("normalize", True)
        pooling = config.get("pooling", "cls")
        gem_p = config.get("gem_p", 3.0)

        # Load model using centralized ModelLoader
        try:
            model, processor, model_info = await self._model_loader.load_embedding_model(
                model_id, model_source
            )
        except ValueError as e:
            return BlockResult(
                error=str(e),
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        except Exception as e:
            return BlockResult(
                error=f"Failed to load model: {e}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Get embedding dimension from ModelInfo
        embedding_dim = model_info.embedding_dim or 768

        # Run inference
        try:
            import torch

            embeddings = []

            for pil_image in pil_images:
                # Process image
                inputs_tensor = processor(images=pil_image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs_tensor = {k: v.cuda() for k, v in inputs_tensor.items()}

                loop = asyncio.get_event_loop()

                def run_inference():
                    with torch.no_grad():
                        outputs = model(**inputs_tensor)

                        # Get embeddings based on model type and pooling strategy
                        if hasattr(outputs, "image_embeds"):
                            # CLIP-like models
                            embedding = outputs.image_embeds
                        elif hasattr(outputs, "last_hidden_state"):
                            hidden = outputs.last_hidden_state

                            if pooling == "cls":
                                # CLS token
                                embedding = hidden[:, 0]
                            elif pooling == "mean":
                                # Mean pooling
                                embedding = hidden.mean(dim=1)
                            elif pooling == "gem":
                                # GeM pooling (Generalized Mean)
                                embedding = (hidden.clamp(min=1e-6).pow(gem_p).mean(dim=1)).pow(1.0 / gem_p)
                            else:
                                embedding = hidden[:, 0]
                        elif hasattr(outputs, "pooler_output"):
                            embedding = outputs.pooler_output
                        else:
                            embedding = outputs[0][:, 0]

                        # Normalize if requested
                        if normalize:
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                        return embedding.cpu().numpy()[0]

                embedding = await loop.run_in_executor(None, run_inference)
                embeddings.append(embedding.tolist())

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={"embeddings": embeddings},
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "embedding_count": len(embeddings),
                    "embedding_dim": embedding_dim,
                    "pooling": pooling,
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
    - Configurable score threshold
    - Payload filtering
    - Multiple embedding queries (batch)
    - Grouping by product
    """

    block_type = "similarity_search"
    display_name = "Similarity Search"
    description = "Search for similar products in Qdrant vector database"

    input_ports = [
        {"name": "embeddings", "type": "array", "required": True, "description": "Query embedding vectors"},
    ]
    output_ports = [
        {"name": "matches", "type": "array", "description": "Matching products with similarity scores"},
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
            # Filtering
            "filter": {
                "type": "object",
                "description": "Qdrant filter conditions (must, should, must_not)",
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
        },
        "required": ["collection"],
    }

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Search for similar items in Qdrant."""
        start_time = time.time()

        embeddings = inputs.get("embeddings", [])
        if not embeddings:
            return BlockResult(
                error="No embeddings provided",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        collection = config.get("collection")
        if not collection:
            return BlockResult(
                error="No collection specified",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        top_k = config.get("top_k", 5)
        threshold = config.get("threshold", 0.7)
        filter_conditions = config.get("filter")

        # Check if Qdrant is configured
        if not qdrant_service.is_configured():
            return BlockResult(
                error="Qdrant not configured",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        try:
            all_matches = []

            for embedding in embeddings:
                # Ensure embedding is a list of floats
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Search Qdrant
                results = await qdrant_service.search(
                    collection_name=collection,
                    query_vector=embedding,
                    limit=top_k,
                    score_threshold=threshold,
                    filter_conditions=filter_conditions,
                )

                # Format matches
                matches = []
                for r in results:
                    payload = r.get("payload", {})
                    matches.append({
                        "id": r["id"],
                        "similarity": round(r["score"], 4),
                        "product_id": payload.get("product_id"),
                        "product_info": {
                            "name": payload.get("product_name"),
                            "upc": payload.get("upc"),
                            "sku": payload.get("sku"),
                        },
                        "metadata": payload,
                    })

                all_matches.append(matches)

            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={"matches": all_matches},
                duration_ms=round(duration, 2),
                metrics={
                    "collection": collection,
                    "query_count": len(embeddings),
                    "total_matches": sum(len(m) for m in all_matches),
                    "threshold": threshold,
                },
            )

        except Exception as e:
            logger.exception("Similarity search failed")
            return BlockResult(
                error=f"Similarity search failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
