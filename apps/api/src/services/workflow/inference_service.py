"""
Workflow Inference Service

Lightweight abstraction layer for GPU inference via RunPod.
Provides simple interface for detection, classification, and embedding tasks.

Architecture:
    Workflow Block → InferenceService → RunPod Unified Worker (GPU)

Future enhancements (not in v1):
- Result caching with Redis
- Request batching
- Hot/warm/cold tier routing
- Local inference fallback
"""

import base64
import io
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
import httpx

from services.runpod import runpod_service, EndpointType
from services.workflow.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Simple inference service for workflow blocks.

    All ML inference goes through unified RunPod worker.
    Models are auto-discovered from database (pretrained + trained).
    """

    def __init__(self):
        self.model_loader = get_model_loader()

    async def detect(
        self,
        model_id: str,
        image: Image.Image,
        confidence: float = 0.5,
        iou: float = 0.45,
        max_detections: int = 300,
        model_source: str = "pretrained",
        input_size: Optional[int] = None,
        agnostic_nms: bool = False,
        # Open-vocabulary detection params
        text_prompt: Optional[str] = None,  # For Grounding DINO: "person. car. dog."
        text_queries: Optional[List[str]] = None,  # For OWL-ViT: ["a cat", "a dog"]
        # Image URL for direct passing to worker (avoids base64 conversion)
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run object detection inference.

        Supports both fixed-vocabulary (YOLO, RT-DETR, D-FINE) and
        open-vocabulary detection (Grounding DINO, OWL-ViT).

        Args:
            model_id: Model ID from wf_pretrained_models or od_trained_models
            image: PIL Image to detect objects in
            confidence: Minimum confidence threshold (0-1)
            iou: IoU threshold for NMS (0-1)
            max_detections: Maximum number of detections to return
            model_source: "pretrained" or "trained"
            input_size: Input resolution (e.g., 640)
            agnostic_nms: Use class-agnostic NMS
            text_prompt: Text prompt for Grounding DINO (e.g., "person. car. dog.")
            text_queries: Text queries for OWL-ViT (e.g., ["a photo of a cat"])

        Returns:
            {
                "detections": [
                    {
                        "id": 0,
                        "class_name": "person",
                        "class_id": 0,
                        "confidence": 0.95,
                        "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                        "area": 0.02,
                    },
                    ...
                ],
                "count": 5,
                "image_size": {"width": 1920, "height": 1080},
            }

        Raises:
            ValueError: If model not found
            RuntimeError: If inference fails
        """
        # Get model info from database
        model_info = await self.model_loader.get_detection_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Detection model not found: {model_id} (source: {model_source})")

        logger.info(
            f"Running detection inference: model={model_info.name} ({model_info.model_type}), "
            f"source={model_source}, confidence={confidence}"
        )

        # Determine which endpoint and format to use
        # For trained/roboflow models, use OD_ANNOTATION endpoint (proven to work)
        # For open-vocab models (grounding_dino, etc), also use OD_ANNOTATION
        use_od_annotation = model_source == "trained" or text_prompt or text_queries

        if use_od_annotation:
            # Use OD_ANNOTATION endpoint format (same as od/ai.py)
            # Prefer image_url if provided, otherwise use base64
            if image_url and image_url.startswith(("http://", "https://")):
                final_image_url = image_url
            else:
                # Fallback to base64 data URL (may not work with all workers)
                image_b64 = self._image_to_base64(image)
                final_image_url = f"data:image/jpeg;base64,{image_b64}"
                logger.warning("Using base64 data URL - may not work with OD_ANNOTATION worker")

            if model_source == "trained":
                # Roboflow/trained model format
                # Get classes as list from class_mapping
                classes = []
                if model_info.class_mapping:
                    # class_mapping is {idx: class_name}
                    max_idx = max(model_info.class_mapping.keys()) if model_info.class_mapping else -1
                    classes = [model_info.class_mapping.get(i, f"class_{i}") for i in range(max_idx + 1)]

                job_input = {
                    "task": "detect",
                    "model": "roboflow",
                    "model_config": {
                        "checkpoint_url": model_info.checkpoint_url,
                        "architecture": model_info.model_type,
                        "classes": classes,
                    },
                    "image_url": final_image_url,
                    "box_threshold": confidence,
                }
            else:
                # Open-vocab model (grounding_dino, florence2, etc)
                job_input = {
                    "task": "detect",
                    "model": model_id,  # grounding_dino, florence2, etc
                    "image_url": final_image_url,
                    "text_prompt": text_prompt,
                    "box_threshold": confidence,
                    "text_threshold": 0.25,
                }

            endpoint_type = EndpointType.OD_ANNOTATION
        else:
            # Use INFERENCE endpoint for pretrained YOLO models
            job_input = {
                "task": "detection",
                "model_id": model_id,
                "model_source": model_source,
                "model_type": model_info.model_type,
                "checkpoint_url": model_info.checkpoint_url,
                "class_mapping": model_info.class_mapping,
                "image": self._image_to_base64(image),
                "config": {
                    "confidence": confidence,
                    "iou_threshold": iou,
                    "max_detections": max_detections,
                    "agnostic_nms": agnostic_nms,
                }
            }
            if input_size:
                job_input["config"]["input_size"] = input_size
            endpoint_type = EndpointType.INFERENCE

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=endpoint_type,
                input_data=job_input,
                timeout=120,
            )

            # Check for job failure
            if result.get("status") == "FAILED":
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Detection job failed: {error_msg}")
                raise RuntimeError(f"Detection inference failed: {error_msg}")

            output = result.get("output", {})

            # Handle OD_ANNOTATION response format
            if use_od_annotation:
                if output.get("status") == "error":
                    error_msg = output.get("traceback", "Unknown error")
                    logger.error(f"Detection inference failed: {error_msg[:200]}")
                    raise RuntimeError(f"Detection inference failed")

                # Convert OD_ANNOTATION predictions to workflow format
                predictions = output.get("predictions", [])
                img_width, img_height = image.size

                detections = []
                for i, pred in enumerate(predictions):
                    bbox = pred.get("bbox", {})
                    # OD_ANNOTATION returns absolute pixel coords
                    x1 = bbox.get("x", 0)
                    y1 = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)

                    # Convert to normalized coords
                    detections.append({
                        "id": i,
                        "class_name": pred.get("label", "unknown"),
                        "class_id": pred.get("class_id", i),
                        "confidence": pred.get("confidence", 0),
                        "bbox": {
                            "x1": x1 / img_width,
                            "y1": y1 / img_height,
                            "x2": (x1 + w) / img_width,
                            "y2": (y1 + h) / img_height,
                        },
                        "bbox_abs": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x1 + w,
                            "y2": y1 + h,
                        },
                    })

                logger.info(f"Detection complete: {len(detections)} detections")
                return {
                    "detections": detections,
                    "count": len(detections),
                    "image_size": {"width": img_width, "height": img_height},
                }
            else:
                # Handle INFERENCE endpoint response
                if not output.get("success"):
                    error_msg = output.get("error", result.get("error", "Unknown error"))
                    logger.error(f"Detection inference failed: {error_msg}")
                    raise RuntimeError(f"Detection inference failed: {error_msg}")

                detection_result = output.get("result", {})
                if "detections" not in detection_result:
                    raise RuntimeError("Invalid response: missing 'detections' field")

                logger.info(
                    f"Detection complete: {detection_result['count']} detections, "
                    f"took {output.get('metadata', {}).get('inference_time_ms', 0):.0f}ms"
                )
                return detection_result

        except httpx.HTTPStatusError as e:
            logger.error(f"RunPod HTTP error: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")
        except httpx.TimeoutException:
            logger.error(f"RunPod timeout after 120s")
            raise RuntimeError("Inference timeout - GPU worker may be cold starting")
        except RuntimeError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected inference error")
            raise RuntimeError(f"Inference error: {e}")

    async def classify(
        self,
        model_id: str,
        image: Image.Image,
        top_k: int = 5,
        model_source: str = "trained",
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run image classification inference.

        Args:
            model_id: Model ID from wf_pretrained_models or cls_trained_models
            image: PIL Image to classify
            top_k: Number of top predictions to return
            model_source: "pretrained" or "trained"
            threshold: Minimum confidence threshold (0-1)

        Returns:
            {
                "predictions": [
                    {
                        "class_name": "cat",
                        "class_id": 0,
                        "confidence": 0.98,
                    },
                    ...
                ],
                "top_class": "cat",
                "top_confidence": 0.98,
            }

        Raises:
            ValueError: If model not found
            RuntimeError: If inference fails
        """
        # Get model info from database
        model_info = await self.model_loader.get_classification_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Classification model not found: {model_id} (source: {model_source})")

        logger.info(
            f"Running classification inference: model={model_info.name} ({model_info.model_type}), "
            f"source={model_source}, top_k={top_k}"
        )

        # Prepare job input
        job_input = {
            "task": "classification",
            "model_id": model_id,
            "model_source": model_source,
            "model_type": model_info.model_type,
            "checkpoint_url": model_info.checkpoint_url,
            "class_mapping": model_info.class_mapping,
            "num_classes": model_info.config.get("num_classes") if model_info.config else None,
            "image": self._image_to_base64(image),
            "config": {
                "top_k": top_k,
                "threshold": threshold,
            }
        }

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                timeout=120,  # Increased for cold start
            )

            # RunPod returns: {status, output: {success, result, metadata}}
            output = result.get("output", {})
            if not output.get("success"):
                error_msg = output.get("error", result.get("error", "Unknown error"))
                logger.error(f"Classification inference failed: {error_msg}")
                raise RuntimeError(f"Classification inference failed: {error_msg}")

            # Validate response structure
            classification_result = output.get("result", {})
            if "predictions" not in classification_result:
                raise RuntimeError("Invalid response: missing 'predictions' field")

            logger.info(
                f"Classification complete: top_class={classification_result.get('top_class', 'unknown')}, "
                f"confidence={classification_result.get('top_confidence', 0.0):.2f}"
            )

            return classification_result

        except httpx.HTTPStatusError as e:
            logger.error(f"RunPod HTTP error: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")
        except httpx.TimeoutException:
            logger.error(f"RunPod timeout after 120s")
            raise RuntimeError("Inference timeout - GPU worker may be cold starting")
        except Exception as e:
            logger.exception(f"Unexpected inference error")
            raise RuntimeError(f"Inference error: {e}")

    async def embed(
        self,
        model_id: str,
        image: Image.Image,
        model_source: str = "pretrained",
        normalize: bool = True,
        pooling: str = "cls",
        input_size: int = 224,
        gem_p: float = 3.0,
        multi_scale: bool = False,
        multi_scale_factors: list = None,
        multi_scale_agg: str = "concat",
        pca_enabled: bool = False,
        pca_dim: int = 256,
        output_format: str = "vector",
    ) -> Dict[str, Any]:
        """
        Extract image embedding via RunPod GPU worker.

        Args:
            model_id: Model ID from wf_pretrained_models or trained_models
            image: PIL Image to extract embedding from
            model_source: "pretrained" or "trained"
            normalize: Normalize embedding to unit length
            pooling: Pooling strategy ("cls", "mean", "gem")
            input_size: Input image size for model
            gem_p: Power parameter for GeM pooling
            multi_scale: Enable multi-scale extraction
            multi_scale_factors: Scale factors for multi-scale (e.g., [1.0, 0.75, 0.5])
            multi_scale_agg: Aggregation method ("concat", "mean", "max")
            pca_enabled: Enable PCA dimensionality reduction
            pca_dim: Target dimension for PCA
            output_format: Output format ("vector", "base64", "vector_with_meta")

        Returns:
            {
                "embedding": [0.1, 0.2, ...],  # Vector of floats or base64 string
                "embedding_dim": 768,
                "normalized": True,
            }

        Raises:
            ValueError: If model not found
            RuntimeError: If inference fails
        """
        # Get model info from database
        model_info = await self.model_loader.get_embedding_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Embedding model not found: {model_id} (source: {model_source})")

        logger.info(
            f"Running embedding extraction: model={model_info.name} ({model_info.model_type}), "
            f"source={model_source}, dim={model_info.embedding_dim}"
        )

        # Prepare job input with all SOTA options
        job_input = {
            "task": "embedding",
            "model_id": model_id,
            "model_source": model_source,
            "model_type": model_info.model_type,
            "checkpoint_url": model_info.checkpoint_url,
            "embedding_dim": model_info.embedding_dim,
            "image": self._image_to_base64(image),
            "config": {
                "normalize": normalize,
                "pooling": pooling,
                "input_size": input_size,
                "gem_p": gem_p,
                "multi_scale": multi_scale,
                "multi_scale_factors": multi_scale_factors or [1.0],
                "multi_scale_agg": multi_scale_agg,
                "pca_enabled": pca_enabled,
                "pca_dim": pca_dim,
                "output_format": output_format,
            }
        }

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                timeout=120,  # Increased for cold start
            )

            # RunPod returns: {status, output: {success, result, metadata}}
            output = result.get("output", {})
            if not output.get("success"):
                error_msg = output.get("error", result.get("error", "Unknown error"))
                logger.error(f"Embedding extraction failed: {error_msg}")
                raise RuntimeError(f"Embedding extraction failed: {error_msg}")

            # Validate response structure
            embedding_result = output.get("result", {})
            if "embedding" not in embedding_result:
                raise RuntimeError("Invalid response: missing 'embedding' field")

            logger.info(
                f"Embedding extraction complete: dim={embedding_result.get('embedding_dim', 0)}"
            )

            return embedding_result

        except httpx.HTTPStatusError as e:
            logger.error(f"RunPod HTTP error: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")
        except httpx.TimeoutException:
            logger.error(f"RunPod timeout after 120s")
            raise RuntimeError("Inference timeout - GPU worker may be cold starting")
        except Exception as e:
            logger.exception(f"Unexpected inference error")
            raise RuntimeError(f"Inference error: {e}")

    async def batch_detect(
        self,
        model_id: str,
        images: List[Image.Image],
        confidence: float = 0.5,
        iou: float = 0.45,
        model_source: str = "pretrained",
    ) -> List[Dict[str, Any]]:
        """
        Run batch detection on multiple images.

        Note: This is a simple loop for now. Future versions may implement
        true batching on the worker side for efficiency.

        Args:
            model_id: Model ID
            images: List of PIL Images
            confidence: Confidence threshold
            iou: IoU threshold
            model_source: "pretrained" or "trained"

        Returns:
            List of detection results (one per image)
        """
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = await self.detect(
                model_id=model_id,
                image=image,
                confidence=confidence,
                iou=iou,
                model_source=model_source,
            )
            results.append(result)
        return results

    async def segment(
        self,
        model_id: str,
        image: Image.Image,
        model_source: str = "pretrained",
        input_boxes: Optional[List[List[float]]] = None,
        input_points: Optional[List[List[float]]] = None,
        input_labels: Optional[List[int]] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SAM segmentation inference.

        Args:
            model_id: Model ID from wf_pretrained_models (sam2-base, sam2-large, etc.)
            image: PIL Image to segment
            model_source: "pretrained" or "trained"
            input_boxes: List of bounding boxes [[x1,y1,x2,y2], ...] in pixel coordinates
            input_points: List of point coordinates [[x, y], ...]
            input_labels: List of point labels (1=foreground, 0=background)
            multimask_output: Return multiple mask candidates per prompt

        Returns:
            {
                "masks": [
                    {
                        "id": 0,
                        "score": 0.98,
                        "area": 12345,
                        "area_ratio": 0.05,
                        "rle": {"counts": [...], "size": [H, W]},
                        "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8},
                    },
                    ...
                ],
                "count": 5,
                "image_size": {"width": 1920, "height": 1080},
            }

        Raises:
            ValueError: If model not found
            RuntimeError: If inference fails
        """
        # Get model info from database
        model_info = await self.model_loader.get_segmentation_model_info(model_id, model_source)
        if not model_info:
            raise ValueError(f"Segmentation model not found: {model_id} (source: {model_source})")

        logger.info(
            f"Running segmentation inference: model={model_info.name} ({model_info.model_type}), "
            f"source={model_source}"
        )

        # Prepare job input
        job_input = {
            "task": "segmentation",
            "model_id": model_id,
            "model_source": model_source,
            "model_type": model_info.model_type,
            "checkpoint_url": model_info.checkpoint_url,
            "image": self._image_to_base64(image),
            "input_boxes": input_boxes,
            "input_points": input_points,
            "input_labels": input_labels,
            "config": {
                "multimask_output": multimask_output,
            }
        }

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                timeout=120,
            )

            output = result.get("output", {})
            if not output.get("success"):
                error_msg = output.get("error", result.get("error", "Unknown error"))
                logger.error(f"Segmentation inference failed: {error_msg}")
                raise RuntimeError(f"Segmentation inference failed: {error_msg}")

            segmentation_result = output.get("result", {})
            if "masks" not in segmentation_result:
                raise RuntimeError("Invalid response: missing 'masks' field")

            logger.info(
                f"Segmentation complete: {segmentation_result['count']} masks"
            )

            return segmentation_result

        except httpx.HTTPStatusError as e:
            logger.error(f"RunPod HTTP error: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")
        except httpx.TimeoutException:
            logger.error(f"RunPod timeout after 120s")
            raise RuntimeError("Inference timeout - GPU worker may be cold starting")
        except Exception as e:
            logger.exception(f"Unexpected inference error")
            raise RuntimeError(f"Inference error: {e}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64-encoded JPEG string
        """
        buffer = io.BytesIO()

        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha as mask
            image = rgb_image
        elif image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Save as JPEG with high quality
        image.save(buffer, format="JPEG", quality=95)

        # Encode to base64
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64


# Singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """
    Get the singleton inference service instance.

    Returns:
        InferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
