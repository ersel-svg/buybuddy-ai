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
    ) -> Dict[str, Any]:
        """
        Run object detection inference.

        Args:
            model_id: Model ID from wf_pretrained_models or od_trained_models
            image: PIL Image to detect objects in
            confidence: Minimum confidence threshold (0-1)
            iou: IoU threshold for NMS (0-1)
            max_detections: Maximum number of detections to return
            model_source: "pretrained" or "trained"
            input_size: Input resolution (e.g., 640)
            agnostic_nms: Use class-agnostic NMS

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

        # Prepare job input
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

        # Add optional params
        if input_size:
            job_input["config"]["input_size"] = input_size

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                timeout=120,  # Increased for cold start (can take 30-60s)
            )

            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Detection inference failed: {error_msg}")
                raise RuntimeError(f"Detection inference failed: {error_msg}")

            # Validate response structure
            detection_result = result.get("result", {})
            if "detections" not in detection_result:
                raise RuntimeError("Invalid response: missing 'detections' field")

            logger.info(
                f"Detection complete: {detection_result['count']} detections, "
                f"took {result.get('metadata', {}).get('inference_time_ms', 0):.0f}ms"
            )

            return detection_result

        except httpx.HTTPStatusError as e:
            logger.error(f"RunPod HTTP error: {e}")
            raise RuntimeError(f"RunPod request failed: {e}")
        except httpx.TimeoutException:
            logger.error(f"RunPod timeout after 60s")
            raise RuntimeError("Inference timeout - GPU worker may be cold starting")
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

            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Classification inference failed: {error_msg}")
                raise RuntimeError(f"Classification inference failed: {error_msg}")

            # Validate response structure
            classification_result = result.get("result", {})
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
    ) -> Dict[str, Any]:
        """
        Extract image embedding.

        Args:
            model_id: Model ID from wf_pretrained_models or trained_models
            image: PIL Image to extract embedding from
            model_source: "pretrained" or "trained"
            normalize: Normalize embedding to unit length
            pooling: Pooling strategy ("cls", "mean", "gem")

        Returns:
            {
                "embedding": [0.1, 0.2, ...],  # Vector of floats
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

        # Prepare job input
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
            }
        }

        # Submit to RunPod
        try:
            result = await runpod_service.submit_job_sync(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                timeout=120,  # Increased for cold start
            )

            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Embedding extraction failed: {error_msg}")
                raise RuntimeError(f"Embedding extraction failed: {error_msg}")

            # Validate response structure
            embedding_result = result.get("result", {})
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
