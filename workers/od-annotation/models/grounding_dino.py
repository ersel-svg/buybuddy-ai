"""
Grounding DINO Model Wrapper using HuggingFace Transformers.

SOTA open-vocabulary object detection with text prompts.
Auto-downloads model from HuggingFace Hub.
"""

import io
from typing import Any
import httpx
from PIL import Image
import torch
from loguru import logger

from .base import BaseModel


class GroundingDINOModel(BaseModel):
    """
    Grounding DINO: Open-Set Object Detection with Text Prompts.

    Uses HuggingFace transformers for automatic model download.

    Input format for text_prompt:
    - Multiple classes: "shelf . product . price tag"
    - Single class: "product"

    Classes are separated by " . " (space dot space)
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        **kwargs,  # Accept extra args for compatibility
    ):
        super().__init__(device)
        self.model_id = model_id
        self._processor = None

    def _load_model(self) -> Any:
        """Load Grounding DINO model from HuggingFace."""
        logger.info(f"Loading Grounding DINO from {self.model_id}...")

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device)

            logger.info(f"Grounding DINO loaded successfully on {self.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            raise

    @property
    def processor(self):
        """Get processor, loading model if needed."""
        if self._processor is None:
            _ = self.model  # This triggers _load_model
        return self._processor

    def predict(
        self,
        image_url: str,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        **kwargs,
    ) -> list[dict]:
        """
        Run Grounding DINO detection.

        Args:
            image_url: URL of the image
            text_prompt: Classes to detect, separated by " . "
            box_threshold: Box confidence threshold
            text_threshold: Text matching threshold

        Returns:
            List of predictions with bbox, label, confidence
        """
        # Download image
        logger.debug(f"Downloading image: {image_url[:80]}...")
        response = httpx.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Get image dimensions
        img_width, img_height = image.size

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        # Note: parameter is 'threshold' not 'box_threshold'
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(img_height, img_width)],
        )[0]

        # Convert to our format
        predictions = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box

            # Convert to normalized format [x, y, width, height]
            predictions.append({
                "bbox": {
                    "x": float(x1 / img_width),
                    "y": float(y1 / img_height),
                    "width": float((x2 - x1) / img_width),
                    "height": float((y2 - y1) / img_height),
                },
                "label": label.strip(),
                "confidence": float(score),
            })

        logger.debug(f"Grounding DINO found {len(predictions)} objects")
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
        Run detection with Non-Maximum Suppression.

        Useful when text prompt contains similar/overlapping concepts.
        """
        import supervision as sv
        import numpy as np

        # Get raw predictions
        predictions = self.predict(
            image_url=image_url,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if not predictions:
            return predictions

        # Convert to supervision format for NMS
        boxes = []
        confidences = []
        class_ids = []
        labels = []
        label_to_id = {}

        for pred in predictions:
            bbox = pred["bbox"]
            boxes.append([
                bbox["x"],
                bbox["y"],
                bbox["x"] + bbox["width"],
                bbox["y"] + bbox["height"],
            ])
            confidences.append(pred["confidence"])

            label = pred["label"]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            class_ids.append(label_to_id[label])
            labels.append(label)

        # Apply NMS
        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )

        detections = detections.with_nms(threshold=nms_threshold)

        # Convert back to our format
        nms_predictions = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            nms_predictions.append({
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "label": labels[i],
                "confidence": float(detections.confidence[i]),
            })

        logger.debug(f"After NMS: {len(nms_predictions)} objects (was {len(predictions)})")
        return nms_predictions
