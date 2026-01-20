"""
SAM (Segment Anything Model) Wrapper using HuggingFace Transformers.

Interactive segmentation with point and box prompts.
Auto-downloads model from HuggingFace Hub.
"""

import io
import torch
import numpy as np
from typing import Any, Optional
import httpx
from PIL import Image
from loguru import logger

from .base import BaseSegmentationModel


class SAM2Model(BaseSegmentationModel):
    """
    SAM: Segment Anything Model via HuggingFace Transformers.

    Supports:
    - Point prompts: Click on object to segment
    - Box prompts: Draw box around object for refined segmentation

    Returns masks that can be converted to bounding boxes.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        device: str = "cuda",
        **kwargs,  # Accept extra args for compatibility
    ):
        super().__init__(device)
        self.model_id = model_id
        self._processor = None

    def _load_model(self) -> Any:
        """Load SAM model from HuggingFace."""
        logger.info(f"Loading SAM from {self.model_id}...")

        try:
            from transformers import SamModel, SamProcessor

            self._processor = SamProcessor.from_pretrained(self.model_id)
            model = SamModel.from_pretrained(self.model_id).to(self.device)

            logger.info(f"SAM loaded successfully on {self.device}")
            return model

        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
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
        SAM doesn't use text prompts for detection.
        This method is here for interface compatibility.
        """
        logger.warning("SAM predict() called - SAM doesn't do text-based detection")
        return []

    def segment_point(
        self,
        image_url: str,
        point: tuple[float, float],
        label: int = 1,
        return_mask: bool = True,
        **kwargs,
    ) -> dict:
        """
        Segment object at a point.

        Args:
            image_url: URL of the image
            point: (x, y) in normalized 0-1 coords
            label: 1 for foreground, 0 for background
            return_mask: Whether to return the mask as base64

        Returns:
            Dict with bbox, confidence, and optionally mask
        """
        # Download image
        logger.debug(f"Downloading image: {image_url[:80]}...")
        response = httpx.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        width, height = image.size

        # Convert normalized point to absolute coords
        point_x = int(point[0] * width)
        point_y = int(point[1] * height)
        input_points = [[[point_x, point_y]]]
        input_labels = [[label]]

        logger.debug(f"Segmenting at point: ({point_x}, {point_y})")

        # Process inputs
        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Convert to float32 before numpy (BFloat16 not supported)
        scores = outputs.iou_scores.cpu().float().numpy()[0][0]

        # Get best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[0][0][best_idx].numpy()
        best_score = float(scores[best_idx])

        # Convert mask to bbox
        bbox_abs = self.mask_to_bbox(best_mask)

        if bbox_abs is None:
            logger.warning("Empty mask produced")
            return {
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "confidence": 0,
                "mask": None,
            }

        # Normalize bbox
        x1, y1, x2, y2 = bbox_abs
        bbox = {
            "x": x1 / width,
            "y": y1 / height,
            "width": (x2 - x1) / width,
            "height": (y2 - y1) / height,
        }

        result = {
            "bbox": bbox,
            "confidence": best_score,
        }

        if return_mask:
            result["mask"] = self.mask_to_base64(best_mask)

        return result

    def segment_box(
        self,
        image_url: str,
        box: list[float],
        return_mask: bool = True,
        **kwargs,
    ) -> dict:
        """
        Segment object within a box.

        Args:
            image_url: URL of the image
            box: [x, y, width, height] in normalized 0-1 coords
            return_mask: Whether to return the mask as base64

        Returns:
            Dict with refined bbox, confidence, and optionally mask
        """
        # Download image
        logger.debug(f"Downloading image: {image_url[:80]}...")
        response = httpx.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        width, height = image.size

        # Convert normalized box to absolute coords (xyxy format)
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int((box[0] + box[2]) * width)
        y2 = int((box[1] + box[3]) * height)
        input_boxes = [[[x1, y1, x2, y2]]]

        logger.debug(f"Segmenting in box: ({x1}, {y1}, {x2}, {y2})")

        # Process inputs
        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Convert to float32 before numpy (BFloat16 not supported)
        scores = outputs.iou_scores.cpu().float().numpy()[0][0]

        # Get best mask
        best_idx = np.argmax(scores)
        best_mask = masks[0][0][best_idx].numpy()
        best_score = float(scores[best_idx])

        # Convert mask to refined bbox
        bbox_abs = self.mask_to_bbox(best_mask)

        if bbox_abs is None:
            logger.warning("Empty mask produced")
            return {
                "bbox": {"x": box[0], "y": box[1], "width": box[2], "height": box[3]},
                "confidence": 0,
                "mask": None,
            }

        # Normalize refined bbox
        rx1, ry1, rx2, ry2 = bbox_abs
        refined_bbox = {
            "x": rx1 / width,
            "y": ry1 / height,
            "width": (rx2 - rx1) / width,
            "height": (ry2 - ry1) / height,
        }

        result = {
            "bbox": refined_bbox,
            "confidence": best_score,
        }

        if return_mask:
            result["mask"] = self.mask_to_base64(best_mask)

        return result

    def segment_multi_point(
        self,
        image_url: str,
        points: list[tuple[float, float]],
        labels: list[int],
        return_mask: bool = True,
        **kwargs,
    ) -> dict:
        """
        Segment using multiple point prompts.

        Args:
            image_url: URL of the image
            points: List of (x, y) in normalized 0-1 coords
            labels: List of labels (1=foreground, 0=background)
            return_mask: Whether to return mask

        Returns:
            Dict with bbox, confidence, and optionally mask
        """
        # Download image
        response = httpx.get(image_url, timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        width, height = image.size

        # Convert points to absolute coords
        input_points = [[
            [int(p[0] * width), int(p[1] * height)]
            for p in points
        ]]
        input_labels = [labels]

        # Process inputs
        inputs = self.processor(
            image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # Convert to float32 before numpy (BFloat16 not supported)
        scores = outputs.iou_scores.cpu().float().numpy()[0][0]

        # Get best mask
        best_idx = np.argmax(scores)
        best_mask = masks[0][0][best_idx].numpy()
        best_score = float(scores[best_idx])

        # Convert mask to bbox
        bbox_abs = self.mask_to_bbox(best_mask)

        if bbox_abs is None:
            return {
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "confidence": 0,
                "mask": None,
            }

        x1, y1, x2, y2 = bbox_abs
        bbox = {
            "x": x1 / width,
            "y": y1 / height,
            "width": (x2 - x1) / width,
            "height": (y2 - y1) / height,
        }

        result = {
            "bbox": bbox,
            "confidence": best_score,
        }

        if return_mask:
            result["mask"] = self.mask_to_base64(best_mask)

        return result
