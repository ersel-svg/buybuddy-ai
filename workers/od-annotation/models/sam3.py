"""
SAM3 Model Wrapper with Text Prompt Support.

SAM3 is a video segmentation model that supports text prompts for grounding.
Unlike SAM2, SAM3 can find objects using natural language descriptions.

Based on: workers/segmentation-preview/src/preview.py
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional
import io

import torch
import numpy as np
import cv2
from PIL import Image
from loguru import logger
from scipy import ndimage

from .base import BaseSegmentationModel


class SAM3Model(BaseSegmentationModel):
    """
    SAM3: Segment Anything Model 3 with Text Prompt Support.

    Unlike SAM2, SAM3 can:
    - Accept text prompts like "red can", "bottle", "product"
    - Find and segment objects based on natural language
    - Also supports point prompts for refinement

    Note: SAM3 is designed for video, so we create a 1-frame "video"
    from the input image for processing.
    """

    # Minimum area threshold as fraction of image (0.001 = 0.1% of image)
    MIN_AREA_FRACTION = 0.0005
    # Maximum number of predictions to return
    MAX_PREDICTIONS = 500

    def __init__(
        self,
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ):
        super().__init__(device)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._video_predictor = None
        self.temp_dir = Path(tempfile.gettempdir()) / "sam3-annotation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> Any:
        """Load SAM3 video predictor."""
        logger.info("Loading SAM3 model...")

        try:
            # Authenticate with HuggingFace for model download
            if self.hf_token:
                from huggingface_hub import login
                login(token=self.hf_token, add_to_git_credential=False)
                logger.info("HuggingFace authenticated")
            else:
                logger.warning("HF_TOKEN not set, SAM3 download may fail")

            from sam3.model_builder import build_sam3_video_predictor
            self._video_predictor = build_sam3_video_predictor()

            logger.info("SAM3 model loaded successfully")

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU Memory used: {mem:.1f} GB")

            return self._video_predictor

        except ImportError as e:
            logger.error(f"SAM3 import error: {e}")
            raise RuntimeError("SAM3 not available. Make sure sam3 package is installed.")
        except Exception as e:
            error_msg = str(e)
            # Check for gated model access error
            if "401" in error_msg or "gated" in error_msg.lower() or "restricted" in error_msg.lower():
                logger.error(
                    "SAM3 access denied. This is a gated model. To use SAM3:\n"
                    "  1. Accept the license at: https://huggingface.co/facebook/sam3\n"
                    "  2. Set HF_TOKEN environment variable with your HuggingFace token\n"
                    "  3. Rebuild the Docker image"
                )
                raise RuntimeError(
                    "SAM3 access denied. Accept license at https://huggingface.co/facebook/sam3 "
                    "and set HF_TOKEN environment variable."
                )
            logger.error(f"SAM3 load error: {e}")
            raise

    @property
    def video_predictor(self):
        """Get video predictor (lazy load)."""
        if self._video_predictor is None:
            self._load_model()
        return self._video_predictor

    def predict(
        self,
        image_url: str,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Run SAM3 detection using text prompt.

        Returns MULTIPLE predictions - one for each detected object instance.
        SAM3 returns separate masks for each object with confidence scores.

        Args:
            image_url: URL of the image
            text_prompt: Text description of object to find (e.g., "red can", "product")
            box_threshold: Minimum confidence threshold (0-1)
            text_threshold: Not used (for API compatibility)

        Returns:
            List of predictions with bbox, label, confidence
        """
        if not text_prompt:
            logger.warning("SAM3 requires a text prompt for detection")
            return []

        # Download image
        image = self.download_image(image_url)
        width, height = image.size
        image_area = width * height

        # Create temporary video file from single image
        video_path = self._create_temp_video(image)

        try:
            # Run SAM3 segmentation - get ALL masks with scores
            masks_with_scores = self._segment_with_text_all_masks_scored(
                video_path, text_prompt, width, height
            )

            if not masks_with_scores:
                logger.warning(f"SAM3: No objects found for prompt '{text_prompt}'")
                return []

            predictions = []
            min_area = int(image_area * self.MIN_AREA_FRACTION)

            # Process each mask (SAM3 returns one mask per detected object)
            for mask, confidence in masks_with_scores:
                # Skip if below threshold
                if confidence < box_threshold:
                    continue

                # Get bounding box from mask
                bbox_abs = self.mask_to_bbox(mask)
                if bbox_abs is None:
                    continue

                x1, y1, x2, y2 = bbox_abs

                # Check minimum area
                area = np.sum(mask)
                if area < min_area:
                    continue

                predictions.append({
                    "bbox": {
                        "x": x1 / width,
                        "y": y1 / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                    },
                    "label": text_prompt,
                    "confidence": round(confidence, 3),
                })

            # Sort by confidence (highest first) and limit
            predictions.sort(key=lambda p: p["confidence"], reverse=True)
            predictions = predictions[:self.MAX_PREDICTIONS]

            logger.info(f"SAM3 found {len(predictions)} objects for prompt '{text_prompt}'")
            return predictions

        finally:
            # Cleanup temp video
            self._cleanup_temp_video(video_path)

    def segment_point(
        self,
        image_url: str,
        point: tuple[float, float],
        label: int = 1,
        return_mask: bool = True,
    ) -> dict:
        """
        Segment object at a point using SAM3.

        Args:
            image_url: URL of the image
            point: (x, y) in normalized 0-1 coords
            label: 1 for foreground, 0 for background
            return_mask: Whether to return the mask as base64

        Returns:
            Dict with bbox, confidence, and optionally mask
        """
        image = self.download_image(image_url)
        width, height = image.size

        # Convert normalized point to pixel coords
        point_px = (int(point[0] * width), int(point[1] * height))

        video_path = self._create_temp_video(image)

        try:
            mask = self._segment_with_point(
                video_path, point_px, label, width, height
            )

            if mask is None:
                return {
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "confidence": 0,
                    "mask": None,
                }

            bbox_abs = self.mask_to_bbox(mask)

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
                "confidence": 1.0,
            }

            if return_mask:
                result["mask"] = self.mask_to_base64(mask)

            return result

        finally:
            self._cleanup_temp_video(video_path)

    def segment_box(
        self,
        image_url: str,
        box: list[float],
        return_mask: bool = True,
    ) -> dict:
        """
        Segment object within a box using SAM3.
        Uses box center as point prompt.

        Args:
            image_url: URL of the image
            box: [x, y, width, height] in normalized 0-1 coords
            return_mask: Whether to return the mask

        Returns:
            Dict with refined bbox and optionally mask
        """
        # Use box center as point prompt
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2

        return self.segment_point(
            image_url=image_url,
            point=(center_x, center_y),
            label=1,
            return_mask=return_mask,
        )

    def segment_with_text_and_point(
        self,
        image_url: str,
        text_prompt: str,
        points: list[tuple[float, float]],
        labels: list[int],
        return_mask: bool = True,
    ) -> dict:
        """
        Segment using both text prompt and point prompts for refinement.

        Args:
            image_url: URL of the image
            text_prompt: Text description of object
            points: List of (x, y) in normalized 0-1 coords
            labels: List of labels (1=foreground, 0=background)
            return_mask: Whether to return mask

        Returns:
            Dict with bbox and optionally mask
        """
        image = self.download_image(image_url)
        width, height = image.size

        # Convert points to pixel coords
        points_px = [(int(p[0] * width), int(p[1] * height)) for p in points]

        video_path = self._create_temp_video(image)

        try:
            mask = self._segment_with_text_and_points(
                video_path, text_prompt, points_px, labels, width, height
            )

            if mask is None:
                return {
                    "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                    "confidence": 0,
                    "mask": None,
                }

            bbox_abs = self.mask_to_bbox(mask)

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
                "confidence": 1.0,
            }

            if return_mask:
                result["mask"] = self.mask_to_base64(mask)

            return result

        finally:
            self._cleanup_temp_video(video_path)

    def _create_temp_video(self, image: Image.Image) -> Path:
        """Create a temporary 1-frame video from image for SAM3."""
        job_id = os.urandom(8).hex()
        video_path = self.temp_dir / f"{job_id}.mp4"

        # Convert PIL to numpy BGR
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]

        # Write single-frame video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (width, height))
        out.write(frame)
        out.release()

        return video_path

    def _cleanup_temp_video(self, video_path: Path):
        """Remove temporary video file."""
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp video: {e}")

    def _find_connected_components(
        self,
        mask: np.ndarray,
        min_area: int,
    ) -> list[tuple[tuple[int, int, int, int], int, np.ndarray]]:
        """
        Find connected components in a binary mask.

        Args:
            mask: Binary mask array (H, W)
            min_area: Minimum area in pixels for a component

        Returns:
            List of (bbox, area, component_mask) tuples
            bbox is (x1, y1, x2, y2) in absolute pixels
        """
        # Label connected components
        labeled_array, num_features = ndimage.label(mask)

        components = []
        for label_id in range(1, num_features + 1):
            # Extract this component's mask
            component_mask = (labeled_array == label_id).astype(np.uint8)

            # Calculate area
            area = np.sum(component_mask)
            if area < min_area:
                continue

            # Get bounding box
            rows = np.any(component_mask, axis=1)
            cols = np.any(component_mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            bbox = (int(x1), int(y1), int(x2), int(y2))
            components.append((bbox, area, component_mask))

        # Sort by area (largest first)
        components.sort(key=lambda x: x[1], reverse=True)

        return components

    def _segment_with_text_all_masks_scored(
        self,
        video_path: Path,
        text_prompt: str,
        width: int,
        height: int,
    ) -> list[tuple[np.ndarray, float]]:
        """Run SAM3 segmentation with text prompt - returns ALL masks with confidence scores."""
        # Start session
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]

        try:
            # Add text prompt
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text_prompt,
                )
            )

            # Propagate (just 1 frame)
            propagate_result = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )

            # Extract ALL masks with their confidence scores
            return self._extract_all_masks_with_scores(propagate_result)

        finally:
            # Close session
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _segment_with_text_all_masks(
        self,
        video_path: Path,
        text_prompt: str,
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        """Run SAM3 segmentation with text prompt - returns ALL masks (without scores)."""
        results = self._segment_with_text_all_masks_scored(video_path, text_prompt, width, height)
        return [mask for mask, _ in results]

    def _segment_with_text(
        self,
        video_path: Path,
        text_prompt: str,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """Run SAM3 segmentation with text prompt (legacy - returns single mask)."""
        masks = self._segment_with_text_all_masks(video_path, text_prompt, width, height)
        return masks[0] if masks else None

    def _segment_with_point(
        self,
        video_path: Path,
        point: tuple[int, int],
        label: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """Run SAM3 segmentation with point prompt."""
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]

        try:
            # Add generic text prompt first (SAM3 requirement)
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text="object",
                )
            )

            # Initial propagation to build cache
            initial_propagate = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )
            for _ in initial_propagate:
                pass

            # Add point prompt
            points_tensor = torch.tensor([[point[0], point[1]]], dtype=torch.float32)
            labels_tensor = torch.tensor([label], dtype=torch.int32)

            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=1,
                    points=points_tensor,
                    point_labels=labels_tensor,
                )
            )

            # Final propagation
            propagate_result = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )

            return self._extract_mask_from_result(propagate_result)

        finally:
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _segment_with_text_and_points(
        self,
        video_path: Path,
        text_prompt: str,
        points: list[tuple[int, int]],
        labels: list[int],
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """Run SAM3 with text prompt + point refinement."""
        response = self.video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(video_path),
            )
        )
        session_id = response["session_id"]

        try:
            # Add text prompt
            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text_prompt,
                )
            )

            # Initial propagation
            initial_propagate = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )
            for _ in initial_propagate:
                pass

            # Add point prompts for refinement
            points_list = [[p[0], p[1]] for p in points]
            points_tensor = torch.tensor(points_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int32)

            self.video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=1,
                    points=points_tensor,
                    point_labels=labels_tensor,
                )
            )

            # Final propagation
            propagate_result = self.video_predictor.propagate_in_video(
                session_id=session_id,
                propagation_direction="forward",
                start_frame_idx=0,
                max_frame_num_to_track=1,
            )

            return self._extract_mask_from_result(propagate_result)

        finally:
            self.video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_all_masks_with_scores(self, propagate_result) -> list[tuple[np.ndarray, float]]:
        """
        Extract ALL masks with their confidence scores from SAM3 propagation result.

        SAM3 returns separate masks for each detected object, along with probability scores.

        Returns:
            List of (mask, confidence) tuples
        """
        results = []

        if hasattr(propagate_result, "__iter__") and not isinstance(propagate_result, (dict, str)):
            for item in propagate_result:
                outputs = None

                if isinstance(item, dict):
                    outputs = item.get("outputs", item)
                elif isinstance(item, tuple) and len(item) > 1:
                    outputs = item[1] if isinstance(item[1], dict) else None

                if outputs is None:
                    continue

                masks_tensor = outputs.get("out_binary_masks")
                probs = outputs.get("out_probs")

                if masks_tensor is None or not hasattr(masks_tensor, 'shape'):
                    continue

                # Convert to numpy if tensor
                if hasattr(masks_tensor, "cpu"):
                    masks_tensor = masks_tensor.cpu().numpy()
                if probs is not None and hasattr(probs, "cpu"):
                    probs = probs.cpu().numpy()

                # Handle different shapes: (N, H, W) or (N, 1, H, W)
                if len(masks_tensor.shape) == 4:
                    masks_tensor = masks_tensor.squeeze(1)

                num_masks = masks_tensor.shape[0]
                logger.debug(f"SAM3 returned {num_masks} masks")

                # Extract each mask with its confidence
                for i in range(num_masks):
                    mask = masks_tensor[i].squeeze().astype(np.uint8)
                    if not np.any(mask):  # Skip empty masks
                        continue

                    # Get confidence from probs array
                    confidence = float(probs[i]) if probs is not None and i < len(probs) else 1.0
                    results.append((mask, confidence))

        logger.debug(f"Extracted {len(results)} masks with scores from SAM3 result")
        return results

    def _extract_all_masks_from_result(self, propagate_result) -> list[np.ndarray]:
        """Extract ALL masks from SAM3 propagation result (without scores)."""
        results = self._extract_all_masks_with_scores(propagate_result)
        return [mask for mask, _ in results]

    def _extract_mask_from_result(self, propagate_result) -> Optional[np.ndarray]:
        """Extract single mask from SAM3 propagation result (legacy)."""
        masks = self._extract_all_masks_from_result(propagate_result)
        return masks[0] if masks else None
