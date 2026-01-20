"""
Base model class for OD Annotation Worker.

All model wrappers should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import io
import httpx
from PIL import Image
import numpy as np
from loguru import logger

from config import config


class BaseModel(ABC):
    """Abstract base class for all detection/segmentation models."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    @property
    def model(self) -> Any:
        """Lazy load model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the model. Override in subclass."""
        pass

    @abstractmethod
    def predict(
        self,
        image_url: str,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        Run prediction on an image.

        Args:
            image_url: URL of the image to process
            text_prompt: Text prompt for detection (model-specific)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            List of predictions, each with:
            - bbox: [x, y, width, height] in normalized 0-1 coords
            - label: Class label string
            - confidence: Confidence score 0-1
        """
        pass

    def download_image(self, image_url: str) -> Image.Image:
        """Download image from URL and return PIL Image."""
        logger.debug(f"Downloading image: {image_url}")

        try:
            with httpx.Client(timeout=config.image_download_timeout) as client:
                response = client.get(image_url)
                response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large
            max_size = config.max_image_size
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.debug(f"Resized image to: {new_size}")

            return image

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise

    def image_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array."""
        return np.array(image)

    @staticmethod
    def normalize_bbox(
        bbox: tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> dict[str, float]:
        """
        Convert absolute bbox to normalized 0-1 coords.

        Args:
            bbox: (x1, y1, x2, y2) in absolute pixels
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            Dict with x, y, width, height in 0-1 normalized coords
        """
        x1, y1, x2, y2 = bbox
        return {
            "x": x1 / image_width,
            "y": y1 / image_height,
            "width": (x2 - x1) / image_width,
            "height": (y2 - y1) / image_height,
        }

    @staticmethod
    def denormalize_bbox(
        bbox: dict[str, float],
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Convert normalized bbox to absolute pixel coords.

        Args:
            bbox: Dict with x, y, width, height in 0-1 coords
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            (x1, y1, x2, y2) in absolute pixels
        """
        x1 = int(bbox["x"] * image_width)
        y1 = int(bbox["y"] * image_height)
        x2 = int((bbox["x"] + bbox["width"]) * image_width)
        y2 = int((bbox["y"] + bbox["height"]) * image_height)
        return x1, y1, x2, y2


class BaseSegmentationModel(BaseModel):
    """Base class for segmentation models (SAM)."""

    @abstractmethod
    def segment_point(
        self,
        image_url: str,
        point: tuple[float, float],
        label: int = 1,
        return_mask: bool = True,
    ) -> dict:
        """
        Segment object at a point.

        Args:
            image_url: URL of the image
            point: (x, y) in normalized 0-1 coords
            label: 1 for foreground, 0 for background
            return_mask: Whether to return the mask as base64

        Returns:
            Dict with bbox and optionally mask
        """
        pass

    @abstractmethod
    def segment_box(
        self,
        image_url: str,
        box: list[float],
        return_mask: bool = True,
    ) -> dict:
        """
        Segment object within a box.

        Args:
            image_url: URL of the image
            box: [x, y, width, height] in normalized 0-1 coords
            return_mask: Whether to return the mask as base64

        Returns:
            Dict with refined bbox and optionally mask
        """
        pass

    @staticmethod
    def mask_to_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """
        Convert binary mask to bounding box.

        Args:
            mask: Binary mask array (H, W)

        Returns:
            (x1, y1, x2, y2) or None if mask is empty
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def mask_to_base64(mask: np.ndarray) -> str:
        """Convert mask array to base64 PNG string."""
        import base64

        # Convert to PIL Image
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_uint8, mode='L')

        # Encode to PNG
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode('utf-8')
