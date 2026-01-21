"""
Mosaic Augmentation for Object Detection.

Mosaic combines 4 training images into one by placing them in a 2x2 grid
with a random center point. This provides several benefits:
1. Effectively increases batch size 4x
2. Improves detection of small objects
3. Provides context diversity
4. Acts as regularization

The bounding boxes from all 4 images are transformed to the new
coordinate space.

Example:
    ┌─────────┬─────────┐
    │ Image 1 │ Image 2 │
    │         │         │
    ├─────────┼─────────┤ ← Random center point
    │ Image 3 │ Image 4 │
    │         │         │
    └─────────┴─────────┘

Reference:
    Bochkovskiy et al., "YOLOv4" (arXiv 2020)
"""

import random
from typing import List, Tuple, Dict, Any, Optional, Callable

import numpy as np
import torch


class MosaicAugmentation:
    """
    Mosaic augmentation that combines 4 images into one.

    Args:
        img_size: Output image size (height, width) or single int for square
        center_ratio_range: Range for random center point (default: 0.5 to 1.5)
        min_bbox_size: Minimum bbox size after transform (filter smaller ones)
        fill_value: Fill value for empty regions (default: 114, gray)

    Usage:
        mosaic = MosaicAugmentation(img_size=640)

        # In dataset __getitem__:
        if random.random() < mosaic_prob:
            indices = [idx] + [random.randint(0, len(dataset)-1) for _ in range(3)]
            images = [load_image(i) for i in indices]
            targets = [load_target(i) for i in indices]
            image, target = mosaic(images, targets)
    """

    def __init__(
        self,
        img_size: int = 640,
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        min_bbox_size: int = 2,
        fill_value: int = 114,
    ):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.fill_value = fill_value

    def __call__(
        self,
        images: List[np.ndarray],
        targets: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mosaic augmentation.

        Args:
            images: List of 4 images as numpy arrays [H, W, C]
            targets: List of 4 target dicts, each with:
                - 'boxes': [N, 4] boxes in xyxy format (absolute coords)
                - 'labels': [N] class labels

        Returns:
            Tuple of (mosaic_image, merged_target)
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        assert len(targets) == 4, "Mosaic requires exactly 4 targets"

        h, w = self.img_size

        # Random center point
        center_x = int(random.uniform(
            w * self.center_ratio_range[0],
            w * self.center_ratio_range[1]
        ))
        center_y = int(random.uniform(
            h * self.center_ratio_range[0],
            h * self.center_ratio_range[1]
        ))

        # Clamp center to valid range
        center_x = np.clip(center_x, w // 4, w * 3 // 4)
        center_y = np.clip(center_y, h // 4, h * 3 // 4)

        # Create output image
        mosaic_img = np.full((h, w, 3), self.fill_value, dtype=np.uint8)

        # Placement positions for each quadrant
        # (x1, y1, x2, y2) in output, (x1, y1, x2, y2) crop from input
        placements = [
            # Top-left
            ((0, 0, center_x, center_y), "top_left"),
            # Top-right
            ((center_x, 0, w, center_y), "top_right"),
            # Bottom-left
            ((0, center_y, center_x, h), "bottom_left"),
            # Bottom-right
            ((center_x, center_y, w, h), "bottom_right"),
        ]

        all_boxes = []
        all_labels = []

        for idx, (img, target) in enumerate(zip(images, targets)):
            out_coords, position = placements[idx]
            out_x1, out_y1, out_x2, out_y2 = out_coords
            out_w = out_x2 - out_x1
            out_h = out_y2 - out_y1

            # Get image dimensions
            img_h, img_w = img.shape[:2]

            # Calculate scaling to fit in quadrant
            scale = min(out_w / img_w, out_h / img_h)

            # Resized dimensions
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            # Resize image
            resized_img = self._resize_image(img, (new_w, new_h))

            # Calculate paste position (center in quadrant)
            paste_x = out_x1 + (out_w - new_w) // 2
            paste_y = out_y1 + (out_h - new_h) // 2

            # Paste into mosaic
            mosaic_img[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_img

            # Transform boxes
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))

            if len(boxes) > 0:
                # Scale boxes
                boxes = boxes.copy().astype(np.float32)
                boxes[:, [0, 2]] *= scale  # x coords
                boxes[:, [1, 3]] *= scale  # y coords

                # Offset boxes
                boxes[:, [0, 2]] += paste_x
                boxes[:, [1, 3]] += paste_y

                # Clip to mosaic bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

                # Filter invalid boxes
                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                valid = (box_w >= self.min_bbox_size) & (box_h >= self.min_bbox_size)

                boxes = boxes[valid]
                labels = labels[valid] if len(labels) > 0 else labels

                all_boxes.append(boxes)
                all_labels.append(labels)

        # Merge all boxes and labels
        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_labels = np.concatenate(all_labels, axis=0)
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.array([], dtype=np.int64)

        merged_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
        }

        return mosaic_img, merged_target

    def _resize_image(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize image using cv2 or PIL."""
        try:
            import cv2
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(size, Image.BILINEAR)
            return np.array(pil_img)


class Mosaic9Augmentation:
    """
    Mosaic-9 augmentation that combines 9 images into a 3x3 grid.

    Similar to Mosaic but with more images for even greater diversity.
    Useful for very small object detection.
    """

    def __init__(
        self,
        img_size: int = 640,
        min_bbox_size: int = 2,
        fill_value: int = 114,
    ):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.min_bbox_size = min_bbox_size
        self.fill_value = fill_value

    def __call__(
        self,
        images: List[np.ndarray],
        targets: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply mosaic-9 augmentation with 9 images in 3x3 grid."""
        assert len(images) == 9, "Mosaic-9 requires exactly 9 images"

        h, w = self.img_size
        cell_h, cell_w = h // 3, w // 3

        mosaic_img = np.full((h, w, 3), self.fill_value, dtype=np.uint8)
        all_boxes = []
        all_labels = []

        for idx, (img, target) in enumerate(zip(images, targets)):
            row = idx // 3
            col = idx % 3

            out_x1 = col * cell_w
            out_y1 = row * cell_h
            out_x2 = out_x1 + cell_w
            out_y2 = out_y1 + cell_h

            img_h, img_w = img.shape[:2]
            scale = min(cell_w / img_w, cell_h / img_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            resized_img = self._resize_image(img, (new_w, new_h))

            paste_x = out_x1 + (cell_w - new_w) // 2
            paste_y = out_y1 + (cell_h - new_h) // 2

            mosaic_img[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized_img

            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))

            if len(boxes) > 0:
                boxes = boxes.copy().astype(np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + paste_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + paste_y

                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                valid = (box_w >= self.min_bbox_size) & (box_h >= self.min_bbox_size)

                all_boxes.append(boxes[valid])
                all_labels.append(labels[valid] if len(labels) > 0 else labels)

        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_labels = np.concatenate(all_labels, axis=0)
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.array([], dtype=np.int64)

        return mosaic_img, {'boxes': merged_boxes, 'labels': merged_labels}

    def _resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
