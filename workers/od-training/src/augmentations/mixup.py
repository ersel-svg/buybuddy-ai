"""
MixUp Augmentation for Object Detection.

MixUp blends two images together with a mixing ratio sampled from
a Beta distribution. For object detection, we keep boxes from both
images (unlike classification where labels are also mixed).

Formula:
    mixed_image = α × image_a + (1 - α) × image_b
    α ~ Beta(beta_param, beta_param)

Benefits:
1. Regularization effect
2. Smoother decision boundaries
3. Reduces overconfidence

Reference:
    Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    Bochkovskiy et al., "YOLOv4" (arXiv 2020) - for detection adaptation
"""

import random
from typing import Tuple, Dict, Any, Optional

import numpy as np


class MixUpAugmentation:
    """
    MixUp augmentation for object detection.

    Blends two images and concatenates their annotations.

    Args:
        alpha: Beta distribution parameter (default: 8.0)
            - Higher values = mixing ratios closer to 0.5
            - Lower values = more extreme ratios
        min_ratio: Minimum mixing ratio (default: 0.0)
        max_ratio: Maximum mixing ratio (default: 1.0)

    Usage:
        mixup = MixUpAugmentation(alpha=8.0)

        # In dataset, after regular augmentation:
        if random.random() < mixup_prob:
            idx2 = random.randint(0, len(dataset) - 1)
            image2, target2 = dataset[idx2]
            image, target = mixup(image, target, image2, target2)
    """

    def __init__(
        self,
        alpha: float = 8.0,
        min_ratio: float = 0.0,
        max_ratio: float = 1.0,
    ):
        self.alpha = alpha
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(
        self,
        image1: np.ndarray,
        target1: Dict[str, Any],
        image2: np.ndarray,
        target2: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply MixUp augmentation.

        Args:
            image1: First image [H, W, C]
            target1: First target dict with 'boxes' and 'labels'
            image2: Second image [H, W, C]
            target2: Second target dict with 'boxes' and 'labels'

        Returns:
            Tuple of (mixed_image, merged_target)
        """
        # Sample mixing ratio from Beta distribution
        ratio = np.random.beta(self.alpha, self.alpha)

        # Clamp to specified range
        ratio = np.clip(ratio, self.min_ratio, self.max_ratio)

        # Ensure images have same size
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        if (h1, w1) != (h2, w2):
            # Resize image2 to match image1
            image2 = self._resize_image(image2, (w1, h1))

            # Scale boxes of image2
            scale_x = w1 / w2
            scale_y = h1 / h2
            boxes2 = target2.get('boxes', np.array([])).copy()
            if len(boxes2) > 0:
                boxes2[:, [0, 2]] *= scale_x
                boxes2[:, [1, 3]] *= scale_y
                target2 = {**target2, 'boxes': boxes2}

        # Mix images
        mixed_image = (ratio * image1.astype(np.float32) +
                      (1 - ratio) * image2.astype(np.float32))
        mixed_image = mixed_image.astype(np.uint8)

        # Merge targets (keep all boxes from both images)
        boxes1 = target1.get('boxes', np.zeros((0, 4)))
        boxes2 = target2.get('boxes', np.zeros((0, 4)))
        labels1 = target1.get('labels', np.array([]))
        labels2 = target2.get('labels', np.array([]))

        if len(boxes1) > 0 and len(boxes2) > 0:
            merged_boxes = np.concatenate([boxes1, boxes2], axis=0)
            merged_labels = np.concatenate([labels1, labels2], axis=0)
        elif len(boxes1) > 0:
            merged_boxes = boxes1
            merged_labels = labels1
        elif len(boxes2) > 0:
            merged_boxes = boxes2
            merged_labels = labels2
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.array([], dtype=np.int64)

        merged_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'mixup_ratio': ratio,  # Store for potential use
        }

        return mixed_image, merged_target

    def _resize_image(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize image."""
        try:
            import cv2
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))


class CutMixAugmentation:
    """
    CutMix augmentation for object detection.

    Instead of blending entire images, CutMix cuts a patch from one
    image and pastes it onto another. More suitable for detection than
    standard MixUp as it preserves local structure.

    Args:
        alpha: Beta distribution parameter for patch size
        min_area_ratio: Minimum patch area ratio
        max_area_ratio: Maximum patch area ratio

    Reference:
        Yun et al., "CutMix" (ICCV 2019)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_area_ratio: float = 0.1,
        max_area_ratio: float = 0.5,
    ):
        self.alpha = alpha
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def __call__(
        self,
        image1: np.ndarray,
        target1: Dict[str, Any],
        image2: np.ndarray,
        target2: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply CutMix augmentation.

        Pastes a rectangular region from image2 onto image1.
        Boxes from image2 that fall within the pasted region are added.
        """
        h, w = image1.shape[:2]

        # Ensure image2 is same size
        if image2.shape[:2] != (h, w):
            image2 = self._resize_image(image2, (w, h))
            scale_x = w / image2.shape[1]
            scale_y = h / image2.shape[0]
            boxes2 = target2.get('boxes', np.array([])).copy()
            if len(boxes2) > 0:
                boxes2[:, [0, 2]] *= scale_x
                boxes2[:, [1, 3]] *= scale_y
                target2 = {**target2, 'boxes': boxes2}

        # Sample patch size using Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = np.clip(lam, self.min_area_ratio, self.max_area_ratio)

        # Calculate patch dimensions
        cut_w = int(w * np.sqrt(lam))
        cut_h = int(h * np.sqrt(lam))

        # Random patch location
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Paste patch from image2 to image1
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        # Get boxes from image1 (filter those that overlap significantly with cut region)
        boxes1 = target1.get('boxes', np.zeros((0, 4))).copy()
        labels1 = target1.get('labels', np.array([]))

        # Get boxes from image2 (only those within the pasted region)
        boxes2 = target2.get('boxes', np.zeros((0, 4))).copy()
        labels2 = target2.get('labels', np.array([]))

        # Filter boxes1: keep boxes that don't overlap too much with cut region
        if len(boxes1) > 0:
            keep1 = self._filter_boxes_by_region(boxes1, x1, y1, x2, y2, keep_outside=True)
            boxes1 = boxes1[keep1]
            labels1 = labels1[keep1] if len(labels1) > 0 else labels1

        # Filter boxes2: keep boxes that are mostly inside cut region
        if len(boxes2) > 0:
            keep2 = self._filter_boxes_by_region(boxes2, x1, y1, x2, y2, keep_outside=False)
            # Clip boxes2 to cut region
            boxes2 = boxes2[keep2]
            labels2 = labels2[keep2] if len(labels2) > 0 else labels2
            if len(boxes2) > 0:
                boxes2[:, [0, 2]] = np.clip(boxes2[:, [0, 2]], x1, x2)
                boxes2[:, [1, 3]] = np.clip(boxes2[:, [1, 3]], y1, y2)

        # Merge boxes
        if len(boxes1) > 0 and len(boxes2) > 0:
            merged_boxes = np.concatenate([boxes1, boxes2], axis=0)
            merged_labels = np.concatenate([labels1, labels2], axis=0)
        elif len(boxes1) > 0:
            merged_boxes = boxes1
            merged_labels = labels1
        elif len(boxes2) > 0:
            merged_boxes = boxes2
            merged_labels = labels2
        else:
            merged_boxes = np.zeros((0, 4), dtype=np.float32)
            merged_labels = np.array([], dtype=np.int64)

        merged_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'cutmix_region': (x1, y1, x2, y2),
        }

        return mixed_image, merged_target

    def _filter_boxes_by_region(
        self,
        boxes: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        keep_outside: bool = True,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Filter boxes based on overlap with region."""
        # Calculate intersection with region
        inter_x1 = np.maximum(boxes[:, 0], x1)
        inter_y1 = np.maximum(boxes[:, 1], y1)
        inter_x2 = np.minimum(boxes[:, 2], x2)
        inter_y2 = np.minimum(boxes[:, 3], y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        overlap_ratio = inter_area / (box_area + 1e-7)

        if keep_outside:
            # Keep boxes with less than threshold overlap
            return overlap_ratio < threshold
        else:
            # Keep boxes with more than threshold overlap
            return overlap_ratio > threshold

    def _resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
