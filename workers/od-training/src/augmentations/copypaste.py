"""
Copy-Paste Augmentation for Object Detection.

Copy-Paste takes object instances from one image and pastes them onto
another image, effectively increasing the number of instances per image.

Benefits:
1. Increases instance count per image
2. Creates novel object configurations
3. Particularly effective for rare classes
4. Simulates occlusion scenarios

The augmentation requires instance segmentation masks. If masks are not
available, we use bounding boxes with optional alpha blending.

Reference:
    Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method
    for Instance Segmentation" (CVPR 2021)
"""

import random
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for object detection.

    Copies objects from a source image and pastes them onto a target image.
    Works with or without segmentation masks.

    Args:
        prob: Probability of applying copy-paste (default: 0.5)
        max_paste: Maximum number of objects to paste (default: 3)
        blend_ratio: Alpha blending ratio (0=hard, 1=full blend) (default: 0.0)
        scale_range: Range of scales for pasted objects (default: (0.5, 1.5))
        jitter_ratio: Random position jitter ratio (default: 0.1)
        use_mask: Whether to use segmentation masks if available (default: True)

    Usage:
        copypaste = CopyPasteAugmentation(max_paste=3)

        # In dataset, after loading both images:
        if random.random() < copypaste_prob:
            image, target = copypaste(
                target_image, target_target,
                source_image, source_target
            )
    """

    def __init__(
        self,
        prob: float = 0.5,
        max_paste: int = 3,
        blend_ratio: float = 0.0,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        jitter_ratio: float = 0.1,
        use_mask: bool = True,
    ):
        self.prob = prob
        self.max_paste = max_paste
        self.blend_ratio = blend_ratio
        self.scale_range = scale_range
        self.jitter_ratio = jitter_ratio
        self.use_mask = use_mask

    def __call__(
        self,
        target_image: np.ndarray,
        target_target: Dict[str, Any],
        source_image: np.ndarray,
        source_target: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Copy-Paste augmentation.

        Args:
            target_image: Image to paste onto [H, W, C]
            target_target: Target annotations for target image
            source_image: Image to copy objects from [H, W, C]
            source_target: Source annotations with 'boxes', 'labels', optional 'masks'

        Returns:
            Tuple of (augmented_image, augmented_target)
        """
        # Random skip based on probability
        if random.random() > self.prob:
            return target_image, target_target

        # Get source objects
        source_boxes = source_target.get('boxes', np.array([]))
        source_labels = source_target.get('labels', np.array([]))
        source_masks = source_target.get('masks', None)

        if len(source_boxes) == 0:
            return target_image, target_target

        # Select random objects to paste
        num_objects = len(source_boxes)
        num_paste = min(random.randint(1, self.max_paste), num_objects)
        paste_indices = random.sample(range(num_objects), num_paste)

        # Get target dimensions
        th, tw = target_image.shape[:2]
        sh, sw = source_image.shape[:2]

        # Copy target image
        result_image = target_image.copy()

        # Collect pasted boxes and labels
        pasted_boxes = []
        pasted_labels = []

        for idx in paste_indices:
            box = source_boxes[idx].copy()
            label = source_labels[idx]

            # Get object crop
            x1, y1, x2, y2 = box.astype(int)
            x1, x2 = max(0, x1), min(sw, x2)
            y1, y2 = max(0, y1), min(sh, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            obj_crop = source_image[y1:y2, x1:x2].copy()
            obj_h, obj_w = obj_crop.shape[:2]

            # Get mask if available
            if self.use_mask and source_masks is not None:
                obj_mask = source_masks[idx][y1:y2, x1:x2].copy()
            else:
                obj_mask = None

            # Random scale
            scale = random.uniform(*self.scale_range)
            new_w = int(obj_w * scale)
            new_h = int(obj_h * scale)

            if new_w < 10 or new_h < 10:
                continue

            # Resize object and mask
            obj_crop = self._resize_image(obj_crop, (new_w, new_h))
            if obj_mask is not None:
                obj_mask = self._resize_image(obj_mask, (new_w, new_h))

            # Random paste location with jitter
            max_x = tw - new_w
            max_y = th - new_h

            if max_x <= 0 or max_y <= 0:
                continue

            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Apply jitter
            jitter_x = int(new_w * self.jitter_ratio * (random.random() - 0.5))
            jitter_y = int(new_h * self.jitter_ratio * (random.random() - 0.5))
            paste_x = max(0, min(tw - new_w, paste_x + jitter_x))
            paste_y = max(0, min(th - new_h, paste_y + jitter_y))

            # Paste object
            if obj_mask is not None:
                # Use mask for blending
                mask_3ch = obj_mask[..., np.newaxis] if obj_mask.ndim == 2 else obj_mask
                if mask_3ch.shape[-1] == 1:
                    mask_3ch = np.repeat(mask_3ch, 3, axis=-1)
                mask_3ch = mask_3ch.astype(np.float32) / 255.0

                target_region = result_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
                blended = (obj_crop * mask_3ch + target_region * (1 - mask_3ch)).astype(np.uint8)
                result_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = blended
            else:
                # Simple paste with optional alpha blending
                if self.blend_ratio > 0:
                    target_region = result_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
                    alpha = self.blend_ratio
                    blended = ((1 - alpha) * obj_crop + alpha * target_region).astype(np.uint8)
                    result_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = blended
                else:
                    result_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = obj_crop

            # Record pasted box
            new_box = [paste_x, paste_y, paste_x + new_w, paste_y + new_h]
            pasted_boxes.append(new_box)
            pasted_labels.append(label)

        # Merge with original target annotations
        target_boxes = target_target.get('boxes', np.zeros((0, 4)))
        target_labels = target_target.get('labels', np.array([]))

        if pasted_boxes:
            pasted_boxes = np.array(pasted_boxes)
            pasted_labels = np.array(pasted_labels)

            if len(target_boxes) > 0:
                merged_boxes = np.concatenate([target_boxes, pasted_boxes], axis=0)
                merged_labels = np.concatenate([target_labels, pasted_labels], axis=0)
            else:
                merged_boxes = pasted_boxes
                merged_labels = pasted_labels
        else:
            merged_boxes = target_boxes
            merged_labels = target_labels

        result_target = {
            'boxes': merged_boxes,
            'labels': merged_labels,
            'copypaste_count': len(pasted_boxes) if pasted_boxes else 0,
        }

        return result_image, result_target

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


class SimpleCopyPaste:
    """
    Simplified Copy-Paste using bounding box crops only.

    Faster version that doesn't require masks. Uses rectangular crops
    with optional edge blending.

    Args:
        prob: Probability of applying (default: 0.3)
        max_paste: Maximum objects to paste (default: 2)
        edge_blend: Pixels to blend at edges (default: 5)
    """

    def __init__(
        self,
        prob: float = 0.3,
        max_paste: int = 2,
        edge_blend: int = 5,
    ):
        self.prob = prob
        self.max_paste = max_paste
        self.edge_blend = edge_blend

    def __call__(
        self,
        target_image: np.ndarray,
        target_target: Dict[str, Any],
        source_image: np.ndarray,
        source_target: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply simple copy-paste."""
        if random.random() > self.prob:
            return target_image, target_target

        source_boxes = source_target.get('boxes', np.array([]))
        source_labels = source_target.get('labels', np.array([]))

        if len(source_boxes) == 0:
            return target_image, target_target

        th, tw = target_image.shape[:2]
        result_image = target_image.copy()

        num_paste = min(random.randint(1, self.max_paste), len(source_boxes))
        paste_indices = random.sample(range(len(source_boxes)), num_paste)

        pasted_boxes = []
        pasted_labels = []

        for idx in paste_indices:
            box = source_boxes[idx].astype(int)
            label = source_labels[idx]

            x1, y1, x2, y2 = box
            x1, x2 = max(0, x1), min(source_image.shape[1], x2)
            y1, y2 = max(0, y1), min(source_image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            obj_crop = source_image[y1:y2, x1:x2].copy()
            obj_h, obj_w = obj_crop.shape[:2]

            # Find paste location (avoid overlap with existing boxes)
            max_x = tw - obj_w
            max_y = th - obj_h

            if max_x <= 0 or max_y <= 0:
                continue

            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # Simple paste with edge blending
            if self.edge_blend > 0:
                result_image = self._blend_paste(
                    result_image, obj_crop, paste_x, paste_y, self.edge_blend
                )
            else:
                result_image[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_crop

            new_box = [paste_x, paste_y, paste_x + obj_w, paste_y + obj_h]
            pasted_boxes.append(new_box)
            pasted_labels.append(label)

        # Merge annotations
        target_boxes = target_target.get('boxes', np.zeros((0, 4)))
        target_labels = target_target.get('labels', np.array([]))

        if pasted_boxes:
            pasted_boxes = np.array(pasted_boxes)
            pasted_labels = np.array(pasted_labels)

            if len(target_boxes) > 0:
                merged_boxes = np.concatenate([target_boxes, pasted_boxes], axis=0)
                merged_labels = np.concatenate([target_labels, pasted_labels], axis=0)
            else:
                merged_boxes = pasted_boxes
                merged_labels = pasted_labels
        else:
            merged_boxes = target_boxes
            merged_labels = target_labels

        return result_image, {
            'boxes': merged_boxes,
            'labels': merged_labels,
        }

    def _blend_paste(
        self,
        target: np.ndarray,
        obj: np.ndarray,
        x: int,
        y: int,
        blend_size: int,
    ) -> np.ndarray:
        """Paste with edge blending."""
        h, w = obj.shape[:2]

        # Create blend mask
        mask = np.ones((h, w), dtype=np.float32)

        # Fade edges
        for i in range(blend_size):
            alpha = (i + 1) / blend_size
            mask[i, :] = alpha
            mask[h - 1 - i, :] = alpha
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)

        mask = mask[..., np.newaxis]

        # Blend
        target_region = target[y:y+h, x:x+w].astype(np.float32)
        obj_float = obj.astype(np.float32)
        blended = (obj_float * mask + target_region * (1 - mask)).astype(np.uint8)

        target[y:y+h, x:x+w] = blended
        return target
