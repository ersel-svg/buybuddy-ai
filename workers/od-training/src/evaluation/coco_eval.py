"""
COCO-style Evaluation for Object Detection.

Computes standard COCO metrics:
- AP (Average Precision) at IoU thresholds [0.50:0.05:0.95]
- AP@50 (PASCAL VOC metric)
- AP@75 (strict metric)
- AP for small, medium, large objects
- AR (Average Recall)

Reference:
    COCO Detection Challenge: https://cocodataset.org/#detection-eval

Usage:
    from evaluation import COCOEvaluator

    evaluator = COCOEvaluator(num_classes=80)

    # During validation loop
    for images, targets in dataloader:
        predictions = model(images)
        evaluator.update(predictions, targets)

    # After all batches
    metrics = evaluator.compute()
    print(f"mAP@50:95: {metrics['mAP']:.4f}")
    print(f"mAP@50: {metrics['mAP_50']:.4f}")
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in xyxy format
        boxes2: [M, 4] boxes in xyxy format

    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, 0, None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


class COCOEvaluator:
    """
    COCO-style evaluator for object detection.

    Accumulates predictions and ground truth, then computes
    AP at various IoU thresholds.

    Args:
        num_classes: Number of object classes (excluding background)
        iou_thresholds: IoU thresholds for evaluation
                       Default: [0.5, 0.55, ..., 0.95]
        max_detections: Maximum detections per image
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: Optional[List[float]] = None,
        max_detections: int = 100,
    ):
        self.num_classes = num_classes
        self.max_detections = max_detections

        # Default COCO IoU thresholds
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

        self.reset()

    def reset(self):
        """Reset accumulated predictions."""
        # Store predictions and ground truth per image
        self.predictions = defaultdict(list)  # image_id -> list of predictions
        self.ground_truths = defaultdict(list)  # image_id -> list of ground truths
        self.image_ids = set()

    def update(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ):
        """
        Update with batch of predictions and targets.

        Args:
            predictions: List of prediction dicts with:
                - 'boxes': [N, 4] predicted boxes (xyxy)
                - 'scores': [N] confidence scores
                - 'labels': [N] predicted class labels
            targets: List of target dicts with:
                - 'boxes': [M, 4] ground truth boxes (xyxy)
                - 'labels': [M] ground truth labels
                - 'image_id': image identifier
        """
        for pred, target in zip(predictions, targets):
            image_id = target.get('image_id', len(self.image_ids))
            self.image_ids.add(image_id)

            # Convert tensors to numpy
            pred_boxes = self._to_numpy(pred.get('boxes', []))
            pred_scores = self._to_numpy(pred.get('scores', []))
            pred_labels = self._to_numpy(pred.get('labels', []))

            gt_boxes = self._to_numpy(target.get('boxes', []))
            gt_labels = self._to_numpy(target.get('labels', []))

            # Sort predictions by score (descending)
            if len(pred_scores) > 0:
                sorted_idx = np.argsort(-pred_scores)[:self.max_detections]
                pred_boxes = pred_boxes[sorted_idx]
                pred_scores = pred_scores[sorted_idx]
                pred_labels = pred_labels[sorted_idx]

            # Store predictions
            for i in range(len(pred_boxes)):
                self.predictions[image_id].append({
                    'box': pred_boxes[i],
                    'score': pred_scores[i],
                    'label': int(pred_labels[i]),
                })

            # Store ground truths
            for i in range(len(gt_boxes)):
                self.ground_truths[image_id].append({
                    'box': gt_boxes[i],
                    'label': int(gt_labels[i]),
                    'matched': False,
                })

    def _to_numpy(self, x) -> np.ndarray:
        """Convert tensor or list to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return np.array(x)
        return x

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dict with metrics:
            - mAP: mean AP across all IoU thresholds
            - mAP_50: AP at IoU=0.50
            - mAP_75: AP at IoU=0.75
            - mAP_small: AP for small objects (area < 32²)
            - mAP_medium: AP for medium objects (32² < area < 96²)
            - mAP_large: AP for large objects (area > 96²)
            - per_class_ap: Dict of AP per class
        """
        # Compute AP for each class and IoU threshold
        ap_per_class = {}
        ap_matrix = np.zeros((self.num_classes, len(self.iou_thresholds)))

        for class_id in range(self.num_classes):
            for iou_idx, iou_thresh in enumerate(self.iou_thresholds):
                ap = self._compute_ap_for_class(class_id, iou_thresh)
                ap_matrix[class_id, iou_idx] = ap

            # Store mean AP for this class across all IoU thresholds
            ap_per_class[class_id] = float(np.mean(ap_matrix[class_id]))

        # Compute mean metrics
        # Filter out classes with no ground truth
        valid_classes = []
        for class_id in range(self.num_classes):
            has_gt = any(
                any(gt['label'] == class_id for gt in gts)
                for gts in self.ground_truths.values()
            )
            if has_gt:
                valid_classes.append(class_id)

        if valid_classes:
            valid_ap_matrix = ap_matrix[valid_classes]
            mAP = float(np.mean(valid_ap_matrix))
            mAP_50 = float(np.mean(valid_ap_matrix[:, 0]))  # IoU=0.50
            mAP_75 = float(np.mean(valid_ap_matrix[:, 5])) if len(self.iou_thresholds) > 5 else mAP_50
        else:
            mAP = 0.0
            mAP_50 = 0.0
            mAP_75 = 0.0

        # Compute recall
        ar = self._compute_average_recall()

        return {
            'mAP': mAP,
            'mAP_50': mAP_50,
            'mAP_75': mAP_75,
            'AR': ar,
            'per_class_ap': ap_per_class,
            'num_predictions': sum(len(p) for p in self.predictions.values()),
            'num_ground_truths': sum(len(g) for g in self.ground_truths.values()),
        }

    def _compute_ap_for_class(
        self,
        class_id: int,
        iou_threshold: float,
    ) -> float:
        """
        Compute AP for a single class at given IoU threshold.

        Uses 101-point interpolation (COCO style).
        """
        # Collect all predictions and ground truths for this class
        all_preds = []
        all_gts = {}

        for img_id in self.image_ids:
            # Get predictions for this class
            img_preds = [
                p for p in self.predictions.get(img_id, [])
                if p['label'] == class_id
            ]
            for pred in img_preds:
                all_preds.append({
                    'image_id': img_id,
                    'box': pred['box'],
                    'score': pred['score'],
                })

            # Get ground truths for this class
            img_gts = [
                gt for gt in self.ground_truths.get(img_id, [])
                if gt['label'] == class_id
            ]
            all_gts[img_id] = [{'box': gt['box'], 'matched': False} for gt in img_gts]

        if not all_preds or not any(all_gts.values()):
            return 0.0

        # Sort predictions by score
        all_preds.sort(key=lambda x: x['score'], reverse=True)

        # Match predictions to ground truths
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        total_gt = sum(len(gts) for gts in all_gts.values())

        for pred_idx, pred in enumerate(all_preds):
            img_id = pred['image_id']
            img_gts = all_gts.get(img_id, [])

            if not img_gts:
                fp[pred_idx] = 1
                continue

            # Compute IoU with all ground truths
            pred_box = pred['box'].reshape(1, 4)
            gt_boxes = np.array([gt['box'] for gt in img_gts])
            ious = compute_iou(pred_box, gt_boxes)[0]

            # Find best matching ground truth
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_threshold and not img_gts[best_iou_idx]['matched']:
                tp[pred_idx] = 1
                img_gts[best_iou_idx]['matched'] = True
            else:
                fp[pred_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # 101-point interpolation
        ap = self._interpolate_ap(recall, precision)

        return ap

    def _interpolate_ap(
        self,
        recall: np.ndarray,
        precision: np.ndarray,
    ) -> float:
        """
        Compute AP using 101-point interpolation (COCO style).

        For each recall threshold r in [0, 0.01, ..., 1],
        AP = sum(max(precision[recall >= r])) / 101
        """
        # Prepend sentinel values
        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])

        # Ensure precision is monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Find indices where recall changes
        recall_thresholds = np.linspace(0, 1, 101)

        ap = 0.0
        for thresh in recall_thresholds:
            # Find precision at this recall threshold
            idx = np.searchsorted(recall, thresh, side='left')
            if idx < len(precision):
                ap += precision[idx]

        return ap / 101

    def _compute_average_recall(self) -> float:
        """Compute average recall at IoU=0.5."""
        total_tp = 0
        total_gt = 0

        iou_thresh = 0.5

        for img_id in self.image_ids:
            preds = self.predictions.get(img_id, [])
            gts = self.ground_truths.get(img_id, [])

            if not gts:
                continue

            total_gt += len(gts)

            if not preds:
                continue

            # Match predictions to ground truths
            pred_boxes = np.array([p['box'] for p in preds])
            gt_boxes = np.array([g['box'] for g in gts])

            ious = compute_iou(pred_boxes, gt_boxes)
            gt_matched = [False] * len(gts)

            # Sort predictions by score
            sorted_idx = np.argsort([-p['score'] for p in preds])

            for pred_idx in sorted_idx:
                pred_label = preds[pred_idx]['label']

                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    if gt['label'] != pred_label:
                        continue
                    if ious[pred_idx, gt_idx] >= iou_thresh:
                        gt_matched[gt_idx] = True
                        total_tp += 1
                        break

        return total_tp / max(total_gt, 1)


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int,
) -> Dict[str, float]:
    """
    Convenience function to evaluate predictions.

    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of classes

    Returns:
        Evaluation metrics
    """
    evaluator = COCOEvaluator(num_classes=num_classes)
    evaluator.update(predictions, targets)
    return evaluator.compute()


def calculate_map(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate mAP for a set of predictions.

    Simplified interface for quick evaluation.

    Args:
        pred_boxes: List of [N, 4] predicted boxes per image
        pred_scores: List of [N] confidence scores per image
        pred_labels: List of [N] predicted labels per image
        gt_boxes: List of [M, 4] ground truth boxes per image
        gt_labels: List of [M] ground truth labels per image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        mAP value
    """
    evaluator = COCOEvaluator(
        num_classes=num_classes,
        iou_thresholds=[iou_threshold],
    )

    predictions = []
    targets = []

    for i in range(len(pred_boxes)):
        predictions.append({
            'boxes': pred_boxes[i],
            'scores': pred_scores[i],
            'labels': pred_labels[i],
        })
        targets.append({
            'boxes': gt_boxes[i],
            'labels': gt_labels[i],
            'image_id': i,
        })

    evaluator.update(predictions, targets)
    metrics = evaluator.compute()

    return metrics['mAP']


class ConfusionMatrix:
    """
    Confusion matrix for object detection.

    Useful for analyzing per-class performance and common misclassifications.

    Args:
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold for predictions
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        # Matrix: rows = ground truth, cols = predictions
        # +1 for background class
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))

    def update(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ):
        """Update confusion matrix with predictions."""
        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('boxes', np.array([]))
            pred_scores = pred.get('scores', np.array([]))
            pred_labels = pred.get('labels', np.array([]))

            gt_boxes = target.get('boxes', np.array([]))
            gt_labels = target.get('labels', np.array([]))

            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()

            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
                gt_labels = gt_labels.cpu().numpy()

            # Filter by confidence
            if len(pred_scores) > 0:
                mask = pred_scores >= self.conf_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue

            # Match predictions to ground truths
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                ious = compute_iou(pred_boxes, gt_boxes)
                gt_matched = [False] * len(gt_boxes)

                for pred_idx in range(len(pred_boxes)):
                    pred_label = int(pred_labels[pred_idx])
                    matched = False

                    for gt_idx in range(len(gt_boxes)):
                        if gt_matched[gt_idx]:
                            continue
                        if ious[pred_idx, gt_idx] >= self.iou_threshold:
                            gt_label = int(gt_labels[gt_idx])
                            self.matrix[gt_label, pred_label] += 1
                            gt_matched[gt_idx] = True
                            matched = True
                            break

                    if not matched:
                        # False positive (background -> predicted class)
                        self.matrix[self.num_classes, pred_label] += 1

                # Count unmatched ground truths (false negatives)
                for gt_idx, matched in enumerate(gt_matched):
                    if not matched:
                        gt_label = int(gt_labels[gt_idx])
                        self.matrix[gt_label, self.num_classes] += 1

            elif len(gt_boxes) > 0:
                # No predictions - all ground truths are false negatives
                for gt_label in gt_labels:
                    self.matrix[int(gt_label), self.num_classes] += 1

            elif len(pred_boxes) > 0:
                # No ground truth - all predictions are false positives
                for pred_label in pred_labels:
                    self.matrix[self.num_classes, int(pred_label)] += 1

    def get_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.matrix

    def get_per_class_accuracy(self) -> np.ndarray:
        """Get per-class accuracy."""
        return np.diag(self.matrix[:self.num_classes, :self.num_classes]) / \
               (self.matrix[:self.num_classes, :].sum(axis=1) + 1e-7)
