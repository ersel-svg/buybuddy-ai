"""
Test for class-agnostic NMS functions.
Run: python test_nms.py
"""

import sys
sys.path.insert(0, "src")

from schemas.od import BBox, AIPrediction
from api.v1.od.ai import _compute_iou, _apply_class_agnostic_nms


def test_compute_iou_no_overlap():
    """Two boxes with no overlap should have IoU = 0."""
    box1 = BBox(x=0.0, y=0.0, width=0.1, height=0.1)
    box2 = BBox(x=0.5, y=0.5, width=0.1, height=0.1)

    iou = _compute_iou(box1, box2)
    assert iou == 0.0, f"Expected 0.0, got {iou}"
    print("  [PASS] No overlap -> IoU = 0")


def test_compute_iou_full_overlap():
    """Identical boxes should have IoU = 1."""
    box1 = BBox(x=0.1, y=0.1, width=0.2, height=0.2)
    box2 = BBox(x=0.1, y=0.1, width=0.2, height=0.2)

    iou = _compute_iou(box1, box2)
    assert abs(iou - 1.0) < 0.001, f"Expected 1.0, got {iou}"
    print("  [PASS] Full overlap -> IoU = 1")


def test_compute_iou_partial_overlap():
    """Partial overlap should give IoU between 0 and 1."""
    box1 = BBox(x=0.0, y=0.0, width=0.2, height=0.2)
    box2 = BBox(x=0.1, y=0.1, width=0.2, height=0.2)

    # box1: (0,0) to (0.2, 0.2), area = 0.04
    # box2: (0.1, 0.1) to (0.3, 0.3), area = 0.04
    # intersection: (0.1, 0.1) to (0.2, 0.2), area = 0.01
    # union: 0.04 + 0.04 - 0.01 = 0.07
    # IoU = 0.01 / 0.07 = 0.142857...

    iou = _compute_iou(box1, box2)
    expected = 0.01 / 0.07
    assert abs(iou - expected) < 0.001, f"Expected {expected:.4f}, got {iou:.4f}"
    print(f"  [PASS] Partial overlap -> IoU = {iou:.4f}")


def test_nms_removes_duplicates():
    """NMS should remove duplicate boxes at same location."""
    predictions = [
        AIPrediction(bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.2), label="product", confidence=0.9),
        AIPrediction(bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.2), label="shelf", confidence=0.8),
        AIPrediction(bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.2), label="item", confidence=0.7),
    ]

    result = _apply_class_agnostic_nms(predictions, iou_threshold=0.5)

    assert len(result) == 1, f"Expected 1, got {len(result)}"
    assert result[0].confidence == 0.9, "Should keep highest confidence"
    assert result[0].label == "product", "Should keep 'product' (highest conf)"
    print(f"  [PASS] 3 duplicate boxes -> 1 (kept highest conf: {result[0].label})")


def test_nms_keeps_separate_boxes():
    """NMS should keep boxes that don't overlap."""
    predictions = [
        AIPrediction(bbox=BBox(x=0.0, y=0.0, width=0.1, height=0.1), label="a", confidence=0.9),
        AIPrediction(bbox=BBox(x=0.5, y=0.5, width=0.1, height=0.1), label="b", confidence=0.8),
        AIPrediction(bbox=BBox(x=0.8, y=0.8, width=0.1, height=0.1), label="c", confidence=0.7),
    ]

    result = _apply_class_agnostic_nms(predictions, iou_threshold=0.5)

    assert len(result) == 3, f"Expected 3, got {len(result)}"
    print("  [PASS] 3 separate boxes -> 3 (all kept)")


def test_nms_mixed_scenario():
    """Real-world scenario: some overlap, some separate."""
    predictions = [
        # Group 1: Two overlapping boxes (should keep highest)
        AIPrediction(bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.2), label="shelf", confidence=0.95),
        AIPrediction(bbox=BBox(x=0.12, y=0.12, width=0.2, height=0.2), label="product", confidence=0.85),

        # Group 2: Separate box
        AIPrediction(bbox=BBox(x=0.6, y=0.6, width=0.15, height=0.15), label="price", confidence=0.75),

        # Group 3: Three overlapping boxes (should keep highest)
        AIPrediction(bbox=BBox(x=0.4, y=0.1, width=0.1, height=0.1), label="tag", confidence=0.60),
        AIPrediction(bbox=BBox(x=0.4, y=0.1, width=0.1, height=0.1), label="label", confidence=0.55),
        AIPrediction(bbox=BBox(x=0.4, y=0.1, width=0.1, height=0.1), label="text", confidence=0.50),
    ]

    result = _apply_class_agnostic_nms(predictions, iou_threshold=0.5)

    # Should keep: shelf (0.95), price (0.75), tag (0.60)
    assert len(result) == 3, f"Expected 3, got {len(result)}"

    labels = [p.label for p in result]
    assert "shelf" in labels, "Should keep 'shelf'"
    assert "price" in labels, "Should keep 'price'"
    assert "tag" in labels, "Should keep 'tag'"

    print(f"  [PASS] 6 boxes (2 groups overlap + 1 separate) -> 3 kept: {labels}")


def test_nms_empty_list():
    """Empty list should return empty list."""
    result = _apply_class_agnostic_nms([], iou_threshold=0.5)
    assert result == [], f"Expected [], got {result}"
    print("  [PASS] Empty list -> empty list")


def test_nms_single_item():
    """Single item should be kept."""
    predictions = [
        AIPrediction(bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.2), label="product", confidence=0.9),
    ]

    result = _apply_class_agnostic_nms(predictions, iou_threshold=0.5)
    assert len(result) == 1, f"Expected 1, got {len(result)}"
    print("  [PASS] Single item -> kept")


if __name__ == "__main__":
    print("\n=== Testing IoU Computation ===")
    test_compute_iou_no_overlap()
    test_compute_iou_full_overlap()
    test_compute_iou_partial_overlap()

    print("\n=== Testing Class-Agnostic NMS ===")
    test_nms_removes_duplicates()
    test_nms_keeps_separate_boxes()
    test_nms_mixed_scenario()
    test_nms_empty_list()
    test_nms_single_item()

    print("\n" + "=" * 40)
    print("ALL TESTS PASSED!")
    print("=" * 40 + "\n")
