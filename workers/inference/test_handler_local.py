"""
Local test script for inference worker handler.

Tests the handler without Docker/RunPod to validate logic.
This is for development testing only - does not test GPU code paths.
"""

import sys
import base64
import io
from PIL import Image, ImageDraw
import json


def create_test_image(size=(640, 480), draw_objects=True):
    """Create a simple test image with some objects."""
    img = Image.new("RGB", size, color=(255, 255, 255))

    if draw_objects:
        draw = ImageDraw.Draw(img)
        # Draw some rectangles (simulate objects)
        draw.rectangle([100, 100, 200, 200], fill=(255, 0, 0))  # Red square
        draw.rectangle([300, 200, 450, 350], fill=(0, 0, 255))  # Blue rectangle
        draw.ellipse([400, 50, 500, 150], fill=(0, 255, 0))  # Green circle

    return img


def image_to_base64(image):
    """Convert PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def test_detection():
    """Test detection task."""
    print("\n" + "="*60)
    print("TEST 1: Detection (pretrained YOLO)")
    print("="*60)

    test_image = create_test_image()
    image_b64 = image_to_base64(test_image)

    job_input = {
        "task": "detection",
        "model_id": "test-model",
        "model_source": "pretrained",
        "model_type": "yolo11n",  # Smallest YOLO for testing
        "checkpoint_url": None,
        "class_mapping": None,
        "image": image_b64,
        "config": {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "input_size": 640,
        }
    }

    print(f"Input: {job_input['model_type']}, source={job_input['model_source']}")
    print(f"Image size: {test_image.size}")

    try:
        from handler import handler
        result = handler({"input": job_input})

        print(f"\n✅ Result success: {result.get('success', False)}")

        if result.get("success"):
            res_data = result["result"]
            print(f"Detections: {res_data.get('count', 0)}")
            print(f"Image size: {res_data.get('image_size', {})}")
            print(f"Inference time: {result.get('metadata', {}).get('inference_time_ms', 0):.0f}ms")
            print(f"Model cached: {result.get('metadata', {}).get('cached', False)}")

            # Validate response structure
            assert "detections" in res_data, "Missing 'detections' field"
            assert "count" in res_data, "Missing 'count' field"
            assert "image_size" in res_data, "Missing 'image_size' field"
            assert isinstance(res_data["detections"], list), "'detections' should be a list"

            if res_data["detections"]:
                det = res_data["detections"][0]
                print(f"\nFirst detection: {det.get('class_name', 'unknown')} "
                      f"({det.get('confidence', 0):.2f})")

                # Validate detection structure
                assert "class_name" in det, "Detection missing 'class_name'"
                assert "class_id" in det, "Detection missing 'class_id'"
                assert "confidence" in det, "Detection missing 'confidence'"
                assert "bbox" in det, "Detection missing 'bbox'"
                assert "area" in det, "Detection missing 'area'"

            print("\n✅ All validation checks passed!")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            if "traceback" in result:
                print(f"\n{result['traceback']}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_classification():
    """Test classification task."""
    print("\n" + "="*60)
    print("TEST 2: Classification (pretrained ViT)")
    print("="*60)

    test_image = create_test_image(draw_objects=False)
    image_b64 = image_to_base64(test_image)

    job_input = {
        "task": "classification",
        "model_id": "test-model",
        "model_source": "pretrained",
        "model_type": "vit-tiny",  # Smallest ViT for testing
        "checkpoint_url": None,
        "class_mapping": None,
        "num_classes": None,
        "image": image_b64,
        "config": {
            "top_k": 5,
            "threshold": 0.0,
        }
    }

    print(f"Input: {job_input['model_type']}, source={job_input['model_source']}")
    print(f"Image size: {test_image.size}")

    try:
        from handler import handler
        result = handler({"input": job_input})

        print(f"\n✅ Result success: {result.get('success', False)}")

        if result.get("success"):
            res_data = result["result"]
            print(f"Predictions: {len(res_data.get('predictions', []))}")
            print(f"Top class: {res_data.get('top_class', 'unknown')}")
            print(f"Confidence: {res_data.get('top_confidence', 0):.2f}")
            print(f"Inference time: {result.get('metadata', {}).get('inference_time_ms', 0):.0f}ms")
            print(f"Model cached: {result.get('metadata', {}).get('cached', False)}")

            # Validate response structure
            assert "predictions" in res_data, "Missing 'predictions' field"
            assert "top_class" in res_data, "Missing 'top_class' field"
            assert "top_confidence" in res_data, "Missing 'top_confidence' field"
            assert isinstance(res_data["predictions"], list), "'predictions' should be a list"

            if res_data["predictions"]:
                pred = res_data["predictions"][0]
                print(f"\nTop prediction: {pred.get('class_name', 'unknown')} "
                      f"({pred.get('confidence', 0):.2f})")

                # Validate prediction structure
                assert "class_name" in pred, "Prediction missing 'class_name'"
                assert "class_id" in pred, "Prediction missing 'class_id'"
                assert "confidence" in pred, "Prediction missing 'confidence'"

            print("\n✅ All validation checks passed!")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            if "traceback" in result:
                print(f"\n{result['traceback']}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_embedding():
    """Test embedding task."""
    print("\n" + "="*60)
    print("TEST 3: Embedding (pretrained DINOv2)")
    print("="*60)

    test_image = create_test_image(draw_objects=False)
    image_b64 = image_to_base64(test_image)

    job_input = {
        "task": "embedding",
        "model_id": "test-model",
        "model_source": "pretrained",
        "model_type": "dinov2-small",  # Smallest DINOv2
        "checkpoint_url": None,
        "embedding_dim": 384,
        "image": image_b64,
        "config": {
            "normalize": True,
            "pooling": "cls",
        }
    }

    print(f"Input: {job_input['model_type']}, source={job_input['model_source']}")
    print(f"Image size: {test_image.size}")

    try:
        from handler import handler
        result = handler({"input": job_input})

        print(f"\n✅ Result success: {result.get('success', False)}")

        if result.get("success"):
            res_data = result["result"]
            print(f"Embedding dim: {res_data.get('embedding_dim', 0)}")
            print(f"Normalized: {res_data.get('normalized', False)}")
            print(f"Inference time: {result.get('metadata', {}).get('inference_time_ms', 0):.0f}ms")
            print(f"Model cached: {result.get('metadata', {}).get('cached', False)}")

            # Validate response structure
            assert "embedding" in res_data, "Missing 'embedding' field"
            assert "embedding_dim" in res_data, "Missing 'embedding_dim' field"
            assert "normalized" in res_data, "Missing 'normalized' field"
            assert isinstance(res_data["embedding"], list), "'embedding' should be a list"

            embedding = res_data["embedding"]
            print(f"Embedding length: {len(embedding)}")
            print(f"Sample values: {embedding[:5]}...")

            # Validate embedding
            assert len(embedding) == res_data["embedding_dim"], "Embedding dimension mismatch"
            assert all(isinstance(x, (int, float)) for x in embedding[:10]), "Invalid embedding values"

            # Check normalization
            if res_data["normalized"]:
                import math
                norm = math.sqrt(sum(x**2 for x in embedding))
                print(f"L2 norm: {norm:.4f} (should be ~1.0)")
                assert abs(norm - 1.0) < 0.01, f"Not normalized: norm={norm}"

            print("\n✅ All validation checks passed!")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown')}")
            if "traceback" in result:
                print(f"\n{result['traceback']}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_input_validation():
    """Test input validation."""
    print("\n" + "="*60)
    print("TEST 4: Input Validation")
    print("="*60)

    from handler import handler

    # Test missing task
    result = handler({"input": {}})
    assert not result["success"], "Should fail on missing task"
    print("✅ Missing task validation works")

    # Test invalid task
    result = handler({"input": {"task": "invalid"}})
    assert not result["success"], "Should fail on invalid task"
    print("✅ Invalid task validation works")

    # Test missing model_type
    result = handler({"input": {"task": "detection"}})
    assert not result["success"], "Should fail on missing model_type"
    print("✅ Missing model_type validation works")

    # Test missing image
    result = handler({"input": {"task": "detection", "model_type": "yolo11n"}})
    assert not result["success"], "Should fail on missing image"
    print("✅ Missing image validation works")

    # Test invalid image
    result = handler({"input": {
        "task": "detection",
        "model_type": "yolo11n",
        "image": "not-valid-base64!!!"
    }})
    assert not result["success"], "Should fail on invalid image"
    print("✅ Invalid image validation works")

    print("\n✅ All validation tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("INFERENCE WORKER - LOCAL HANDLER TESTS")
    print("="*60)
    print("\nNOTE: This tests handler logic only.")
    print("GPU/CUDA features require actual hardware.")
    print("Some models will be downloaded on first run (~100-500MB).")
    print("\n" + "="*60)

    results = []

    # Input validation (fast, no model loading)
    results.append(("Input Validation", test_input_validation()))

    # Ask user which tests to run
    print("\nWhich inference tests do you want to run?")
    print("1. Detection (downloads ~6MB YOLO11n)")
    print("2. Classification (downloads ~20MB ViT-tiny)")
    print("3. Embedding (downloads ~80MB DINOv2-small)")
    print("4. All of the above")
    print("5. Skip inference tests")

    choice = input("\nChoice (1-5) [5]: ").strip() or "5"

    if choice in ["1", "4"]:
        results.append(("Detection", test_detection()))

    if choice in ["2", "4"]:
        results.append(("Classification", test_classification()))

    if choice in ["3", "4"]:
        results.append(("Embedding", test_embedding()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
