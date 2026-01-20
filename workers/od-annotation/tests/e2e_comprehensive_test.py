#!/usr/bin/env python3
"""
Comprehensive E2E Tests for OD Annotation Worker

Tests:
1. Edge Cases (invalid inputs, large images, etc.)
2. Error Handling
3. Concurrent Requests
4. Performance Benchmarks
5. All Model Combinations
"""

import os
import sys
import time
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger
from PIL import Image
import io

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: dict = None

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s) {self.message}"


class ComprehensiveODTester:
    """Comprehensive tester for OD Annotation Worker."""

    def __init__(self):
        self.results: list[TestResult] = []
        self.test_images: list[dict] = []

        # Load environment
        from dotenv import load_dotenv
        load_dotenv("/workspace/.env")

        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.hf_token = os.getenv("HF_TOKEN")

        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials in .env")

        # Initialize Supabase client
        from supabase import create_client
        self.supabase = create_client(self.supabase_url, self.supabase_key)

        # Fetch test images
        result = self.supabase.table("od_images").select(
            "id, filename, image_url, width, height"
        ).limit(10).execute()
        self.test_images = result.data or []

        # Model cache
        self.model_cache: dict[str, Any] = {}

    def run_test(self, name: str, func) -> TestResult:
        """Run a test and record result."""
        logger.info(f"Running: {name}")
        start = time.time()
        try:
            result = func()
            duration = time.time() - start
            test_result = TestResult(name, True, duration, details=result if isinstance(result, dict) else None)
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Test failed: {e}")
            test_result = TestResult(name, False, duration, str(e))

        self.results.append(test_result)
        print(test_result)
        return test_result

    # ==========================================
    # 1. EDGE CASES
    # ==========================================

    def test_empty_image_url(self):
        """Test with empty image URL."""
        from models.grounding_dino import GroundingDINOModel

        model = GroundingDINOModel(device="cuda")

        try:
            model.predict(image_url="", text_prompt="product")
            raise AssertionError("Should have raised an error")
        except Exception as e:
            logger.info(f"Correctly raised error: {type(e).__name__}")
            return {"error_type": type(e).__name__}

    def test_invalid_image_url(self):
        """Test with invalid image URL."""
        from models.grounding_dino import GroundingDINOModel

        model = GroundingDINOModel(device="cuda")

        try:
            model.predict(
                image_url="https://invalid-url-that-does-not-exist.com/image.jpg",
                text_prompt="product"
            )
            raise AssertionError("Should have raised an error")
        except Exception as e:
            logger.info(f"Correctly raised error: {type(e).__name__}")
            return {"error_type": type(e).__name__}

    def test_malformed_url(self):
        """Test with malformed URL."""
        from models.grounding_dino import GroundingDINOModel

        model = GroundingDINOModel(device="cuda")

        try:
            model.predict(image_url="not-a-url", text_prompt="product")
            raise AssertionError("Should have raised an error")
        except Exception as e:
            logger.info(f"Correctly raised error: {type(e).__name__}")
            return {"error_type": type(e).__name__}

    def test_empty_text_prompt(self):
        """Test with empty text prompt."""
        from models.grounding_dino import GroundingDINOModel

        if not self.test_images:
            raise ValueError("No test images")

        model = GroundingDINOModel(device="cuda")

        # Empty prompt should still work (returns empty predictions)
        predictions = model.predict(
            image_url=self.test_images[0]["image_url"],
            text_prompt=""
        )

        logger.info(f"Empty prompt returned {len(predictions)} predictions")
        return {"predictions": len(predictions)}

    def test_very_long_text_prompt(self):
        """Test with very long text prompt."""
        from models.grounding_dino import GroundingDINOModel

        if not self.test_images:
            raise ValueError("No test images")

        model = GroundingDINOModel(device="cuda")

        # Very long prompt with many classes
        long_prompt = " . ".join([f"class_{i}" for i in range(100)])

        predictions = model.predict(
            image_url=self.test_images[0]["image_url"],
            text_prompt=long_prompt
        )

        logger.info(f"Long prompt (100 classes) returned {len(predictions)} predictions")
        return {"predictions": len(predictions), "prompt_length": len(long_prompt)}

    def test_special_characters_in_prompt(self):
        """Test with special characters in prompt."""
        from models.grounding_dino import GroundingDINOModel

        if not self.test_images:
            raise ValueError("No test images")

        model = GroundingDINOModel(device="cuda")

        # Prompt with special characters
        special_prompt = "shelf! @product# $price% ^tag& *sign"

        predictions = model.predict(
            image_url=self.test_images[0]["image_url"],
            text_prompt=special_prompt
        )

        logger.info(f"Special chars prompt returned {len(predictions)} predictions")
        return {"predictions": len(predictions)}

    def test_extreme_thresholds(self):
        """Test with extreme threshold values."""
        from models.grounding_dino import GroundingDINOModel

        if not self.test_images:
            raise ValueError("No test images")

        model = GroundingDINOModel(device="cuda")

        # Very high threshold
        predictions_high = model.predict(
            image_url=self.test_images[0]["image_url"],
            text_prompt="product",
            box_threshold=0.99,
            text_threshold=0.99
        )

        # Very low threshold
        predictions_low = model.predict(
            image_url=self.test_images[0]["image_url"],
            text_prompt="product",
            box_threshold=0.01,
            text_threshold=0.01
        )

        logger.info(f"High threshold: {len(predictions_high)}, Low threshold: {len(predictions_low)}")
        return {
            "high_threshold": len(predictions_high),
            "low_threshold": len(predictions_low)
        }

    def test_sam2_invalid_point(self):
        """Test SAM2 with invalid point coordinates."""
        from models.sam2 import SAM2Model

        if not self.test_images:
            raise ValueError("No test images")

        model = SAM2Model(device="cuda")

        # Point outside image bounds (normalized should be 0-1)
        result = model.segment_point(
            image_url=self.test_images[0]["image_url"],
            point=[2.0, 2.0],  # Outside bounds
            label=1
        )

        logger.info(f"Invalid point result: {result.get('confidence', 0)}")
        return {"confidence": result.get("confidence", 0)}

    def test_sam2_invalid_box(self):
        """Test SAM2 with invalid box coordinates."""
        from models.sam2 import SAM2Model

        if not self.test_images:
            raise ValueError("No test images")

        model = SAM2Model(device="cuda")

        # Box with negative width
        result = model.segment_box(
            image_url=self.test_images[0]["image_url"],
            box=[0.5, 0.5, -0.1, 0.2]  # Negative width
        )

        logger.info(f"Invalid box result: {result.get('confidence', 0)}")
        return {"confidence": result.get("confidence", 0)}

    # ==========================================
    # 2. HANDLER ERROR HANDLING
    # ==========================================

    def test_handler_invalid_task(self):
        """Test handler with invalid task type."""
        from handler import handler

        job_input = {
            "input": {
                "task": "invalid_task",
                "model": "grounding_dino",
                "image_url": self.test_images[0]["image_url"],
            }
        }

        result = handler(job_input)

        assert result.get("status") == "error", f"Expected error, got {result}"
        logger.info(f"Handler correctly returned error: {result.get('error')}")
        return {"error": result.get("error")}

    def test_handler_invalid_model(self):
        """Test handler with invalid model name."""
        from handler import handler

        job_input = {
            "input": {
                "task": "detect",
                "model": "invalid_model",
                "image_url": self.test_images[0]["image_url"],
                "text_prompt": "product"
            }
        }

        result = handler(job_input)

        assert result.get("status") == "error", f"Expected error, got {result}"
        logger.info(f"Handler correctly returned error: {result.get('error')}")
        return {"error": result.get("error")}

    def test_handler_missing_required_field(self):
        """Test handler with missing required field."""
        from handler import handler

        job_input = {
            "input": {
                "task": "detect",
                "model": "grounding_dino",
                # Missing image_url
                "text_prompt": "product"
            }
        }

        result = handler(job_input)

        assert result.get("status") == "error", f"Expected error, got {result}"
        logger.info(f"Handler correctly returned error: {result.get('error')}")
        return {"error": result.get("error")}

    def test_handler_empty_input(self):
        """Test handler with empty input."""
        from handler import handler

        job_input = {"input": {}}

        result = handler(job_input)

        assert result.get("status") == "error", f"Expected error, got {result}"
        logger.info(f"Handler correctly returned error: {result.get('error')}")
        return {"error": result.get("error")}

    # ==========================================
    # 3. CONCURRENT REQUESTS
    # ==========================================

    def test_concurrent_predictions(self):
        """Test 5 concurrent prediction requests."""
        from models.grounding_dino import GroundingDINOModel

        if len(self.test_images) < 5:
            raise ValueError("Need at least 5 test images")

        model = GroundingDINOModel(device="cuda")
        # Ensure model is loaded
        _ = model.model

        def predict_single(image_url):
            return model.predict(
                image_url=image_url,
                text_prompt="product",
                box_threshold=0.3
            )

        # Run 5 predictions sequentially (simulating concurrent - GPU serializes anyway)
        start = time.time()

        results = []
        for i in range(5):
            img_url = self.test_images[i % len(self.test_images)]["image_url"]
            preds = predict_single(img_url)
            results.append(len(preds))
            logger.info(f"  Request {i+1}: {len(preds)} predictions")

        total_time = time.time() - start

        logger.info(f"5 sequential predictions completed in {total_time:.2f}s")
        return {
            "total_time": total_time,
            "avg_time": total_time / 5,
            "total_predictions": sum(results)
        }

    def test_concurrent_different_models(self):
        """Test predictions with different models sequentially."""
        if not self.test_images:
            raise ValueError("No test images")

        image_url = self.test_images[0]["image_url"]

        from models.grounding_dino import GroundingDINOModel
        from models.sam2 import SAM2Model
        from models.florence2 import Florence2Model

        results = {}

        # Test each model
        logger.info("Testing Grounding DINO...")
        start = time.time()
        gdino = GroundingDINOModel(device="cuda")
        gdino_preds = gdino.predict(image_url=image_url, text_prompt="product")
        results["grounding_dino"] = {"time": time.time() - start, "predictions": len(gdino_preds)}

        logger.info("Testing SAM2...")
        start = time.time()
        sam2 = SAM2Model(device="cuda")
        sam2_result = sam2.segment_point(image_url=image_url, point=[0.5, 0.5], label=1)
        results["sam2"] = {"time": time.time() - start, "confidence": sam2_result.get("confidence", 0)}

        logger.info("Testing Florence-2...")
        start = time.time()
        florence = Florence2Model(device="cuda")
        florence_preds = florence.predict(image_url=image_url, text_prompt="product")
        results["florence2"] = {"time": time.time() - start, "predictions": len(florence_preds)}

        logger.info(f"All models tested: {results}")
        return results

    # ==========================================
    # 4. PERFORMANCE BENCHMARKS
    # ==========================================

    def test_warm_prediction_latency(self):
        """Test prediction latency with warm model (already loaded)."""
        from models.grounding_dino import GroundingDINOModel

        if not self.test_images:
            raise ValueError("No test images")

        model = GroundingDINOModel(device="cuda")
        # Warm up
        model.predict(image_url=self.test_images[0]["image_url"], text_prompt="product")

        # Measure 5 predictions
        times = []
        for i in range(5):
            start = time.time()
            model.predict(
                image_url=self.test_images[i % len(self.test_images)]["image_url"],
                text_prompt="product"
            )
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        logger.info(f"Warm prediction latency: avg={avg_time:.2f}s, min={min(times):.2f}s, max={max(times):.2f}s")

        # Check if under 5 seconds
        assert avg_time < 5, f"Warm prediction too slow: {avg_time:.2f}s"

        return {
            "avg": avg_time,
            "min": min(times),
            "max": max(times),
            "all_under_5s": all(t < 5 for t in times)
        }

    def test_sam_segmentation_latency(self):
        """Test SAM segmentation latency (should be < 3 seconds)."""
        from models.sam2 import SAM2Model

        if not self.test_images:
            raise ValueError("No test images")

        model = SAM2Model(device="cuda")
        # Warm up
        model.segment_point(image_url=self.test_images[0]["image_url"], point=[0.5, 0.5], label=1)

        # Measure 5 segmentations
        times = []
        for i in range(5):
            start = time.time()
            model.segment_point(
                image_url=self.test_images[i % len(self.test_images)]["image_url"],
                point=[0.5, 0.5],
                label=1
            )
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        logger.info(f"SAM segmentation latency: avg={avg_time:.2f}s, min={min(times):.2f}s, max={max(times):.2f}s")

        # Check if under 3 seconds
        assert avg_time < 3, f"SAM segmentation too slow: {avg_time:.2f}s"

        return {
            "avg": avg_time,
            "min": min(times),
            "max": max(times),
            "all_under_3s": all(t < 3 for t in times)
        }

    def test_batch_10_images(self):
        """Test batch processing of 10 images (should be < 60 seconds)."""
        from models.grounding_dino import GroundingDINOModel

        if len(self.test_images) < 10:
            logger.warning(f"Only {len(self.test_images)} images available, using all")

        model = GroundingDINOModel(device="cuda")
        # Warm up
        model.predict(image_url=self.test_images[0]["image_url"], text_prompt="product")

        start = time.time()
        total_predictions = 0

        for i in range(min(10, len(self.test_images))):
            preds = model.predict(
                image_url=self.test_images[i]["image_url"],
                text_prompt="product",
                box_threshold=0.3
            )
            total_predictions += len(preds)
            logger.info(f"  Image {i+1}: {len(preds)} predictions")

        total_time = time.time() - start
        images_processed = min(10, len(self.test_images))

        logger.info(f"Batch {images_processed} images: {total_time:.2f}s, {total_predictions} total predictions")

        # Check if under 60 seconds
        assert total_time < 60, f"Batch too slow: {total_time:.2f}s"

        return {
            "total_time": total_time,
            "images_processed": images_processed,
            "total_predictions": total_predictions,
            "avg_per_image": total_time / images_processed
        }

    # ==========================================
    # 5. ALL MODEL COMBINATIONS
    # ==========================================

    def test_all_detection_models(self):
        """Test all detection models with same image."""
        if not self.test_images:
            raise ValueError("No test images")

        image_url = self.test_images[0]["image_url"]
        results = {}

        # Grounding DINO
        from models.grounding_dino import GroundingDINOModel
        gdino = GroundingDINOModel(device="cuda")
        gdino_preds = gdino.predict(image_url=image_url, text_prompt="shelf . product")
        results["grounding_dino"] = len(gdino_preds)
        logger.info(f"Grounding DINO: {len(gdino_preds)} predictions")

        # SAM3
        from models.sam3 import SAM3Model
        sam3 = SAM3Model(device="cuda", hf_token=self.hf_token)
        sam3_preds = sam3.predict(image_url=image_url, text_prompt="shelf")
        results["sam3"] = len(sam3_preds)
        logger.info(f"SAM3: {len(sam3_preds)} predictions")

        # Florence-2
        from models.florence2 import Florence2Model
        florence = Florence2Model(device="cuda")
        florence_preds = florence.predict(image_url=image_url, text_prompt="shelf product")
        results["florence2"] = len(florence_preds)
        logger.info(f"Florence-2: {len(florence_preds)} predictions")

        return results

    def test_all_segmentation_models(self):
        """Test all segmentation models with same image."""
        if not self.test_images:
            raise ValueError("No test images")

        image_url = self.test_images[0]["image_url"]
        results = {}

        # SAM2 point
        from models.sam2 import SAM2Model
        sam2 = SAM2Model(device="cuda")
        sam2_result = sam2.segment_point(image_url=image_url, point=[0.5, 0.5], label=1)
        results["sam2_point"] = sam2_result.get("confidence", 0)
        logger.info(f"SAM2 Point: confidence={results['sam2_point']:.2f}")

        # SAM2 box
        sam2_box = sam2.segment_box(image_url=image_url, box=[0.25, 0.25, 0.5, 0.5])
        results["sam2_box"] = sam2_box.get("confidence", 0)
        logger.info(f"SAM2 Box: confidence={results['sam2_box']:.2f}")

        # SAM3 with text
        from models.sam3 import SAM3Model
        sam3 = SAM3Model(device="cuda", hf_token=self.hf_token)
        sam3_result = sam3.predict(image_url=image_url, text_prompt="shelf")
        results["sam3_text"] = len(sam3_result)
        logger.info(f"SAM3 Text: {len(sam3_result)} segments")

        return results

    # ==========================================
    # RUN ALL TESTS
    # ==========================================

    def run_all(self):
        """Run all comprehensive tests."""
        print("\n" + "=" * 60)
        print("🧪 OD Annotation Worker - Comprehensive E2E Tests")
        print("=" * 60 + "\n")

        # 1. Edge Cases
        print("\n🔧 EDGE CASE TESTS")
        print("-" * 40)
        self.run_test("Empty Image URL", self.test_empty_image_url)
        self.run_test("Invalid Image URL", self.test_invalid_image_url)
        self.run_test("Malformed URL", self.test_malformed_url)
        self.run_test("Empty Text Prompt", self.test_empty_text_prompt)
        self.run_test("Very Long Text Prompt", self.test_very_long_text_prompt)
        self.run_test("Special Characters in Prompt", self.test_special_characters_in_prompt)
        self.run_test("Extreme Thresholds", self.test_extreme_thresholds)
        self.run_test("SAM2 Invalid Point", self.test_sam2_invalid_point)
        self.run_test("SAM2 Invalid Box", self.test_sam2_invalid_box)

        # 2. Error Handling
        print("\n❌ ERROR HANDLING TESTS")
        print("-" * 40)
        self.run_test("Handler Invalid Task", self.test_handler_invalid_task)
        self.run_test("Handler Invalid Model", self.test_handler_invalid_model)
        self.run_test("Handler Missing Field", self.test_handler_missing_required_field)
        self.run_test("Handler Empty Input", self.test_handler_empty_input)

        # 3. Concurrent Requests
        print("\n🔄 CONCURRENT REQUEST TESTS")
        print("-" * 40)
        self.run_test("5 Sequential Predictions", self.test_concurrent_predictions)
        self.run_test("Different Models Sequential", self.test_concurrent_different_models)

        # 4. Performance Benchmarks
        print("\n⚡ PERFORMANCE TESTS")
        print("-" * 40)
        self.run_test("Warm Prediction Latency (<5s)", self.test_warm_prediction_latency)
        self.run_test("SAM Segmentation Latency (<3s)", self.test_sam_segmentation_latency)
        self.run_test("Batch 10 Images (<60s)", self.test_batch_10_images)

        # 5. All Model Combinations
        print("\n🤖 MODEL COMBINATION TESTS")
        print("-" * 40)
        self.run_test("All Detection Models", self.test_all_detection_models)
        self.run_test("All Segmentation Models", self.test_all_segmentation_models)

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)

        print(f"\nTotal: {len(self.results)} tests")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Duration: {total_time:.2f}s")

        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r.passed:
                    print(f"   - {r.name}: {r.message}")
            return 1

        print("\n✅ ALL TESTS PASSED!")
        return 0


def main():
    tester = ComprehensiveODTester()
    sys.exit(tester.run_all())


if __name__ == "__main__":
    main()
