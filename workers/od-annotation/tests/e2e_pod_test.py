#!/usr/bin/env python3
"""
Comprehensive E2E Test for OD Annotation Worker on GPU Pod

Tests all models:
- Grounding DINO: Text→bbox detection
- SAM2: Point/box segmentation
- SAM3: Text-guided segmentation
- Florence-2: Versatile vision tasks

Uses real images from Supabase database.
"""

import os
import sys
import time
import json
import base64
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


class ODAnnotationTester:
    """End-to-end tester for OD Annotation Worker."""

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
    # Setup Tests
    # ==========================================

    def test_supabase_connection(self):
        """Test Supabase connection."""
        result = self.supabase.table("od_images").select("id", count="exact").limit(1).execute()
        count = result.count or 0
        logger.info(f"Supabase connected. Found {count} images in od_images.")
        return {"image_count": count}

    def test_fetch_test_images(self):
        """Fetch test images from Supabase."""
        result = self.supabase.table("od_images").select(
            "id, filename, image_url, width, height"
        ).limit(5).execute()

        self.test_images = result.data or []

        if not self.test_images:
            raise ValueError("No images found in od_images table!")

        logger.info(f"Fetched {len(self.test_images)} test images")
        for img in self.test_images:
            logger.info(f"  - {img['filename']} ({img['width']}x{img['height']})")

        return {"images": [img["filename"] for img in self.test_images]}

    def test_download_image(self):
        """Test downloading an image."""
        if not self.test_images:
            raise ValueError("No test images available")

        image_url = self.test_images[0]["image_url"]
        logger.info(f"Downloading: {image_url[:80]}...")

        response = httpx.get(image_url, timeout=30)
        response.raise_for_status()

        # Verify it's a valid image
        img = Image.open(io.BytesIO(response.content))
        logger.info(f"Downloaded image: {img.size[0]}x{img.size[1]}, mode={img.mode}")

        return {"size": img.size, "mode": img.mode}

    # ==========================================
    # Model Loading Tests
    # ==========================================

    def test_load_grounding_dino(self):
        """Test loading Grounding DINO model."""
        from models.grounding_dino import GroundingDINOModel

        logger.info("Loading Grounding DINO...")
        start = time.time()
        model = GroundingDINOModel(device="cuda")
        load_time = time.time() - start

        self.model_cache["grounding_dino"] = model
        logger.info(f"Grounding DINO loaded in {load_time:.2f}s")

        return {"load_time": load_time}

    def test_load_sam2(self):
        """Test loading SAM2 model."""
        from models.sam2 import SAM2Model

        logger.info("Loading SAM2...")
        start = time.time()
        model = SAM2Model(device="cuda")
        load_time = time.time() - start

        self.model_cache["sam2"] = model
        logger.info(f"SAM2 loaded in {load_time:.2f}s")

        return {"load_time": load_time}

    def test_load_sam3(self):
        """Test loading SAM3 model."""
        from models.sam3 import SAM3Model

        logger.info("Loading SAM3...")
        start = time.time()
        model = SAM3Model(device="cuda", hf_token=self.hf_token)
        load_time = time.time() - start

        self.model_cache["sam3"] = model
        logger.info(f"SAM3 loaded in {load_time:.2f}s")

        return {"load_time": load_time}

    def test_load_florence2(self):
        """Test loading Florence-2 model."""
        from models.florence2 import Florence2Model

        logger.info("Loading Florence-2...")
        start = time.time()
        model = Florence2Model(device="cuda")
        load_time = time.time() - start

        self.model_cache["florence2"] = model
        logger.info(f"Florence-2 loaded in {load_time:.2f}s")

        return {"load_time": load_time}

    # ==========================================
    # Inference Tests
    # ==========================================

    def test_grounding_dino_inference(self):
        """Test Grounding DINO detection."""
        if "grounding_dino" not in self.model_cache:
            raise ValueError("Grounding DINO not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["grounding_dino"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running Grounding DINO detection...")
        start = time.time()
        predictions = model.predict(
            image_url=image_url,
            text_prompt="shelf . product . price tag",
            box_threshold=0.3,
            text_threshold=0.25,
        )
        inference_time = time.time() - start

        logger.info(f"Found {len(predictions)} objects in {inference_time:.2f}s")
        for pred in predictions[:5]:
            logger.info(f"  - {pred['label']}: {pred['confidence']:.2f} @ {pred['bbox']}")

        return {
            "inference_time": inference_time,
            "predictions_count": len(predictions),
            "predictions": predictions[:3],
        }

    def test_sam2_point_segmentation(self):
        """Test SAM2 point segmentation."""
        if "sam2" not in self.model_cache:
            raise ValueError("SAM2 not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["sam2"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running SAM2 point segmentation...")
        start = time.time()
        result = model.segment_point(
            image_url=image_url,
            point=[0.5, 0.5],  # Center of image
            label=1,
            return_mask=True,
        )
        inference_time = time.time() - start

        logger.info(f"Segmentation complete in {inference_time:.2f}s")
        logger.info(f"  bbox: {result['bbox']}")
        logger.info(f"  confidence: {result.get('confidence', 'N/A')}")
        logger.info(f"  has_mask: {'mask' in result and result['mask'] is not None}")

        return {
            "inference_time": inference_time,
            "bbox": result["bbox"],
            "has_mask": "mask" in result and result["mask"] is not None,
        }

    def test_sam2_box_segmentation(self):
        """Test SAM2 box segmentation."""
        if "sam2" not in self.model_cache:
            raise ValueError("SAM2 not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["sam2"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running SAM2 box segmentation...")
        start = time.time()
        result = model.segment_box(
            image_url=image_url,
            box=[0.25, 0.25, 0.5, 0.5],  # Center region
            return_mask=True,
        )
        inference_time = time.time() - start

        logger.info(f"Box segmentation complete in {inference_time:.2f}s")
        logger.info(f"  bbox: {result['bbox']}")

        return {
            "inference_time": inference_time,
            "bbox": result["bbox"],
        }

    def test_sam3_text_segmentation(self):
        """Test SAM3 text-guided segmentation."""
        if "sam3" not in self.model_cache:
            raise ValueError("SAM3 not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["sam3"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running SAM3 text-guided segmentation...")
        start = time.time()
        predictions = model.predict(
            image_url=image_url,
            text_prompt="shelf",
            box_threshold=0.3,
        )
        inference_time = time.time() - start

        logger.info(f"Found {len(predictions)} segments in {inference_time:.2f}s")
        for pred in predictions[:3]:
            logger.info(f"  - {pred['label']}: {pred['confidence']:.2f}")

        return {
            "inference_time": inference_time,
            "predictions_count": len(predictions),
        }

    def test_florence2_detection(self):
        """Test Florence-2 detection."""
        if "florence2" not in self.model_cache:
            raise ValueError("Florence-2 not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["florence2"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running Florence-2 detection...")
        start = time.time()
        # Florence-2 predict() handles task selection based on text_prompt
        # Empty prompt = general OD, with prompt = phrase grounding
        predictions = model.predict(
            image_url=image_url,
            text_prompt="shelf product",  # Phrase grounding
        )
        inference_time = time.time() - start

        logger.info(f"Found {len(predictions)} objects in {inference_time:.2f}s")
        for pred in predictions[:3]:
            logger.info(f"  - {pred['label']}: {pred.get('confidence', 1.0):.2f}")

        return {
            "inference_time": inference_time,
            "predictions_count": len(predictions),
        }

    def test_florence2_dense_caption(self):
        """Test Florence-2 dense region caption generation."""
        if "florence2" not in self.model_cache:
            raise ValueError("Florence-2 not loaded")
        if not self.test_images:
            raise ValueError("No test images")

        model = self.model_cache["florence2"]
        image_url = self.test_images[0]["image_url"]

        logger.info("Running Florence-2 dense region captioning...")
        start = time.time()
        regions = model.dense_region_caption(image_url=image_url)
        inference_time = time.time() - start

        logger.info(f"Generated {len(regions)} region captions in {inference_time:.2f}s")
        for region in regions[:3]:
            logger.info(f"  - {region.get('caption', 'N/A')[:50]}...")

        return {
            "inference_time": inference_time,
            "regions_count": len(regions),
        }

    # ==========================================
    # Batch Processing Tests
    # ==========================================

    def test_batch_detection(self):
        """Test batch detection with multiple images."""
        if "grounding_dino" not in self.model_cache:
            raise ValueError("Grounding DINO not loaded")
        if len(self.test_images) < 2:
            raise ValueError("Need at least 2 test images")

        model = self.model_cache["grounding_dino"]

        logger.info(f"Running batch detection on {len(self.test_images)} images...")
        start = time.time()

        results = []
        for img in self.test_images:
            img_start = time.time()
            predictions = model.predict(
                image_url=img["image_url"],
                text_prompt="product",
                box_threshold=0.3,
            )
            img_time = time.time() - img_start
            results.append({
                "filename": img["filename"],
                "predictions": len(predictions),
                "time": img_time,
            })
            logger.info(f"  {img['filename']}: {len(predictions)} objects in {img_time:.2f}s")

        total_time = time.time() - start
        total_predictions = sum(r["predictions"] for r in results)

        logger.info(f"Batch complete: {total_predictions} total predictions in {total_time:.2f}s")

        return {
            "total_time": total_time,
            "images_processed": len(results),
            "total_predictions": total_predictions,
            "avg_time_per_image": total_time / len(results),
        }

    # ==========================================
    # Database Operations Tests
    # ==========================================

    def test_create_annotation(self):
        """Test creating an annotation in database."""
        if not self.test_images:
            raise ValueError("No test images")

        # First, get or create a test dataset
        dataset_result = self.supabase.table("od_datasets").select("id").limit(1).execute()

        if not dataset_result.data:
            # Create a test dataset
            create_result = self.supabase.table("od_datasets").insert({
                "name": f"E2E Test Dataset {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "description": "Created by E2E test",
                "annotation_type": "bbox",
            }).execute()
            dataset_id = create_result.data[0]["id"]
            logger.info(f"Created test dataset: {dataset_id}")
        else:
            dataset_id = dataset_result.data[0]["id"]
            logger.info(f"Using existing dataset: {dataset_id}")

        # Get or create a test class
        class_result = self.supabase.table("od_classes").select("id").eq("name", "e2e_test_class").limit(1).execute()

        if not class_result.data:
            create_class = self.supabase.table("od_classes").insert({
                "name": "e2e_test_class",
                "display_name": "E2E Test Class",
                "color": "#FF5733",
                "category": "test",
            }).execute()
            class_id = create_class.data[0]["id"]
            logger.info(f"Created test class: {class_id}")
        else:
            class_id = class_result.data[0]["id"]
            logger.info(f"Using existing class: {class_id}")

        # Create annotation
        image_id = self.test_images[0]["id"]

        # Note: Schema uses separate bbox columns, not JSONB
        annotation_data = {
            "dataset_id": dataset_id,
            "image_id": image_id,
            "class_id": class_id,
            "bbox_x": 0.1,
            "bbox_y": 0.1,
            "bbox_width": 0.2,
            "bbox_height": 0.2,
            "is_ai_generated": True,
            "confidence": 0.95,
            "ai_model": "e2e_test",
        }

        result = self.supabase.table("od_annotations").insert(annotation_data).execute()
        annotation_id = result.data[0]["id"]
        logger.info(f"Created annotation: {annotation_id}")

        # Store for cleanup
        self._test_annotation_id = annotation_id
        self._test_dataset_id = dataset_id
        self._test_class_id = class_id

        return {
            "annotation_id": annotation_id,
            "dataset_id": dataset_id,
            "class_id": class_id,
        }

    def test_read_annotation(self):
        """Test reading annotation from database."""
        if not hasattr(self, "_test_annotation_id"):
            raise ValueError("No test annotation created")

        result = self.supabase.table("od_annotations").select("*").eq(
            "id", self._test_annotation_id
        ).single().execute()

        annotation = result.data
        logger.info(f"Read annotation: {annotation['id']}")
        logger.info(f"  bbox: ({annotation['bbox_x']}, {annotation['bbox_y']}, {annotation['bbox_width']}, {annotation['bbox_height']})")
        logger.info(f"  ai_model: {annotation['ai_model']}")

        return {"annotation": annotation}

    def test_delete_annotation(self):
        """Test deleting annotation from database."""
        if not hasattr(self, "_test_annotation_id"):
            raise ValueError("No test annotation created")

        self.supabase.table("od_annotations").delete().eq(
            "id", self._test_annotation_id
        ).execute()

        logger.info(f"Deleted annotation: {self._test_annotation_id}")

        return {"deleted": self._test_annotation_id}

    # ==========================================
    # Handler Integration Test
    # ==========================================

    def test_handler_detect(self):
        """Test the main handler with detect task."""
        from handler import handler

        if not self.test_images:
            raise ValueError("No test images")

        job_input = {
            "input": {
                "task": "detect",
                "model": "grounding_dino",
                "image_url": self.test_images[0]["image_url"],
                "text_prompt": "shelf . product",
                "box_threshold": 0.3,
            }
        }

        logger.info("Testing handler with detect task...")
        start = time.time()
        result = handler(job_input)
        duration = time.time() - start

        assert result.get("status") != "error", f"Handler error: {result.get('error')}"
        logger.info(f"Handler returned {len(result.get('predictions', []))} predictions in {duration:.2f}s")

        return {
            "duration": duration,
            "predictions_count": len(result.get("predictions", [])),
            "status": result.get("status"),
        }

    def test_handler_segment(self):
        """Test the main handler with segment task."""
        from handler import handler

        if not self.test_images:
            raise ValueError("No test images")

        job_input = {
            "input": {
                "task": "segment",
                "model": "sam2",
                "image_url": self.test_images[0]["image_url"],
                "prompt_type": "point",
                "point": [0.5, 0.5],
                "label": 1,
            }
        }

        logger.info("Testing handler with segment task...")
        start = time.time()
        result = handler(job_input)
        duration = time.time() - start

        assert result.get("status") != "error", f"Handler error: {result.get('error')}"
        logger.info(f"Handler returned bbox: {result.get('bbox')} in {duration:.2f}s")

        return {
            "duration": duration,
            "bbox": result.get("bbox"),
            "status": result.get("status"),
        }

    def test_handler_batch(self):
        """Test the main handler with batch task."""
        from handler import handler

        if len(self.test_images) < 2:
            raise ValueError("Need at least 2 test images")

        images = [
            {"id": img["id"], "url": img["image_url"]}
            for img in self.test_images[:3]
        ]

        job_input = {
            "input": {
                "task": "batch",
                "model": "grounding_dino",
                "images": images,
                "text_prompt": "product",
                "box_threshold": 0.3,
            }
        }

        logger.info(f"Testing handler with batch task ({len(images)} images)...")
        start = time.time()
        result = handler(job_input)
        duration = time.time() - start

        assert result.get("status") in ["success", "partial"], f"Handler error: {result.get('error')}"
        logger.info(f"Handler processed {result.get('successful', 0)} images in {duration:.2f}s")

        return {
            "duration": duration,
            "successful": result.get("successful"),
            "failed": result.get("failed"),
            "status": result.get("status"),
        }

    # ==========================================
    # Run All Tests
    # ==========================================

    def run_all(self):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("🧪 OD Annotation Worker - E2E Test Suite")
        print("=" * 60 + "\n")

        # Setup tests
        print("\n📦 SETUP TESTS")
        print("-" * 40)
        self.run_test("Supabase Connection", self.test_supabase_connection)
        self.run_test("Fetch Test Images", self.test_fetch_test_images)
        self.run_test("Download Image", self.test_download_image)

        # Model loading tests
        print("\n🔧 MODEL LOADING TESTS")
        print("-" * 40)
        self.run_test("Load Grounding DINO", self.test_load_grounding_dino)
        self.run_test("Load SAM2", self.test_load_sam2)
        self.run_test("Load SAM3", self.test_load_sam3)
        self.run_test("Load Florence-2", self.test_load_florence2)

        # Inference tests
        print("\n🎯 INFERENCE TESTS")
        print("-" * 40)
        self.run_test("Grounding DINO Inference", self.test_grounding_dino_inference)
        self.run_test("SAM2 Point Segmentation", self.test_sam2_point_segmentation)
        self.run_test("SAM2 Box Segmentation", self.test_sam2_box_segmentation)
        self.run_test("SAM3 Text Segmentation", self.test_sam3_text_segmentation)
        self.run_test("Florence-2 Detection", self.test_florence2_detection)
        self.run_test("Florence-2 Dense Caption", self.test_florence2_dense_caption)

        # Batch tests
        print("\n📦 BATCH PROCESSING TESTS")
        print("-" * 40)
        self.run_test("Batch Detection", self.test_batch_detection)

        # Database tests
        print("\n💾 DATABASE OPERATIONS TESTS")
        print("-" * 40)
        self.run_test("Create Annotation", self.test_create_annotation)
        self.run_test("Read Annotation", self.test_read_annotation)
        self.run_test("Delete Annotation", self.test_delete_annotation)

        # Handler integration tests
        print("\n🔌 HANDLER INTEGRATION TESTS")
        print("-" * 40)
        self.run_test("Handler Detect", self.test_handler_detect)
        self.run_test("Handler Segment", self.test_handler_segment)
        self.run_test("Handler Batch", self.test_handler_batch)

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
    tester = ODAnnotationTester()
    sys.exit(tester.run_all())


if __name__ == "__main__":
    main()
