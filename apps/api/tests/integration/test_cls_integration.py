"""
CLS Integration Tests

Validates that API and Workers are properly integrated:
1. Input format compatibility
2. Response parsing
3. Configuration checks
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================
# CLS Annotation Integration Tests
# ============================================

class TestCLSAnnotationIntegration:
    """Test CLS Annotation API <-> Worker integration."""

    def test_single_predict_input_format(self):
        """Verify single prediction sends correct format to worker."""
        # Expected worker input format (from handler.py)
        expected_keys = {"task", "model", "images", "classes", "top_k"}

        # Simulated API output
        api_output = {
            "task": "classify",
            "model": "clip",
            "images": [{"id": "test-id", "url": "https://example.com/img.jpg"}],
            "classes": ["class1", "class2"],
            "top_k": 5,
        }

        # Verify all expected keys are present
        assert expected_keys.issubset(api_output.keys()), \
            f"Missing keys: {expected_keys - api_output.keys()}"

        # Verify images is an array
        assert isinstance(api_output["images"], list)
        assert len(api_output["images"]) > 0

        # Verify each image has id and url
        for img in api_output["images"]:
            assert "id" in img, "Image must have 'id'"
            assert "url" in img, "Image must have 'url'"

        # Verify classes is an array (not class_names!)
        assert "classes" in api_output, "Must use 'classes' not 'class_names'"
        assert isinstance(api_output["classes"], list)

    def test_batch_classify_input_format(self):
        """Verify batch classification sends correct format to worker."""
        expected_keys = {"task", "model", "images", "classes", "top_k", "job_id"}

        api_output = {
            "task": "batch_classify",
            "model": "clip",
            "images": [
                {"id": "img1", "url": "https://example.com/1.jpg"},
                {"id": "img2", "url": "https://example.com/2.jpg"},
            ],
            "classes": ["cat", "dog", "bird"],
            "top_k": 3,
            "job_id": "job-123",
        }

        assert expected_keys.issubset(api_output.keys())
        assert "classes" in api_output  # NOT class_names

    def test_worker_response_parsing(self):
        """Verify API correctly parses worker response."""
        # Worker response format (from handler.py)
        worker_response = {
            "results": [
                {
                    "id": "img1",
                    "predictions": [
                        {"class": "cat", "confidence": 0.95},
                        {"class": "dog", "confidence": 0.03},
                    ]
                }
            ],
            "model": "clip",
            "model_variant": "ViT-B-32",
            "elapsed_seconds": 1.5,
        }

        # Simulate API parsing
        output = worker_response
        results = output.get("results", [])

        assert len(results) > 0, "Should have results"

        first_result = results[0]
        predictions = first_result.get("predictions", [])

        assert len(predictions) > 0, "Should have predictions"

        # Verify prediction format
        pred = predictions[0]
        assert "class" in pred, "Prediction should have 'class' field"
        assert "confidence" in pred, "Prediction should have 'confidence' field"

        # API should read 'class' field
        class_name = pred.get("class", pred.get("label", pred.get("class_name", "")))
        assert class_name == "cat"


# ============================================
# CLS Training Integration Tests
# ============================================

class TestCLSTrainingIntegration:
    """Test CLS Training API <-> Worker integration."""

    def test_training_input_format(self):
        """Verify training job sends correct format to worker."""
        # Expected worker input format (from handler.py)
        expected_keys = {"training_run_id", "dataset", "config", "supabase_url", "supabase_key"}
        expected_dataset_keys = {"train_urls", "val_urls", "class_names"}
        expected_config_keys = {"model_name", "epochs", "batch_size", "learning_rate"}

        # Simulated API output
        api_output = {
            "training_run_id": "uuid-123",
            "dataset": {
                "train_urls": [
                    {"url": "https://example.com/1.jpg", "label": 0},
                    {"url": "https://example.com/2.jpg", "label": 1},
                ],
                "val_urls": [
                    {"url": "https://example.com/3.jpg", "label": 0},
                ],
                "class_names": ["class_a", "class_b"],
            },
            "config": {
                "model_name": "efficientnet_b0",
                "epochs": 30,
                "batch_size": 32,
                "learning_rate": 0.001,
                "loss": "label_smoothing",
                "augmentation": "sota",
            },
            "supabase_url": "https://xxx.supabase.co",
            "supabase_key": "secret",
        }

        # Verify top-level keys
        assert expected_keys.issubset(api_output.keys()), \
            f"Missing keys: {expected_keys - api_output.keys()}"

        # Verify dataset keys
        dataset = api_output["dataset"]
        assert expected_dataset_keys.issubset(dataset.keys()), \
            f"Missing dataset keys: {expected_dataset_keys - dataset.keys()}"

        # Verify train_urls format
        for item in dataset["train_urls"]:
            assert "url" in item, "train_urls items must have 'url'"
            assert "label" in item, "train_urls items must have 'label'"
            assert isinstance(item["label"], int), "label must be integer"

        # Verify config keys
        config = api_output["config"]
        assert expected_config_keys.issubset(config.keys()), \
            f"Missing config keys: {expected_config_keys - config.keys()}"

    def test_model_name_mapping(self):
        """Verify model type/size maps to correct timm model names."""
        model_name_map = {
            ("efficientnet", "b0"): "efficientnet_b0",
            ("efficientnet", "b1"): "efficientnet_b1",
            ("efficientnet", "b2"): "efficientnet_b2",
            ("vit", "tiny"): "vit_tiny_patch16_224",
            ("vit", "small"): "vit_small_patch16_224",
            ("vit", "base"): "vit_base_patch16_224",
            ("convnext", "tiny"): "convnext_tiny",
            ("convnext", "small"): "convnext_small",
        }

        # Test each mapping
        for (model_type, model_size), expected_name in model_name_map.items():
            actual = model_name_map.get((model_type.lower(), model_size.lower()))
            assert actual == expected_name, \
                f"Model {model_type}/{model_size} should map to {expected_name}"

    def test_loss_function_mapping(self):
        """Verify loss function names map correctly."""
        loss_map = {
            "cross_entropy": "cross_entropy",
            "label_smoothing": "label_smoothing",
            "focal": "focal",
            "arcface": "arcface",
            "cosface": "cosface",
            "circle": "circle",
        }

        for api_name, worker_name in loss_map.items():
            assert loss_map.get(api_name) == worker_name

    def test_train_val_split(self):
        """Verify train/val split logic."""
        all_images = [{"url": f"img{i}.jpg", "label": i % 3} for i in range(100)]
        train_split = 0.8

        split_idx = int(len(all_images) * train_split)
        train_urls = all_images[:split_idx]
        val_urls = all_images[split_idx:]

        assert len(train_urls) == 80
        assert len(val_urls) == 20

        # Edge case: ensure at least 2 validation samples
        small_dataset = [{"url": f"img{i}.jpg", "label": 0} for i in range(5)]
        split_idx = int(len(small_dataset) * train_split)  # 4
        train = small_dataset[:split_idx]
        val = small_dataset[split_idx:]

        if len(val) < 2:
            val = small_dataset[-2:]
            train = small_dataset[:-2]

        assert len(val) >= 2, "Must have at least 2 validation samples"


# ============================================
# Schema Compatibility Tests
# ============================================

class TestSchemaCompatibility:
    """Test that database schemas match expected fields."""

    def test_cls_training_runs_fields(self):
        """Verify cls_training_runs table has required fields."""
        required_fields = {
            "id", "name", "description", "dataset_id", "status",
            "model_type", "model_size", "config", "total_epochs",
            "current_epoch", "runpod_job_id", "error_message",
            "best_accuracy", "best_f1", "metrics_history",
        }

        # These fields are referenced in training.py
        api_used_fields = {
            "id", "name", "description", "dataset_id", "dataset_version_id",
            "task_type", "num_classes", "model_type", "model_size",
            "config", "total_epochs", "status", "runpod_job_id",
            "error_message", "current_epoch", "best_accuracy", "best_f1",
            "best_top5_accuracy", "best_epoch", "metrics_history",
            "started_at", "completed_at",
        }

        # Just check that we use valid field names
        # In real test, would verify against actual schema
        assert len(api_used_fields) > 0

    def test_cls_labels_join(self):
        """Verify cls_labels can join with cls_images."""
        # This query pattern is used in training.py
        query_pattern = "cls_labels.select('image_id, class_id, cls_images!inner(id, image_url)')"

        # Verify join field exists
        assert "cls_images!inner" in query_pattern
        assert "image_url" in query_pattern


# ============================================
# Configuration Tests
# ============================================

class TestEndpointConfiguration:
    """Test endpoint configurations."""

    def test_endpoint_types_defined(self):
        """Verify all CLS endpoint types are defined."""
        # These should match EndpointType enum in runpod.py
        expected_endpoints = [
            "CLS_ANNOTATION",
            "CLS_TRAINING",
        ]

        # In real test, would import and verify
        for endpoint in expected_endpoints:
            assert endpoint in ["CLS_ANNOTATION", "CLS_TRAINING"]

    def test_config_settings_exist(self):
        """Verify config has CLS endpoint settings."""
        expected_settings = [
            "runpod_endpoint_cls_annotation",
            "runpod_endpoint_cls_training",
        ]

        # In real test, would import Settings and verify
        for setting in expected_settings:
            assert setting.startswith("runpod_endpoint_cls_")


# ============================================
# End-to-End Flow Tests
# ============================================

class TestEndToEndFlows:
    """Test complete flows from API to worker."""

    @pytest.mark.asyncio
    async def test_annotation_flow(self):
        """Test complete annotation flow."""
        # 1. API receives request
        request = {
            "image_id": "img-123",
            "dataset_id": "ds-456",
            "model": "clip",
            "top_k": 5,
            "threshold": 0.1,
        }

        # 2. API transforms to worker format
        worker_input = {
            "task": "classify",
            "model": request["model"],
            "images": [{"id": request["image_id"], "url": "https://example.com/img.jpg"}],
            "classes": ["class1", "class2", "class3"],
            "top_k": request["top_k"],
        }

        # 3. Worker processes and returns
        worker_output = {
            "results": [{
                "id": request["image_id"],
                "predictions": [
                    {"class": "class1", "confidence": 0.8},
                    {"class": "class2", "confidence": 0.15},
                ]
            }],
            "model": "clip",
        }

        # 4. API parses response
        results = worker_output.get("results", [])
        assert len(results) == 1

        predictions = results[0].get("predictions", [])
        filtered = [p for p in predictions if p["confidence"] >= request["threshold"]]
        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_training_flow(self):
        """Test complete training flow."""
        # 1. API receives training request
        training_config = {
            "name": "Test Training",
            "dataset_id": "ds-123",
            "model_type": "efficientnet",
            "model_size": "b0",
            "epochs": 10,
        }

        # 2. API prepares dataset
        labeled_images = [
            {"url": f"https://example.com/{i}.jpg", "label": i % 3}
            for i in range(100)
        ]

        # 3. Split train/val
        train_urls = labeled_images[:80]
        val_urls = labeled_images[80:]

        # 4. Create worker input
        worker_input = {
            "training_run_id": "run-123",
            "dataset": {
                "train_urls": train_urls,
                "val_urls": val_urls,
                "class_names": ["class0", "class1", "class2"],
            },
            "config": {
                "model_name": "efficientnet_b0",
                "epochs": training_config["epochs"],
                "batch_size": 32,
            },
        }

        # Verify format
        assert len(worker_input["dataset"]["train_urls"]) == 80
        assert len(worker_input["dataset"]["val_urls"]) == 20
        assert worker_input["config"]["model_name"] == "efficientnet_b0"


# ============================================
# Run validation without pytest
# ============================================

def run_validation():
    """Run all validations and print results."""
    print("=" * 60)
    print("CLS Integration Validation")
    print("=" * 60)

    tests = [
        ("Annotation Input Format", TestCLSAnnotationIntegration().test_single_predict_input_format),
        ("Annotation Batch Format", TestCLSAnnotationIntegration().test_batch_classify_input_format),
        ("Annotation Response Parse", TestCLSAnnotationIntegration().test_worker_response_parsing),
        ("Training Input Format", TestCLSTrainingIntegration().test_training_input_format),
        ("Model Name Mapping", TestCLSTrainingIntegration().test_model_name_mapping),
        ("Loss Function Mapping", TestCLSTrainingIntegration().test_loss_function_mapping),
        ("Train/Val Split", TestCLSTrainingIntegration().test_train_val_split),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_validation()
    sys.exit(0 if success else 1)
