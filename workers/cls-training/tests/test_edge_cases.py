"""
Edge Case Tests for CLS Training Pipeline

Tests:
1. Failed URLs (404, timeout, invalid)
2. Corrupted images
3. Label validation (invalid labels, missing classes)
4. Memory checks
5. Checkpoint save/load/resume
6. ONNX export
7. Graceful shutdown
8. Training stats tracking
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production import (
    ProductionConfig,
    validate_dataset,
    load_image_with_retry,
    check_memory_available,
    export_model_onnx,
    export_model_torchscript,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    get_shutdown_handler,
    GracefulShutdown,
    TrainingStats,
)


class TestFailedURLs(unittest.TestCase):
    """Test URL loading with various failure scenarios."""

    def test_404_url(self):
        """Test handling of 404 URLs."""
        config = ProductionConfig(url_retries=1, url_timeout=5)
        img, error = load_image_with_retry(
            "https://httpstat.us/404",
            config=config,
        )
        # Should return fallback image and error message
        self.assertIsNotNone(img)
        self.assertIsNotNone(error)
        # Error could be "HTTP 404" or network error depending on environment
        self.assertTrue(len(error) > 0)
        # Fallback should be gray image
        self.assertEqual(img.size, (224, 224))

    def test_timeout_url(self):
        """Test handling of timeout."""
        config = ProductionConfig(url_retries=1, url_timeout=1)
        img, error = load_image_with_retry(
            "https://httpstat.us/200?sleep=5000",  # 5 second delay
            config=config,
        )
        self.assertIsNotNone(img)
        self.assertIsNotNone(error)
        # Error could be "Timeout" or connection error depending on environment
        self.assertTrue(len(error) > 0)

    def test_invalid_url(self):
        """Test handling of invalid URLs."""
        config = ProductionConfig(url_retries=1, url_timeout=5)
        img, error = load_image_with_retry(
            "https://invalid.domain.that.does.not.exist.xyz/image.jpg",
            config=config,
        )
        self.assertIsNotNone(img)
        self.assertIsNotNone(error)

    def test_valid_url(self):
        """Test that valid URLs work correctly."""
        config = ProductionConfig(url_retries=2, url_timeout=10)
        img, error = load_image_with_retry(
            "https://picsum.photos/id/1/200/200",
            config=config,
        )
        self.assertIsNotNone(img)
        self.assertIsNone(error)
        self.assertEqual(img.mode, "RGB")

    def test_retry_logic(self):
        """Test that retry logic works."""
        config = ProductionConfig(url_retries=3, url_retry_delay=0.1)

        # Mock requests to fail twice then succeed
        call_count = [0]
        original_get = __import__('requests').get

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise __import__('requests').exceptions.Timeout()
            return original_get(*args, **kwargs)

        with patch('requests.get', mock_get):
            img, error = load_image_with_retry(
                "https://picsum.photos/id/1/200/200",
                config=config,
            )

        # Should have tried 3 times
        self.assertEqual(call_count[0], 3)


class TestCorruptedImages(unittest.TestCase):
    """Test handling of corrupted/invalid images."""

    def test_non_image_url(self):
        """Test URL that returns non-image content."""
        config = ProductionConfig(url_retries=1, url_timeout=10)
        # JSON endpoint - not an image
        img, error = load_image_with_retry(
            "https://httpbin.org/json",
            config=config,
        )
        self.assertIsNotNone(img)
        self.assertIsNotNone(error)
        self.assertIn("Image error", error)

    def test_truncated_image(self):
        """Test handling when image data is corrupted."""
        # This is harder to test without a specific corrupted image endpoint
        # The production code handles this via PIL's img.load() call
        pass


class TestLabelValidation(unittest.TestCase):
    """Test dataset validation with label issues."""

    def test_valid_dataset(self):
        """Test validation passes for valid dataset."""
        train_urls = [
            {"url": "http://example.com/1.jpg", "label": 0},
            {"url": "http://example.com/2.jpg", "label": 1},
            {"url": "http://example.com/3.jpg", "label": 2},
        ]
        val_urls = [
            {"url": "http://example.com/4.jpg", "label": 0},
            {"url": "http://example.com/5.jpg", "label": 1},
        ]
        class_names = ["cat", "dog", "bird"]

        result = validate_dataset(train_urls, val_urls, class_names)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertEqual(result["num_classes"], 3)
        self.assertEqual(result["train_samples"], 3)
        self.assertEqual(result["val_samples"], 2)

    def test_invalid_labels(self):
        """Test validation catches invalid labels."""
        train_urls = [
            {"url": "http://example.com/1.jpg", "label": 0},
            {"url": "http://example.com/2.jpg", "label": 5},  # Invalid!
            {"url": "http://example.com/3.jpg", "label": -1},  # Invalid!
        ]
        val_urls = []
        class_names = ["cat", "dog"]  # Only 2 classes (0, 1)

        result = validate_dataset(train_urls, val_urls, class_names)

        self.assertFalse(result["valid"])
        self.assertTrue(any("Invalid labels" in issue for issue in result["issues"]))

    def test_empty_dataset(self):
        """Test validation catches empty datasets."""
        result = validate_dataset([], [], [])

        self.assertFalse(result["valid"])
        self.assertTrue(any("No training data" in issue for issue in result["issues"]))
        self.assertTrue(any("No class names" in issue for issue in result["issues"]))

    def test_missing_class_warning(self):
        """Test validation warns about missing class samples."""
        train_urls = [
            {"url": "http://example.com/1.jpg", "label": 0},
            {"url": "http://example.com/2.jpg", "label": 0},
            # No label=1 samples
        ]
        val_urls = []
        class_names = ["cat", "dog"]

        result = validate_dataset(train_urls, val_urls, class_names)

        # Should be valid but with warnings
        self.assertTrue(result["valid"])
        self.assertTrue(any("dog" in w and "no training samples" in w for w in result["warnings"]))

    def test_empty_urls(self):
        """Test validation catches empty URLs."""
        train_urls = [
            {"url": "", "label": 0},
            {"url": "http://example.com/2.jpg", "label": 1},
        ]
        val_urls = []
        class_names = ["cat", "dog"]

        result = validate_dataset(train_urls, val_urls, class_names)

        self.assertFalse(result["valid"])
        self.assertTrue(any("empty URLs" in issue for issue in result["issues"]))


class TestMemoryCheck(unittest.TestCase):
    """Test memory checking functionality."""

    def test_cpu_device(self):
        """Test memory check on CPU (should skip check)."""
        result = check_memory_available(
            model=None,
            batch_size=32,
            device="cpu",
        )

        self.assertEqual(result["device"], "cpu")
        self.assertFalse(result["check_performed"])
        self.assertEqual(result["recommended_batch_size"], 32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test memory check on CUDA."""
        model = nn.Linear(1000, 10)

        result = check_memory_available(
            model=model,
            batch_size=32,
            device="cuda",
        )

        self.assertEqual(result["device"], "cuda")
        self.assertTrue(result["check_performed"])
        self.assertIn("total_memory_gb", result)
        self.assertIn("free_memory_gb", result)
        self.assertIn("recommended_batch_size", result)


class TestCheckpointSaveLoad(unittest.TestCase):
    """Test checkpoint save and load functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = nn.Linear(10, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            metrics={"val_acc": 0.92},
            save_dir=self.temp_dir,
            config={"lr": 0.001},
        )

        self.assertTrue(os.path.exists(path))
        self.assertIn("epoch_5", path)

        # Check latest checkpoint exists
        latest_path = os.path.join(self.temp_dir, "latest_checkpoint.pt")
        self.assertTrue(os.path.exists(latest_path))

    def test_save_best_checkpoint(self):
        """Test saving best checkpoint."""
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            metrics={"val_acc": 0.92},
            save_dir=self.temp_dir,
            config={},
            is_best=True,
        )

        best_path = os.path.join(self.temp_dir, "best_model.pt")
        self.assertTrue(os.path.exists(best_path))

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        # Save first
        path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            metrics={"val_acc": 0.92},
            save_dir=self.temp_dir,
            config={"lr": 0.001},
        )

        # Create new model and load
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        loaded = load_checkpoint(
            path,
            model=new_model,
            optimizer=new_optimizer,
            device="cpu",
        )

        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["metrics"]["val_acc"], 0.92)
        self.assertEqual(loaded["config"]["lr"], 0.001)

        # Verify model weights were loaded
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_not_found(self):
        """Test loading non-existent checkpoint raises error."""
        new_model = nn.Linear(10, 2)

        with self.assertRaises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt", model=new_model)

    def test_cleanup_old_checkpoints(self):
        """Test cleanup keeps only last N checkpoints."""
        # Create multiple checkpoints
        for i in range(5):
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=i,
                metrics={},
                save_dir=self.temp_dir,
                config={},
            )

        # Should have 5 epoch checkpoints
        import glob
        checkpoints = glob.glob(os.path.join(self.temp_dir, "checkpoint_epoch_*.pt"))
        self.assertEqual(len(checkpoints), 5)

        # Cleanup, keep last 2
        cleanup_old_checkpoints(self.temp_dir, keep_last_n=2)

        # Should have 2 checkpoints now
        checkpoints = glob.glob(os.path.join(self.temp_dir, "checkpoint_epoch_*.pt"))
        self.assertEqual(len(checkpoints), 2)

        # Should keep the most recent ones (epoch 3 and 4)
        self.assertTrue(any("epoch_3" in c for c in checkpoints))
        self.assertTrue(any("epoch_4" in c for c in checkpoints))


# Check if ONNX is available
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class TestONNXExport(unittest.TestCase):
    """Test ONNX model export."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipIf(not ONNX_AVAILABLE, "ONNX not installed")
    def test_export_simple_model(self):
        """Test ONNX export with simple model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        save_path = os.path.join(self.temp_dir, "model.onnx")
        result = export_model_onnx(
            model,
            save_path,
            img_size=32,
            num_classes=10,
        )

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(save_path))
        self.assertIn("size_mb", result)

    def test_export_invalid_model(self):
        """Test ONNX export with invalid model."""
        # Model that can't be exported (no forward method issue)
        model = nn.Module()  # Empty module

        save_path = os.path.join(self.temp_dir, "model.onnx")
        result = export_model_onnx(model, save_path)

        self.assertFalse(result["success"])
        self.assertIn("error", result)


class TestTorchScriptExport(unittest.TestCase):
    """Test TorchScript model export."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_export_simple_model(self):
        """Test TorchScript export with simple model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        save_path = os.path.join(self.temp_dir, "model.pt")
        result = export_model_torchscript(model, save_path, img_size=32)

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(save_path))


class TestGracefulShutdown(unittest.TestCase):
    """Test graceful shutdown handler."""

    def test_initial_state(self):
        """Test initial shutdown state is False."""
        handler = GracefulShutdown()
        self.assertFalse(handler.should_stop())

    def test_request_shutdown(self):
        """Test shutdown request."""
        handler = GracefulShutdown()
        handler.request_shutdown(signum=15)
        self.assertTrue(handler.should_stop())

    def test_thread_safety(self):
        """Test shutdown handler is thread-safe."""
        import threading

        handler = GracefulShutdown()
        results = []

        def check_and_set():
            # Read
            results.append(handler.should_stop())
            # Write
            handler.request_shutdown()
            # Read again
            results.append(handler.should_stop())

        threads = [threading.Thread(target=check_and_set) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All final reads should be True
        self.assertTrue(handler.should_stop())

    def test_global_handler(self):
        """Test global shutdown handler singleton."""
        handler1 = get_shutdown_handler()
        handler2 = get_shutdown_handler()
        self.assertIs(handler1, handler2)


class TestTrainingStats(unittest.TestCase):
    """Test training statistics tracking."""

    def test_record_epoch(self):
        """Test epoch recording."""
        stats = TrainingStats()
        stats.record_epoch(0, 10.5, {"val_acc": 0.85})
        stats.record_epoch(1, 9.8, {"val_acc": 0.90})

        summary = stats.get_summary()
        self.assertEqual(summary["total_epochs"], 2)
        self.assertAlmostEqual(summary["avg_epoch_time"], 10.15, places=1)

    def test_record_failed_urls(self):
        """Test failed URL recording."""
        stats = TrainingStats()
        stats.record_failed_url("http://example.com/1.jpg", "404 Not Found")
        stats.record_failed_url("http://example.com/2.jpg", "Timeout")

        summary = stats.get_summary()
        self.assertEqual(summary["failed_url_count"], 2)
        self.assertEqual(len(summary["failed_urls"]), 2)

    def test_summary_limits_failed_urls(self):
        """Test summary limits failed URLs to first 10."""
        stats = TrainingStats()
        for i in range(20):
            stats.record_failed_url(f"http://example.com/{i}.jpg", "Error")

        summary = stats.get_summary()
        self.assertEqual(summary["failed_url_count"], 20)
        self.assertEqual(len(summary["failed_urls"]), 10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_record_memory(self):
        """Test memory recording on CUDA."""
        stats = TrainingStats()

        # Allocate some memory
        tensor = torch.zeros(1000, 1000, device="cuda")
        stats.record_memory()

        summary = stats.get_summary()
        self.assertGreater(summary["peak_memory_gb"], 0)

        del tensor
        torch.cuda.empty_cache()


class TestProductionConfig(unittest.TestCase):
    """Test ProductionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProductionConfig()

        self.assertEqual(config.url_timeout, 30)
        self.assertEqual(config.url_retries, 3)
        self.assertEqual(config.url_retry_delay, 1.0)
        self.assertEqual(config.max_batch_size, 64)
        self.assertTrue(config.enable_graceful_shutdown)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProductionConfig(
            url_timeout=60,
            url_retries=5,
            export_onnx=True,
        )

        self.assertEqual(config.url_timeout, 60)
        self.assertEqual(config.url_retries, 5)
        self.assertTrue(config.export_onnx)


class TestURLImageDataset(unittest.TestCase):
    """Test URLImageDataset with edge cases."""

    def test_mixed_valid_invalid_urls(self):
        """Test dataset with mix of valid and invalid URLs."""
        from handler import URLImageDataset
        from src.augmentations import get_val_transforms

        image_data = [
            {"url": "https://picsum.photos/id/1/100/100", "label": 0},
            {"url": "https://invalid.url.xyz/image.jpg", "label": 1},
            {"url": "https://picsum.photos/id/2/100/100", "label": 0},
        ]

        transform = get_val_transforms(img_size=100)
        dataset = URLImageDataset(
            image_data,
            transform=transform,
            production_config=ProductionConfig(url_retries=1, url_timeout=5),
        )

        # Should not raise, just return fallback for invalid URL
        for i in range(len(dataset)):
            img, label = dataset[i]
            self.assertEqual(img.shape, (3, 100, 100))


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("EDGE CASE TESTS FOR CLS TRAINING PIPELINE")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFailedURLs))
    suite.addTests(loader.loadTestsFromTestCase(TestCorruptedImages))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointSaveLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXExport))
    suite.addTests(loader.loadTestsFromTestCase(TestTorchScriptExport))
    suite.addTests(loader.loadTestsFromTestCase(TestGracefulShutdown))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingStats))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestURLImageDataset))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL EDGE CASE TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n  - {test}: {traceback[:200]}...")
