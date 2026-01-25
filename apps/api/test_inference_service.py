"""
Unit tests for InferenceService.

Tests the InferenceService with mocked dependencies.
Run with: pytest test_inference_service.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
import io
import base64


# Mock imports
@pytest.fixture
def mock_runpod_service():
    """Mock RunPod service."""
    with patch("services.workflow.inference_service.runpod_service") as mock:
        yield mock


@pytest.fixture
def mock_model_loader():
    """Mock ModelLoader."""
    with patch("services.workflow.inference_service.get_model_loader") as mock:
        loader = MagicMock()
        mock.return_value = loader
        yield loader


@pytest.fixture
def inference_service(mock_model_loader):
    """Create InferenceService instance with mocked dependencies."""
    from services.workflow.inference_service import InferenceService
    return InferenceService()


@pytest.fixture
def test_image():
    """Create a test image."""
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    return img


class TestDetection:
    """Test detection inference."""

    @pytest.mark.asyncio
    async def test_detect_success(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test successful detection."""
        # Mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "YOLO11n"
        mock_model_info.model_type = "yolo11n"
        mock_model_info.checkpoint_url = None
        mock_model_info.class_mapping = {"0": "person", "1": "car"}
        mock_model_loader.get_detection_model_info = AsyncMock(return_value=mock_model_info)

        # Mock RunPod response
        mock_response = {
            "success": True,
            "result": {
                "detections": [
                    {
                        "id": 0,
                        "class_name": "person",
                        "class_id": 0,
                        "confidence": 0.95,
                        "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
                        "area": 0.02,
                    }
                ],
                "count": 1,
                "image_size": {"width": 640, "height": 480},
            },
            "metadata": {
                "inference_time_ms": 150,
                "cached": False,
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        # Run detection
        result = await inference_service.detect(
            model_id="test-model-id",
            image=test_image,
            confidence=0.5,
        )

        # Verify
        assert result["count"] == 1
        assert len(result["detections"]) == 1
        assert result["detections"][0]["class_name"] == "person"
        assert result["detections"][0]["confidence"] == 0.95

        # Verify model loader was called correctly
        mock_model_loader.get_detection_model_info.assert_called_once_with("test-model-id", "pretrained")

        # Verify RunPod was called
        assert mock_runpod_service.submit_job_sync.called

    @pytest.mark.asyncio
    async def test_detect_model_not_found(self, inference_service, mock_model_loader, test_image):
        """Test detection with model not found."""
        mock_model_loader.get_detection_model_info = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Detection model not found"):
            await inference_service.detect(
                model_id="nonexistent-model",
                image=test_image,
            )

    @pytest.mark.asyncio
    async def test_detect_inference_failed(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test detection inference failure."""
        # Mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "YOLO11n"
        mock_model_info.model_type = "yolo11n"
        mock_model_info.checkpoint_url = None
        mock_model_info.class_mapping = {}
        mock_model_loader.get_detection_model_info = AsyncMock(return_value=mock_model_info)

        # Mock RunPod error response
        mock_response = {
            "success": False,
            "error": "GPU out of memory",
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Detection inference failed"):
            await inference_service.detect(
                model_id="test-model-id",
                image=test_image,
            )

    @pytest.mark.asyncio
    async def test_detect_invalid_response(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test detection with invalid response format."""
        mock_model_info = MagicMock()
        mock_model_info.name = "YOLO11n"
        mock_model_info.model_type = "yolo11n"
        mock_model_info.checkpoint_url = None
        mock_model_info.class_mapping = {}
        mock_model_loader.get_detection_model_info = AsyncMock(return_value=mock_model_info)

        # Mock invalid response (missing 'detections' field)
        mock_response = {
            "success": True,
            "result": {
                "count": 0,  # Missing 'detections' field!
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Invalid response: missing 'detections' field"):
            await inference_service.detect(
                model_id="test-model-id",
                image=test_image,
            )


class TestClassification:
    """Test classification inference."""

    @pytest.mark.asyncio
    async def test_classify_success(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test successful classification."""
        # Mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "Product Classifier"
        mock_model_info.model_type = "vit-base"
        mock_model_info.checkpoint_url = "https://example.com/model.pt"
        mock_model_info.class_mapping = {"0": "cat", "1": "dog"}
        mock_model_info.config = {"num_classes": 2}
        mock_model_loader.get_classification_model_info = AsyncMock(return_value=mock_model_info)

        # Mock RunPod response
        mock_response = {
            "success": True,
            "result": {
                "predictions": [
                    {"class_name": "cat", "class_id": 0, "confidence": 0.98},
                    {"class_name": "dog", "class_id": 1, "confidence": 0.02},
                ],
                "top_class": "cat",
                "top_confidence": 0.98,
            },
            "metadata": {
                "inference_time_ms": 200,
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        # Run classification
        result = await inference_service.classify(
            model_id="test-model-id",
            image=test_image,
            top_k=5,
            model_source="trained",
        )

        # Verify
        assert result["top_class"] == "cat"
        assert result["top_confidence"] == 0.98
        assert len(result["predictions"]) == 2

    @pytest.mark.asyncio
    async def test_classify_invalid_response(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test classification with invalid response."""
        mock_model_info = MagicMock()
        mock_model_info.name = "Classifier"
        mock_model_info.model_type = "vit-base"
        mock_model_info.checkpoint_url = None
        mock_model_info.class_mapping = {}
        mock_model_info.config = {}
        mock_model_loader.get_classification_model_info = AsyncMock(return_value=mock_model_info)

        # Missing 'predictions' field
        mock_response = {
            "success": True,
            "result": {
                "top_class": "cat",
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Invalid response: missing 'predictions' field"):
            await inference_service.classify(
                model_id="test-model-id",
                image=test_image,
            )


class TestEmbedding:
    """Test embedding extraction."""

    @pytest.mark.asyncio
    async def test_embed_success(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test successful embedding extraction."""
        # Mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "DINOv2-base"
        mock_model_info.model_type = "dinov2-base"
        mock_model_info.checkpoint_url = None
        mock_model_info.embedding_dim = 768
        mock_model_loader.get_embedding_model_info = AsyncMock(return_value=mock_model_info)

        # Mock RunPod response
        mock_response = {
            "success": True,
            "result": {
                "embedding": [0.1] * 768,
                "embedding_dim": 768,
                "normalized": True,
            },
            "metadata": {
                "inference_time_ms": 300,
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        # Run embedding extraction
        result = await inference_service.embed(
            model_id="test-model-id",
            image=test_image,
            normalize=True,
        )

        # Verify
        assert result["embedding_dim"] == 768
        assert len(result["embedding"]) == 768
        assert result["normalized"] is True

    @pytest.mark.asyncio
    async def test_embed_invalid_response(self, inference_service, mock_model_loader, mock_runpod_service, test_image):
        """Test embedding with invalid response."""
        mock_model_info = MagicMock()
        mock_model_info.name = "DINOv2"
        mock_model_info.model_type = "dinov2-base"
        mock_model_info.checkpoint_url = None
        mock_model_info.embedding_dim = 768
        mock_model_loader.get_embedding_model_info = AsyncMock(return_value=mock_model_info)

        # Missing 'embedding' field
        mock_response = {
            "success": True,
            "result": {
                "embedding_dim": 768,
            }
        }
        mock_runpod_service.submit_job_sync = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Invalid response: missing 'embedding' field"):
            await inference_service.embed(
                model_id="test-model-id",
                image=test_image,
            )


class TestImageConversion:
    """Test image conversion utilities."""

    def test_image_to_base64_rgb(self, inference_service):
        """Test RGB image to base64."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        b64 = inference_service._image_to_base64(img)

        # Should be valid base64
        assert isinstance(b64, str)
        assert len(b64) > 0

        # Should be decodable
        img_bytes = base64.b64decode(b64)
        decoded_img = Image.open(io.BytesIO(img_bytes))
        assert decoded_img.mode == "RGB"
        assert decoded_img.size == (100, 100)

    def test_image_to_base64_rgba(self, inference_service):
        """Test RGBA image conversion (should convert to RGB)."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        b64 = inference_service._image_to_base64(img)

        # Decode and verify it's RGB
        img_bytes = base64.b64decode(b64)
        decoded_img = Image.open(io.BytesIO(img_bytes))
        assert decoded_img.mode == "RGB"

    def test_image_to_base64_grayscale(self, inference_service):
        """Test grayscale image conversion."""
        img = Image.new("L", (100, 100), color=128)
        b64 = inference_service._image_to_base64(img)

        # Should work
        assert isinstance(b64, str)
        assert len(b64) > 0


class TestBatchDetection:
    """Test batch detection."""

    @pytest.mark.asyncio
    async def test_batch_detect(self, inference_service, mock_model_loader, mock_runpod_service):
        """Test batch detection on multiple images."""
        # Create test images
        images = [
            Image.new("RGB", (100, 100), color=(255, 0, 0)),
            Image.new("RGB", (100, 100), color=(0, 255, 0)),
        ]

        # Mock model info
        mock_model_info = MagicMock()
        mock_model_info.name = "YOLO11n"
        mock_model_info.model_type = "yolo11n"
        mock_model_info.checkpoint_url = None
        mock_model_info.class_mapping = {}
        mock_model_loader.get_detection_model_info = AsyncMock(return_value=mock_model_info)

        # Mock RunPod responses
        mock_runpod_service.submit_job_sync = AsyncMock(return_value={
            "success": True,
            "result": {
                "detections": [],
                "count": 0,
                "image_size": {"width": 100, "height": 100},
            }
        })

        # Run batch detection
        results = await inference_service.batch_detect(
            model_id="test-model-id",
            images=images,
            confidence=0.5,
        )

        # Verify
        assert len(results) == 2
        assert all("detections" in r for r in results)

        # Should have called RunPod twice
        assert mock_runpod_service.submit_job_sync.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
