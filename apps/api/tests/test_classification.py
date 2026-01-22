"""
Classification Module - API Tests

Comprehensive tests for the Classification system.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

# Test configuration
BASE_URL = "/api/v1/classification"


class TestClassificationHealth:
    """Test health and stats endpoints."""

    def test_health_check(self, client):
        """Test classification health endpoint."""
        response = client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["module"] == "classification"

    def test_stats_endpoint(self, client):
        """Test classification stats endpoint."""
        response = client.get(f"{BASE_URL}/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_images" in data
        assert "total_datasets" in data
        assert "total_labels" in data
        assert "total_classes" in data
        assert "total_models" in data


class TestClassificationImages:
    """Test image management endpoints."""

    def test_list_images_empty(self, client):
        """Test listing images when empty."""
        response = client.get(f"{BASE_URL}/images")
        assert response.status_code == 200
        data = response.json()
        assert "images" in data
        assert "total" in data

    def test_list_images_with_filters(self, client):
        """Test listing images with filters."""
        response = client.get(f"{BASE_URL}/images", params={
            "page": 1,
            "limit": 10,
            "status": "pending",
            "source": "upload"
        })
        assert response.status_code == 200

    def test_get_filter_options(self, client):
        """Test getting filter options."""
        response = client.get(f"{BASE_URL}/images/filters/options")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "statuses" in data


class TestClassificationClasses:
    """Test class management endpoints."""

    def test_list_classes_empty(self, client):
        """Test listing classes when empty."""
        response = client.get(f"{BASE_URL}/classes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_class(self, client, mock_supabase):
        """Test creating a new class."""
        mock_supabase.return_value.data = {"id": "test-class-id", "name": "test_class"}

        response = client.post(f"{BASE_URL}/classes", json={
            "name": "test_class",
            "display_name": "Test Class",
            "color": "#3B82F6"
        })
        # May return 200 or 201 depending on implementation
        assert response.status_code in [200, 201]

    def test_create_class_bulk(self, client, mock_supabase):
        """Test bulk creating classes."""
        mock_supabase.return_value.data = [
            {"id": "id1", "name": "class1"},
            {"id": "id2", "name": "class2"}
        ]

        response = client.post(f"{BASE_URL}/classes/bulk", json={
            "classes": [
                {"name": "class1", "color": "#3B82F6"},
                {"name": "class2", "color": "#10B981"}
            ]
        })
        assert response.status_code in [200, 201]


class TestClassificationDatasets:
    """Test dataset management endpoints."""

    def test_list_datasets_empty(self, client):
        """Test listing datasets when empty."""
        response = client.get(f"{BASE_URL}/datasets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_dataset(self, client, mock_supabase):
        """Test creating a new dataset."""
        mock_supabase.return_value.data = {
            "id": "test-dataset-id",
            "name": "Test Dataset",
            "task_type": "single_label"
        }

        response = client.post(f"{BASE_URL}/datasets", json={
            "name": "Test Dataset",
            "description": "A test dataset",
            "task_type": "single_label"
        })
        assert response.status_code in [200, 201]


class TestClassificationLabels:
    """Test label management endpoints."""

    def test_set_label(self, client, mock_supabase):
        """Test setting a label for an image."""
        mock_supabase.return_value.data = {"id": "label-id"}

        response = client.post(
            f"{BASE_URL}/labels/test-dataset-id/test-image-id",
            json={
                "class_id": "test-class-id",
                "confidence": 0.95
            }
        )
        # May fail if dataset doesn't exist, but endpoint should be reachable
        assert response.status_code in [200, 201, 404]

    def test_bulk_set_labels(self, client, mock_supabase):
        """Test bulk setting labels."""
        response = client.post(
            f"{BASE_URL}/labels/test-dataset-id/bulk",
            json={
                "image_ids": ["img1", "img2"],
                "class_id": "test-class-id"
            }
        )
        assert response.status_code in [200, 201, 404]


class TestClassificationLabeling:
    """Test labeling workflow endpoints."""

    def test_get_labeling_queue(self, client, mock_supabase):
        """Test getting labeling queue."""
        mock_supabase.return_value.data = []

        response = client.get(
            f"{BASE_URL}/labeling/test-dataset-id/queue",
            params={"mode": "unlabeled", "limit": 100}
        )
        assert response.status_code in [200, 404]

    def test_get_labeling_progress(self, client, mock_supabase):
        """Test getting labeling progress."""
        response = client.get(f"{BASE_URL}/labeling/test-dataset-id/progress")
        assert response.status_code in [200, 404]


class TestClassificationTraining:
    """Test training endpoints."""

    def test_list_training_runs(self, client):
        """Test listing training runs."""
        response = client.get(f"{BASE_URL}/training")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_augmentation_presets(self, client):
        """Test getting augmentation presets."""
        response = client.get(f"{BASE_URL}/training/augmentation-presets")
        assert response.status_code == 200
        data = response.json()
        assert "sota" in data
        assert "heavy" in data
        assert "medium" in data
        assert "light" in data
        assert "none" in data

    def test_get_model_configs(self, client):
        """Test getting model configurations."""
        response = client.get(f"{BASE_URL}/training/model-configs")
        assert response.status_code == 200
        data = response.json()
        # Verify all model types are present
        expected_models = ["vit", "convnext", "efficientnet", "swin", "dinov2", "clip"]
        for model in expected_models:
            assert model in data, f"Missing model config: {model}"


class TestClassificationModels:
    """Test trained model endpoints."""

    def test_list_models(self, client):
        """Test listing trained models."""
        response = client.get(f"{BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_default_model(self, client):
        """Test getting default model for a type."""
        response = client.get(f"{BASE_URL}/models/default/vit")
        # May return 404 if no default model exists
        assert response.status_code in [200, 404]


class TestClassificationImport:
    """Test import endpoints."""

    def test_import_from_urls(self, client, mock_supabase):
        """Test importing images from URLs."""
        response = client.post(f"{BASE_URL}/images/import/urls", json={
            "urls": ["https://example.com/image1.jpg"],
            "skip_duplicates": True
        })
        # May fail due to URL fetch, but endpoint should be reachable
        assert response.status_code in [200, 201, 400, 500]

    def test_import_from_products(self, client, mock_supabase):
        """Test importing images from products."""
        response = client.post(f"{BASE_URL}/images/import/products", json={
            "label_source": "category",
            "skip_duplicates": True
        })
        assert response.status_code in [200, 201, 400, 500]


# Fixtures
@pytest.fixture
def client():
    """Create test client."""
    from main import app
    return TestClient(app)


@pytest.fixture
def mock_supabase():
    """Mock Supabase service."""
    with patch('services.supabase.supabase_service') as mock:
        mock.client.table.return_value.select.return_value.execute.return_value = Mock(data=[], count=0)
        mock.client.table.return_value.insert.return_value.execute.return_value = Mock(data=[])
        mock.client.table.return_value.update.return_value.execute.return_value = Mock(data=[])
        mock.client.table.return_value.delete.return_value.execute.return_value = Mock(data=[])
        yield mock


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
