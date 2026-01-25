"""
Pytest configuration and fixtures for BuyBuddy AI tests.
"""

import pytest
import os


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end (requires credentials)")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests by default unless --run-e2e is passed."""
    if not config.getoption("--run-e2e", default=False):
        skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run e2e tests (requires credentials)",
    )


@pytest.fixture
def mock_supabase_url():
    """Provide mock Supabase URL."""
    return os.environ.get("SUPABASE_URL", "https://example.supabase.co")


@pytest.fixture
def mock_qdrant_url():
    """Provide mock Qdrant URL."""
    return os.environ.get("QDRANT_URL", "https://qdrant.example.com")


@pytest.fixture
def sample_source_config():
    """Provide sample source_config for tests."""
    return {
        "type": "both",
        "filters": {
            "has_embedding": False,
            "product_source": "all",
        },
        "frame_selection": "first",
        "max_frames": 10,
    }


@pytest.fixture
def sample_job_input(sample_source_config, mock_supabase_url, mock_qdrant_url):
    """Provide sample job input for tests."""
    return {
        "job_id": "test-job-123",
        "source_config": sample_source_config,
        "model_type": "dinov2-base",
        "embedding_dim": 768,
        "collection_name": "test_collection",
        "cutout_collection": "test_cutouts",
        "supabase_url": mock_supabase_url,
        "supabase_service_key": "test-key",
        "qdrant_url": mock_qdrant_url,
        "qdrant_api_key": "test-key",
    }
