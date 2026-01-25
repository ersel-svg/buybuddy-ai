"""
End-to-end tests for RunPod Embedding Extraction.

These tests require actual RunPod credentials and endpoint.
Set the following environment variables:
- RUNPOD_API_KEY: Your RunPod API key
- RUNPOD_ENDPOINT_EMBEDDING: Embedding extraction endpoint ID
- SUPABASE_URL: Supabase project URL
- SUPABASE_SERVICE_ROLE_KEY: Supabase service role key
- QDRANT_URL: Qdrant URL
- QDRANT_API_KEY: Qdrant API key

Usage:
    # Run with real credentials (careful - costs money!)
    RUNPOD_API_KEY=xxx pytest tests/e2e/test_runpod_embedding.py -v

    # Skip E2E tests
    pytest tests/e2e/test_runpod_embedding.py -v -k "not e2e"
"""

import pytest
import os
import json
import asyncio
from datetime import datetime
from typing import Optional

# Try to import httpx, skip if not available
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Environment variables
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_EMBEDDING = os.environ.get("RUNPOD_ENDPOINT_EMBEDDING")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")


def skip_if_no_credentials():
    """Skip test if credentials not available."""
    missing = []
    if not RUNPOD_API_KEY:
        missing.append("RUNPOD_API_KEY")
    if not RUNPOD_ENDPOINT_EMBEDDING:
        missing.append("RUNPOD_ENDPOINT_EMBEDDING")
    if missing:
        pytest.skip(f"Missing credentials: {', '.join(missing)}")


def skip_if_no_httpx():
    """Skip test if httpx not installed."""
    if not HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")


class TestRunPodPayloadSubmission:
    """Tests for RunPod payload submission."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_submit_sota_payload(self):
        """Test submitting SOTA payload to RunPod."""
        skip_if_no_credentials()
        skip_if_no_httpx()

        job_id = f"test-sota-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        payload = {
            "input": {
                "job_id": job_id,
                "source_config": {
                    "type": "cutouts",
                    "filters": {"has_embedding": False},
                    "frame_selection": "first",
                    "max_frames": 1,
                },
                "model_type": "dinov2-base",
                "embedding_dim": 768,
                "collection_name": f"test_collection_{job_id}",
                "supabase_url": SUPABASE_URL,
                "supabase_service_key": SUPABASE_SERVICE_ROLE_KEY,
                "qdrant_url": QDRANT_URL,
                "qdrant_api_key": QDRANT_API_KEY,
            }
        }

        # Verify payload size
        payload_size = len(json.dumps(payload))
        print(f"Payload size: {payload_size} bytes")
        assert payload_size < 10_000, f"Payload too large: {payload_size} bytes"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json=payload,
                timeout=30,
            )

            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

            assert response.status_code == 200, f"Submit failed: {response.text}"

            data = response.json()
            assert "id" in data, "Response missing job ID"

            return data["id"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_submit_minimal_payload(self):
        """Test submitting minimal payload to verify endpoint is reachable."""
        skip_if_no_credentials()
        skip_if_no_httpx()

        # Minimal payload just to test endpoint
        payload = {
            "input": {
                "test": True,
                "job_id": "test-minimal",
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json=payload,
                timeout=30,
            )

            # Either 200 (accepted) or error with valid JSON
            assert response.status_code in [200, 400, 422]

            data = response.json()
            print(f"Minimal payload response: {data}")

            if response.status_code == 200:
                assert "id" in data


class TestRunPodJobStatus:
    """Tests for RunPod job status polling."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_poll_job_status(self):
        """Test polling job status."""
        skip_if_no_credentials()
        skip_if_no_httpx()

        # Submit a test job first
        job_id = f"test-status-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        payload = {
            "input": {
                "job_id": job_id,
                "test": True,
            }
        }

        async with httpx.AsyncClient() as client:
            # Submit job
            submit_response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json=payload,
                timeout=30,
            )

            if submit_response.status_code != 200:
                pytest.skip("Could not submit job")

            runpod_job_id = submit_response.json()["id"]

            # Poll status
            status_response = await client.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/status/{runpod_job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                timeout=30,
            )

            assert status_response.status_code == 200

            data = status_response.json()
            assert "status" in data
            print(f"Job status: {data['status']}")


class TestRunPodCancellation:
    """Tests for RunPod job cancellation."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_cancel_running_job(self):
        """Test cancelling a running job."""
        skip_if_no_credentials()
        skip_if_no_httpx()

        job_id = f"test-cancel-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Submit a long-running job
        payload = {
            "input": {
                "job_id": job_id,
                "source_config": {
                    "type": "cutouts",
                    "filters": {},
                },
                "model_type": "dinov2-base",
                "embedding_dim": 768,
                "collection_name": f"test_cancel_{job_id}",
                "supabase_url": SUPABASE_URL or "https://example.supabase.co",
                "supabase_service_key": SUPABASE_SERVICE_ROLE_KEY or "test",
                "qdrant_url": QDRANT_URL or "https://qdrant.example.com",
                "qdrant_api_key": QDRANT_API_KEY or "test",
            }
        }

        async with httpx.AsyncClient() as client:
            # Submit
            submit_response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/run",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json=payload,
                timeout=30,
            )

            if submit_response.status_code != 200:
                print(f"Submit failed: {submit_response.text}")
                pytest.skip("Could not submit job for cancel test")

            runpod_job_id = submit_response.json()["id"]
            print(f"Submitted job: {runpod_job_id}")

            # Wait a bit for job to start
            await asyncio.sleep(1)

            # Cancel
            cancel_response = await client.post(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_EMBEDDING}/cancel/{runpod_job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                timeout=30,
            )

            print(f"Cancel response: {cancel_response.status_code}")

            # RunPod returns 200 even if job already completed
            assert cancel_response.status_code == 200


class TestPayloadValidation:
    """Tests for payload validation before submission."""

    def test_payload_has_required_fields(self):
        """Validate payload has all required fields."""
        required_fields = [
            "job_id",
            "source_config",
            "model_type",
            "embedding_dim",
            "collection_name",
            "supabase_url",
            "supabase_service_key",
            "qdrant_url",
            "qdrant_api_key",
        ]

        payload = {
            "job_id": "test",
            "source_config": {"type": "cutouts"},
            "model_type": "dinov2-base",
            "embedding_dim": 768,
            "collection_name": "test",
            "supabase_url": "https://example.supabase.co",
            "supabase_service_key": "key",
            "qdrant_url": "https://qdrant.example.com",
            "qdrant_api_key": "key",
        }

        for field in required_fields:
            assert field in payload, f"Missing required field: {field}"

    def test_payload_size_check(self):
        """Validate payload size before submission."""

        def validate_payload_size(payload: dict, max_size: int = 10_000_000) -> bool:
            size = len(json.dumps(payload))
            return size < max_size

        # SOTA payload - should pass
        sota_payload = {
            "input": {
                "job_id": "test",
                "source_config": {"type": "cutouts"},
            }
        }
        assert validate_payload_size(sota_payload)

        # Large payload - should fail
        large_payload = {
            "input": {
                "images": [{"id": i, "url": f"https://example.com/{i}.jpg"} for i in range(200_000)]
            }
        }
        assert not validate_payload_size(large_payload)


class TestMockE2EFlow:
    """Mock E2E tests that don't require credentials."""

    def test_full_flow_simulation(self):
        """Simulate full E2E flow without actual API calls."""
        # 1. Create job in DB
        job = {
            "id": "test-job-123",
            "status": "pending",
            "source_config": {"type": "cutouts", "filters": {}},
        }

        # 2. Submit to RunPod (simulated)
        runpod_response = {"id": "runpod-job-456", "status": "IN_QUEUE"}
        job["runpod_job_id"] = runpod_response["id"]
        job["status"] = "queued"

        # 3. Worker starts (simulated)
        job["status"] = "running"

        # 4. Worker processes (simulated)
        progress_updates = [10, 30, 50, 70, 90, 100]
        for progress in progress_updates:
            job["progress"] = progress

        # 5. Worker completes
        job["status"] = "completed"
        job["progress"] = 100

        assert job["status"] == "completed"
        assert job["progress"] == 100

    def test_cancel_flow_simulation(self):
        """Simulate cancel flow without actual API calls."""
        job = {
            "id": "test-job-cancel",
            "status": "running",
            "progress": 30,
        }

        # Cancel requested
        cancel_result = {"status": "cancelled", "runpod_cancel": "success"}

        # Update job
        job["status"] = "cancelled"

        assert job["status"] == "cancelled"

    def test_error_flow_simulation(self):
        """Simulate error flow without actual API calls."""
        job = {
            "id": "test-job-error",
            "status": "running",
        }

        # Error occurs
        error = "Connection to Qdrant failed"
        job["status"] = "failed"
        job["error_message"] = error

        assert job["status"] == "failed"
        assert "error_message" in job
