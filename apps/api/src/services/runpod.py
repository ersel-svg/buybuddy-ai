"""Runpod API Service for job orchestration."""

import httpx
from typing import Optional, Any
from enum import Enum

from config import settings


class EndpointType(str, Enum):
    """Runpod endpoint types."""
    VIDEO = "video"
    AUGMENTATION = "augmentation"
    TRAINING = "training"
    EMBEDDING = "embedding"


class RunpodService:
    """Service class for Runpod API operations."""

    BASE_URL = "https://api.runpod.ai/v2"

    def __init__(self) -> None:
        self.api_key = settings.runpod_api_key
        self.endpoints = {
            EndpointType.VIDEO: settings.runpod_endpoint_video,
            EndpointType.AUGMENTATION: settings.runpod_endpoint_augmentation,
            EndpointType.TRAINING: settings.runpod_endpoint_training,
            EndpointType.EMBEDDING: settings.runpod_endpoint_embedding,
        }

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_endpoint_id(self, endpoint_type: EndpointType) -> str:
        """Get endpoint ID for a given type."""
        endpoint_id = self.endpoints.get(endpoint_type)
        if not endpoint_id:
            raise ValueError(f"Endpoint not configured for type: {endpoint_type}")
        return endpoint_id

    async def submit_job(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Submit a job to Runpod endpoint.

        Args:
            endpoint_type: Type of worker endpoint
            input_data: Input data for the job
            webhook_url: Optional webhook URL for completion callback

        Returns:
            Runpod job response with id and status
        """
        endpoint_id = self._get_endpoint_id(endpoint_type)

        payload: dict[str, Any] = {"input": input_data}
        if webhook_url:
            payload["webhook"] = webhook_url

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/{endpoint_id}/run",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    async def submit_job_sync(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        timeout: int = 300,
    ) -> dict[str, Any]:
        """
        Submit a job and wait for completion (synchronous mode).

        Args:
            endpoint_type: Type of worker endpoint
            input_data: Input data for the job
            timeout: Maximum wait time in seconds (default 5 minutes)

        Returns:
            Job result including output
        """
        endpoint_id = self._get_endpoint_id(endpoint_type)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/{endpoint_id}/runsync",
                headers=self._get_headers(),
                json={"input": input_data},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()

    async def get_job_status(
        self,
        endpoint_type: EndpointType,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Get job status from Runpod.

        Args:
            endpoint_type: Type of worker endpoint
            job_id: Runpod job ID

        Returns:
            Job status response
        """
        endpoint_id = self._get_endpoint_id(endpoint_type)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint_id}/status/{job_id}",
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    async def cancel_job(
        self,
        endpoint_type: EndpointType,
        job_id: str,
    ) -> dict[str, Any]:
        """
        Cancel a running job on Runpod.

        Args:
            endpoint_type: Type of worker endpoint
            job_id: Runpod job ID

        Returns:
            Cancellation response
        """
        endpoint_id = self._get_endpoint_id(endpoint_type)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/{endpoint_id}/cancel/{job_id}",
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self, endpoint_type: EndpointType) -> dict[str, Any]:
        """
        Check endpoint health.

        Args:
            endpoint_type: Type of worker endpoint

        Returns:
            Health status response
        """
        endpoint_id = self._get_endpoint_id(endpoint_type)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint_id}/health",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            return response.json()

    def is_configured(self, endpoint_type: Optional[EndpointType] = None) -> bool:
        """
        Check if Runpod is configured.

        Args:
            endpoint_type: Optional specific endpoint to check

        Returns:
            True if configured, False otherwise
        """
        if not self.api_key:
            return False

        if endpoint_type:
            return bool(self.endpoints.get(endpoint_type))

        return any(self.endpoints.values())


# Singleton instance
runpod_service = RunpodService()
