"""Runpod API Service for job orchestration."""

import asyncio
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
    PREVIEW = "preview"  # Segmentation preview (single frame)
    OD_ANNOTATION = "od_annotation"  # OD AI annotation (Grounding DINO, SAM, Florence)
    OD_TRAINING = "od_training"  # OD model training (RF-DETR, RT-DETR, YOLO-NAS)


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
            EndpointType.PREVIEW: settings.runpod_endpoint_preview,
            EndpointType.OD_ANNOTATION: settings.runpod_endpoint_od_annotation,
            EndpointType.OD_TRAINING: settings.runpod_endpoint_od_training,
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

    async def submit_and_wait(
        self,
        endpoint_type: EndpointType,
        input_data: dict[str, Any],
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """
        Submit a job and poll until completion (async polling mode).

        More reliable than runsync for long-running jobs as it avoids
        RunPod's server-side timeout limitations.

        Args:
            endpoint_type: Type of worker endpoint
            input_data: Input data for the job
            timeout: Maximum wait time in seconds (default 5 minutes)
            poll_interval: Seconds between status checks (default 2s)

        Returns:
            Job result including output

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        # Submit the job
        submit_response = await self.submit_job(endpoint_type, input_data)
        job_id = submit_response.get("id")

        if not job_id:
            raise RuntimeError(f"Failed to submit job: {submit_response}")

        print(f"[RunPod] Job submitted: {job_id}")

        # Poll for completion
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            status_response = await self.get_job_status(endpoint_type, job_id)
            status = status_response.get("status")

            print(f"[RunPod] Job {job_id} status: {status} ({elapsed:.1f}s)")

            if status == "COMPLETED":
                return status_response

            if status == "FAILED":
                error = status_response.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")

            if status == "CANCELLED":
                raise RuntimeError("Job was cancelled")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

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

    async def start_training_job(
        self,
        training_run_id: str,
        model_type: str,
        config: dict[str, Any],
        training_images: Optional[dict[str, list[dict]]] = None,
        checkpoint_url: Optional[str] = None,
        start_epoch: int = 0,
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Start a training job on RunPod.

        Args:
            training_run_id: ID of the training run in database
            model_type: Model type (e.g., "dinov2-base")
            config: Training configuration dict
            training_images: Dict mapping product_id to list of images with URLs
            checkpoint_url: Optional URL to resume from checkpoint
            start_epoch: Epoch to start from (for resume)
            webhook_url: Optional webhook URL for completion callback

        Returns:
            RunPod job response with id and status
        """
        input_data = {
            "training_run_id": training_run_id,
            "model_type": model_type,
            "config": config,
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_service_role_key,
            "hf_token": settings.hf_token,
        }

        # Add training images if provided (new format with URLs)
        if training_images:
            input_data["training_images"] = training_images

        # Add resume parameters if provided
        if checkpoint_url:
            input_data["checkpoint_url"] = checkpoint_url
            input_data["start_epoch"] = start_epoch

        return await self.submit_job(
            endpoint_type=EndpointType.TRAINING,
            input_data=input_data,
            webhook_url=webhook_url,
        )

    async def start_embedding_job(
        self,
        job_id: str,
        model_type: str,
        product_ids: list[str],
        collection_name: str,
        purpose: str = "matching",
        frame_selection: str = "first",
        max_frames: int = 1,
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Start an embedding extraction job on RunPod.

        Args:
            job_id: ID of the embedding job in database
            model_type: Model type (e.g., "dinov2-base")
            product_ids: List of product IDs to extract embeddings for
            collection_name: Qdrant collection name
            purpose: Purpose of extraction (matching, training, evaluation)
            frame_selection: Frame selection strategy (first, all, key_frames, interval)
            max_frames: Maximum frames per product
            webhook_url: Optional webhook URL for completion callback

        Returns:
            RunPod job response with id and status
        """
        input_data = {
            "job_id": job_id,
            "model_type": model_type,
            "product_ids": product_ids,
            "collection_name": collection_name,
            "purpose": purpose,
            "frame_selection": frame_selection,
            "max_frames": max_frames,
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_service_role_key,
            "qdrant_url": settings.qdrant_url,
            "qdrant_api_key": settings.qdrant_api_key,
            "hf_token": settings.hf_token,
        }

        return await self.submit_job(
            endpoint_type=EndpointType.EMBEDDING,
            input_data=input_data,
            webhook_url=webhook_url,
        )

    async def start_evaluation_job(
        self,
        training_run_id: str,
        checkpoint_id: str,
        checkpoint_url: str,
        model_type: str,
        test_product_ids: list[str],
        webhook_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Start a model evaluation job on RunPod.

        Args:
            training_run_id: ID of the training run
            checkpoint_id: ID of the checkpoint to evaluate
            checkpoint_url: URL to the checkpoint file
            model_type: Model type
            test_product_ids: List of test product IDs
            webhook_url: Optional webhook URL

        Returns:
            RunPod job response with id and status
        """
        input_data = {
            "training_run_id": training_run_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_url": checkpoint_url,
            "model_type": model_type,
            "test_product_ids": test_product_ids,
            "mode": "evaluate",
            "supabase_url": settings.supabase_url,
            "supabase_key": settings.supabase_service_role_key,
            "hf_token": settings.hf_token,
        }

        return await self.submit_job(
            endpoint_type=EndpointType.TRAINING,
            input_data=input_data,
            webhook_url=webhook_url,
        )


# Singleton instance
runpod_service = RunpodService()
