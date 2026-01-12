"""Supabase client wrapper for database operations."""

from functools import lru_cache
from typing import Optional, Any
from datetime import datetime
import json

from supabase import create_client, Client

from config import settings


@lru_cache
def get_supabase_client() -> Client:
    """Get cached Supabase client."""
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise ValueError("Supabase URL and key must be configured")
    return create_client(settings.supabase_url, settings.supabase_anon_key)


class SupabaseService:
    """Service class for Supabase operations."""

    def __init__(self) -> None:
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    # =========================================
    # Products
    # =========================================

    async def get_products(
        self,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get products with pagination and filters."""
        query = self.client.table("products").select("*", count="exact")

        if search:
            query = query.or_(
                f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
            )
        if status:
            query = query.eq("status", status)
        if category:
            query = query.eq("category", category)

        # Pagination
        start = (page - 1) * limit
        end = start + limit - 1
        query = query.range(start, end).order("created_at", desc=True)

        response = query.execute()
        return {
            "items": response.data,
            "total": response.count or 0,
            "page": page,
            "limit": limit,
        }

    async def get_product(self, product_id: str) -> Optional[dict[str, Any]]:
        """Get single product by ID."""
        response = (
            self.client.table("products")
            .select("*")
            .eq("id", product_id)
            .single()
            .execute()
        )
        return response.data

    async def create_product(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new product."""
        response = self.client.table("products").insert(data).execute()
        return response.data[0]

    async def update_product(
        self, product_id: str, data: dict[str, Any], expected_version: Optional[int] = None
    ) -> dict[str, Any]:
        """Update product with optimistic locking."""
        query = self.client.table("products").update({
            **data,
            "updated_at": datetime.utcnow().isoformat(),
            "version": (expected_version or 0) + 1,
        }).eq("id", product_id)

        if expected_version is not None:
            query = query.eq("version", expected_version)

        response = query.execute()

        if not response.data:
            raise ValueError("Product was modified by another user")

        return response.data[0]

    async def delete_product(self, product_id: str) -> None:
        """Delete product."""
        self.client.table("products").delete().eq("id", product_id).execute()

    async def delete_products(self, product_ids: list[str]) -> int:
        """Bulk delete products."""
        response = (
            self.client.table("products")
            .delete()
            .in_("id", product_ids)
            .execute()
        )
        return len(response.data)

    async def get_product_categories(self) -> list[str]:
        """Get distinct product categories."""
        response = (
            self.client.table("products")
            .select("category")
            .not_.is_("category", "null")
            .execute()
        )
        categories = set(item["category"] for item in response.data if item["category"])
        return sorted(list(categories))

    # =========================================
    # Datasets
    # =========================================

    async def get_datasets(self) -> list[dict[str, Any]]:
        """Get all datasets."""
        response = (
            self.client.table("datasets")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def get_dataset(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get dataset with products."""
        # Get dataset
        dataset_response = (
            self.client.table("datasets")
            .select("*")
            .eq("id", dataset_id)
            .single()
            .execute()
        )

        if not dataset_response.data:
            return None

        dataset = dataset_response.data

        # Get products in dataset
        products_response = (
            self.client.table("dataset_products")
            .select("product_id, products(*)")
            .eq("dataset_id", dataset_id)
            .execute()
        )

        dataset["products"] = [
            item["products"] for item in products_response.data if item.get("products")
        ]

        return dataset

    async def create_dataset(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new dataset."""
        response = self.client.table("datasets").insert({
            "name": data["name"],
            "description": data.get("description"),
            "product_count": 0,
        }).execute()
        return response.data[0]

    async def update_dataset(
        self, dataset_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update dataset."""
        response = (
            self.client.table("datasets")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", dataset_id)
            .execute()
        )
        return response.data[0]

    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete dataset and its product associations."""
        # Delete associations first
        self.client.table("dataset_products").delete().eq("dataset_id", dataset_id).execute()
        # Delete dataset
        self.client.table("datasets").delete().eq("id", dataset_id).execute()

    async def add_products_to_dataset(
        self, dataset_id: str, product_ids: list[str]
    ) -> int:
        """Add products to dataset."""
        # Get existing products to avoid duplicates
        existing = (
            self.client.table("dataset_products")
            .select("product_id")
            .eq("dataset_id", dataset_id)
            .in_("product_id", product_ids)
            .execute()
        )
        existing_ids = {item["product_id"] for item in existing.data}

        # Insert only new products
        new_ids = [pid for pid in product_ids if pid not in existing_ids]
        if new_ids:
            records = [{"dataset_id": dataset_id, "product_id": pid} for pid in new_ids]
            self.client.table("dataset_products").insert(records).execute()

            # Update product count
            await self._update_dataset_product_count(dataset_id)

        return len(new_ids)

    async def remove_product_from_dataset(
        self, dataset_id: str, product_id: str
    ) -> None:
        """Remove product from dataset."""
        self.client.table("dataset_products").delete().eq(
            "dataset_id", dataset_id
        ).eq("product_id", product_id).execute()

        await self._update_dataset_product_count(dataset_id)

    async def _update_dataset_product_count(self, dataset_id: str) -> None:
        """Update the product count for a dataset."""
        count_response = (
            self.client.table("dataset_products")
            .select("*", count="exact")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        count = count_response.count or 0

        self.client.table("datasets").update({"product_count": count}).eq(
            "id", dataset_id
        ).execute()

    # =========================================
    # Jobs
    # =========================================

    async def get_jobs(self, job_type: Optional[str] = None) -> list[dict[str, Any]]:
        """Get jobs, optionally filtered by type."""
        query = self.client.table("jobs").select("*").order("created_at", desc=True)

        if job_type:
            query = query.eq("type", job_type)

        response = query.limit(100).execute()
        return response.data

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get single job."""
        response = (
            self.client.table("jobs").select("*").eq("id", job_id).single().execute()
        )
        return response.data

    async def create_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new job."""
        response = self.client.table("jobs").insert({
            "type": data["type"],
            "status": "pending",
            "progress": 0,
            "config": data.get("config", {}),
        }).execute()
        return response.data[0]

    async def update_job(self, job_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update job status."""
        response = (
            self.client.table("jobs")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", job_id)
            .execute()
        )
        return response.data[0]

    # =========================================
    # Training Jobs
    # =========================================

    async def get_training_jobs(self) -> list[dict[str, Any]]:
        """Get training jobs with dataset info."""
        response = (
            self.client.table("training_jobs")
            .select("*, datasets(name)")
            .order("created_at", desc=True)
            .execute()
        )

        # Flatten dataset name
        for job in response.data:
            if job.get("datasets"):
                job["dataset_name"] = job["datasets"]["name"]
                del job["datasets"]

        return response.data

    async def create_training_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create training job."""
        # Create base job
        job = await self.create_job({"type": "training", "config": data})

        # Create training-specific record
        training_job = {
            "id": job["id"],
            "dataset_id": data["dataset_id"],
            "epochs": data.get("epochs", 30),
            "batch_size": data.get("batch_size", 32),
            "learning_rate": data.get("learning_rate", 0.0001),
        }
        self.client.table("training_jobs").insert(training_job).execute()

        return {**job, **training_job}

    # =========================================
    # Videos
    # =========================================

    async def get_videos(self) -> list[dict[str, Any]]:
        """Get all videos."""
        response = (
            self.client.table("videos")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def sync_videos_from_buybuddy(self, videos: list[dict[str, Any]]) -> int:
        """Sync videos from Buybuddy API."""
        if not videos:
            return 0

        # Get existing barcodes
        barcodes = [v["barcode"] for v in videos]
        existing = (
            self.client.table("videos")
            .select("barcode")
            .in_("barcode", barcodes)
            .execute()
        )
        existing_barcodes = {item["barcode"] for item in existing.data}

        # Insert new videos
        new_videos = [v for v in videos if v["barcode"] not in existing_barcodes]
        if new_videos:
            self.client.table("videos").insert(new_videos).execute()

        return len(new_videos)

    # =========================================
    # Model Artifacts
    # =========================================

    async def get_models(self) -> list[dict[str, Any]]:
        """Get all model artifacts."""
        response = (
            self.client.table("model_artifacts")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def activate_model(self, model_id: str) -> dict[str, Any]:
        """Activate a model and deactivate others."""
        # Deactivate all models
        self.client.table("model_artifacts").update({"is_active": False}).execute()

        # Activate selected model
        response = (
            self.client.table("model_artifacts")
            .update({"is_active": True})
            .eq("id", model_id)
            .execute()
        )
        return response.data[0]

    # =========================================
    # Embedding Indexes
    # =========================================

    async def get_embedding_indexes(self) -> list[dict[str, Any]]:
        """Get all embedding indexes."""
        response = (
            self.client.table("embedding_indexes")
            .select("*, model_artifacts(name)")
            .order("created_at", desc=True)
            .execute()
        )

        # Flatten model name
        for idx in response.data:
            if idx.get("model_artifacts"):
                idx["model_name"] = idx["model_artifacts"]["name"]
                del idx["model_artifacts"]

        return response.data

    async def create_embedding_index(
        self, name: str, model_id: str
    ) -> dict[str, Any]:
        """Create new embedding index."""
        response = self.client.table("embedding_indexes").insert({
            "name": name,
            "model_artifact_id": model_id,
            "vector_count": 0,
            "index_path": f"/indexes/{name.lower().replace(' ', '_')}.faiss",
        }).execute()
        return response.data[0]

    # =========================================
    # Real Images (for matching)
    # =========================================

    async def get_real_images(self, product_id: str) -> list[dict[str, Any]]:
        """Get real images for a product."""
        try:
            response = (
                self.client.table("product_real_images")
                .select("*")
                .eq("product_id", product_id)
                .order("created_at", desc=True)
                .execute()
            )
            return response.data
        except Exception:
            # Table may not exist yet
            return []

    async def get_real_image_count(self, product_id: str) -> int:
        """Get count of real images for a product."""
        try:
            response = (
                self.client.table("product_real_images")
                .select("*", count="exact")
                .eq("product_id", product_id)
                .execute()
            )
            return response.count or 0
        except Exception:
            # Table may not exist yet
            return 0

    async def add_real_images(
        self, product_id: str, image_urls: list[str]
    ) -> int:
        """Add real images to a product."""
        if not image_urls:
            return 0

        records = [
            {"product_id": product_id, "image_url": url}
            for url in image_urls
        ]
        response = self.client.table("product_real_images").insert(records).execute()
        return len(response.data)

    async def remove_real_images(
        self, product_id: str, image_ids: list[str]
    ) -> None:
        """Remove real images from a product."""
        self.client.table("product_real_images").delete().eq(
            "product_id", product_id
        ).in_("id", image_ids).execute()

    # =========================================
    # Storage
    # =========================================

    async def upload_file(
        self, bucket: str, path: str, file_data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload file to Supabase Storage."""
        self.client.storage.from_(bucket).upload(
            path, file_data, {"content-type": content_type}
        )
        return self.get_public_url(bucket, path)

    def get_public_url(self, bucket: str, path: str) -> str:
        """Get public URL for a file."""
        return f"{settings.supabase_url}/storage/v1/object/public/{bucket}/{path}"

    async def delete_file(self, bucket: str, path: str) -> None:
        """Delete file from storage."""
        self.client.storage.from_(bucket).remove([path])


# Singleton instance
supabase_service = SupabaseService()
