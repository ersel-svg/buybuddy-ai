"""
RunPod Serverless Handler for Embedding Extraction.

SOTA Features (matching OD/CLS training pattern):
- Direct Supabase writes (no webhook dependency)
- Direct Qdrant upserts (worker handles storage)
- Progress tracking per batch
- Cancellation support (checks job status between batches)
- Graceful shutdown handling
- Retry logic for DB operations

Input (New Format - Phase 2):
{
    "job_id": "uuid",                    # Embedding job ID in Supabase
    "model_type": "dinov2-base",         # Model type
    "embedding_dim": 768,                # Expected dimension
    "collection_name": "prod_dinov2",    # Qdrant collection name
    "purpose": "matching",               # matching | evaluation | production
    "checkpoint_url": "https://...",     # Optional: For fine-tuned models

    # ALL images for this job
    "images": [
        {
            "id": "product_123_synthetic_0",
            "url": "https://storage.googleapis.com/...",
            "type": "product" | "cutout",
            "metadata": {...}
        },
        ...
    ],

    # Credentials for direct DB/vector store access
    "supabase_url": "https://...",
    "supabase_service_key": "eyJ...",
    "qdrant_url": "https://...",
    "qdrant_api_key": "...",

    # Optional: Collection metadata to create on completion
    "collection_metadata": {
        "collection_type": "production",
        "source_type": "all",
        "embedding_model_id": "model_uuid",
        "image_types": ["synthetic", "real"]
    }
}

Output:
{
    "status": "success" | "cancelled" | "error",
    "processed_count": 1000,
    "failed_count": 5,
    "total_images": 1005,
    "embedding_dim": 768,
}
"""

import os
import signal
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

import runpod


# ===========================================
# Graceful Shutdown Handling
# ===========================================
_shutdown_requested = False
_current_job_id = None
_supabase_client = None


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[SHUTDOWN] Signal {signum} received, initiating graceful shutdown...")

    # Update embedding job status if we have one
    if _current_job_id and _supabase_client:
        try:
            _supabase_client.table("embedding_jobs").update({
                "status": "cancelled",
                "error_message": "Job cancelled due to server shutdown",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", _current_job_id).execute()
            print(f"[SHUTDOWN] Embedding job {_current_job_id} marked as cancelled")
        except Exception as e:
            print(f"[SHUTDOWN] Failed to update job: {e}")


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


# Register signal handlers
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ===========================================
# Load environment variables from .env file
# ===========================================
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_paths = [
        Path("/workspace/.env"),
        Path(__file__).parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            for line in env_path.read_text().strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            break


# Load .env on module import
load_env_file()


# ===========================================
# Supabase Client
# ===========================================
def get_supabase_client(url: str, key: str):
    """Create Supabase client with provided credentials."""
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not url or not key:
        print("[WARNING] Supabase credentials not provided, progress updates disabled")
        return None

    from supabase import create_client
    _supabase_client = create_client(url, key)
    return _supabase_client


# ===========================================
# Qdrant Client
# ===========================================
def get_qdrant_client(url: str, api_key: str):
    """Create Qdrant client with provided credentials."""
    if not url or not api_key:
        print("[WARNING] Qdrant credentials not provided")
        return None

    from qdrant_client import QdrantClient
    return QdrantClient(url=url, api_key=api_key)


# ===========================================
# Update Embedding Job
# ===========================================
def update_embedding_job(
    client,
    job_id: str,
    status: str = None,
    processed_images: int = None,
    error_message: str = None,
    max_retries: int = 3,
):
    """
    Update embedding job directly in Supabase with retry logic.
    """
    import time

    if not client:
        print(f"[WARNING] Cannot update job: Supabase client not available")
        return False

    last_error = None
    for attempt in range(max_retries):
        try:
            # Build update data
            update_data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            if status:
                update_data["status"] = status

            if processed_images is not None:
                update_data["processed_images"] = processed_images

            if error_message:
                update_data["error_message"] = error_message

            if status == "running" and "started_at" not in update_data:
                # Check if started_at already set
                check_result = client.table("embedding_jobs").select("started_at").eq("id", job_id).single().execute()
                if check_result.data and not check_result.data.get("started_at"):
                    update_data["started_at"] = datetime.now(timezone.utc).isoformat()

            if status == "completed":
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Update job
            client.table("embedding_jobs").update(update_data).eq("id", job_id).execute()
            print(f"[Supabase] Job updated: status={status}, processed={processed_images}")

            return True

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[WARNING] Update failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Failed to update job after {max_retries} attempts: {e}")

    return False


def check_job_cancelled(client, job_id: str) -> bool:
    """Check if job has been cancelled by user."""
    if not client:
        return False

    try:
        result = client.table("embedding_jobs").select("status").eq("id", job_id).single().execute()
        if result.data and result.data.get("status") == "cancelled":
            return True
    except Exception as e:
        print(f"[WARNING] Failed to check job status: {e}")

    return False


# ===========================================
# Embedding Extractor
# ===========================================
from extractor import get_extractor

# Cache extractor instance
_extractor_cache: dict = {}


def get_cached_extractor(model_type: str, checkpoint_url: Optional[str] = None):
    """Get or create cached extractor instance."""
    cache_key = f"{model_type}:{checkpoint_url or 'default'}"

    if cache_key not in _extractor_cache:
        print(f"Loading model: {model_type}")
        _extractor_cache[cache_key] = get_extractor(
            model_type=model_type,
            checkpoint_url=checkpoint_url,
        )
        print(f"Model loaded! Embedding dim: {_extractor_cache[cache_key].embedding_dim}")

    return _extractor_cache[cache_key]


# ===========================================
# Main Handler
# ===========================================
def handler(job):
    """
    Main handler for RunPod serverless.

    Supports TWO modes:
    1. STATEFUL (New - Phase 2): job_id + credentials → Direct DB/Qdrant writes
    2. STATELESS (Legacy): images only → Return embeddings to API
    """
    global _current_job_id

    job_input = job.get("input", {}) if job else {}

    # Detect mode based on input
    has_job_id = "job_id" in job_input
    has_supabase = "supabase_url" in job_input and "supabase_service_key" in job_input
    has_qdrant = "qdrant_url" in job_input and "qdrant_api_key" in job_input

    if has_job_id and has_supabase and has_qdrant:
        # STATEFUL MODE - Phase 2
        return handler_stateful(job_input)
    else:
        # STATELESS MODE - Legacy (backward compatible)
        return handler_stateless(job_input)


def handler_stateful(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateful handler - Direct DB/Qdrant writes.

    Follows OD/CLS training pattern:
    - Direct Supabase writes for progress
    - Direct Qdrant upserts for embeddings
    - Cancellation support
    - Graceful shutdown
    """
    global _current_job_id

    # Extract parameters
    job_id = job_input.get("job_id")
    model_type = job_input.get("model_type", "dinov2-base")
    embedding_dim = job_input.get("embedding_dim", 768)
    collection_name = job_input.get("collection_name")
    purpose = job_input.get("purpose", "matching")
    checkpoint_url = job_input.get("checkpoint_url")
    images = job_input.get("images", [])

    # Credentials
    supabase_url = job_input.get("supabase_url")
    supabase_key = job_input.get("supabase_service_key")
    qdrant_url = job_input.get("qdrant_url")
    qdrant_api_key = job_input.get("qdrant_api_key")

    # Optional
    collection_metadata = job_input.get("collection_metadata")
    batch_size = job_input.get("batch_size", 50)
    hf_token = job_input.get("hf_token") or os.environ.get("HF_TOKEN")

    # Set HF token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Track current job for shutdown handling
    _current_job_id = job_id

    print(f"\n{'=' * 60}")
    print(f"EMBEDDING EXTRACTION (STATEFUL MODE)")
    print(f"  Job ID: {job_id}")
    print(f"  Images: {len(images)}")
    print(f"  Model: {model_type}")
    print(f"  Collection: {collection_name}")
    print(f"  Purpose: {purpose}")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    # Validate
    if not images:
        return {"status": "error", "error": "No images provided"}

    if not collection_name:
        return {"status": "error", "error": "collection_name is required"}

    total_images = len(images)
    processed_count = 0
    failed_count = 0

    try:
        # Initialize clients
        supabase = get_supabase_client(supabase_url, supabase_key)
        qdrant = get_qdrant_client(qdrant_url, qdrant_api_key)

        if not qdrant:
            raise ValueError("Failed to initialize Qdrant client")

        # Update job status to running
        update_embedding_job(supabase, job_id, status="running")

        # Check for early shutdown
        if is_shutdown_requested():
            return {"status": "cancelled", "error": "Server shutdown"}

        # Load extractor
        extractor = get_cached_extractor(model_type, checkpoint_url)

        # Process in batches
        for i in range(0, total_images, batch_size):
            # Check for cancellation
            if check_job_cancelled(supabase, job_id):
                print(f"[INFO] Job {job_id} was cancelled, stopping processing")
                return {
                    "status": "cancelled",
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "total_images": total_images,
                }

            # Check for shutdown
            if is_shutdown_requested():
                return {"status": "cancelled", "error": "Server shutdown"}

            batch = images[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_images + batch_size - 1) // batch_size

            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} images)")

            try:
                # Extract embeddings
                urls = [img["url"] for img in batch]
                batch_results = extractor.extract_batch_from_urls(urls, batch_size=batch_size)
                url_to_embedding = {url: emb for url, emb in batch_results}

                # Prepare Qdrant points
                from qdrant_client.models import PointStruct
                import uuid

                points = []
                for img in batch:
                    embedding = url_to_embedding.get(img["url"])

                    if embedding is not None:
                        # Create point with metadata as payload
                        point_id = img["id"]

                        # Use UUID if ID is not valid
                        try:
                            # Try to use as UUID
                            point_id_parsed = str(uuid.UUID(point_id)) if "-" in point_id else point_id
                        except:
                            # Generate UUID from string
                            point_id_parsed = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

                        payload = {
                            "id": img["id"],
                            "type": img.get("type", "unknown"),
                            "url": img["url"],
                            **(img.get("metadata", {})),
                        }

                        points.append(PointStruct(
                            id=point_id_parsed,
                            vector=embedding.tolist(),
                            payload=payload,
                        ))
                    else:
                        failed_count += 1
                        print(f"  Failed: {img['id']}")

                # Upsert to Qdrant
                if points:
                    qdrant.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=True,
                    )
                    processed_count += len(points)
                    print(f"  Upserted {len(points)} points to Qdrant")

                # Update cutout_images if applicable
                if supabase:
                    for img in batch:
                        if img.get("type") == "cutout" and img["url"] in url_to_embedding:
                            try:
                                supabase.table("cutout_images").update({
                                    "has_embedding": True,
                                    "embedding_model_id": collection_metadata.get("embedding_model_id") if collection_metadata else None,
                                    "qdrant_point_id": img["id"],
                                }).eq("id", img["id"]).execute()
                            except Exception as e:
                                print(f"  [WARNING] Failed to update cutout {img['id']}: {e}")

            except Exception as e:
                print(f"  Batch error: {e}")
                failed_count += len(batch)
                traceback.print_exc()

            # Update progress
            update_embedding_job(
                supabase,
                job_id,
                processed_images=processed_count + failed_count,
            )

        # Create collection metadata if provided
        if collection_metadata and supabase:
            try:
                # Check if collection already exists
                existing = supabase.table("embedding_collections").select("id").eq("name", collection_name).execute()

                if existing.data:
                    # Update existing
                    supabase.table("embedding_collections").update({
                        **collection_metadata,
                        "vector_count": processed_count,
                        "last_sync_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("name", collection_name).execute()
                else:
                    # Insert new
                    supabase.table("embedding_collections").insert({
                        "name": collection_name,
                        **collection_metadata,
                        "vector_count": processed_count,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }).execute()

                print(f"[INFO] Collection metadata saved for {collection_name}")
            except Exception as e:
                print(f"[WARNING] Failed to save collection metadata: {e}")

        # Update embedding model vector count
        if collection_metadata and collection_metadata.get("embedding_model_id") and supabase:
            try:
                # Get actual count from Qdrant
                collection_info = qdrant.get_collection(collection_name)
                vector_count = collection_info.points_count

                supabase.table("embedding_models").update({
                    "qdrant_vector_count": vector_count,
                }).eq("id", collection_metadata["embedding_model_id"]).execute()

                print(f"[INFO] Updated model vector count: {vector_count}")
            except Exception as e:
                print(f"[WARNING] Failed to update model vector count: {e}")

        # Mark job as completed
        update_embedding_job(
            supabase,
            job_id,
            status="completed",
            processed_images=processed_count,
        )

        print(f"\n{'=' * 60}")
        print(f"COMPLETED")
        print(f"  Processed: {processed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {total_images}")
        print(f"{'=' * 60}\n")

        return {
            "status": "success",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_images": total_images,
            "embedding_dim": extractor.embedding_dim,
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\nERROR: {error_msg}")
        print(error_trace)

        # Update job status to failed
        if supabase:
            update_embedding_job(
                supabase,
                job_id,
                status="failed",
                error_message=error_msg,
            )

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
            "processed_count": processed_count,
            "failed_count": failed_count,
        }

    finally:
        _current_job_id = None


def handler_stateless(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stateless handler - Legacy mode (backward compatible).

    Just extracts embeddings and returns them. API handles storage.
    """
    try:
        # Validate input
        images = job_input.get("images", [])

        if not images:
            return {
                "status": "error",
                "error": "No images provided. Expected: {\"images\": [{\"id\": \"...\", \"url\": \"...\"}]}"
            }

        model_type = job_input.get("model_type", "dinov2-base")
        checkpoint_url = job_input.get("checkpoint_url")
        batch_size = job_input.get("batch_size", 16)

        # Set HuggingFace token
        hf_token = job_input.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        print(f"\n{'=' * 60}")
        print(f"EMBEDDING EXTRACTION (STATELESS MODE)")
        print(f"  Images: {len(images)}")
        print(f"  Model: {model_type}")
        print(f"  Batch size: {batch_size}")
        print(f"{'=' * 60}\n")

        # Initialize extractor
        extractor = get_cached_extractor(model_type, checkpoint_url)

        # Extract embeddings
        results = []
        failed_ids = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            urls = [img["url"] for img in batch]

            print(f"Processing batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")

            try:
                batch_results = extractor.extract_batch_from_urls(urls, batch_size=batch_size)
                url_to_embedding = {url: emb for url, emb in batch_results}
            except Exception as e:
                print(f"Batch extraction error: {e}")
                url_to_embedding = {}

            for img in batch:
                embedding = url_to_embedding.get(img["url"])

                if embedding is not None:
                    result_item = {
                        "id": img["id"],
                        "type": img.get("type", "unknown"),
                        "domain": img.get("domain", "unknown"),
                        "vector": embedding.tolist(),
                    }
                    # Pass through additional metadata
                    for key in ["product_id", "frame_index", "category", "is_primary", "external_id", "barcode"]:
                        if key in img:
                            result_item[key] = img[key]

                    results.append(result_item)
                else:
                    failed_ids.append(img["id"])
                    print(f"  Failed: {img['id']}")

        print(f"\n{'=' * 60}")
        print(f"COMPLETED")
        print(f"  Processed: {len(results)}")
        print(f"  Failed: {len(failed_ids)}")
        print(f"{'=' * 60}\n")

        return {
            "status": "success",
            "embeddings": results,
            "processed_count": len(results),
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids,
            "embedding_dim": extractor.embedding_dim,
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\nERROR: {error_msg}")
        print(error_trace)

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }


if __name__ == "__main__":
    print("Starting Embedding Extraction Worker...")
    print(f"CUDA Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    runpod.serverless.start({"handler": handler})
