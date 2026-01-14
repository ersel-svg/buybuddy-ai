"""
RunPod Serverless Handler for Embedding Extraction.

Input:
{
    "job_id": "uuid-...",           # Embedding job ID in database
    "model_id": "uuid-...",         # Embedding model ID
    "model_type": "dinov2-base",    # Model type
    "checkpoint_url": "...",        # Optional: custom model URL
    "embedding_dim": 768,           # Embedding dimension
    "qdrant_collection": "...",     # Qdrant collection name
    "job_type": "full",             # full or incremental
    "source": "cutouts",            # cutouts, products, or both
    "batch_size": 16,               # Batch size for processing
}

Output:
{
    "status": "success",
    "job_id": "uuid-...",
    "processed_count": 1234,
    "failed_count": 5,
    "qdrant_vectors": 1234,
}
"""

import os
import runpod
import traceback
import uuid
from typing import Optional
from datetime import datetime

# Supabase client
from supabase import create_client, Client

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Local extractor
from extractor import get_extractor

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_KEY", ""))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

# Singletons
supabase_client: Optional[Client] = None
qdrant_client: Optional[QdrantClient] = None


def get_supabase() -> Optional[Client]:
    """Get or create Supabase client."""
    global supabase_client
    if supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Supabase init error: {e}")
    return supabase_client


def get_qdrant() -> Optional[QdrantClient]:
    """Get or create Qdrant client."""
    global qdrant_client
    if qdrant_client is None and QDRANT_URL:
        try:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            )
        except Exception as e:
            print(f"Qdrant init error: {e}")
    return qdrant_client


def update_job_status(
    job_id: str,
    status: str,
    processed: int = 0,
    total: int = 0,
    error: Optional[str] = None,
):
    """Update embedding job status in database."""
    client = get_supabase()
    if not client:
        return

    try:
        update_data = {
            "status": status,
            "processed_images": processed,
        }
        if status == "running" and total > 0:
            update_data["total_images"] = total
            update_data["started_at"] = datetime.utcnow().isoformat()
        if status == "completed":
            update_data["completed_at"] = datetime.utcnow().isoformat()
        if error:
            update_data["error_message"] = error

        client.table("embedding_jobs").update(update_data).eq("id", job_id).execute()
        print(f"Job {job_id} status: {status} ({processed}/{total})")
    except Exception as e:
        print(f"Job status update error: {e}")


def ensure_qdrant_collection(collection_name: str, embedding_dim: int):
    """Ensure Qdrant collection exists with correct config."""
    client = get_qdrant()
    if not client:
        raise RuntimeError("Qdrant client not initialized")

    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} exists")
    except Exception:
        # Create new collection
        print(f"Creating collection {collection_name} (dim={embedding_dim})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )


def upsert_to_qdrant(
    collection_name: str,
    points: list[tuple[str, list[float], dict]],
):
    """
    Upsert points to Qdrant.

    Args:
        collection_name: Target collection
        points: List of (id, vector, payload) tuples
    """
    client = get_qdrant()
    if not client:
        raise RuntimeError("Qdrant client not initialized")

    qdrant_points = [
        qmodels.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        for point_id, vector, payload in points
    ]

    client.upsert(
        collection_name=collection_name,
        points=qdrant_points,
    )


def fetch_cutouts(
    job_type: str,
    model_id: str,
    limit: int = 10000,
) -> list[dict]:
    """Fetch cutout images from Supabase."""
    client = get_supabase()
    if not client:
        return []

    query = client.table("cutout_images").select("id, image_url, predicted_upc, has_embedding, qdrant_point_id")

    if job_type == "incremental":
        # Only cutouts without embeddings for this model
        query = query.eq("has_embedding", False)

    result = query.limit(limit).execute()
    return result.data if result.data else []


def update_cutout_embedding(cutout_id: str, model_id: str, point_id: str):
    """Update cutout record with embedding info."""
    client = get_supabase()
    if not client:
        return

    try:
        client.table("cutout_images").update({
            "has_embedding": True,
            "embedding_model_id": model_id,
            "qdrant_point_id": point_id,
        }).eq("id", cutout_id).execute()
    except Exception as e:
        print(f"Cutout update error: {e}")


def update_model_vector_count(model_id: str, collection_name: str):
    """Update model's Qdrant vector count."""
    client = get_supabase()
    qdrant = get_qdrant()
    if not client or not qdrant:
        return

    try:
        info = qdrant.get_collection(collection_name)
        count = info.points_count

        client.table("embedding_models").update({
            "qdrant_vector_count": count,
        }).eq("id", model_id).execute()

        print(f"Model {model_id} vector count: {count}")
    except Exception as e:
        print(f"Model update error: {e}")


def handler(job):
    """Main handler for RunPod serverless."""
    job_input = job.get("input", {}) if job else {}
    embedding_job_id = job_input.get("job_id")

    try:
        # ========================================
        # VALIDATE INPUT
        # ========================================
        if not embedding_job_id:
            return {"status": "error", "error": "job_id is required"}

        model_id = job_input.get("model_id")
        model_type = job_input.get("model_type", "dinov2-base")
        checkpoint_url = job_input.get("checkpoint_url")
        embedding_dim = job_input.get("embedding_dim", 768)
        qdrant_collection = job_input.get("qdrant_collection")
        job_type = job_input.get("job_type", "full")
        source = job_input.get("source", "cutouts")
        batch_size = job_input.get("batch_size", 16)

        if not qdrant_collection:
            return {"status": "error", "error": "qdrant_collection is required"}

        print(f"\n{'=' * 60}")
        print(f"EMBEDDING EXTRACTION JOB")
        print(f"  Job ID: {embedding_job_id}")
        print(f"  Model: {model_type} (dim={embedding_dim})")
        print(f"  Collection: {qdrant_collection}")
        print(f"  Type: {job_type}, Source: {source}")
        print(f"{'=' * 60}\n")

        # ========================================
        # INITIALIZE
        # ========================================
        update_job_status(embedding_job_id, "running")

        # Ensure Qdrant collection exists
        ensure_qdrant_collection(qdrant_collection, embedding_dim)

        # Initialize extractor
        extractor = get_extractor(
            model_type=model_type,
            checkpoint_url=checkpoint_url,
        )

        # ========================================
        # FETCH IMAGES
        # ========================================
        all_images = []

        if source in ["cutouts", "both"]:
            cutouts = fetch_cutouts(job_type, model_id)
            for c in cutouts:
                all_images.append({
                    "type": "cutout",
                    "id": c["id"],
                    "url": c["image_url"],
                    "metadata": {
                        "source": "cutout",
                        "cutout_id": c["id"],
                        "predicted_upc": c.get("predicted_upc"),
                    },
                })
            print(f"Fetched {len(cutouts)} cutouts")

        # TODO: Add product frames if source includes "products"

        total_images = len(all_images)
        update_job_status(embedding_job_id, "running", 0, total_images)

        if total_images == 0:
            update_job_status(embedding_job_id, "completed", 0, 0)
            return {
                "status": "success",
                "job_id": embedding_job_id,
                "processed_count": 0,
                "failed_count": 0,
                "message": "No images to process",
            }

        # ========================================
        # EXTRACT & UPLOAD
        # ========================================
        processed = 0
        failed = 0

        for i in range(0, total_images, batch_size):
            batch = all_images[i : i + batch_size]
            urls = [img["url"] for img in batch]

            # Extract embeddings
            results = extractor.extract_batch_from_urls(urls, batch_size=batch_size)

            # Map results back to images
            url_to_embedding = {url: emb for url, emb in results}

            # Prepare Qdrant points
            points = []
            for img in batch:
                embedding = url_to_embedding.get(img["url"])
                if embedding is None:
                    failed += 1
                    continue

                # Generate UUID for Qdrant point
                point_id = str(uuid.uuid4())

                points.append((
                    point_id,
                    embedding.tolist(),
                    img["metadata"],
                ))

                # Update cutout record
                if img["type"] == "cutout":
                    update_cutout_embedding(img["id"], model_id, point_id)

                processed += 1

            # Upsert batch to Qdrant
            if points:
                upsert_to_qdrant(qdrant_collection, points)

            # Update progress
            update_job_status(embedding_job_id, "running", processed, total_images)
            print(f"Progress: {processed}/{total_images} ({failed} failed)")

        # ========================================
        # FINALIZE
        # ========================================
        update_model_vector_count(model_id, qdrant_collection)
        update_job_status(embedding_job_id, "completed", processed, total_images)

        print(f"\n{'=' * 60}")
        print(f"COMPLETED: {processed} processed, {failed} failed")
        print(f"{'=' * 60}\n")

        return {
            "status": "success",
            "job_id": embedding_job_id,
            "processed_count": processed,
            "failed_count": failed,
            "total_images": total_images,
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\nERROR: {error_msg}")
        print(error_trace)

        if embedding_job_id:
            update_job_status(embedding_job_id, "failed", error=error_msg)

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
            "job_id": embedding_job_id,
        }


if __name__ == "__main__":
    print("Starting Embedding Extraction Worker...")
    print(f"CUDA Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    runpod.serverless.start({"handler": handler})
