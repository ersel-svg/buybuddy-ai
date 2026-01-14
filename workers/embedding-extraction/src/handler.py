"""
RunPod Serverless Handler for Embedding Extraction.

Simple stateless worker - extracts embeddings from image URLs and returns them.
No database connections - API handles storage.

Input:
{
    "images": [
        {"id": "uuid-1", "url": "https://...", "type": "cutout"},
        {"id": "uuid-2", "url": "https://...", "type": "product"},
    ],
    "model_type": "dinov2-base",  # Optional, default: dinov2-base
    "batch_size": 16,              # Optional, default: 16
}

Output:
{
    "status": "success",
    "embeddings": [
        {"id": "uuid-1", "type": "cutout", "vector": [0.1, 0.2, ...]},
        {"id": "uuid-2", "type": "product", "vector": [0.3, 0.4, ...]},
    ],
    "processed_count": 2,
    "failed_count": 0,
    "failed_ids": [],
    "embedding_dim": 768,
}
"""

import os
import runpod
import traceback
from typing import Optional

# Local extractor
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


def handler(job):
    """
    Main handler for RunPod serverless.

    Extracts embeddings from image URLs and returns them.
    No database connections - pure compute.
    """
    job_input = job.get("input", {}) if job else {}

    try:
        # ========================================
        # VALIDATE INPUT
        # ========================================
        images = job_input.get("images", [])

        if not images:
            return {
                "status": "error",
                "error": "No images provided. Expected: {\"images\": [{\"id\": \"...\", \"url\": \"...\"}]}"
            }

        model_type = job_input.get("model_type", "dinov2-base")
        checkpoint_url = job_input.get("checkpoint_url")
        batch_size = job_input.get("batch_size", 16)

        print(f"\n{'=' * 60}")
        print(f"EMBEDDING EXTRACTION")
        print(f"  Images: {len(images)}")
        print(f"  Model: {model_type}")
        print(f"  Batch size: {batch_size}")
        print(f"{'=' * 60}\n")

        # ========================================
        # INITIALIZE EXTRACTOR
        # ========================================
        extractor = get_cached_extractor(model_type, checkpoint_url)

        # ========================================
        # EXTRACT EMBEDDINGS
        # ========================================
        results = []
        failed_ids = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            urls = [img["url"] for img in batch]

            print(f"Processing batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")

            # Extract embeddings for batch
            try:
                batch_results = extractor.extract_batch_from_urls(urls, batch_size=batch_size)
                url_to_embedding = {url: emb for url, emb in batch_results}
            except Exception as e:
                print(f"Batch extraction error: {e}")
                url_to_embedding = {}

            # Map results back to images
            for img in batch:
                embedding = url_to_embedding.get(img["url"])

                if embedding is not None:
                    results.append({
                        "id": img["id"],
                        "type": img.get("type", "unknown"),
                        "vector": embedding.tolist(),
                    })
                else:
                    failed_ids.append(img["id"])
                    print(f"  Failed: {img['id']} - {img['url'][:50]}...")

        # ========================================
        # RETURN RESULTS
        # ========================================
        processed_count = len(results)
        failed_count = len(failed_ids)

        print(f"\n{'=' * 60}")
        print(f"COMPLETED")
        print(f"  Processed: {processed_count}")
        print(f"  Failed: {failed_count}")
        print(f"{'=' * 60}\n")

        return {
            "status": "success",
            "embeddings": results,
            "processed_count": processed_count,
            "failed_count": failed_count,
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
