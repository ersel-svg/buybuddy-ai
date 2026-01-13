"""
Runpod Serverless Handler for Video Segmentation

Input:
{
    "video_url": "https://...",
    "barcode": "123456789",
    "video_id": 12345,
    "product_id": "uuid-...",  # Our system's product UUID
}

Output (returned to RunPod, sent via webhook automatically):
{
    "status": "success",
    "barcode": "123456789",
    "product_id": "uuid-...",
    "metadata": {...},
    "frame_count": 177,
    "frames_url": "https://..."
}

Note: RunPod serverless automatically sends webhook on job completion.
No manual callbacks needed - RunPod handles this with the webhook_url
specified when submitting the job.
"""

import runpod
import traceback
import os
from pipeline import ProductPipeline

# Pipeline singleton - loaded once on cold start
pipeline = None


def get_pipeline():
    """Get or create pipeline singleton."""
    global pipeline
    if pipeline is None:
        print("=" * 60)
        print("COLD START - Loading pipeline...")
        print("=" * 60)
        pipeline = ProductPipeline()
        print("Pipeline ready!")
        print("=" * 60)
    return pipeline


def handler(job):
    """Main handler for Runpod serverless."""
    job_input = job.get("input", {}) if job else {}
    product_id = None  # Define early so it's available in except block

    try:
        # Validate input
        video_url = job_input.get("video_url")
        if not video_url:
            return {"status": "error", "error": "video_url is required"}

        barcode = job_input.get("barcode", "unknown")
        video_id = job_input.get("video_id")
        product_id = job_input.get("product_id")

        # Validate product_id is provided (required for storage)
        if not product_id:
            return {"status": "error", "error": "product_id is required. Create product before processing."}

        print(f"\n{'=' * 60}")
        print(f"Processing: {barcode}")
        print(f"Video URL: {video_url[:80]}...")
        print(f"Product ID: {product_id}")
        print(f"{'=' * 60}\n")

        # Get pipeline and process
        pipe = get_pipeline()
        result = pipe.process(
            video_url=video_url,
            barcode=barcode,
            video_id=video_id,
            product_id=product_id,
        )

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: {barcode}")
        print(f"Frames: {result['frame_count']}")
        print(f"{'=' * 60}\n")

        # Return result - RunPod will send this via webhook automatically
        return {
            "status": "success",
            "barcode": barcode,
            "video_id": video_id,
            "product_id": product_id,
            "metadata": result["metadata"],
            "frame_count": result["frame_count"],
            "frames_url": result.get("frames_url"),
            "primary_image_url": result.get("primary_image_url"),
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n{'=' * 60}")
        print(f"ERROR: {error_msg}")
        print(error_trace)
        print(f"{'=' * 60}\n")

        # Return error with product_id so webhook can update product status
        # Note: product_id may be None if error occurred before validation
        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
            "product_id": product_id,  # Include for webhook to reset product status
        }


# Health check for local testing
def health_check():
    """Simple health check."""
    return {"status": "healthy", "worker": "video-segmentation"}


# Start Runpod serverless worker
if __name__ == "__main__":
    print("Starting Video Segmentation Worker...")
    print(f"CUDA Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    runpod.serverless.start({"handler": handler})
