"""
Runpod Serverless Handler for Video Segmentation

Input:
{
    "video_url": "https://...",
    "barcode": "123456789",
    "video_id": 12345,
    "product_id": "uuid-...",  # Our system's product UUID
    "sample_rate": 1,  # Optional: extract every Nth frame (1 = every frame)
    "max_frames": null,  # Optional: maximum frames to extract (null = all)
    "gemini_model": "gemini-2.0-flash",  # Optional: Gemini model for metadata extraction
    "custom_prompts": ["red can", "bottle"],  # Optional: override Gemini grounding prompts
    "points": [{"x": 0.5, "y": 0.3, "label": 1}]  # Optional: point prompts for SAM3
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

# Supabase for job status updates
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Pipeline singleton - loaded once on cold start
pipeline = None
supabase_client = None


def get_supabase():
    """Get or create Supabase client singleton."""
    global supabase_client
    if supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Supabase init error: {e}")
    return supabase_client


def update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    """Update job status in database."""
    client = get_supabase()
    if not client:
        return
    try:
        update_data = {"status": status}
        if status == "completed":
            update_data["progress"] = 100
        if result:
            update_data["result"] = result
        if error:
            update_data["error"] = error
        client.table("jobs").update(update_data).eq("runpod_job_id", job_id).execute()
        print(f"Job status updated to: {status}")
    except Exception as e:
        print(f"Job status update error: {e}")


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
    job_id = job.get("id", "unknown")
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
        sample_rate = job_input.get("sample_rate")  # Optional: extract every Nth frame
        max_frames = job_input.get("max_frames")  # Optional: max frames to extract
        gemini_model = job_input.get("gemini_model")  # Optional: Gemini model for metadata
        custom_prompts = job_input.get("custom_prompts")  # Optional: override Gemini grounding prompts
        points = job_input.get("points")  # Optional: point prompts for SAM3

        # Validate product_id is provided (required for storage)
        if not product_id:
            return {"status": "error", "error": "product_id is required. Create product before processing."}

        print(f"\n{'=' * 60}")
        print(f"Processing: {barcode}")
        print(f"Video URL: {video_url[:80]}...")
        print(f"Product ID: {product_id}")
        print(f"Sample Rate: {sample_rate or 'config default'}")
        print(f"Max Frames: {max_frames or 'config default (all)'}")
        print(f"Gemini Model: {gemini_model or 'config default (gemini-2.0-flash)'}")
        if custom_prompts:
            print(f"Custom Prompts: {custom_prompts}")
        if points:
            print(f"Point Prompts: {len(points)} points")
        print(f"{'=' * 60}\n")

        # Get pipeline and process
        pipe = get_pipeline()
        result = pipe.process(
            video_url=video_url,
            barcode=barcode,
            video_id=video_id,
            product_id=product_id,
            sample_rate=sample_rate,
            max_frames=max_frames,
            gemini_model=gemini_model,
            custom_prompts=custom_prompts,
            points=points,
        )

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: {barcode}")
        print(f"Frames: {result['frame_count']}")
        print(f"{'=' * 60}\n")

        # Build result
        final_result = {
            "status": "success",
            "barcode": barcode,
            "video_id": video_id,
            "product_id": product_id,
            "sample_rate": sample_rate,
            "max_frames": max_frames,
            "gemini_model": gemini_model,
            "metadata": result["metadata"],
            "frame_count": result["frame_count"],
            "frames_url": result.get("frames_url"),
            "primary_image_url": result.get("primary_image_url"),
        }

        # Update job status to completed
        update_job_status(job_id, "completed", result=final_result)

        return final_result

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n{'=' * 60}")
        print(f"ERROR: {error_msg}")
        print(error_trace)
        print(f"{'=' * 60}\n")

        # Update job status to failed
        update_job_status(job_id, "failed", error=error_msg)

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


# Concurrency modifier - only allow 1 job per worker at a time
# This prevents OOM errors when multiple jobs try to use GPU simultaneously
def concurrency_modifier(current_concurrency: int) -> int:
    """
    Limit concurrent jobs to 1 per worker.
    SAM3 uses ~17-20GB GPU memory, so we can't run multiple jobs on a 24GB GPU.
    """
    return 1  # Max 1 concurrent job per worker


# Start Runpod serverless worker
if __name__ == "__main__":
    print("Starting Video Segmentation Worker...")
    print(f"CUDA Available: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
    })
