"""
Runpod Serverless Handler

Input:
{
    "video_url": "https://...",
    "barcode": "123456789",
    "video_id": 12345
}

Output:
{
    "status": "success",
    "barcode": "123456789",
    "metadata": {...},
    "frame_count": 177,
    "frames": ["base64...", ...] or "frames_url": "https://..."
}
"""

import runpod
import traceback
from pipeline import ProductPipeline

# Pipeline singleton - loaded once on cold start
pipeline = None


def get_pipeline():
    """Get or create pipeline singleton."""
    global pipeline
    if pipeline is None:
        print("="*60)
        print("COLD START - Loading pipeline...")
        print("="*60)
        pipeline = ProductPipeline()
        print("Pipeline ready!")
        print("="*60)
    return pipeline


def handler(job):
    """
    Main handler for Runpod serverless.
    """
    try:
        job_input = job.get("input", {})
        
        # Validate input
        video_url = job_input.get("video_url")
        if not video_url:
            return {"status": "error", "error": "video_url is required"}
        
        barcode = job_input.get("barcode", "unknown")
        video_id = job_input.get("video_id")
        
        print(f"\n{'='*60}")
        print(f"Processing: {barcode}")
        print(f"Video URL: {video_url[:80]}...")
        print(f"{'='*60}\n")
        
        # Get pipeline
        pipe = get_pipeline()
        
        # Run pipeline
        result = pipe.process(
            video_url=video_url,
            barcode=barcode,
            video_id=video_id
        )
        
        print(f"\n{'='*60}")
        print(f"SUCCESS: {barcode}")
        print(f"Frames: {result['frame_count']}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "barcode": barcode,
            "video_id": video_id,
            "metadata": result["metadata"],
            "frame_count": result["frame_count"],
            "frames_url": result.get("frames_url"),
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"ERROR: {error_msg}")
        print(error_trace)
        print(f"{'='*60}\n")
        
        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace
        }


# Start Runpod serverless worker
if __name__ == "__main__":
    print("Starting Runpod Serverless Worker...")
    runpod.serverless.start({"handler": handler})
