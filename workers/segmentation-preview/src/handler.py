"""
Runpod Serverless Handler for Segmentation Preview

This worker generates a preview segmentation mask for a single frame,
allowing users to verify the segmentation before processing the full video.

Input:
{
    "video_url": "https://...",
    "text_prompts": ["red can", "bottle"],  # Optional: text prompts for SAM3
    "points": [{"x": 0.5, "y": 0.3, "label": 1}]  # Optional: point prompts
}

Output:
{
    "status": "success",
    "mask_image": "base64-encoded-png",
    "mask_stats": {
        "pixel_count": 12345,
        "coverage_percent": 8.5,
        "width": 1920,
        "height": 1080
    }
}
"""

import runpod
import traceback
from preview import PreviewPipeline

# Pipeline singleton - loaded once on cold start
pipeline = None


def get_pipeline():
    """Get or create pipeline singleton."""
    global pipeline
    if pipeline is None:
        print("=" * 60)
        print("COLD START - Loading preview pipeline...")
        print("=" * 60)
        pipeline = PreviewPipeline()
        print("Pipeline ready!")
        print("=" * 60)
    return pipeline


def handler(job):
    """Main handler for Runpod serverless."""
    job_input = job.get("input", {}) if job else {}

    try:
        # Validate input
        video_url = job_input.get("video_url")
        if not video_url:
            return {"status": "error", "error": "video_url is required"}

        text_prompts = job_input.get("text_prompts", [])
        points = job_input.get("points", [])

        # Need at least one prompt type
        if not text_prompts and not points:
            return {"status": "error", "error": "At least one text_prompt or point is required"}

        print(f"\n{'=' * 60}")
        print(f"Preview request:")
        print(f"Video URL: {video_url[:80]}...")
        if text_prompts:
            print(f"Text Prompts: {text_prompts}")
        if points:
            print(f"Point Prompts: {len(points)} points")
        print(f"{'=' * 60}\n")

        # Get pipeline and generate preview
        pipe = get_pipeline()
        result = pipe.preview(
            video_url=video_url,
            text_prompts=text_prompts,
            points=points,
        )

        print(f"\n{'=' * 60}")
        print(f"SUCCESS: Preview generated")
        print(f"Coverage: {result['mask_stats']['coverage_percent']:.1f}%")
        print(f"{'=' * 60}\n")

        return {
            "status": "success",
            "mask_image": result["mask_image"],
            "mask_stats": result["mask_stats"],
        }

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n{'=' * 60}")
        print(f"ERROR: {error_msg}")
        print(error_trace)
        print(f"{'=' * 60}\n")

        return {
            "status": "error",
            "error": error_msg,
            "traceback": error_trace,
        }


# Concurrency modifier - only allow 1 job per worker at a time
def concurrency_modifier(current_concurrency: int) -> int:
    """Limit concurrent jobs to 1 per worker."""
    return 1


# Start Runpod serverless worker
if __name__ == "__main__":
    print("Starting Segmentation Preview Worker...")
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
    })
