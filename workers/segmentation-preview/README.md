# Segmentation Preview Worker

A lightweight RunPod serverless worker for single-frame segmentation preview. This worker runs SAM3 on just the first frame of a video and returns a mask overlay for user validation before full video reprocessing.

## Purpose

When automatic segmentation produces incorrect results, users can:
1. Provide custom text prompts (e.g., "the red can on the left")
2. Click on the frame to add positive/negative point prompts
3. Preview the segmentation result before committing to full reprocess

This worker handles the preview step - fast feedback on a single frame.

## Input

```json
{
  "video_url": "https://storage.example.com/video.mp4",
  "text_prompts": ["red energy drink can"],
  "points": [
    {"x": 0.45, "y": 0.32, "label": 1},
    {"x": 0.12, "y": 0.88, "label": 0}
  ]
}
```

- `video_url` (required): URL of the video to preview
- `text_prompts` (optional): List of text prompts for SAM3
- `points` (optional): List of point prompts
  - `x`, `y`: Normalized coordinates (0-1)
  - `label`: 1 = positive (include), 0 = negative (exclude)

At least one of `text_prompts` or `points` is required.

## Output

```json
{
  "mask_image": "base64-encoded-png...",
  "first_frame": "base64-encoded-png...",
  "mask_stats": {
    "pixel_count": 45230,
    "coverage_percent": 8.5,
    "width": 1920,
    "height": 1080
  }
}
```

- `mask_image`: First frame with semi-transparent green mask overlay
- `first_frame`: Original first frame (for comparison)
- `mask_stats`: Statistics about the detected mask

## Deployment

### Build Docker Image

```bash
docker build -t segmentation-preview:latest .
```

### Deploy to RunPod

1. Push image to Docker Hub or RunPod registry
2. Create new Serverless Endpoint on RunPod
3. Select GPU (RTX 3090 or better recommended)
4. Set endpoint ID in API configuration

### Environment Variables

No required environment variables. SAM3 model is loaded from HuggingFace automatically.

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Test handler
python -c "
from src.preview import PreviewPipeline
pipe = PreviewPipeline()
result = pipe.preview(
    video_url='https://example.com/test.mp4',
    text_prompts=['can'],
)
print(f'Coverage: {result[\"mask_stats\"][\"coverage_percent\"]}%')
"
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   API        │────▶│  This Worker    │
│   (Click/Type)  │     │  /preview    │     │  (SAM3 1-frame) │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                              ┌───────▼───────┐
                                              │ mask_image    │
                                              │ (base64 PNG)  │
                                              └───────────────┘
```

This is separate from the main video-segmentation worker to:
- Avoid affecting production pipeline stability
- Enable faster iteration on preview features
- Keep preview logic simple and focused
