# Video Segmentation Worker

Runpod serverless worker for processing product videos.

## Pipeline

```
Video URL → Download → Extract Frames → Gemini (metadata) → SAM2 (segmentation) → 518x518 Frames → Supabase Storage
```

## Input

```json
{
  "video_url": "https://drive.google.com/...",
  "barcode": "8690504012345",
  "video_id": 12345,
  "product_id": "uuid-...",
  "job_id": "uuid-..."
}
```

## Output

```json
{
  "status": "success",
  "barcode": "8690504012345",
  "product_id": "uuid-...",
  "metadata": {
    "brand_info": { "brand_name": "Coca-Cola", ... },
    "product_identity": { "product_name": "Classic", ... },
    ...
  },
  "frame_count": 177,
  "frames_url": "https://xxx.supabase.co/storage/v1/object/public/frames/uuid/",
  "primary_image_url": "https://xxx.supabase.co/storage/v1/object/public/frames/uuid/frame_0088.png"
}
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `HF_TOKEN` | HuggingFace token (for SAM2 models) | Yes |
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_KEY` | Supabase service role key | Yes |
| `CALLBACK_URL` | Backend API URL for webhooks | Optional |

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="..."
export SUPABASE_URL="..."
export SUPABASE_KEY="..."

# Test locally
python src/handler.py
```

## Docker Build

```bash
# Build image
docker build -t video-segmentation-worker .

# Run locally
docker run --gpus all \
  -e GEMINI_API_KEY="..." \
  -e SUPABASE_URL="..." \
  -e SUPABASE_KEY="..." \
  video-segmentation-worker
```

## Deploy to Runpod

1. Push image to Docker Hub or GitHub Container Registry
2. Create new Serverless Endpoint on Runpod
3. Select GPU type (recommended: RTX 4090 or A100)
4. Set environment variables
5. Deploy

## Webhook Callback

When `job_id` is provided and `CALLBACK_URL` is set, the worker sends status updates:

```json
POST {CALLBACK_URL}/api/v1/webhooks/runpod
{
  "job_id": "uuid-...",
  "type": "video_processing",
  "status": "processing" | "completed" | "failed",
  "result": { ... },  // on success
  "error": "..."      // on failure
}
```

## Processing Steps

1. **Download**: Fetch video from URL
2. **Extract Frames**: Get all frames from video (max 500, configurable)
3. **Gemini Metadata**: Extract brand, product name, category, nutrition facts
4. **SAM2 Segmentation**: Segment product from background using text prompt
5. **Post-process**: Crop, center, resize to 518x518 (DINOv2 optimal)
6. **Upload**: Save frames and metadata to Supabase Storage
7. **Update DB**: Update product record with metadata and URLs
