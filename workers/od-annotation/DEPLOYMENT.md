# OD Annotation Worker - RunPod Deployment Guide

## Overview

This worker provides AI-powered object detection and segmentation:
- **Grounding DINO**: Text→bbox detection (SOTA open-vocabulary)
- **SAM 2.1**: Interactive point/box segmentation
- **SAM 3**: Text-guided segmentation
- **Florence-2**: Microsoft's versatile vision model

## Prerequisites

1. RunPod account with GPU serverless access
2. Docker Hub account (for pushing image)
3. Hugging Face token (for gated models like SAM3)

## Build & Push Docker Image

### 1. Build locally (for testing)

```bash
cd workers/od-annotation

# Build image
docker build -t od-annotation-worker:latest .

# Test locally (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  od-annotation-worker:latest
```

### 2. Push to Docker Hub

```bash
# Tag for Docker Hub
docker tag od-annotation-worker:latest your-dockerhub/od-annotation-worker:v1.0.0

# Push
docker push your-dockerhub/od-annotation-worker:v1.0.0
```

## RunPod Serverless Setup

### 1. Create New Template

Go to RunPod Console → Serverless → Templates → New Template

| Field | Value |
|-------|-------|
| Template Name | `od-annotation-worker` |
| Container Image | `your-dockerhub/od-annotation-worker:v1.0.0` |
| Container Disk | `20 GB` (for model weights) |
| Volume Disk | `0 GB` (optional) |

### 2. Environment Variables

Add these environment variables in the template:

```
DEVICE=cuda
LOG_LEVEL=INFO
HF_TOKEN=your_huggingface_token
PRELOAD_MODELS=grounding_dino,sam2
DEFAULT_BOX_THRESHOLD=0.3
DEFAULT_TEXT_THRESHOLD=0.25
```

### 3. Create Endpoint

Go to Serverless → Endpoints → New Endpoint

| Field | Value |
|-------|-------|
| Endpoint Name | `od-annotation` |
| Template | `od-annotation-worker` |
| GPU Type | `RTX 4090` or `A100` (24GB+ VRAM) |
| Workers | Active: 0, Max: 3 |
| Idle Timeout | 5 seconds |
| Flash Boot | Enabled |

### 4. Get Endpoint ID

After creation, copy the Endpoint ID (e.g., `abc123xyz`)

## Configure Backend API

### 1. Update Environment Variables

Add to your `.env` file:

```bash
# RunPod Configuration
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_OD_ANNOTATION=abc123xyz
```

### 2. Verify Configuration

```python
# In Python shell
from config import settings
print(settings.runpod_endpoint_od_annotation)  # Should show endpoint ID
```

## Testing

### 1. Test Worker Locally

```bash
# Send test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "detect",
      "model": "grounding_dino",
      "image_url": "https://example.com/test.jpg",
      "text_prompt": "shelf . product"
    }
  }'
```

### 2. Test via RunPod Console

Use the "Test" button in RunPod Console:

```json
{
  "input": {
    "task": "detect",
    "model": "grounding_dino",
    "image_url": "https://your-image-url.jpg",
    "text_prompt": "shelf . product . price tag",
    "box_threshold": 0.3
  }
}
```

### 3. Test via API

```bash
# Run API tests
cd workers/od-annotation
python tests/test_api.py --api-url http://localhost:8000/api/v1/od -v
```

## Task Types

### Detect (Single Image)

```json
{
  "input": {
    "task": "detect",
    "model": "grounding_dino",
    "image_url": "https://...",
    "text_prompt": "shelf . product . price tag",
    "box_threshold": 0.3,
    "text_threshold": 0.25
  }
}
```

### Segment (Interactive)

```json
{
  "input": {
    "task": "segment",
    "model": "sam2",
    "image_url": "https://...",
    "prompt_type": "point",
    "point": [0.5, 0.3],
    "label": 1
  }
}
```

### Batch (Multiple Images)

```json
{
  "input": {
    "task": "batch",
    "model": "grounding_dino",
    "images": [
      {"id": "img1", "url": "https://..."},
      {"id": "img2", "url": "https://..."}
    ],
    "text_prompt": "product",
    "box_threshold": 0.3
  }
}
```

## GPU Requirements

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| Grounding DINO | ~4GB | RTX 3090+ |
| SAM 2.1 | ~6GB | RTX 3090+ |
| SAM 3 | ~8GB | RTX 4090/A100 |
| Florence-2 | ~8GB | RTX 4090/A100 |
| All Models | ~16GB | RTX 4090/A100 |

## Monitoring

### RunPod Dashboard

- Check worker logs in RunPod Console
- Monitor GPU utilization
- Track cold start times

### API Logs

```bash
# Check API logs for RunPod calls
tail -f logs/api.log | grep -i runpod
```

## Troubleshooting

### Cold Start Issues

1. Enable Flash Boot in endpoint settings
2. Increase `PRELOAD_MODELS` for frequently used models
3. Use larger GPU for faster model loading

### Memory Errors

1. Reduce batch size
2. Clear model cache between requests
3. Use smaller model variants

### Timeout Issues

1. Increase endpoint timeout (default: 300s)
2. Use batch endpoint for multiple images
3. Check image download speed

## Cost Optimization

1. **Idle Timeout**: Set to 5s to minimize idle costs
2. **Active Workers**: Keep at 0, let auto-scaling handle demand
3. **GPU Selection**: Use RTX 4090 for best price/performance
4. **Preload Models**: Only preload frequently used models

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-20 | Initial release with GDINO, SAM2, SAM3, Florence-2 |
