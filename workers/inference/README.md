# Unified Inference Worker

RunPod serverless worker for all inference tasks: detection, classification, and embedding.

## Features

- **Single worker** handles all model types
- **Model caching** - models stay in memory between requests
- **Checkpoint caching** - downloads cached locally
- **GPU accelerated** - uses CUDA when available
- **Auto-discovery** - models auto-loaded from database

## Supported Models

### Detection
- **Pretrained**: YOLO11, YOLOv8-10 (Ultralytics)
- **Trained**: RT-DETR, D-FINE, YOLO-NAS (from `od_trained_models`)

### Classification
- **Pretrained**: ImageNet models (ViT, ConvNeXt, EfficientNet, Swin)
- **Trained**: Custom models (from `cls_trained_models`)

### Embedding
- **Pretrained**: DINOv2, CLIP, SigLIP
- **Trained**: Fine-tuned embeddings (from `trained_models`)

## Input Format

```json
{
  "task": "detection | classification | embedding",
  "model_id": "uuid",
  "model_source": "pretrained | trained",
  "model_type": "yolo11n | vit | dinov2",
  "checkpoint_url": "https://...",
  "class_mapping": {"0": "person", "1": "car"},
  "image": "base64_encoded_jpeg",
  "config": {
    "confidence": 0.5,
    "top_k": 5,
    "normalize": true
  }
}
```

## Output Format

### Detection
```json
{
  "success": true,
  "result": {
    "detections": [
      {
        "id": 0,
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.95,
        "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
        "area": 0.02
      }
    ],
    "count": 5,
    "image_size": {"width": 1920, "height": 1080}
  },
  "metadata": {
    "inference_time_ms": 250,
    "cached": true
  }
}
```

### Classification
```json
{
  "success": true,
  "result": {
    "predictions": [
      {"class_name": "cat", "class_id": 0, "confidence": 0.98}
    ],
    "top_class": "cat",
    "top_confidence": 0.98
  }
}
```

### Embedding
```json
{
  "success": true,
  "result": {
    "embedding": [0.1, 0.2, ...],
    "embedding_dim": 768,
    "normalized": true
  }
}
```

## Deployment

### Build Docker Image

```bash
cd workers/inference
docker build -t inference-worker:latest .
```

### Test Locally

```bash
docker run --gpus all -p 8000:8000 \
  -e RUNPOD_API_KEY=your-key \
  inference-worker:latest
```

### Deploy to RunPod

1. Push image to Docker Hub or RunPod's registry
2. Create serverless endpoint in RunPod dashboard
3. Configure:
   - GPU: RTX 3090 or better
   - Min Workers: 0 (serverless)
   - Max Workers: 10
   - GPU Memory: 12GB+
   - Timeout: 300s

4. Copy endpoint ID to `.env`:
```bash
RUNPOD_INFERENCE_ENDPOINT_ID=your-endpoint-id
```

## Performance

### Cold Start
- First request: ~30-60s (model download + loading)
- Subsequent requests: ~2-5s (model cached)

### Inference Times (RTX 3090)
- Detection (YOLO11): 50-200ms
- Classification (ViT): 100-300ms
- Embedding (DINOv2): 200-500ms

### Optimization
- **Keep workers warm**: Set min workers to 1-2 in RunPod
- **Preload models**: Set `PRELOAD_MODELS` env var
- **Batch requests**: Use batch endpoints (future enhancement)

## Monitoring

Worker logs show:
- Model cache hits/misses
- Download progress
- Inference times
- GPU utilization

Example log:
```
Loading model: task=detection, type=yolo11n, source=pretrained
Using cached model: detection:pretrained:yolo11n:pretrained
Inference complete in 150ms
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller model variants
- Increase GPU memory allocation

### Slow First Request
- Normal for cold start
- Consider using warm pool (min workers > 0)
- Preload common models

### Model Not Found
- Check model_id exists in database
- Verify checkpoint_url is accessible
- Check model_source ("pretrained" vs "trained")

## Testing

Test detection:
```bash
python test_inference.py --task detection --model yolo11n
```

Test classification:
```bash
python test_inference.py --task classification --model vit-base
```

Test embedding:
```bash
python test_inference.py --task embedding --model dinov2-base
```

## Cost Estimation

RunPod serverless pricing (RTX 3090):
- Compute: ~$0.50/hour
- Idle: $0 (serverless)

Monthly cost estimate:
- 1000 requests/day @ 5s each: ~$20/month
- 10,000 requests/day: ~$200/month
- Always-on (1 worker): ~$360/month

## Future Enhancements

- [ ] Request batching
- [ ] Result caching with Redis
- [ ] SAHI tiled inference
- [ ] TTA (test-time augmentation)
- [ ] Multi-scale embedding
- [ ] Model quantization (FP16/INT8)
- [ ] Hot model tier routing
