# Inference Worker - Testing Guide

This guide covers how to validate that the inference implementation is working correctly.

## Test Strategy

We have 3 levels of testing:

1. **Unit Tests** - Test InferenceService logic with mocked dependencies
2. **Handler Tests** - Test worker handler logic without Docker
3. **Integration Tests** - Test end-to-end with Docker/RunPod

## 1. Unit Tests (InferenceService)

Tests the API-side service with mocked RunPod and ModelLoader.

### Location
```
apps/api/test_inference_service.py
```

### Requirements
```bash
cd apps/api
pip install pytest pytest-asyncio pillow
```

### Run Tests
```bash
cd apps/api
pytest test_inference_service.py -v
```

### What's Tested
- ✅ Detection inference success case
- ✅ Classification inference success case
- ✅ Embedding extraction success case
- ✅ Model not found error handling
- ✅ Inference failure error handling
- ✅ Invalid response validation
- ✅ Image format conversion (RGB, RGBA, grayscale)
- ✅ Batch detection

### Expected Output
```
test_inference_service.py::TestDetection::test_detect_success PASSED
test_inference_service.py::TestDetection::test_detect_model_not_found PASSED
test_inference_service.py::TestDetection::test_detect_inference_failed PASSED
test_inference_service.py::TestDetection::test_detect_invalid_response PASSED
test_inference_service.py::TestClassification::test_classify_success PASSED
test_inference_service.py::TestClassification::test_classify_invalid_response PASSED
test_inference_service.py::TestEmbedding::test_embed_success PASSED
test_inference_service.py::TestEmbedding::test_embed_invalid_response PASSED
test_inference_service.py::TestImageConversion::test_image_to_base64_rgb PASSED
test_inference_service.py::TestImageConversion::test_image_to_base64_rgba PASSED
test_inference_service.py::TestImageConversion::test_image_to_base64_grayscale PASSED
test_inference_service.py::TestBatchDetection::test_batch_detect PASSED

======================== 12 passed in 0.5s ========================
```

## 2. Handler Tests (Worker Logic)

Tests the worker handler without Docker/RunPod infrastructure.

### Location
```
workers/inference/test_handler_local.py
```

### Requirements
```bash
cd workers/inference
pip install -r requirements.txt
```

### Run Tests
```bash
cd workers/inference
python test_handler_local.py
```

### What's Tested
- ✅ Input validation (missing fields, invalid values)
- ✅ Detection with pretrained YOLO11n (~6MB download)
- ✅ Classification with pretrained ViT-tiny (~20MB download)
- ✅ Embedding with pretrained DINOv2-small (~80MB download)
- ✅ Response format validation
- ✅ Error handling
- ✅ Model caching (second run should be faster)

### Expected Output
```
============================================================
INFERENCE WORKER - LOCAL HANDLER TESTS
============================================================

NOTE: This tests handler logic only.
GPU/CUDA features require actual hardware.
Some models will be downloaded on first run (~100-500MB).

============================================================

Which inference tests do you want to run?
1. Detection (downloads ~6MB YOLO11n)
2. Classification (downloads ~20MB ViT-tiny)
3. Embedding (downloads ~80MB DINOv2-small)
4. All of the above
5. Skip inference tests

Choice (1-5) [5]: 1

============================================================
TEST 1: Detection (pretrained YOLO)
============================================================
Input: yolo11n, source=pretrained
Image size: (640, 480)

Loading model: detection:pretrained:yolo11n:pretrained
Model loaded in 2500ms

✅ Result success: True
Detections: 2
Image size: {'width': 640, 'height': 480}
Inference time: 150ms
Model cached: False

First detection: person (0.85)

✅ All validation checks passed!

============================================================
TEST SUMMARY
============================================================
✅ PASS: Input Validation
✅ PASS: Detection

============================================================
✅ ALL TESTS PASSED!
============================================================
```

### Notes
- First run will download models (this is normal)
- CPU inference is slow (~1-5s), GPU would be much faster
- Model caching works - second run should reuse loaded models
- If you see "Model cached: True", caching is working!

## 3. Docker Tests (Container)

Test the worker in a Docker container (local GPU recommended).

### Build Image
```bash
cd workers/inference
docker build -t inference-worker:test .
```

### Run Container (GPU)
```bash
docker run --gpus all \
  -p 8000:8000 \
  -e RUNPOD_API_KEY=test \
  inference-worker:test
```

### Run Container (CPU only)
```bash
docker run \
  -p 8000:8000 \
  -e RUNPOD_API_KEY=test \
  inference-worker:test
```

### Test with curl
```bash
# Detection test
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "detection",
      "model_type": "yolo11n",
      "model_source": "pretrained",
      "image": "<base64_image>",
      "config": {"confidence": 0.5}
    }
  }'
```

### Expected Output
```json
{
  "success": true,
  "result": {
    "detections": [...],
    "count": 5,
    "image_size": {"width": 1920, "height": 1080}
  },
  "metadata": {
    "task": "detection",
    "model_type": "yolo11n",
    "inference_time_ms": 250,
    "cached": false,
    "device": "cuda"
  }
}
```

## 4. Integration Tests (End-to-End)

These require:
- ✅ RunPod endpoint deployed
- ✅ Database with test models
- ✅ API server running

### Run E2E Tests
```bash
# Coming soon - requires full deployment
cd apps/api
pytest tests/integration/test_workflow_inference.py -v
```

## Validation Checklist

Before deploying to production, verify:

### Unit Tests
- [ ] All 12 pytest tests pass
- [ ] No warnings or errors
- [ ] Test coverage >80%

### Handler Tests
- [ ] Input validation works correctly
- [ ] Detection inference produces valid output
- [ ] Classification inference produces valid output
- [ ] Embedding extraction produces valid output
- [ ] Model caching works (second run faster)
- [ ] Error handling catches invalid inputs

### Docker Tests
- [ ] Image builds successfully
- [ ] Container starts without errors
- [ ] Health check passes
- [ ] Detection inference works
- [ ] GPU is detected (if available)
- [ ] Models are cached between requests

### Response Format Validation
- [ ] Detection output has: `detections`, `count`, `image_size`
- [ ] Classification output has: `predictions`, `top_class`, `top_confidence`
- [ ] Embedding output has: `embedding`, `embedding_dim`, `normalized`
- [ ] Each detection has: `id`, `class_name`, `class_id`, `confidence`, `bbox`, `area`
- [ ] BBox coordinates are normalized (0-1 range)

### Performance Validation
- [ ] Cold start completes within 120s timeout
- [ ] Warm inference <5s
- [ ] Detection inference <500ms (GPU)
- [ ] Memory usage stable (no leaks)
- [ ] Models stay cached between requests

## Known Issues & Limitations

### Current Implementation
1. **No batch inference** - Processes one image at a time
2. **No result caching** - Every request runs inference
3. **No annotation overlay** - DetectionBlock handles this
4. **Sequential batch processing** - `batch_detect()` is a simple loop

### Future Enhancements
1. True batch inference on worker
2. Redis-based result caching
3. Request queuing and prioritization
4. Hot/warm/cold model tier routing
5. TTA and multi-scale inference

## Troubleshooting

### Tests fail with "Module not found"
```bash
# Make sure you're in the right directory
cd apps/api  # for unit tests
cd workers/inference  # for handler tests

# Install dependencies
pip install -r requirements.txt
```

### Tests timeout
- Increase timeout in test configuration
- Check internet connection (models download on first run)
- CPU inference is slow - consider running minimal tests only

### GPU not detected in Docker
```bash
# Install nvidia-docker
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Test GPU access
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Models download slowly
- Normal on first run (100-500MB total)
- Models are cached in `/tmp/checkpoints`
- Subsequent runs will be faster

### Response format mismatch
- Check worker handler output format
- Verify DetectionBlock expectations
- Run response validation tests

## Test Results Location

Test outputs are saved to:
- Unit test reports: `apps/api/pytest_report.html`
- Handler test logs: `workers/inference/test_output.log`
- Docker logs: `docker logs <container_id>`

## Continuous Integration

For CI/CD pipelines:

```bash
# Run unit tests only (fast, no GPU needed)
pytest apps/api/test_inference_service.py -v --junitxml=test-results.xml

# Run handler validation tests (skip inference)
cd workers/inference && python test_handler_local.py

# Build Docker image
docker build -t inference-worker:$CI_COMMIT_SHA workers/inference/
```

## Next Steps

After validation:
1. Deploy worker to RunPod
2. Configure endpoint ID in `.env`
3. Update workflow blocks to use InferenceService
4. Run end-to-end workflow tests
5. Monitor production metrics
