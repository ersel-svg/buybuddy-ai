# Workflow Block System - SOTA Architecture Analysis

**Date:** 2026-01-26
**Analyst:** Claude Sonnet 4.5
**Purpose:** Identify architectural issues and design SOTA, scalable inference system

---

## 🔍 Executive Summary

**Current Status:** ❌ **CRITICAL ARCHITECTURAL ISSUES**

The workflow block system has a fundamental design flaw: **Model blocks (Detection, Classification, Embedding) are doing LOCAL inference instead of delegating to the RunPod GPU worker.**

### Key Findings

| Component | Current State | Target State | Status |
|-----------|--------------|--------------|---------|
| DetectionBlock | ❌ Local ModelLoader | ✅ InferenceService → RunPod | **BROKEN** |
| ClassificationBlock | ❌ Local ModelLoader | ✅ InferenceService → RunPod | **BROKEN** |
| EmbeddingBlock | ❌ Local torch inference | ✅ InferenceService → RunPod | **BROKEN** |
| InferenceService | ✅ Exists & correct | ✅ Already has all methods | **READY** |
| RunPod Worker | ✅ Supports all models | ✅ GPU inference working | **READY** |

**Impact:**
- ❌ Tests failing due to missing local dependencies (ultralytics, torch)
- ❌ No GPU acceleration (trying to run on API server CPU)
- ❌ Cannot scale to 100s-1000s of requests
- ❌ Worker sitting idle while API tries local inference

---

## 📊 Block Inventory

### Available Blocks (24 total)

#### 1. **Input Blocks** (2)
- [ImageInputBlock](../apps/api/src/services/workflow/blocks/input_blocks.py) - ✅ Working
- [ParameterInputBlock](../apps/api/src/services/workflow/blocks/input_blocks.py) - ✅ Working

#### 2. **Model Blocks** (5) - ❌ **ALL NEED FIXING**
- **DetectionBlock** - Lines 97-813 - ❌ Uses ModelLoader (local)
- **ClassificationBlock** - Lines 815-1340 - ❌ Uses ModelLoader (local)
- **EmbeddingBlock** - Lines 1342-1750 - ❌ Uses local torch inference
- **SegmentationBlock** - Lines 1752+ - ❌ Not checked but likely same issue
- **SimilaritySearchBlock** - Lines 2065+ - ⚠️ Uses Qdrant (OK, different service)

#### 3. **Transform Blocks** (11) - ✅ All CPU-based (correct)
- CropBlock, BlurRegionBlock, ResizeBlock, TileBlock, StitchBlock
- RotateFlipBlock, NormalizeBlock, SmoothingBlock
- DrawBoxesBlock, DrawMasksBlock, HeatmapBlock, ComparisonBlock

#### 4. **Logic Blocks** (6) - ✅ All CPU-based (correct)
- ConditionBlock, FilterBlock, ForEachBlock
- CollectBlock, MapBlock, GridBuilderBlock

#### 5. **Output Blocks** (4) - ✅ All CPU-based (correct)
- JsonOutputBlock, APIResponseBlock, WebhookBlock, AggregationBlock

---

## 🔴 Critical Issues

### Issue #1: DetectionBlock Uses Local Inference

**File:** [model_blocks.py:464-813](../apps/api/src/services/workflow/blocks/model_blocks.py:464)

**Current Implementation:**
```python
async def execute(self, inputs, config, context):
    # ❌ WRONG: Tries to load model locally
    model, model_info = await self._model_loader.load_detection_model(
        model_id, model_source
    )

    # ❌ WRONG: Runs local inference (lines 543-805)
    results = model.predict(
        image,
        conf=confidence,
        iou=iou_threshold,
        # ... lots of local inference code
    )
```

**Problem:**
1. Calls `ModelLoader.load_detection_model()` which does `from ultralytics import YOLO`
2. Ultralytics not installed on API server (intentionally - should be worker-only)
3. Tries to run inference on API server CPU (slow, not scalable)
4. Worker has GPU and ultralytics but never gets called

**Evidence:** Test failures show:
```
Failed to load model: ultralytics package not installed
```

---

### Issue #2: ClassificationBlock Uses Local Inference

**File:** [model_blocks.py:815-1340](../apps/api/src/services/workflow/blocks/model_blocks.py:815)

**Current Implementation:**
```python
async def execute(self, inputs, config, context):
    # ❌ WRONG: Local model loading
    model, processor, model_info = await self._model_loader.load_classification_model(
        model_id, model_source
    )

    # ❌ WRONG: Local torch inference (lines 1100-1310)
    loop = asyncio.get_event_loop()
    logits = await loop.run_in_executor(
        None, lambda: model(**inputs_tensor).logits
    )
```

**Same architectural issue** - trying to run transformers models locally.

---

### Issue #3: EmbeddingBlock Uses Local Inference

**File:** [model_blocks.py:1342-1750](../apps/api/src/services/workflow/blocks/model_blocks.py:1342)

**Current Implementation:**
```python
async def execute(self, inputs, config, context):
    # ❌ WRONG: Local model loading
    model, processor, model_info = await self._model_loader.load_embedding_model(
        model_id, model_source
    )

    # ❌ WRONG: Local torch inference (lines 1557-1700)
    inputs_tensor = processor(images=resized_image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs_tensor = {k: v.cuda() for k, v in inputs_tensor.items()}
        model.cuda()

    with torch.no_grad():
        outputs = model(**inputs_tensor)
```

**Surprisingly:** This was the only test that "worked" but only because:
1. Test report says it took 9.14s for DINOv2
2. Looking at code, this is **LOCAL inference**, not worker!
3. Must have torch/transformers installed temporarily during testing

---

## ✅ What's Already Correct

### InferenceService - Ready to Use

**File:** [inference_service.py:1-419](../apps/api/src/services/workflow/inference_service.py)

Already has perfect methods:

```python
class InferenceService:
    """All ML inference goes through unified RunPod worker."""

    async def detect(
        self, model_id, image, confidence=0.5, iou=0.45,
        max_detections=300, model_source="pretrained", ...
    ) -> Dict[str, Any]:
        """Run detection via RunPod GPU worker."""
        # 1. Get model info from DB
        model_info = await self.model_loader.get_detection_model_info(...)

        # 2. Prepare job for worker
        job_input = {
            "task": "detection",
            "model_id": model_id,
            "model_type": model_info.model_type,
            "checkpoint_url": model_info.checkpoint_url,
            "class_mapping": model_info.class_mapping,
            "image": self._image_to_base64(image),
            "config": {"confidence": confidence, "iou_threshold": iou, ...}
        }

        # 3. Submit to RunPod worker (GPU)
        result = await runpod_service.submit_job_sync(
            endpoint_type=EndpointType.INFERENCE,
            input_data=job_input,
            timeout=120,
        )

        return result

    async def classify(...) -> Dict[str, Any]:
        """Run classification via RunPod GPU worker."""
        # Same pattern

    async def embed(...) -> Dict[str, Any]:
        """Run embedding extraction via RunPod GPU worker."""
        # Same pattern
```

**Status:** ✅ Perfect - just needs to be used by blocks!

---

### RunPod Worker - Ready to Serve

**File:** [workers/inference/handler.py](../workers/inference/handler.py)

Already supports:

**Detection Models:**
- ✅ YOLO11, YOLOv8-10 (pretrained via ultralytics)
- ✅ RT-DETR, D-FINE (pretrained via transformers)
- ✅ Custom trained models from `od_trained_models` table

**Classification Models:**
- ✅ ViT, ConvNeXt, EfficientNet, Swin (pretrained)
- ✅ Custom trained models from `cls_trained_models` table

**Embedding Models:**
- ✅ DINOv2, CLIP, SigLIP (pretrained)
- ✅ Fine-tuned models from `trained_models` table

**Worker Features:**
```python
def handler(job: dict) -> dict:
    """Main handler for unified inference worker."""
    task = job_input.get("task")  # detection | classification | embedding

    # Load model (with caching!)
    model, processor = get_or_load_model(
        task=task,
        model_type=model_type,
        model_source=model_source,
        checkpoint_url=checkpoint_url,
    )

    # Run inference on GPU
    if task == "detection":
        result = run_detection(model, processor, image, config)
    elif task == "classification":
        result = run_classification(model, processor, image, config)
    elif task == "embedding":
        result = run_embedding(model, processor, image, config)

    return {"success": True, "result": result, "metadata": {...}}
```

**Status:** ✅ Fully functional - proven by infrastructure tests

---

## 🎯 SOTA Architecture Design

### Principle: **Zero Local Inference**

```
┌─────────────────────────────────────────────────────┐
│                  API Server (FastAPI)                │
│  • No ML libraries (ultralytics, torch)             │
│  • No GPU required                                   │
│  • Only: PIL, numpy, httpx                          │
│  • Stateless - scales horizontally                  │
└─────────────────────────────────────────────────────┘
                       ↓ (HTTP)
┌─────────────────────────────────────────────────────┐
│              InferenceService (Thin Layer)           │
│  • Model metadata from Supabase                     │
│  • Image → base64 encoding                          │
│  • Job preparation & submission                     │
└─────────────────────────────────────────────────────┘
                       ↓ (HTTP)
┌─────────────────────────────────────────────────────┐
│           RunPod Service (Job Orchestration)         │
│  • Endpoint routing (INFERENCE endpoint)            │
│  • Job submission (/runsync)                        │
│  • Status polling                                   │
└─────────────────────────────────────────────────────┘
                       ↓ (HTTPS)
┌─────────────────────────────────────────────────────┐
│           RunPod Serverless GPU Worker              │
│  Endpoint: yz9lgcxh1rdj9o (24GB Pro GPU)           │
│  • Model caching (stays warm)                       │
│  • GPU inference (CUDA)                             │
│  • All ML libraries (ultralytics, transformers)     │
│  • Checkpoint downloading & caching                 │
│  • Supports: Pretrained + Trained models            │
└─────────────────────────────────────────────────────┘
```

### Block Architecture Pattern

**CORRECT Pattern** (what blocks should do):

```python
class DetectionBlock(ModelBlock):
    """Object Detection Block - delegates to GPU worker."""

    def __init__(self):
        super().__init__()
        # ✅ NO ModelLoader - use InferenceService instead
        from services.workflow.inference_service import get_inference_service
        self._inference_service = get_inference_service()

    async def execute(self, inputs, config, context):
        # 1. Load image (local - fast)
        image = await load_image_from_input(inputs.get("image"))

        # 2. Extract config
        model_id = config.get("model_id")
        model_source = config.get("model_source", "pretrained")
        confidence = config.get("confidence", 0.5)

        # 3. ✅ Delegate to InferenceService (GPU worker)
        result = await self._inference_service.detect(
            model_id=model_id,
            image=image,
            confidence=confidence,
            iou=config.get("iou_threshold", 0.45),
            max_detections=config.get("max_detections", 300),
            model_source=model_source,
        )

        # 4. Format outputs (local - fast)
        detections = result["detections"]

        return BlockResult(
            outputs={
                "detections": detections,
                "count": len(detections),
            },
            duration_ms=round((time.time() - start_time) * 1000, 2),
        )
```

**Key Benefits:**
- ✅ No local ML dependencies
- ✅ GPU acceleration via worker
- ✅ Model caching on worker (fast subsequent calls)
- ✅ Horizontally scalable API
- ✅ Works with pretrained + trained models
- ✅ Simple, maintainable code

---

## 📈 Scalability Design

### For 100s-1000s of Images

#### 1. **Queue-Based Architecture** (Current + Needed)

**Current State:**
```
API → InferenceService → RunPod.submit_job_sync() → Worker
                          (blocks until done)
```

**For Scale:**
```
API → InferenceService → RunPod.submit_job() → Returns job_id immediately
                          ↓
                    Webhook callback when done
                          ↓
                    Update workflow execution status
```

**Implementation:**
```python
class InferenceService:
    async def detect_async(
        self,
        model_id,
        image,
        workflow_id,
        execution_id,
        node_id,
        **kwargs
    ):
        """Async detection with webhook callback."""
        # Prepare webhook URL
        webhook_url = f"{settings.api_base_url}/webhooks/inference/{execution_id}/{node_id}"

        # Submit job (non-blocking)
        job = await runpod_service.submit_job(
            endpoint_type=EndpointType.INFERENCE,
            input_data=job_input,
            webhook_url=webhook_url,
        )

        # Store job metadata
        await self._store_pending_job(
            job_id=job["id"],
            workflow_id=workflow_id,
            execution_id=execution_id,
            node_id=node_id,
        )

        return {"status": "pending", "job_id": job["id"]}
```

#### 2. **Batch Processing**

For workflows processing multiple images:

```python
class DetectionBlock:
    async def execute(self, inputs, config, context):
        images = inputs.get("images", [])

        if len(images) > 10:
            # Use batch mode
            results = await self._inference_service.batch_detect(
                model_id=model_id,
                images=images,
                confidence=confidence,
                # ...
            )
        else:
            # Single/small batch
            results = [
                await self._inference_service.detect(model_id, img, ...)
                for img in images
            ]
```

**Worker Side** (future enhancement):
```python
# workers/inference/handler.py
def handler(job: dict) -> dict:
    images = job_input.get("images", [])  # Support batch

    if len(images) > 1:
        # Batch inference
        results = model.predict(images, ...)
    else:
        # Single inference
        result = model.predict(images[0], ...)
```

#### 3. **RunPod Scaling Configuration**

```yaml
# RunPod Endpoint Config
endpoint_id: yz9lgcxh1rdj9o
gpu: 24GB Pro (RTX 4090 / A5000)

# Scaling params
min_workers: 0        # Serverless (cost-efficient)
max_workers: 10       # Scale to 10 concurrent GPUs
idle_timeout: 60s     # Keep warm for 1 minute
max_queue: 100        # Queue up to 100 jobs

# For high traffic periods
min_workers: 2        # Keep 2 GPUs always warm
max_workers: 20       # Scale to 20 concurrent GPUs
```

**Cost Optimization:**
- **Off-peak:** min_workers=0 (pure serverless, $0 idle cost)
- **Peak hours:** min_workers=2-5 (sub-second latency)
- **Cold start:** ~30-60s first request, then <5s cached

#### 4. **Multiple Queues per Workflow**

Different workflow types can use different worker pools:

```python
class InferenceService:
    def __init__(self):
        self.endpoints = {
            "fast": EndpointType.INFERENCE,      # yz9lgcxh1rdj9o (24GB)
            "heavy": EndpointType.INFERENCE_HEAVY, # Future: 48GB GPU
            "batch": EndpointType.INFERENCE_BATCH, # Future: batch-optimized
        }

    async def detect(self, model_id, image, priority="fast", **kwargs):
        """Route to appropriate endpoint based on priority."""
        endpoint = self.endpoints.get(priority, EndpointType.INFERENCE)
        # ...
```

---

## 🔭 Observability & Traceability

### 1. **Workflow Execution Tracking**

**Current:** ✅ Already implemented in `wf_executions` table

```sql
-- Workflow execution metadata
wf_executions (
  id,
  workflow_id,
  status,  -- pending → running → completed/failed
  started_at,
  completed_at,
  duration_ms,
  error_message,
  input_data,
  output_data,
  node_metrics  -- Per-node timing/metrics
)
```

### 2. **Per-Node Metrics** (Enhanced)

**Proposed Addition:**
```python
class BlockResult:
    outputs: dict
    duration_ms: float
    metrics: dict = {
        # Existing
        "model_id": "yolo11n",
        "detection_count": 5,

        # Add inference tracking
        "inference_job_id": "runpod-job-123",
        "inference_time_ms": 250,  # GPU time
        "model_load_time_ms": 50,  # From worker
        "cached": True,             # Model was cached
        "worker_id": "worker-abc",
        "gpu_model": "RTX 4090",

        # Queue metrics
        "queue_wait_ms": 1500,      # Time waiting for worker
        "total_time_ms": 1800,      # End-to-end
    }
```

### 3. **Distributed Tracing**

Add trace IDs that flow through the entire stack:

```python
# API → InferenceService → RunPod → Worker

# Generate trace ID
trace_id = f"wf-{workflow_id}-exec-{execution_id}-node-{node_id}"

# Pass through entire chain
job_input = {
    "task": "detection",
    "trace_id": trace_id,  # ← Flows to worker
    # ...
}

# Worker logs with trace_id
print(f"[{trace_id}] Loading model: yolo11n")
print(f"[{trace_id}] Inference complete in 250ms")

# Results include trace_id
return {
    "success": True,
    "trace_id": trace_id,
    "result": {...},
}
```

**Benefits:**
- Trace a single image through entire workflow
- Debug failures by searching logs for trace_id
- Measure end-to-end latency per node

### 4. **Monitoring Dashboard Metrics**

Track these KPIs:

**Workflow Level:**
- Workflows created / hour
- Workflows executed / hour
- Success rate (%)
- P50, P95, P99 execution times
- Error rate by error type

**Inference Level:**
- Inference jobs / hour (per model type)
- Average GPU inference time (detection, classification, embedding)
- Worker cold start frequency
- Model cache hit rate
- Queue depth (jobs waiting)
- Worker utilization (%)

**Cost Tracking:**
- GPU hours consumed
- Cost per workflow execution
- Cost per 1000 inferences
- Idle vs active time

### 5. **Logging Strategy**

**API Logs:**
```python
logger.info(
    f"Workflow execution started",
    extra={
        "workflow_id": workflow_id,
        "execution_id": execution_id,
        "node_count": len(nodes),
        "user_id": user_id,
    }
)

logger.info(
    f"Inference job submitted",
    extra={
        "execution_id": execution_id,
        "node_id": node_id,
        "model_id": model_id,
        "runpod_job_id": job_id,
    }
)
```

**Worker Logs:**
```python
print(f"[INFERENCE] Task: {task}, Model: {model_type}, Source: {model_source}")
print(f"[CACHE] Model cache {'HIT' if cached else 'MISS'}: {cache_key}")
print(f"[TIMING] Model load: {load_time_ms:.0f}ms, Inference: {inference_time_ms:.0f}ms")
print(f"[RESULT] Detection: {count} objects, Classes: {unique_classes}")
```

---

## 🚀 Implementation Plan

### Phase 1: Fix Model Blocks (Critical - 1 day)

**Files to Modify:**
1. [model_blocks.py:97-813](../apps/api/src/services/workflow/blocks/model_blocks.py:97) - DetectionBlock
2. [model_blocks.py:815-1340](../apps/api/src/services/workflow/blocks/model_blocks.py:815) - ClassificationBlock
3. [model_blocks.py:1342-1750](../apps/api/src/services/workflow/blocks/model_blocks.py:1342) - EmbeddingBlock

**Changes:**
```python
# BEFORE (all 3 blocks)
def __init__(self):
    super().__init__()
    self._model_loader = get_model_loader()  # ❌ Remove

async def execute(self, inputs, config, context):
    model, model_info = await self._model_loader.load_detection_model(...)  # ❌ Remove
    results = model.predict(...)  # ❌ Remove local inference

# AFTER (all 3 blocks)
def __init__(self):
    super().__init__()
    from services.workflow.inference_service import get_inference_service
    self._inference_service = get_inference_service()  # ✅ Add

async def execute(self, inputs, config, context):
    # ✅ Use InferenceService
    result = await self._inference_service.detect(  # or classify() or embed()
        model_id=config.get("model_id"),
        image=image,
        confidence=config.get("confidence", 0.5),
        # ... other config params
    )
```

**Testing:**
```bash
cd apps/api
python tests/test_real_inference.py

# Expected: All 8 tests pass with GPU inference
# ✅ Test 1: YOLO Detection - PASS (worker GPU)
# ✅ Test 2: Detection + Visualization - PASS
# ✅ Test 3: DINOv2 Embedding - PASS (worker GPU)
# ✅ Test 4: Parallel Multi-Model - PASS
# ✅ Test 5: ViT Classification - PASS (worker GPU)
# ✅ Test 6: Detection + Filter - PASS
# ✅ Test 7: Transform Pipeline - PASS
# ✅ Test 8: Conditional Logic - PASS
```

### Phase 2: Enhanced Observability (Optional - 2 days)

1. Add trace_id propagation
2. Add detailed metrics to BlockResult
3. Create monitoring dashboard (Grafana/Datadog)
4. Set up alerting for failures

### Phase 3: Async/Queue Mode (Optional - 3 days)

1. Implement webhook callbacks
2. Add async job tracking
3. Support batch processing
4. Queue management UI

---

## ✅ Success Criteria

### Immediate (Phase 1)
- ✅ All 8 workflow tests pass
- ✅ No local ML inference on API server
- ✅ All inference via RunPod worker (GPU)
- ✅ API can run without torch/ultralytics installed

### Short-term
- ✅ 100 concurrent workflow executions
- ✅ <5s average execution time (cached models)
- ✅ <30s cold start time
- ✅ 99.5% success rate

### Long-term (Scale)
- ✅ 1000+ images/hour throughput
- ✅ <$0.01 cost per inference (serverless)
- ✅ Multi-region worker deployment
- ✅ Auto-scaling based on queue depth

---

## 📝 Recommendations

### DO ✅
1. **Fix all model blocks immediately** - Use InferenceService
2. **Remove local inference code** - Delete ModelLoader usage from blocks
3. **Keep transform blocks CPU-based** - They're fast, no GPU needed
4. **Use RunPod serverless** - Cost-efficient, scales automatically
5. **Add trace IDs** - Essential for debugging at scale
6. **Monitor worker metrics** - Cache hit rate, queue depth, GPU utilization

### DON'T ❌
1. **Don't install ML libraries on API server** - Keep it lightweight
2. **Don't over-engineer** - Start with sync mode (submit_job_sync)
3. **Don't create custom queue system yet** - Use RunPod's built-in queue
4. **Don't optimize prematurely** - Fix architecture first, then optimize
5. **Don't cache on API side** - Worker caching is sufficient

### AVOID ⚠️
1. **Hybrid local/remote** - All or nothing (choose remote)
2. **Multiple inference backends** - Stick with RunPod
3. **Complex retry logic** - RunPod handles this
4. **Manual load balancing** - RunPod auto-scales

---

## 🎯 Conclusion

**Current State:** System has solid foundation (InferenceService, Worker) but blocks are using wrong architecture (local inference).

**Required Action:** **Simple refactor of 3 blocks** to use InferenceService instead of ModelLoader.

**Expected Outcome:**
- ✅ All tests pass
- ✅ GPU acceleration
- ✅ Production-ready scalability
- ✅ <1 day implementation

**This is NOT over-engineering** - it's fixing a fundamental architectural mistake. The SOTA approach is already designed (InferenceService), just needs to be used.
