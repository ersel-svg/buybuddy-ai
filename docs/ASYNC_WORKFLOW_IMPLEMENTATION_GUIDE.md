# Async Workflow Implementation Guide

**Date:** 2026-01-26
**Target:** Production-ready async workflow system
**Pattern:** Based on proven OD Training architecture
**Timeline:** 2-3 days implementation

---

## 📋 Overview

Transform the workflow system from **sync (blocking)** to **async (non-blocking)** GPU inference using the same proven pattern as OD Training.

### Current Problems
❌ API blocks while waiting for GPU
❌ Can't scale beyond ~10 concurrent workflows
❌ No real-time progress tracking
❌ Worker sits idle while API tries local inference

### Solution
✅ API submits job and returns immediately
✅ Worker writes progress to Supabase directly
✅ Frontend polls status (like training dashboard)
✅ Scale to 1000+ concurrent workflows

---

## 🎯 Architecture Comparison

### BEFORE (Sync - Current)
```
POST /workflows/{id}/run
  ↓
API waits for GPU (30-60s)
  ↓
Returns result
```
**Problem:** API thread blocked, can't scale

### AFTER (Async - Target)
```
POST /workflows/{id}/run
  ↓
API submits job → Returns immediately (execution_id)
  ↓
Background task polls wf_inference_jobs
  ↓
Worker writes to Supabase (no webhooks)
  ↓
Frontend polls execution status
```
**Benefit:** 1000s of concurrent executions

---

## 🗄️ Database Changes

### Step 1: Run Migration

```bash
cd apps/api
# Run migration
psql $DATABASE_URL -f migrations/add_inference_jobs_table.sql
```

**What it creates:**
1. `wf_inference_jobs` table - Tracks each inference job
2. Columns in `wf_executions` - Job counters
3. Indexes - Fast status/execution queries
4. Triggers - Auto-update `updated_at`

### Step 2: Verify Schema

```sql
-- Check table exists
SELECT * FROM wf_inference_jobs LIMIT 0;

-- Check new columns
SELECT
    inference_jobs_pending,
    inference_jobs_completed,
    inference_jobs_failed
FROM wf_executions LIMIT 1;
```

---

## 🔧 Code Implementation

### Files to Modify

#### 1. **InferenceService** (Primary Change)
**File:** `apps/api/src/services/workflow/inference_service.py`

**Add these methods:**

```python
async def detect_async(
    self,
    execution_id: str,
    workflow_id: str,
    node_id: str,
    model_id: str,
    image: Image.Image,
    **kwargs
) -> str:
    """Submit detection job, return job_id immediately."""
    # 1. Get model info from DB
    # 2. Create wf_inference_jobs record (status='pending')
    # 3. Submit to RunPod (non-blocking)
    # 4. Update with runpod_job_id (status='queued')
    # 5. Return job_id

async def get_job_result(self, job_id: str, timeout: int = 120) -> Dict:
    """Poll wf_inference_jobs until completed/failed."""
    # Poll every 2s until status changes
    # Return result or error
```

**Similar for:** `classify_async()`, `embed_async()`

#### 2. **DetectionBlock** (Refactor)
**File:** `apps/api/src/services/workflow/blocks/model_blocks.py:97-813`

**REMOVE:**
```python
# ❌ DELETE THIS
model, model_info = await self._model_loader.load_detection_model(...)
results = model.predict(...)
```

**ADD:**
```python
# ✅ ADD THIS
from services.workflow.inference_service import get_inference_service

class DetectionBlock:
    def __init__(self):
        super().__init__()
        self._inference_service = get_inference_service()

    async def execute(self, inputs, config, context):
        # 1. Load image (local)
        image = await load_image_from_input(inputs.get("image"))

        # 2. Submit to worker
        job_id = await self._inference_service.detect_async(
            execution_id=context.execution_id,
            workflow_id=context.workflow_id,
            node_id=context.node_id,
            model_id=config.get("model_id"),
            image=image,
            confidence=config.get("confidence", 0.5),
            # ... other config
        )

        # 3. Wait for result
        result = await self._inference_service.get_job_result(job_id)

        # 4. Return formatted outputs
        return BlockResult(outputs=result["result"], ...)
```

#### 3. **ClassificationBlock** (Refactor)
**File:** Same file, lines 815-1340

**Same pattern as DetectionBlock:**
- Remove `_model_loader` usage
- Add `_inference_service`
- Use `classify_async()` + `get_job_result()`

#### 4. **EmbeddingBlock** (Refactor)
**File:** Same file, lines 1342-1750

**Same pattern:**
- Remove local torch inference
- Use `embed_async()` + `get_job_result()`

#### 5. **Worker Handler** (Add Supabase Writes)
**File:** `workers/inference/handler.py`

**ADD this function:**
```python
def update_inference_job(
    execution_id: str,
    node_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    metadata: Optional[Dict] = None,
):
    """Write job status to Supabase (same as training pattern)."""
    from supabase import create_client

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        return

    client = create_client(supabase_url, supabase_key)

    # Find job by execution_id + node_id
    jobs = client.table("wf_inference_jobs").select("id").eq(
        "execution_id", execution_id
    ).eq("node_id", node_id).execute()

    if not jobs.data:
        return

    job_id = jobs.data[0]["id"]

    # Build update
    update_data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if status == "running":
        update_data["started_at"] = datetime.now(timezone.utc).isoformat()

    if status in ("completed", "failed"):
        update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

    if result:
        update_data["result"] = result

    if error:
        update_data["error_message"] = error

    if duration_ms:
        update_data["duration_ms"] = int(duration_ms)

    if metadata:
        update_data["metadata"] = metadata

    # Update
    client.table("wf_inference_jobs").update(update_data).eq(
        "id", job_id
    ).execute()
```

**UPDATE handler():**
```python
def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    # Extract context
    exec_ctx = job_input.get("execution_context", {})
    execution_id = exec_ctx.get("execution_id")
    node_id = exec_ctx.get("node_id")

    # Set Supabase env
    if exec_ctx.get("supabase_url"):
        os.environ["SUPABASE_URL"] = exec_ctx["supabase_url"]
        os.environ["SUPABASE_SERVICE_KEY"] = exec_ctx["supabase_service_key"]

    try:
        # Update: running
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="running",
            )

        # ... existing inference code ...

        # Update: completed
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="completed",
                result=result,
                duration_ms=inference_time,
                metadata={...}
            )

        return {"success": True, "result": result}

    except Exception as e:
        # Update: failed
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="failed",
                error=str(e),
            )

        return {"success": False, "error": str(e)}
```

#### 6. **ExecutionContext** (Add fields)
**File:** `apps/api/src/services/workflow/base.py`

```python
@dataclass
class ExecutionContext:
    """Execution context passed to blocks."""
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    node_id: Optional[str] = None  # ✅ ADD THIS
    variables: dict = field(default_factory=dict)
```

#### 7. **Workflow Engine** (Pass context)
**File:** `apps/api/src/services/workflow/engine.py`

```python
async def execute(self, workflow, inputs, workflow_id, execution_id):
    # ...

    for node_id in execution_order:
        node = next(n for n in nodes if n["id"] == node_id)

        # ✅ Update context with current node
        context.node_id = node_id

        result = await self._execute_node(node, node_results, inputs, context)
        # ...
```

---

## 🧪 Testing

### Step 1: Unit Test Each Block

```bash
cd apps/api

# Test DetectionBlock
python -c "
import asyncio
from services.workflow.blocks.model_blocks import DetectionBlock
from services.workflow.base import ExecutionContext
from PIL import Image

async def test():
    block = DetectionBlock()
    context = ExecutionContext(
        execution_id='test-exec-123',
        workflow_id='test-wf-456',
        node_id='detect_1'
    )

    # Create test image
    img = Image.new('RGB', (640, 480), color='red')

    result = await block.execute(
        inputs={'image': img},
        config={
            'model_id': 'yolo11n',
            'model_source': 'pretrained',
            'confidence': 0.5,
        },
        context=context
    )

    print(f'Status: {result.error or \"SUCCESS\"}')
    print(f'Detections: {len(result.outputs.get(\"detections\", []))}')

asyncio.run(test())
"
```

### Step 2: Run Full Workflow Tests

```bash
# Run all 8 workflow tests
python tests/test_real_inference.py
```

**Expected Output:**
```
================================================================================
REAL INFERENCE TESTS - Async GPU Worker
================================================================================
Testing 8 comprehensive workflows with async RunPod inference
Started: 2026-01-26T12:00:00
================================================================================

Test 1: Simple YOLO Detection
✓ Workflow created: abc123...
✓ Job submitted: job-456...
✓ Status: completed (Duration: 5240ms)
✅ SUCCESS - 5 detections found

Test 2: Detection + Visualization
✓ Workflow created: def456...
✓ Job submitted: job-789...
✓ Status: completed (Duration: 5180ms)
✅ SUCCESS - Annotated image generated

...

Test 8: Conditional Logic
✓ Workflow created: xyz789...
✓ Job submitted: job-012...
✓ Status: completed (Duration: 5320ms)
✅ SUCCESS - Condition evaluated

================================================================================
TEST SUMMARY
================================================================================
Total: 8
Passed: 8 ✅
Failed: 0
Success Rate: 100.0%
================================================================================
```

### Step 3: Test Concurrent Execution

```python
# test_concurrent.py
import asyncio
import httpx

async def run_workflow(workflow_id, image_base64):
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"http://localhost:8000/api/v1/workflows/{workflow_id}/run",
            json={"input": {"image_base64": image_base64}}
        )
        return resp.json()

async def test_concurrent():
    # Run 50 workflows concurrently
    tasks = [
        run_workflow("workflow-id", "base64-image")
        for _ in range(50)
    ]

    results = await asyncio.gather(*tasks)

    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"Completed: {completed}/50")

asyncio.run(test_concurrent())
```

### Step 4: Monitor Database

```sql
-- Check inference jobs
SELECT
    execution_id,
    node_id,
    task,
    status,
    duration_ms,
    submitted_at,
    completed_at
FROM wf_inference_jobs
ORDER BY submitted_at DESC
LIMIT 20;

-- Check execution summary
SELECT
    e.id,
    e.status,
    e.duration_ms,
    e.inference_jobs_pending,
    e.inference_jobs_completed,
    e.inference_jobs_failed
FROM wf_executions e
ORDER BY e.created_at DESC
LIMIT 10;
```

---

## 📊 Observability

### Metrics to Track

**Workflow Level:**
```sql
-- Success rate by hour
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / COUNT(*), 2) as success_rate
FROM wf_executions
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;
```

**Inference Job Level:**
```sql
-- Average inference time by model
SELECT
    model_id,
    task,
    COUNT(*) as jobs,
    ROUND(AVG(duration_ms), 0) as avg_duration_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms), 0) as p95_duration_ms,
    COUNT(*) FILTER (WHERE (metadata->>'cached')::boolean) as cached_count
FROM wf_inference_jobs
WHERE completed_at > NOW() - INTERVAL '24 hours'
  AND status = 'completed'
GROUP BY model_id, task
ORDER BY jobs DESC;
```

**Worker Utilization:**
```sql
-- Jobs per status
SELECT
    status,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
FROM wf_inference_jobs
WHERE submitted_at > NOW() - INTERVAL '1 hour'
GROUP BY status;
```

### Logging

**API Logs:**
```python
logger.info(
    "Inference job submitted",
    extra={
        "execution_id": execution_id,
        "node_id": node_id,
        "job_id": job_id,
        "model_id": model_id,
        "task": task,
    }
)
```

**Worker Logs:**
```python
print(f"[INFERENCE] Job: {job_id}, Model: {model_id}, Task: {task}")
print(f"[TIMING] Load: {load_time_ms:.0f}ms, Inference: {inference_time_ms:.0f}ms")
print(f"[RESULT] {task}: {result_summary}")
```

---

## 🚀 Deployment

### Step 1: Deploy Worker

```bash
cd workers/inference

# Build and push
docker build -t your-registry/inference-worker:latest .
docker push your-registry/inference-worker:latest

# Update RunPod endpoint with new image
# (via RunPod dashboard or API)
```

### Step 2: Deploy API

```bash
cd apps/api

# Run migration
python scripts/run_migration.py migrations/add_inference_jobs_table.sql

# Deploy API (your deployment process)
# e.g., docker build, k8s apply, etc.
```

### Step 3: Configure RunPod

```yaml
# RunPod Endpoint Settings
endpoint_id: yz9lgcxh1rdj9o

# Scaling
min_workers: 0      # Serverless (cost-efficient)
max_workers: 10     # Scale up to 10 GPUs
idle_timeout: 60s   # Keep warm for 1 min

# For high-traffic periods:
min_workers: 2      # Always warm
max_workers: 20     # More capacity
```

### Step 4: Environment Variables

**Worker:**
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

**API:**
```bash
RUNPOD_API_KEY=your-runpod-key
RUNPOD_INFERENCE_ENDPOINT_ID=yz9lgcxh1rdj9o
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
```

---

## ✅ Success Criteria

### Day 1
- [x] Database migration completed
- [x] InferenceService has async methods
- [x] DetectionBlock refactored
- [x] Worker writes to Supabase
- [x] 1 test passes end-to-end

### Day 2
- [ ] ClassificationBlock refactored
- [ ] EmbeddingBlock refactored
- [ ] All 8 tests pass
- [ ] Concurrent test (10 workflows)

### Day 3
- [ ] Production deployment
- [ ] Monitoring dashboard
- [ ] 100+ concurrent test
- [ ] Documentation complete

### Production Ready
- [ ] 99% success rate
- [ ] <5s average latency (cached)
- [ ] 100+ concurrent executions
- [ ] Cost monitoring active

---

## 🎯 Next Steps

1. **Review this guide** ✅ (you're here!)
2. **Run database migration** → Creates infrastructure
3. **Implement InferenceService async methods** → Core logic
4. **Refactor 3 model blocks** → Use new service
5. **Update worker handler** → Supabase writes
6. **Test thoroughly** → All 8 tests + concurrent
7. **Deploy to production** → Monitor closely

---

## 📚 Related Documentation

- [ASYNC_WORKFLOW_ARCHITECTURE.md](./ASYNC_WORKFLOW_ARCHITECTURE.md) - Full architecture design
- [WORKFLOW_BLOCK_ANALYSIS.md](./WORKFLOW_BLOCK_ANALYSIS.md) - Current state analysis
- [REAL_INFERENCE_TEST_REPORT.md](./REAL_INFERENCE_TEST_REPORT.md) - Test results

---

**This guide is implementation-ready.** Follow step by step and you'll have a production-grade async workflow system in 2-3 days.
