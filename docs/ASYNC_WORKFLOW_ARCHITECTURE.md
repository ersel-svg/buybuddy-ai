# Async Workflow Inference Architecture - SOTA Design

**Date:** 2026-01-26
**Based on:** OD Training async pattern (proven in production)
**Goal:** Scale to 1000s of concurrent workflow executions

---

## 🎯 Core Philosophy

**"Worker writes back, not webhooks"**

Same pattern as OD Training:
1. API submits job to RunPod (non-blocking)
2. Worker writes progress directly to Supabase
3. Frontend polls Supabase for status updates
4. No webhook dependencies = more reliable

---

## 📊 Database Schema

### New: `wf_inference_jobs` Table

```sql
CREATE TABLE wf_inference_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Workflow context
    execution_id UUID NOT NULL REFERENCES wf_executions(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,  -- Node ID in workflow definition

    -- Job details
    task TEXT NOT NULL,  -- 'detection', 'classification', 'embedding'
    model_id TEXT NOT NULL,
    model_source TEXT NOT NULL DEFAULT 'pretrained',
    runpod_job_id TEXT,

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending',  -- pending → queued → running → completed/failed
    progress INT DEFAULT 0,  -- 0-100

    -- Timing
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INT,

    -- Results (lightweight - large data goes to wf_executions.output_data)
    result JSONB,
    error_message TEXT,

    -- Metadata
    metadata JSONB,  -- {worker_id, gpu_model, cached, queue_wait_ms, ...}

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX idx_wf_inference_jobs_execution ON wf_inference_jobs(execution_id);
CREATE INDEX idx_wf_inference_jobs_status ON wf_inference_jobs(status);
CREATE INDEX idx_wf_inference_jobs_runpod ON wf_inference_jobs(runpod_job_id);
CREATE INDEX idx_wf_inference_jobs_submitted ON wf_inference_jobs(submitted_at DESC);
```

### Enhanced: `wf_executions` Table

```sql
-- Already exists, add new fields:
ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS
    inference_jobs_pending INT DEFAULT 0;

ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS
    inference_jobs_completed INT DEFAULT 0;

ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS
    inference_jobs_failed INT DEFAULT 0;
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  • Creates workflow execution                                │
│  • Polls wf_executions.status every 2s                      │
│  • Displays real-time progress per node                      │
└─────────────────────────────────────────────────────────────┘
                           ↓ POST /workflows/{id}/run
┌─────────────────────────────────────────────────────────────┐
│                      API Server (FastAPI)                    │
│  1. Create wf_execution record (status='pending')           │
│  2. Start background task: process_workflow()               │
│  3. Return execution_id immediately                          │
└─────────────────────────────────────────────────────────────┘
                           ↓ BackgroundTasks
┌─────────────────────────────────────────────────────────────┐
│               Workflow Engine (Background Task)              │
│  • Topological sort of nodes                                │
│  • For each node:                                            │
│    - If model block: submit_inference_job() → RunPod        │
│    - If transform block: execute locally (fast)             │
│  • Update wf_execution.status = 'running'                   │
└─────────────────────────────────────────────────────────────┘
                           ↓ RunPod.submit_job()
┌─────────────────────────────────────────────────────────────┐
│               RunPod Inference Worker (GPU)                  │
│  • Receives job from RunPod queue                           │
│  • Updates wf_inference_jobs via Supabase:                  │
│    - status='running', started_at                           │
│  • Loads model (with caching)                               │
│  • Runs GPU inference                                        │
│  • Writes result to Supabase:                               │
│    - wf_inference_jobs.result                               │
│    - wf_inference_jobs.status='completed'                   │
│    - wf_inference_jobs.duration_ms                          │
│  • Returns success to RunPod                                 │
└─────────────────────────────────────────────────────────────┘
                           ↓ Supabase writes
┌─────────────────────────────────────────────────────────────┐
│               Workflow Engine (Polling Loop)                 │
│  • Polls wf_inference_jobs for completion                   │
│  • When all jobs done:                                       │
│    - Collect results                                         │
│    - Execute remaining nodes                                 │
│    - Update wf_execution.status='completed'                 │
│    - Update wf_execution.output_data                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
                   Frontend sees status='completed'
```

---

## 🔧 Implementation

### 1. Enhanced InferenceService

```python
# apps/api/src/services/workflow/inference_service.py

class InferenceService:
    """Async inference service with Supabase job tracking."""

    async def detect_async(
        self,
        execution_id: str,
        workflow_id: str,
        node_id: str,
        model_id: str,
        image: Image.Image,
        confidence: float = 0.5,
        **kwargs
    ) -> str:
        """
        Submit detection job asynchronously.

        Returns:
            job_id: UUID of wf_inference_jobs record
        """
        # 1. Get model info
        model_info = await self.model_loader.get_detection_model_info(
            model_id, kwargs.get("model_source", "pretrained")
        )

        # 2. Prepare job input for worker
        job_input = {
            "task": "detection",
            "model_id": model_id,
            "model_type": model_info.model_type,
            "model_source": kwargs.get("model_source", "pretrained"),
            "checkpoint_url": model_info.checkpoint_url,
            "class_mapping": model_info.class_mapping,
            "image": self._image_to_base64(image),
            "config": {
                "confidence": confidence,
                "iou_threshold": kwargs.get("iou", 0.45),
                "max_detections": kwargs.get("max_detections", 300),
            },
            # NEW: Supabase context for worker writes
            "execution_context": {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "node_id": node_id,
                "supabase_url": settings.supabase_url,
                "supabase_service_key": settings.supabase_service_role_key,
            }
        }

        # 3. Create inference job record
        from services.supabase import supabase_service
        job_record = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "node_id": node_id,
            "task": "detection",
            "model_id": model_id,
            "model_source": kwargs.get("model_source", "pretrained"),
            "status": "pending",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }

        result = supabase_service.client.table("wf_inference_jobs").insert(
            job_record
        ).execute()

        job_id = result.data[0]["id"]

        # 4. Submit to RunPod (async, non-blocking)
        try:
            runpod_result = await runpod_service.submit_job(
                endpoint_type=EndpointType.INFERENCE,
                input_data=job_input,
                webhook_url=None,  # Worker writes to Supabase directly
            )

            # 5. Update with RunPod job ID
            supabase_service.client.table("wf_inference_jobs").update({
                "runpod_job_id": runpod_result.get("id"),
                "status": "queued",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job_id).execute()

            logger.info(
                f"Inference job submitted: job_id={job_id}, "
                f"runpod_job_id={runpod_result.get('id')}, "
                f"node={node_id}, model={model_id}"
            )

        except Exception as e:
            # Mark job as failed
            supabase_service.client.table("wf_inference_jobs").update({
                "status": "failed",
                "error_message": str(e),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job_id).execute()
            raise

        return job_id

    async def get_job_result(self, job_id: str, timeout: int = 120) -> Dict:
        """
        Wait for inference job completion by polling Supabase.

        Args:
            job_id: wf_inference_jobs.id
            timeout: Max wait time in seconds

        Returns:
            Job result dict
        """
        from services.supabase import supabase_service
        import asyncio

        start_time = time.time()
        poll_interval = 2  # seconds

        while time.time() - start_time < timeout:
            # Poll job status
            result = supabase_service.client.table("wf_inference_jobs").select(
                "*"
            ).eq("id", job_id).single().execute()

            job = result.data
            status = job["status"]

            if status == "completed":
                return {
                    "success": True,
                    "result": job["result"],
                    "duration_ms": job["duration_ms"],
                    "metadata": job["metadata"],
                }

            elif status == "failed":
                return {
                    "success": False,
                    "error": job["error_message"],
                }

            # Still running, wait and poll again
            await asyncio.sleep(poll_interval)

        # Timeout
        return {
            "success": False,
            "error": f"Inference job timeout after {timeout}s",
        }

    # Similar methods for classify_async() and embed_async()
```

### 2. Refactored DetectionBlock

```python
# apps/api/src/services/workflow/blocks/model_blocks.py

class DetectionBlock(ModelBlock):
    """
    Object Detection Block - Async GPU Inference

    Uses RunPod worker with Supabase progress tracking.
    No local inference - fully serverless.
    """

    def __init__(self):
        super().__init__()
        from services.workflow.inference_service import get_inference_service
        self._inference_service = get_inference_service()

    async def execute(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        context: ExecutionContext,
    ) -> BlockResult:
        """Run detection via RunPod GPU worker."""
        start_time = time.time()

        # 1. Load image (local - fast)
        image = await load_image_from_input(inputs.get("image"))
        if image is None:
            return BlockResult(
                error="Failed to load input image",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )

        # 2. Extract config
        model_id = config.get("model_id")
        model_source = config.get("model_source", "pretrained")
        confidence = config.get("confidence", 0.5)
        iou = config.get("iou_threshold", 0.45)
        max_det = config.get("max_detections", 300)

        try:
            # 3. Submit to RunPod worker (async)
            job_id = await self._inference_service.detect_async(
                execution_id=context.execution_id,
                workflow_id=context.workflow_id,
                node_id=context.node_id,
                model_id=model_id,
                image=image,
                confidence=confidence,
                iou=iou,
                max_detections=max_det,
                model_source=model_source,
            )

            logger.info(f"Detection job submitted: {job_id}")

            # 4. Wait for completion (polling Supabase)
            result = await self._inference_service.get_job_result(
                job_id, timeout=120
            )

            if not result["success"]:
                return BlockResult(
                    error=f"Detection failed: {result['error']}",
                    duration_ms=round((time.time() - start_time) * 1000, 2),
                )

            # 5. Extract detections
            inference_result = result["result"]
            detections = inference_result.get("detections", [])

            # 6. Return results
            duration = (time.time() - start_time) * 1000

            return BlockResult(
                outputs={
                    "detections": detections,
                    "count": len(detections),
                    "image_size": inference_result.get("image_size"),
                },
                duration_ms=round(duration, 2),
                metrics={
                    "model_id": model_id,
                    "model_source": model_source,
                    "detection_count": len(detections),
                    "confidence_threshold": confidence,
                    "inference_job_id": job_id,
                    "gpu_inference_ms": result.get("duration_ms", 0),
                    "total_ms": round(duration, 2),
                },
            )

        except Exception as e:
            logger.exception("Detection block failed")
            return BlockResult(
                error=f"Detection failed: {str(e)}",
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
```

### 3. Enhanced Workflow Engine

```python
# apps/api/src/services/workflow/engine.py

class WorkflowEngine:
    """
    Executes workflows with async inference support.
    """

    async def execute(
        self,
        workflow: dict,
        inputs: dict,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> dict:
        """
        Execute workflow with async GPU inference.

        Model blocks submit jobs and wait for completion.
        Transform blocks execute locally (fast).
        """
        start_time = time.time()

        # 1. Create execution context
        context = ExecutionContext(
            workflow_id=workflow_id,
            execution_id=execution_id,
            variables={},
        )

        # 2. Update execution status
        if execution_id:
            supabase_service.client.table("wf_executions").update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", execution_id).execute()

        try:
            # 3. Get nodes and edges
            nodes = workflow.get("nodes", [])
            edges = workflow.get("edges", [])

            # 4. Build execution graph
            graph = self._build_graph(nodes, edges)
            execution_order = self._topological_sort(graph, nodes)

            # 5. Execute nodes in order
            node_results = {}
            node_metrics = {}

            for node_id in execution_order:
                node = next(n for n in nodes if n["id"] == node_id)

                # Update context with current node
                context.node_id = node_id

                # Execute node
                logger.info(f"Executing node: {node_id} ({node['type']})")

                result = await self._execute_node(
                    node, node_results, inputs, context
                )

                node_results[node_id] = result
                node_metrics[node_id] = {
                    "duration_ms": result.duration_ms,
                    "output_count": len(result.outputs) if result.outputs else 0,
                }

                if result.error:
                    logger.error(f"Node {node_id} failed: {result.error}")
                    raise RuntimeError(f"Node {node_id} failed: {result.error}")

            # 6. Collect final outputs
            output_nodes = workflow.get("outputs", [])
            final_outputs = {}

            for output_spec in output_nodes:
                source_node = output_spec.get("source_node")
                source_handle = output_spec.get("source_handle")

                if source_node in node_results:
                    result = node_results[source_node]
                    if result.outputs and source_handle in result.outputs:
                        final_outputs[output_spec["name"]] = result.outputs[source_handle]

            # 7. Update execution record
            duration = (time.time() - start_time) * 1000

            if execution_id:
                supabase_service.client.table("wf_executions").update({
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration),
                    "output_data": final_outputs,
                    "node_metrics": node_metrics,
                }).eq("id", execution_id).execute()

            return {
                "success": True,
                "outputs": final_outputs,
                "duration_ms": round(duration),
                "node_metrics": node_metrics,
            }

        except Exception as e:
            logger.exception("Workflow execution failed")

            if execution_id:
                supabase_service.client.table("wf_executions").update({
                    "status": "failed",
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.time() - start_time) * 1000),
                }).eq("id", execution_id).execute()

            return {
                "success": False,
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000),
            }
```

### 4. Enhanced Worker (Writes Back to Supabase)

```python
# workers/inference/handler.py

def update_inference_job(
    execution_id: str,
    node_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None,
    metadata: Optional[Dict] = None,
):
    """
    Update inference job in Supabase (same pattern as training).
    Worker writes directly - no webhook needed.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        print("[WARNING] Supabase not configured, skipping job update")
        return

    from supabase import create_client
    client = create_client(supabase_url, supabase_key)

    try:
        # Find job record
        jobs = client.table("wf_inference_jobs").select("id").eq(
            "execution_id", execution_id
        ).eq("node_id", node_id).execute()

        if not jobs.data:
            print(f"[WARNING] Inference job not found: {execution_id}/{node_id}")
            return

        job_id = jobs.data[0]["id"]

        # Build update data
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

        # Update job
        client.table("wf_inference_jobs").update(update_data).eq(
            "id", job_id
        ).execute()

        print(f"[INFERENCE] Job updated: {job_id} → {status}")

    except Exception as e:
        print(f"[ERROR] Failed to update inference job: {e}")


def handler(job: dict) -> dict:
    """Main inference handler with Supabase writes."""
    job_input = job.get("input", {})

    # Extract execution context
    exec_ctx = job_input.get("execution_context", {})
    execution_id = exec_ctx.get("execution_id")
    node_id = exec_ctx.get("node_id")

    # Set Supabase env for update_inference_job()
    if exec_ctx.get("supabase_url"):
        os.environ["SUPABASE_URL"] = exec_ctx["supabase_url"]
        os.environ["SUPABASE_SERVICE_KEY"] = exec_ctx["supabase_service_key"]

    try:
        # Update status: running
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="running",
            )

        # Load image
        image = load_image_from_base64(job_input["image"])

        # Get model
        start_time = time.time()
        model, processor = get_or_load_model(
            task=job_input["task"],
            model_type=job_input["model_type"],
            model_source=job_input["model_source"],
            checkpoint_url=job_input.get("checkpoint_url"),
        )
        load_time = (time.time() - start_time) * 1000

        # Run inference
        start_time = time.time()
        if job_input["task"] == "detection":
            result = run_detection(model, processor, image, job_input["config"])
        elif job_input["task"] == "classification":
            result = run_classification(model, processor, image, job_input["config"])
        elif job_input["task"] == "embedding":
            result = run_embedding(model, processor, image, job_input["config"])

        inference_time = (time.time() - start_time) * 1000

        # Update status: completed
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="completed",
                result=result,
                duration_ms=inference_time,
                metadata={
                    "model_load_time_ms": load_time,
                    "inference_time_ms": inference_time,
                    "cached": load_time < 1000,  # Cached if loaded <1s
                    "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                }
            )

        return {
            "success": True,
            "result": result,
            "metadata": {
                "inference_time_ms": inference_time,
                "model_load_time_ms": load_time,
            }
        }

    except Exception as e:
        # Update status: failed
        if execution_id and node_id:
            update_inference_job(
                execution_id=execution_id,
                node_id=node_id,
                status="failed",
                error=str(e),
            )

        return {
            "success": False,
            "error": str(e),
        }


# Register handler
runpod.serverless.start({"handler": handler})
```

---

## 📊 Scalability Analysis

### Current Sync Mode (Before)
```
Request → API blocks → Wait for GPU → Return result
            ↑
      Blocks until done
      Can't handle concurrent requests well
```

**Limits:**
- 1 request = 1 GPU worker blocked
- Max ~10 concurrent (limited by API threads)
- No visibility into progress

### Async Mode (After)
```
Request → API submits job → Return immediately
            ↓
        Background task polls
            ↓
        Worker writes to Supabase
            ↓
        Frontend polls status
```

**Benefits:**
- ✅ 1000s of concurrent executions
- ✅ RunPod auto-scales workers (0→10+)
- ✅ API stays lightweight
- ✅ Real-time progress tracking
- ✅ Failed jobs don't block queue

### Cost Model

**Serverless Scaling:**
```yaml
# Low traffic (0-10 req/min)
min_workers: 0
cost: $0/hour idle

# Medium traffic (10-100 req/min)
workers: 1-3 (auto-scaled)
cost: ~$1.50/hour

# High traffic (100-1000 req/min)
workers: 5-10 (auto-scaled)
cost: ~$5/hour

# Spike traffic (1000+ req/min)
workers: 10-20 (auto-scaled)
cost: ~$10/hour
```

**Cost per inference:**
- Detection (YOLO): ~$0.001 (1ms GPU)
- Classification (ViT): ~$0.002 (2ms GPU)
- Embedding (DINOv2): ~$0.003 (3ms GPU)

**Example: 10,000 workflows/day**
- Average 2 model nodes per workflow
- 20,000 inferences/day
- Cost: ~$40/day (~$1,200/month)
- With serverless idle: ~$800/month

---

## ✅ Implementation Checklist

### Phase 1: Database & Service (Day 1)
- [ ] Create `wf_inference_jobs` table
- [ ] Add `inference_jobs_*` columns to `wf_executions`
- [ ] Update InferenceService with async methods
- [ ] Add `get_job_result()` polling method

### Phase 2: Blocks Refactor (Day 1-2)
- [ ] Refactor DetectionBlock to use `detect_async()`
- [ ] Refactor ClassificationBlock to use `classify_async()`
- [ ] Refactor EmbeddingBlock to use `embed_async()`
- [ ] Update all blocks to pass execution context

### Phase 3: Worker Updates (Day 2)
- [ ] Add `update_inference_job()` function to worker
- [ ] Extract execution context from job input
- [ ] Write status updates to Supabase
- [ ] Test worker writes

### Phase 4: Testing (Day 2-3)
- [ ] Run all 8 workflow tests
- [ ] Verify async execution
- [ ] Check Supabase job records
- [ ] Monitor worker logs
- [ ] Test concurrent executions (10+)
- [ ] Test timeout handling
- [ ] Test failure scenarios

### Phase 5: Frontend Integration (Day 3-4)
- [ ] Add polling for execution status
- [ ] Display per-node progress
- [ ] Show inference job details
- [ ] Add cancel functionality
- [ ] Real-time metrics dashboard

---

## 🎯 Success Metrics

### Performance
- ✅ <2s API response time (execution created)
- ✅ <5s for cached model inference
- ✅ <30s for cold start
- ✅ 100+ concurrent workflows

### Reliability
- ✅ 99.5% success rate
- ✅ Automatic retry on worker failure
- ✅ Graceful degradation
- ✅ No webhook dependencies

### Observability
- ✅ Real-time execution status
- ✅ Per-node timing breakdown
- ✅ GPU utilization metrics
- ✅ Cost tracking per workflow

---

## 🚀 Conclusion

This async architecture is **proven in production** (OD training uses same pattern).

**Key advantages:**
1. **Scalable** - 1000s of concurrent executions
2. **Reliable** - Direct Supabase writes (no webhooks)
3. **Observable** - Real-time progress tracking
4. **Cost-efficient** - Serverless auto-scaling
5. **Simple** - Same pattern across all async operations

**Implementation:** ~2-3 days for full async workflow system.

**This is SOTA** - matches industry best practices for serverless ML inference.
