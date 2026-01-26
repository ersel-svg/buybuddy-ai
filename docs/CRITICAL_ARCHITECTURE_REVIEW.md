# Critical Architecture Review - Is This Plan SOTA?

**Date:** 2026-01-26
**Purpose:** Honest critique of proposed async workflow architecture

---

## 🔴 VERDICT: **NOT SOTA** - But Practical

My initial plan has several **anti-patterns** compared to true SOTA systems. Here's an honest assessment:

---

## ❌ Problems with My Original Design

### Problem 1: Database Polling is NOT SOTA

**My Design:**
```
API → Submit job → Poll Supabase every 2s → Get result
```

**Why it's bad:**
- **High latency:** 0-2s delay even after job completes
- **Database load:** Constant SELECT queries (N workflows × 0.5 req/s)
- **Not real-time:** User sees stale data
- **Doesn't scale:** 1000 workflows = 500 queries/second to Supabase

**SOTA Approach (Redis Pub/Sub + WebSocket):**
```
Worker → Publishes to Redis channel → WebSocket pushes to frontend instantly
```

**Why it's better:**
- **Sub-200ms latency** (vs 2000ms polling)
- **No database load** for status updates
- **True real-time** - instant UI updates
- **Scales infinitely** - Redis handles millions of messages/sec

**Source:** [Real-Time Updates with Redis](https://medium.com/@saulojterceiro/real-time-updates-with-redis-how-i-eliminated-polling-and-boosted-my-web-apps-performance-e46a040ef5ee)

---

### Problem 2: No Workflow Orchestration Engine

**My Design:**
```python
# Simple sequential execution
for node in execution_order:
    result = await execute_node(node)
```

**Why it's bad:**
- No durable execution (crash = lost state)
- No automatic retries with backoff
- No timeout handling
- No checkpointing
- No distributed execution

**SOTA Approach (Temporal/Celery):**
```python
# Durable, fault-tolerant execution
@workflow.defn
class InferenceWorkflow:
    @workflow.run
    async def run(self, input):
        # Automatic retry, timeout, checkpointing
        result = await workflow.execute_activity(
            run_detection,
            args=[input],
            retry_policy=RetryPolicy(max_attempts=3),
            start_to_close_timeout=timedelta(minutes=5),
        )
```

**Why it's better:**
- **Exactly-once execution** - guaranteed completion
- **Automatic retry** with exponential backoff
- **State persistence** - survives crashes
- **Distributed execution** - scale across workers
- **Observability** - built-in tracing

**Source:** [Workflow Orchestration: Building Complex AI Pipelines](https://medium.com/@omark.k.aly/workflow-orchestration-building-complex-ai-pipelines-c8504ab8306f)

---

### Problem 3: Direct Supabase Writes from Worker

**My Design:**
```python
# Worker writes directly to Supabase
def handler(job):
    update_inference_job(status="running")  # Direct DB write
    result = run_inference()
    update_inference_job(status="completed", result=result)  # Direct DB write
```

**Why it's bad:**
- **Tight coupling** - Worker depends on Supabase schema
- **Network overhead** - Every status change = HTTP request
- **No batching** - 1 inference = 3+ DB writes
- **Single point of failure** - Supabase down = job fails

**SOTA Approach (Event-driven with Redis/Kafka):**
```python
# Worker publishes events
def handler(job):
    redis.publish("inference:started", {"job_id": job_id})
    result = run_inference()
    redis.publish("inference:completed", {"job_id": job_id, "result": result})

# Separate consumer handles persistence
async def event_consumer():
    async for event in redis.subscribe("inference:*"):
        await batch_write_to_database(event)
```

**Why it's better:**
- **Decoupled** - Worker doesn't know about database
- **Batched writes** - Reduces DB load by 10x
- **Fault tolerant** - Events buffered if DB down
- **Observable** - Events can feed multiple consumers (logs, metrics, UI)

**Source:** [Redis Pub/Sub](https://redis.io/docs/latest/develop/pubsub/)

---

### Problem 4: Sync Waiting in Blocks

**My Design:**
```python
class DetectionBlock:
    async def execute(self, inputs, config, context):
        job_id = await self._inference_service.detect_async(...)

        # ❌ BLOCKS HERE - waiting for job
        result = await self._inference_service.get_job_result(job_id, timeout=120)

        return BlockResult(outputs=result)
```

**Why it's bad:**
- **Not truly async** - Still blocks the workflow execution
- **Resource waste** - API thread held while waiting
- **No parallelism** - Can't run multiple inference blocks concurrently

**SOTA Approach (True async with callbacks/futures):**
```python
class WorkflowEngine:
    async def execute_parallel_nodes(self, nodes):
        # Submit all jobs concurrently
        futures = []
        for node in nodes:
            if is_inference_node(node):
                future = self._inference_service.submit_job(node)
                futures.append((node.id, future))

        # Wait for all concurrently
        results = await asyncio.gather(*[f for _, f in futures])

        return dict(zip([n for n, _ in futures], results))
```

**Why it's better:**
- **True parallelism** - 10 detection nodes = 10 concurrent GPU jobs
- **Better throughput** - Maximizes GPU utilization
- **Lower latency** - Parallel > Sequential

---

### Problem 5: No Cold Start Optimization

**My Design:**
- Relies on RunPod's default cold start handling
- No model preloading strategy
- No warm pool management

**SOTA Approach (Modal/Replicate style):**
```python
# Declarative warm pool
@app.function(
    gpu="A100",
    container_idle_timeout=300,  # Keep warm for 5 min
    allow_concurrent_inputs=10,  # Handle 10 requests per container
    min_containers=2,            # Always keep 2 warm
)
async def detect(image, model_id):
    model = get_cached_model(model_id)  # In-memory cache
    return model.predict(image)
```

**Why it's better:**
- **Sub-second cold starts** - Modal achieves this
- **Request batching** - Multiple inputs per container
- **Smart scaling** - Predictive warm pool management

**Source:** [Modal: High-performance AI infrastructure](https://modal.com/)

---

## ✅ What My Design Got Right

1. **Worker-side model caching** - Good pattern, keep it
2. **Unified inference worker** - Single worker for all tasks is efficient
3. **Supabase for state** - Good for persistence, just don't poll it
4. **RunPod serverless** - Good choice for GPU scaling
5. **Separate inference jobs table** - Good for observability

---

## 🎯 ACTUAL SOTA ARCHITECTURE

Based on research, here's what a truly SOTA system looks like:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  • Connects via WebSocket                                   │
│  • Receives real-time updates                               │
│  • No polling                                               │
└─────────────────────────────────────────────────────────────┘
              ↑ WebSocket (real-time)        ↓ REST (submit)
┌─────────────────────────────────────────────────────────────┐
│                    API Server (FastAPI)                      │
│  • POST /workflows/{id}/run → Creates execution             │
│  • Publishes to Redis: "workflow:execute"                   │
│  • Subscribes to Redis for status → Push via WebSocket      │
└─────────────────────────────────────────────────────────────┘
              ↑↓ Redis Pub/Sub
┌─────────────────────────────────────────────────────────────┐
│                 Workflow Orchestrator                        │
│  (Celery/Temporal - runs as separate service)               │
│  • Consumes "workflow:execute" from Redis                   │
│  • Handles retries, timeouts, checkpointing                 │
│  • Submits inference jobs to RunPod                         │
│  • Publishes progress: "workflow:node:completed"            │
└─────────────────────────────────────────────────────────────┘
              ↓ HTTP (job submit)
┌─────────────────────────────────────────────────────────────┐
│              RunPod Inference Worker (GPU)                   │
│  • Receives job, runs inference                             │
│  • Publishes result to Redis: "inference:completed"         │
│  • Supports batching for high throughput                    │
└─────────────────────────────────────────────────────────────┘
              ↓ (async)
┌─────────────────────────────────────────────────────────────┐
│                Database Writer Service                       │
│  • Subscribes to all events                                 │
│  • Batches writes to Supabase                               │
│  • Handles backpressure                                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Redis** - Event bus for all communication
2. **WebSocket** - Real-time frontend updates
3. **Celery/Temporal** - Workflow orchestration with durability
4. **Batch DB Writer** - Efficient database persistence

---

## 🤔 PRAGMATIC RECOMMENDATION

**Full SOTA is overkill for your current stage.**

Here's a **pragmatic middle ground**:

### Phase 1: Fix the Broken Basics (Day 1)
Keep my original design BUT:
- Remove local inference from blocks ✅
- Use InferenceService → RunPod ✅
- Simple sync execution (ok for now)

### Phase 2: Add Redis Pub/Sub (Day 2-3)
- Worker publishes to Redis instead of Supabase writes
- API subscribes and pushes to frontend via SSE (simpler than WebSocket)
- Batch writer for database persistence

### Phase 3: Add Celery (Week 2 - Optional)
- Only if you need:
  - Retry logic
  - Distributed execution
  - Complex DAG workflows

### Phase 4: Full Temporal (Month 2 - Optional)
- Only if you need:
  - Exactly-once guarantees
  - Long-running workflows (hours/days)
  - Enterprise reliability

---

## 📊 Complexity vs Benefit Trade-off

| Approach | Complexity | Latency | Scale | Reliability |
|----------|-----------|---------|-------|-------------|
| **My Original (Poll)** | Low | 2s | 100s | Medium |
| **Redis Pub/Sub** | Medium | 200ms | 1000s | High |
| **+ Celery** | Medium-High | 200ms | 10000s | Very High |
| **+ Temporal** | High | 200ms | 100000s | Enterprise |

---

## 🎯 REVISED PLAN

### Immediate (Day 1) - Fix Broken
```
DetectionBlock → InferenceService → RunPod
                 (no local inference)
```

### Short-term (Day 2-3) - Add Redis
```
Worker → Redis.publish("inference:done")
API → Redis.subscribe() → SSE push to frontend
Background → Batch write to Supabase
```

### Medium-term (Week 2) - Optional Celery
```
Only if needed for retry/distributed
```

**This is NOT over-engineering. It's the right amount of engineering for scale.**

---

## Sources

- [HackerNoon: Long Polling vs Redis Pub/Sub](https://hackernoon.com/how-to-choose-the-right-real-time-communication-approach-long-polling-vs-redis-pubsub)
- [Medium: Real-Time Updates with Redis](https://medium.com/@saulojterceiro/real-time-updates-with-redis-how-i-eliminated-polling-and-boosted-my-web-apps-performance-e46a040ef5ee)
- [Medium: Workflow Orchestration for AI Pipelines](https://medium.com/@omark.k.aly/workflow-orchestration-building-complex-ai-pipelines-c8504ab8306f)
- [Modal: High-performance AI infrastructure](https://modal.com/)
- [Redis Pub/Sub Documentation](https://redis.io/docs/latest/develop/pubsub/)
- [Celery and Kubernetes for ML Orchestration](https://cfp.in.pycon.org/2025/talk/TK9ULJ/)
