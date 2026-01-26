# 🚀 Workflow System - Kapsamlı Uygulama Planı

> **Hedef:** n8n/Roboflow Workflows seviyesinde production-ready bir visual pipeline builder
>
> **Tahmini Süre:** 6-8 hafta (full-time)
>
> **Öncelik:** Test hızı + Production scalability + Güvenilirlik

---

## 📋 İÇİNDEKİLER

1. [Phase 1: Temel Düzeltmeler](#phase-1-temel-düzeltmeler-1-hafta)
2. [Phase 2: Async & Queue Sistemi](#phase-2-async--queue-sistemi-15-hafta)
3. [Phase 3: Real-time Progress](#phase-3-real-time-progress-1-hafta)
4. [Phase 4: Engine İyileştirmeleri](#phase-4-engine-iyileştirmeleri-15-hafta)
5. [Phase 5: Caching & Performance](#phase-5-caching--performance-1-hafta)
6. [Phase 6: Advanced Features](#phase-6-advanced-features-2-hafta)

---

## Phase 1: Temel Düzeltmeler (1 hafta)

### 1.1 Rate Limiting (2 saat)

**Dosyalar:**
- `apps/api/src/main.py`
- `apps/api/src/config.py`
- `apps/api/src/middleware/rate_limit.py` (yeni)

**Uygulama:**

```python
# config.py - Yeni ayarlar ekle
class Settings(BaseSettings):
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_workflow_runs_per_minute: int = 20
    rate_limit_burst: int = 10
    redis_url: str = ""  # Rate limiting için Redis

# middleware/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri=settings.redis_url or "memory://",
)

# main.py
from middleware.rate_limit import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# executions.py - Workflow run endpoint'ine özel limit
@router.post("/{workflow_id}/run")
@limiter.limit("20/minute")  # Workflow execution için daha düşük limit
async def run_workflow(...):
```

**Gerekli Paketler:**
```
slowapi>=0.1.9
redis>=5.0.0
```

---

### 1.2 Frontend-Backend Config Senkronizasyonu (1 gün)

**Sorun:** Condition, ForEach, Collect block'ları frontend ve backend'de farklı config schema'ya sahip.

**Çözüm Stratejisi:** Backend'i frontend'e eşitle (frontend doğru tasarlanmış)

#### 1.2.1 Condition Block Düzeltmesi

**Dosya:** `apps/api/src/services/workflow/blocks/placeholder_blocks.py`

```python
class ConditionBlock(BaseBlock):
    """
    Condition Block - Frontend-aligned

    Tek koşul değerlendirmesi (frontend ile uyumlu).
    Çoklu koşullar için birden fazla Condition block zincirlenebilir.
    """

    block_type = "condition"
    display_name = "Condition"
    description = "Branch based on single condition"

    input_ports = [
        {"name": "value", "type": "any", "required": True, "description": "Value to evaluate"},
        {"name": "compare_to", "type": "any", "required": False, "description": "Value to compare against"},
    ]

    output_ports = [
        {"name": "true_output", "type": "any", "description": "Output when condition is true"},
        {"name": "false_output", "type": "any", "description": "Output when condition is false"},
        {"name": "result", "type": "boolean", "description": "Condition result"},
        {"name": "matched_conditions", "type": "array", "description": "Which conditions matched"},
    ]

    # Frontend ile AYNI config schema
    config_schema = {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "description": "Field to evaluate (dot notation for nested)",
            },
            "operator": {
                "type": "string",
                "enum": ["eq", "neq", "gt", "gte", "lt", "lte", "in", "nin", "contains", "matches", "exists", "empty"],
                "default": "gt",
            },
            "value": {
                "type": "string",
                "description": "Value to compare against",
            },
            "true_value": {
                "type": "string",
                "default": "pass",
                "description": "Output when true",
            },
            "false_value": {
                "type": "string",
                "default": "fail",
                "description": "Output when false",
            },
        },
    }
```

#### 1.2.2 ForEach Block Düzeltmesi

```python
class ForEachBlock(BaseBlock):
    config_schema = {
        "type": "object",
        "properties": {
            # Frontend ile aynı isim: max_iterations (max_items değil)
            "max_iterations": {
                "type": "number",
                "default": 0,
                "description": "Maximum items to process (0 = all)",
            },
            "parallel": {
                "type": "boolean",
                "default": False,
                "description": "Process items in parallel",
            },
            "batch_size": {
                "type": "number",
                "default": 10,
                "description": "Items per batch for parallel processing",
            },
        },
    }
```

#### 1.2.3 Collect Block Düzeltmesi

```python
class CollectBlock(BaseBlock):
    config_schema = {
        "type": "object",
        "properties": {
            "filter_nulls": {
                "type": "boolean",
                "default": True,
            },
            "flatten": {
                "type": "boolean",
                "default": False,
            },
            # Frontend'de var, backend'de eksikti - EKLENDİ
            "unique": {
                "type": "boolean",
                "default": False,
                "description": "Remove duplicate values",
            },
            "unique_key": {
                "type": "string",
                "description": "Field to use for uniqueness check",
            },
        },
    }

    async def execute(self, inputs, config, context):
        # ... mevcut kod ...

        # Unique desteği ekle
        unique = config.get("unique", False)
        unique_key = config.get("unique_key")

        if unique and results:
            if unique_key:
                # Object array için key-based unique
                seen = set()
                unique_results = []
                for item in results:
                    key = item.get(unique_key) if isinstance(item, dict) else item
                    if key not in seen:
                        seen.add(key)
                        unique_results.append(item)
                results = unique_results
            else:
                # Primitive array için set-based unique
                try:
                    results = list(dict.fromkeys(results))
                except TypeError:
                    # Unhashable items, skip unique
                    pass
```

---

### 1.3 Schema Sync Script (Opsiyonel - 4 saat)

Frontend TypeScript tanımlarından Python schema otomatik generate eden script:

**Dosya:** `scripts/sync_block_schemas.py`

```python
#!/usr/bin/env python3
"""
Frontend block tanımlarından backend schema generate eder.
Çalıştır: python scripts/sync_block_schemas.py
"""

import json
import re
from pathlib import Path

FRONTEND_BLOCKS_DIR = Path("apps/web/src/lib/workflow/blocks")
BACKEND_BLOCKS_DIR = Path("apps/api/src/services/workflow/blocks")
OUTPUT_FILE = BACKEND_BLOCKS_DIR / "generated_schemas.json"

def parse_typescript_block(ts_content: str) -> dict:
    """TypeScript block definition'ı parse et."""
    # configFields array'ini bul
    match = re.search(r'configFields:\s*\[(.*?)\]', ts_content, re.DOTALL)
    if not match:
        return {}

    # Her field'ı parse et
    fields = []
    field_pattern = r'\{\s*key:\s*"(\w+)".*?type:\s*"(\w+)".*?(?:default:\s*([^,}]+))?'
    for m in re.finditer(field_pattern, match.group(1), re.DOTALL):
        fields.append({
            "key": m.group(1),
            "type": m.group(2),
            "default": m.group(3).strip() if m.group(3) else None,
        })

    return {"configFields": fields}

def main():
    schemas = {}

    for ts_file in FRONTEND_BLOCKS_DIR.glob("*.ts"):
        if ts_file.name == "index.ts":
            continue

        content = ts_file.read_text()
        # Her block export'unu bul
        for match in re.finditer(r'export const (\w+Block)', content):
            block_name = match.group(1)
            schema = parse_typescript_block(content)
            if schema:
                schemas[block_name] = schema

    OUTPUT_FILE.write_text(json.dumps(schemas, indent=2))
    print(f"Generated schemas for {len(schemas)} blocks -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
```

---

## Phase 2: Async & Queue Sistemi (1.5 hafta)

### 2.1 Database Schema Güncellemesi (2 saat)

**Dosya:** `infra/supabase/migrations/063_workflow_async_execution.sql`

```sql
-- Workflow Async Execution Support
-- Migration 063: Add async execution capabilities

-- Execution mode enum
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(20) DEFAULT 'sync'
    CHECK (execution_mode IN ('sync', 'async', 'background'));

-- Priority for queue ordering
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 5
    CHECK (priority BETWEEN 1 AND 10);

-- Webhook URL for async completion notification
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS callback_url TEXT;

-- Retry tracking
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3;
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS last_retry_at TIMESTAMPTZ;

-- Progress tracking for real-time updates
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS progress JSONB DEFAULT '{
    "current_node": null,
    "completed_nodes": [],
    "total_nodes": 0,
    "percent": 0
}'::jsonb;

-- Index for queue processing
CREATE INDEX IF NOT EXISTS idx_wf_executions_queue
ON wf_executions(status, priority DESC, created_at ASC)
WHERE status IN ('pending', 'running');

-- Index for async mode
CREATE INDEX IF NOT EXISTS idx_wf_executions_mode
ON wf_executions(execution_mode);

COMMENT ON COLUMN wf_executions.execution_mode IS 'sync=wait, async=return immediately with ID, background=fire-and-forget';
COMMENT ON COLUMN wf_executions.priority IS '1=lowest, 10=highest priority in queue';
```

---

### 2.2 Execution Request Schema Güncellemesi (1 saat)

**Dosya:** `apps/api/src/schemas/workflows.py`

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ExecutionMode(str, Enum):
    SYNC = "sync"       # Wait for result (current behavior)
    ASYNC = "async"     # Return execution_id immediately, poll for result
    BACKGROUND = "background"  # Fire-and-forget with optional webhook

class WorkflowRunRequest(BaseModel):
    """Enhanced workflow run request with execution mode support."""

    # Image input (one required)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

    # Parameters
    parameters: Optional[dict] = None

    # Execution options
    mode: ExecutionMode = ExecutionMode.SYNC
    priority: int = Field(default=5, ge=1, le=10)
    callback_url: Optional[str] = None  # Webhook for async completion
    timeout_seconds: int = Field(default=300, ge=10, le=3600)

    # Retry options
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_on_failure: bool = True

class WorkflowRunResponse(BaseModel):
    """Response for workflow run request."""
    execution_id: str
    status: str
    mode: ExecutionMode

    # Only populated for sync mode
    output_data: Optional[dict] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    # For async/background mode
    status_url: Optional[str] = None
    estimated_wait_seconds: Optional[int] = None
```

---

### 2.3 Background Worker Sistemi (1 gün)

**Dosya:** `apps/api/src/services/workflow/worker.py`

```python
"""
Workflow Background Worker

Async execution queue'dan işleri alıp çalıştırır.
Mevcut LocalJobWorker pattern'ini takip eder.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from services.supabase import supabase_service
from services.workflow import get_workflow_engine

logger = logging.getLogger(__name__)


class WorkflowWorker:
    """
    Background worker for async workflow execution.

    Features:
    - Atomic job claiming (prevents duplicate processing)
    - Priority-based queue
    - Graceful shutdown
    - Retry on failure
    - Progress tracking
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        max_concurrent: int = 5,
        stale_timeout_minutes: int = 30,
    ):
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.stale_timeout_minutes = stale_timeout_minutes
        self._running = False
        self._current_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the worker."""
        self._running = True
        logger.info(f"Workflow worker started (max_concurrent={self.max_concurrent})")

        while self._running:
            try:
                # Claim pending jobs up to max_concurrent
                available_slots = self.max_concurrent - len(self._current_tasks)

                if available_slots > 0:
                    jobs = await self._claim_jobs(available_slots)

                    for job in jobs:
                        task = asyncio.create_task(self._process_job(job))
                        self._current_tasks[job["id"]] = task

                # Clean up completed tasks
                completed = [
                    job_id for job_id, task in self._current_tasks.items()
                    if task.done()
                ]
                for job_id in completed:
                    del self._current_tasks[job_id]

                # Also check for stale jobs
                await self._recover_stale_jobs()

            except Exception as e:
                logger.exception(f"Worker loop error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping workflow worker...")
        self._running = False

        # Wait for current tasks to complete (with timeout)
        if self._current_tasks:
            logger.info(f"Waiting for {len(self._current_tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._current_tasks.values(), return_exceptions=True),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling...")
                for task in self._current_tasks.values():
                    task.cancel()

        logger.info("Workflow worker stopped")

    async def _claim_jobs(self, limit: int) -> list[dict]:
        """
        Atomically claim pending jobs from queue.
        Uses optimistic locking to prevent duplicate claims.
        """
        # Find pending jobs
        result = supabase_service.client.table("wf_executions").select(
            "id, workflow_id, input_data, priority, retry_count, max_retries"
        ).eq(
            "status", "pending"
        ).in_(
            "execution_mode", ["async", "background"]
        ).order(
            "priority", desc=True
        ).order(
            "created_at", desc=False
        ).limit(limit).execute()

        if not result.data:
            return []

        claimed = []
        for job in result.data:
            # Try to claim with atomic update
            update_result = supabase_service.client.table("wf_executions").update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", job["id"]).eq("status", "pending").execute()

            if update_result.data:
                claimed.append(job)
                logger.info(f"Claimed job {job['id'][:8]}... (priority={job['priority']})")

        return claimed

    async def _process_job(self, job: dict):
        """Process a single workflow job."""
        job_id = job["id"]
        workflow_id = job["workflow_id"]

        try:
            # Get workflow definition
            workflow = supabase_service.client.table("wf_workflows").select(
                "definition"
            ).eq("id", workflow_id).single().execute()

            if not workflow.data:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Execute workflow
            engine = get_workflow_engine()
            result = await engine.execute(
                workflow=workflow.data["definition"],
                inputs=job["input_data"],
                workflow_id=workflow_id,
                execution_id=job_id,
                progress_callback=lambda p: self._update_progress(job_id, p),
            )

            # Update completion
            await self._complete_job(job_id, result)

        except Exception as e:
            logger.exception(f"Job {job_id[:8]}... failed: {e}")
            await self._fail_job(job, str(e))

    async def _update_progress(self, job_id: str, progress: dict):
        """Update job progress for real-time tracking."""
        supabase_service.client.table("wf_executions").update({
            "progress": progress,
        }).eq("id", job_id).execute()

    async def _complete_job(self, job_id: str, result: dict):
        """Mark job as completed."""
        supabase_service.client.table("wf_executions").update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": result.get("duration_ms", 0),
            "output_data": result.get("outputs", {}),
            "node_metrics": result.get("metrics", {}),
        }).eq("id", job_id).execute()

        logger.info(f"Job {job_id[:8]}... completed in {result.get('duration_ms', 0)}ms")

        # Send webhook if configured
        await self._send_completion_webhook(job_id, result)

    async def _fail_job(self, job: dict, error: str):
        """Mark job as failed, with retry logic."""
        job_id = job["id"]
        retry_count = job.get("retry_count", 0)
        max_retries = job.get("max_retries", 3)

        if retry_count < max_retries:
            # Schedule retry
            supabase_service.client.table("wf_executions").update({
                "status": "pending",
                "retry_count": retry_count + 1,
                "last_retry_at": datetime.now(timezone.utc).isoformat(),
                "error_message": f"Retry {retry_count + 1}/{max_retries}: {error}",
            }).eq("id", job_id).execute()

            logger.info(f"Job {job_id[:8]}... scheduled for retry ({retry_count + 1}/{max_retries})")
        else:
            # Final failure
            supabase_service.client.table("wf_executions").update({
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error_message": error,
            }).eq("id", job_id).execute()

            logger.error(f"Job {job_id[:8]}... failed permanently after {max_retries} retries")
            await self._send_failure_webhook(job_id, error)

    async def _recover_stale_jobs(self):
        """Recover jobs stuck in 'running' state."""
        from datetime import timedelta

        stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=self.stale_timeout_minutes)

        result = supabase_service.client.table("wf_executions").update({
            "status": "pending",
            "error_message": "Recovered from stale state",
        }).eq(
            "status", "running"
        ).lt(
            "started_at", stale_threshold.isoformat()
        ).execute()

        if result.data:
            logger.warning(f"Recovered {len(result.data)} stale jobs")

    async def _send_completion_webhook(self, job_id: str, result: dict):
        """Send webhook on job completion."""
        job = supabase_service.client.table("wf_executions").select(
            "callback_url"
        ).eq("id", job_id).single().execute()

        if job.data and job.data.get("callback_url"):
            import httpx
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    await client.post(
                        job.data["callback_url"],
                        json={
                            "execution_id": job_id,
                            "status": "completed",
                            "output_data": result.get("outputs"),
                            "duration_ms": result.get("duration_ms"),
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to send completion webhook: {e}")

    async def _send_failure_webhook(self, job_id: str, error: str):
        """Send webhook on job failure."""
        # Similar to _send_completion_webhook
        pass


# Singleton instance
_workflow_worker: Optional[WorkflowWorker] = None


def get_workflow_worker() -> WorkflowWorker:
    """Get singleton worker instance."""
    global _workflow_worker
    if _workflow_worker is None:
        _workflow_worker = WorkflowWorker()
    return _workflow_worker
```

---

### 2.4 API Endpoint Güncellemesi (4 saat)

**Dosya:** `apps/api/src/api/v1/workflows/executions.py`

```python
from schemas.workflows import WorkflowRunRequest, WorkflowRunResponse, ExecutionMode

@router.post("/{workflow_id}/run", response_model=WorkflowRunResponse)
@limiter.limit("20/minute")
async def run_workflow(
    request: Request,
    workflow_id: str,
    data: WorkflowRunRequest,
):
    """
    Run a workflow with configurable execution mode.

    Modes:
    - sync: Wait for completion (default, max 5 min)
    - async: Return immediately, poll /executions/{id} for result
    - background: Fire-and-forget with optional webhook callback
    """

    # Validate workflow exists
    workflow = supabase_service.client.table("wf_workflows").select(
        "id, status, definition"
    ).eq("id", workflow_id).single().execute()

    if not workflow.data:
        raise HTTPException(404, "Workflow not found")
    if workflow.data["status"] == "archived":
        raise HTTPException(400, "Cannot run archived workflow")

    # Create execution record
    execution_data = {
        "workflow_id": workflow_id,
        "status": "pending",
        "execution_mode": data.mode.value,
        "priority": data.priority,
        "callback_url": data.callback_url,
        "max_retries": data.max_retries,
        "input_data": {
            "image_url": data.image_url,
            "image_base64": data.image_base64,
            "parameters": data.parameters or {},
        },
    }

    result = supabase_service.client.table("wf_executions").insert(
        execution_data
    ).execute()

    execution = result.data[0]
    execution_id = execution["id"]

    # Handle based on mode
    if data.mode == ExecutionMode.SYNC:
        # Synchronous execution (existing behavior)
        return await _execute_sync(
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_def=workflow.data["definition"],
            inputs=execution_data["input_data"],
            timeout=data.timeout_seconds,
        )

    elif data.mode == ExecutionMode.ASYNC:
        # Return immediately, worker will process
        return WorkflowRunResponse(
            execution_id=execution_id,
            status="pending",
            mode=data.mode,
            status_url=f"/api/v1/workflows/executions/{execution_id}",
            estimated_wait_seconds=_estimate_wait_time(workflow_id),
        )

    else:  # BACKGROUND
        # Fire and forget
        return WorkflowRunResponse(
            execution_id=execution_id,
            status="pending",
            mode=data.mode,
            status_url=f"/api/v1/workflows/executions/{execution_id}",
        )


async def _execute_sync(
    workflow_id: str,
    execution_id: str,
    workflow_def: dict,
    inputs: dict,
    timeout: int,
) -> WorkflowRunResponse:
    """Execute workflow synchronously with timeout."""

    # Update status to running
    supabase_service.client.table("wf_executions").update({
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", execution_id).execute()

    try:
        engine = get_workflow_engine()

        # Execute with timeout
        result = await asyncio.wait_for(
            engine.execute(
                workflow=workflow_def,
                inputs=inputs,
                workflow_id=workflow_id,
                execution_id=execution_id,
            ),
            timeout=timeout,
        )

        # Update completion
        supabase_service.client.table("wf_executions").update({
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": result.get("duration_ms", 0),
            "output_data": result.get("outputs", {}),
            "node_metrics": result.get("metrics", {}),
        }).eq("id", execution_id).execute()

        return WorkflowRunResponse(
            execution_id=execution_id,
            status="completed",
            mode=ExecutionMode.SYNC,
            output_data=result.get("outputs"),
            duration_ms=result.get("duration_ms"),
        )

    except asyncio.TimeoutError:
        supabase_service.client.table("wf_executions").update({
            "status": "failed",
            "error_message": f"Execution timeout after {timeout}s",
        }).eq("id", execution_id).execute()

        raise HTTPException(408, f"Workflow execution timeout after {timeout}s")

    except Exception as e:
        supabase_service.client.table("wf_executions").update({
            "status": "failed",
            "error_message": str(e),
        }).eq("id", execution_id).execute()

        raise HTTPException(500, f"Workflow execution failed: {e}")


def _estimate_wait_time(workflow_id: str) -> int:
    """Estimate wait time based on queue depth and avg duration."""
    # Count pending jobs ahead
    pending = supabase_service.client.table("wf_executions").select(
        "id", count="exact"
    ).eq("status", "pending").execute()

    # Get avg duration for this workflow
    stats = supabase_service.client.table("wf_workflows").select(
        "avg_duration_ms"
    ).eq("id", workflow_id).single().execute()

    queue_depth = pending.count or 0
    avg_ms = stats.data.get("avg_duration_ms", 5000) if stats.data else 5000

    # Rough estimate: queue_depth * avg_duration / concurrent_workers
    return max(1, int((queue_depth * avg_ms / 1000) / 5))
```

---

### 2.5 main.py Worker Entegrasyonu (1 saat)

**Dosya:** `apps/api/src/main.py`

```python
from services.workflow.worker import get_workflow_worker

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan with workflow worker."""

    # ... existing startup code ...

    # Start workflow worker
    workflow_worker = get_workflow_worker()
    workflow_worker_task = asyncio.create_task(workflow_worker.start())
    print("   ✓ Workflow background worker started")

    yield

    # ... existing shutdown code ...

    # Stop workflow worker
    await workflow_worker.stop()
    workflow_worker_task.cancel()
    print("   ✓ Workflow worker stopped")
```

---

## Phase 3: Real-time Progress (1 hafta)

### 3.1 WebSocket Endpoint (1 gün)

**Dosya:** `apps/api/src/api/v1/workflows/websocket.py`

```python
"""
WebSocket endpoint for real-time workflow execution progress.
"""

import asyncio
import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.supabase import supabase_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Active connections per execution
connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manage WebSocket connections for execution progress."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, execution_id: str, websocket: WebSocket):
        await websocket.accept()
        if execution_id not in self.connections:
            self.connections[execution_id] = set()
        self.connections[execution_id].add(websocket)
        logger.info(f"WS connected: execution={execution_id[:8]}...")

    def disconnect(self, execution_id: str, websocket: WebSocket):
        if execution_id in self.connections:
            self.connections[execution_id].discard(websocket)
            if not self.connections[execution_id]:
                del self.connections[execution_id]
        logger.info(f"WS disconnected: execution={execution_id[:8]}...")

    async def broadcast(self, execution_id: str, message: dict):
        if execution_id not in self.connections:
            return

        dead_connections = set()
        for websocket in self.connections[execution_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.add(websocket)

        # Clean up dead connections
        for ws in dead_connections:
            self.connections[execution_id].discard(ws)


manager = ConnectionManager()


@router.websocket("/executions/{execution_id}/progress")
async def execution_progress(websocket: WebSocket, execution_id: str):
    """
    WebSocket endpoint for execution progress.

    Sends events:
    - node_start: {node_id, node_type, timestamp}
    - node_complete: {node_id, duration_ms, outputs_summary}
    - node_error: {node_id, error}
    - progress: {percent, current_node, completed_nodes}
    - complete: {status, duration_ms, outputs}
    - error: {message}
    """
    await manager.connect(execution_id, websocket)

    try:
        # Send current state
        execution = supabase_service.client.table("wf_executions").select(
            "status, progress, output_data, error_message"
        ).eq("id", execution_id).single().execute()

        if execution.data:
            await websocket.send_json({
                "type": "initial_state",
                "data": execution.data,
            })

        # Keep connection alive and listen for updates
        while True:
            try:
                # Wait for messages (ping/pong or commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30,
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        manager.disconnect(execution_id, websocket)


# Helper to broadcast from engine
async def broadcast_progress(execution_id: str, event_type: str, data: dict):
    """Broadcast progress event to all connected clients."""
    await manager.broadcast(execution_id, {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
```

---

### 3.2 Engine Progress Callback (4 saat)

**Dosya:** `apps/api/src/services/workflow/engine.py`

```python
from typing import Callable, Optional, Awaitable

ProgressCallback = Callable[[dict], Awaitable[None]]

class WorkflowEngine:
    async def execute(
        self,
        workflow: dict,
        inputs: dict,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """Execute workflow with optional progress tracking."""

        self.progress_callback = progress_callback
        self.execution_id = execution_id

        # ... existing setup ...

        # Calculate total nodes for progress
        total_nodes = len(execution_order)
        completed_nodes = []

        for node_id in execution_order:
            node = nodes_map[node_id]
            node_type = node.get("type")

            # Broadcast node start
            await self._emit_progress("node_start", {
                "node_id": node_id,
                "node_type": node_type,
                "node_label": node.get("data", {}).get("label", node_type),
            })

            try:
                # Execute node
                result = await self._execute_node(...)
                completed_nodes.append(node_id)

                # Broadcast node complete
                await self._emit_progress("node_complete", {
                    "node_id": node_id,
                    "duration_ms": result.duration_ms,
                    "outputs_summary": self._summarize_outputs(result.outputs),
                })

                # Broadcast overall progress
                await self._emit_progress("progress", {
                    "percent": round(len(completed_nodes) / total_nodes * 100),
                    "current_node": node_id,
                    "completed_nodes": completed_nodes,
                    "total_nodes": total_nodes,
                })

            except Exception as e:
                await self._emit_progress("node_error", {
                    "node_id": node_id,
                    "error": str(e),
                })
                raise

        # Broadcast completion
        await self._emit_progress("complete", {
            "status": "completed",
            "duration_ms": total_duration,
        })

        return result

    async def _emit_progress(self, event_type: str, data: dict):
        """Emit progress event."""
        if self.progress_callback:
            await self.progress_callback({
                "type": event_type,
                **data,
            })

        # Also broadcast via WebSocket if execution_id is set
        if self.execution_id:
            from api.v1.workflows.websocket import broadcast_progress
            await broadcast_progress(self.execution_id, event_type, data)

    def _summarize_outputs(self, outputs: dict) -> dict:
        """Summarize outputs for progress (avoid sending large data)."""
        summary = {}
        for key, value in outputs.items():
            if isinstance(value, list):
                summary[key] = f"array[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"object({len(value)} keys)"
            elif isinstance(value, str) and len(value) > 100:
                summary[key] = f"string({len(value)} chars)"
            else:
                summary[key] = type(value).__name__
        return summary
```

---

### 3.3 Frontend WebSocket Client (Referans)

**Dosya:** `apps/web/src/hooks/useWorkflowProgress.ts`

```typescript
import { useEffect, useState, useCallback } from 'react';

interface ProgressEvent {
  type: 'node_start' | 'node_complete' | 'node_error' | 'progress' | 'complete' | 'error';
  data: any;
  timestamp: string;
}

interface WorkflowProgress {
  percent: number;
  currentNode: string | null;
  completedNodes: string[];
  totalNodes: number;
  nodeResults: Record<string, any>;
  error: string | null;
  isComplete: boolean;
}

export function useWorkflowProgress(executionId: string | null) {
  const [progress, setProgress] = useState<WorkflowProgress>({
    percent: 0,
    currentNode: null,
    completedNodes: [],
    totalNodes: 0,
    nodeResults: {},
    error: null,
    isComplete: false,
  });

  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!executionId) return;

    const wsUrl = `${process.env.NEXT_PUBLIC_WS_URL}/api/v1/workflows/executions/${executionId}/progress`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const message: ProgressEvent = JSON.parse(event.data);

      switch (message.type) {
        case 'node_start':
          setProgress(p => ({
            ...p,
            currentNode: message.data.node_id,
          }));
          break;

        case 'node_complete':
          setProgress(p => ({
            ...p,
            completedNodes: [...p.completedNodes, message.data.node_id],
            nodeResults: {
              ...p.nodeResults,
              [message.data.node_id]: message.data.outputs_summary,
            },
          }));
          break;

        case 'progress':
          setProgress(p => ({
            ...p,
            percent: message.data.percent,
            totalNodes: message.data.total_nodes,
          }));
          break;

        case 'complete':
          setProgress(p => ({
            ...p,
            isComplete: true,
            percent: 100,
          }));
          break;

        case 'node_error':
        case 'error':
          setProgress(p => ({
            ...p,
            error: message.data.error || message.data.message,
          }));
          break;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    // Ping interval
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 25000);

    return () => {
      clearInterval(pingInterval);
      ws.close();
    };
  }, [executionId]);

  return { progress, isConnected };
}
```

---

## Phase 4: Engine İyileştirmeleri (1.5 hafta)

### 4.1 Conditional Branching (1 gün)

**Sorun:** Şu an tüm node'lar sırayla çalışıyor, condition sonucuna göre branch atlanmıyor.

**Dosya:** `apps/api/src/services/workflow/engine.py`

```python
class WorkflowEngine:
    async def execute(self, ...):
        # ... existing setup ...

        # Track active branches
        active_branches: Set[str] = {"main"}  # Default branch
        skipped_nodes: Set[str] = set()

        for node_id in execution_order:
            node = nodes_map[node_id]
            node_type = node.get("type")

            # Check if this node should be skipped due to branching
            if self._should_skip_node(node_id, edges, context, skipped_nodes):
                skipped_nodes.add(node_id)
                logger.info(f"Skipping node {node_id} (inactive branch)")
                metrics[node_id] = {"skipped": True, "reason": "inactive_branch"}
                continue

            # Execute node
            result = await self._execute_node(...)

            # Handle branching for condition nodes
            if node_type == "condition":
                branch_result = result.outputs.get("result", False)

                # Mark downstream nodes on inactive branch as skipped
                inactive_branch = "false" if branch_result else "true"
                self._mark_inactive_branch(
                    node_id,
                    inactive_branch,
                    edges,
                    nodes_map,
                    skipped_nodes
                )

    def _should_skip_node(
        self,
        node_id: str,
        edges: list,
        context: ExecutionContext,
        skipped_nodes: Set[str],
    ) -> bool:
        """Check if node should be skipped due to inactive branch."""

        # Find all incoming edges
        incoming_edges = [e for e in edges if e.get("target") == node_id]

        if not incoming_edges:
            return False  # Input nodes always run

        # If ALL incoming sources are skipped, skip this node too
        all_skipped = all(
            e.get("source") in skipped_nodes
            for e in incoming_edges
        )

        return all_skipped

    def _mark_inactive_branch(
        self,
        condition_node_id: str,
        inactive_handle: str,  # "true" or "false"
        edges: list,
        nodes_map: dict,
        skipped_nodes: Set[str],
    ):
        """Mark all nodes in inactive branch as skipped."""

        # Find edge from inactive output handle
        inactive_edges = [
            e for e in edges
            if e.get("source") == condition_node_id
            and e.get("sourceHandle") == f"{inactive_handle}_output"
        ]

        # BFS to mark all downstream nodes
        to_skip = set()
        queue = [e.get("target") for e in inactive_edges]

        while queue:
            node_id = queue.pop(0)
            if node_id in to_skip:
                continue

            to_skip.add(node_id)

            # Find downstream nodes
            downstream = [
                e.get("target") for e in edges
                if e.get("source") == node_id
            ]
            queue.extend(downstream)

        skipped_nodes.update(to_skip)
        logger.info(f"Marked {len(to_skip)} nodes as skipped (inactive branch from {condition_node_id})")
```

---

### 4.2 Parallel ForEach (1 gün)

**Dosya:** `apps/api/src/services/workflow/engine.py`

```python
import asyncio
from asyncio import Semaphore

class WorkflowEngine:
    async def _execute_foreach_iteration(
        self,
        node_id: str,
        node: dict,
        items: list,
        config: dict,
        context: ExecutionContext,
        loop_nodes: list,
    ) -> list:
        """Execute ForEach with optional parallel processing."""

        parallel = config.get("parallel", False)
        batch_size = config.get("batch_size", 10)
        max_iterations = config.get("max_iterations", 0)

        # Apply max_iterations limit
        if max_iterations > 0:
            items = items[:max_iterations]

        if not parallel:
            # Sequential execution (existing behavior)
            return await self._execute_foreach_sequential(
                items, context, loop_nodes
            )
        else:
            # Parallel execution with semaphore
            return await self._execute_foreach_parallel(
                items, context, loop_nodes, batch_size
            )

    async def _execute_foreach_sequential(
        self,
        items: list,
        context: ExecutionContext,
        loop_nodes: list,
    ) -> list:
        """Sequential iteration (current behavior)."""
        results = []

        for idx, item in enumerate(items):
            # Set iteration context
            context.variables["_foreach_item"] = item
            context.variables["_foreach_index"] = idx
            context.variables["_foreach_total"] = len(items)
            context.variables["_foreach_is_first"] = idx == 0
            context.variables["_foreach_is_last"] = idx == len(items) - 1

            # Execute loop body nodes
            for loop_node_id in loop_nodes:
                await self._execute_node(loop_node_id, ...)

            # Collect result
            results.append(context.variables.get("_foreach_result"))

        return results

    async def _execute_foreach_parallel(
        self,
        items: list,
        context: ExecutionContext,
        loop_nodes: list,
        batch_size: int,
    ) -> list:
        """Parallel iteration with batching."""

        semaphore = Semaphore(batch_size)
        results = [None] * len(items)

        async def process_item(idx: int, item: any):
            async with semaphore:
                # Create isolated context for this iteration
                iter_context = context.copy()
                iter_context.variables["_foreach_item"] = item
                iter_context.variables["_foreach_index"] = idx
                iter_context.variables["_foreach_total"] = len(items)

                # Execute loop body
                for loop_node_id in loop_nodes:
                    await self._execute_node(loop_node_id, iter_context)

                results[idx] = iter_context.variables.get("_foreach_result")

        # Process all items in parallel (limited by semaphore)
        await asyncio.gather(*[
            process_item(idx, item)
            for idx, item in enumerate(items)
        ])

        return results
```

---

### 4.3 Node-Level Testing Endpoint (4 saat)

**Dosya:** `apps/api/src/api/v1/workflows/testing.py`

```python
"""
Node-level testing endpoints for workflow development.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from services.workflow.blocks import get_block
from services.workflow.base import ExecutionContext, BlockResult

router = APIRouter(prefix="/test", tags=["Workflow Testing"])


class NodeTestRequest(BaseModel):
    """Request to test a single node."""
    node_type: str
    config: dict = {}
    mock_inputs: dict = {}

    # Optional: provide image directly
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class NodeTestResponse(BaseModel):
    """Response from node test."""
    success: bool
    outputs: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: float
    metrics: Optional[dict] = None


@router.post("/node", response_model=NodeTestResponse)
async def test_node(request: NodeTestRequest):
    """
    Test a single node in isolation.

    Useful for:
    - Debugging node configuration
    - Validating model outputs
    - Quick iteration during development
    """

    # Get block class
    try:
        block_class = get_block(request.node_type)
    except ValueError:
        raise HTTPException(404, f"Unknown node type: {request.node_type}")

    # Create block instance
    block = block_class()

    # Create mock context
    context = ExecutionContext(
        inputs={
            "image_url": request.image_url,
            "image_base64": request.image_base64,
        },
        nodes={},
        parameters={},
        variables={},
        execution_id="test",
    )

    # Execute block
    try:
        result: BlockResult = await block.execute(
            inputs=request.mock_inputs,
            config=request.config,
            context=context,
        )

        return NodeTestResponse(
            success=result.error is None,
            outputs=result.outputs,
            error=result.error,
            duration_ms=result.duration_ms,
            metrics=result.metrics,
        )

    except Exception as e:
        return NodeTestResponse(
            success=False,
            error=str(e),
            duration_ms=0,
        )


@router.get("/blocks")
async def list_testable_blocks():
    """List all available blocks with their schemas."""
    from services.workflow.blocks import get_block_metadata
    return get_block_metadata()


@router.post("/validate-config")
async def validate_node_config(node_type: str, config: dict):
    """
    Validate node configuration without executing.

    Returns validation errors and warnings.
    """
    try:
        block_class = get_block(node_type)
    except ValueError:
        raise HTTPException(404, f"Unknown node type: {node_type}")

    block = block_class()

    # Check required fields
    errors = []
    warnings = []

    schema = block.config_schema.get("properties", {})
    required = block.config_schema.get("required", [])

    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    for field, value in config.items():
        if field not in schema:
            warnings.append(f"Unknown field: {field}")
        else:
            field_schema = schema[field]
            # Type validation
            expected_type = field_schema.get("type")
            if expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' should be a number")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field '{field}' should be a boolean")
            elif expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' should be a string")

            # Enum validation
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(f"Field '{field}' must be one of: {field_schema['enum']}")

            # Range validation
            if "minimum" in field_schema and value < field_schema["minimum"]:
                errors.append(f"Field '{field}' must be >= {field_schema['minimum']}")
            if "maximum" in field_schema and value > field_schema["maximum"]:
                errors.append(f"Field '{field}' must be <= {field_schema['maximum']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
```

---

## Phase 5: Caching & Performance (1 hafta)

### 5.1 Redis Inference Cache (1 gün)

**Dosya:** `apps/api/src/services/workflow/cache.py`

```python
"""
Inference result caching for workflow blocks.

Uses Redis for distributed caching with TTL.
"""

import hashlib
import json
import logging
from typing import Optional, Any
from datetime import timedelta

import redis.asyncio as redis

from config import settings

logger = logging.getLogger(__name__)


class InferenceCache:
    """
    Redis-based cache for inference results.

    Cache key format: wf:inference:{task}:{model_id}:{image_hash}:{config_hash}
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._enabled = bool(settings.redis_url)

    async def _get_client(self) -> Optional[redis.Redis]:
        """Lazy initialize Redis client."""
        if not self._enabled:
            return None

        if self._client is None:
            self._client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _hash_image(self, image_base64: str) -> str:
        """Create hash of image for cache key."""
        return hashlib.sha256(image_base64.encode()).hexdigest()[:16]

    def _hash_config(self, config: dict) -> str:
        """Create hash of config for cache key."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _make_key(
        self,
        task: str,
        model_id: str,
        image_hash: str,
        config_hash: str,
    ) -> str:
        """Create cache key."""
        return f"wf:inference:{task}:{model_id}:{image_hash}:{config_hash}"

    async def get(
        self,
        task: str,
        model_id: str,
        image_base64: str,
        config: dict,
    ) -> Optional[dict]:
        """Get cached inference result."""
        client = await self._get_client()
        if not client:
            return None

        try:
            key = self._make_key(
                task,
                model_id,
                self._hash_image(image_base64),
                self._hash_config(config),
            )

            cached = await client.get(key)
            if cached:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(cached)

            logger.debug(f"Cache MISS: {key}")
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(
        self,
        task: str,
        model_id: str,
        image_base64: str,
        config: dict,
        result: dict,
        ttl_minutes: int = 60,
    ):
        """Cache inference result."""
        client = await self._get_client()
        if not client:
            return

        try:
            key = self._make_key(
                task,
                model_id,
                self._hash_image(image_base64),
                self._hash_config(config),
            )

            await client.setex(
                key,
                timedelta(minutes=ttl_minutes),
                json.dumps(result),
            )

            logger.debug(f"Cache SET: {key} (TTL={ttl_minutes}m)")

        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    async def invalidate_model(self, model_id: str):
        """Invalidate all cache entries for a model."""
        client = await self._get_client()
        if not client:
            return

        try:
            pattern = f"wf:inference:*:{model_id}:*"
            keys = await client.keys(pattern)
            if keys:
                await client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for model {model_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")


# Singleton
_inference_cache: Optional[InferenceCache] = None


def get_inference_cache() -> InferenceCache:
    """Get singleton cache instance."""
    global _inference_cache
    if _inference_cache is None:
        _inference_cache = InferenceCache()
    return _inference_cache
```

---

### 5.2 Inference Service Cache Entegrasyonu (2 saat)

**Dosya:** `apps/api/src/services/workflow/inference_service.py`

```python
from services.workflow.cache import get_inference_cache

class InferenceService:
    async def detect(
        self,
        model_id: str,
        image: Image.Image,
        confidence: float = 0.5,
        # ... other params ...
        use_cache: bool = True,
        cache_ttl_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Run detection with optional caching."""

        cache = get_inference_cache()
        image_base64 = self._image_to_base64(image)
        config = {
            "confidence": confidence,
            "iou_threshold": iou,
            "max_detections": max_detections,
        }

        # Check cache
        if use_cache:
            cached = await cache.get(
                task="detection",
                model_id=model_id,
                image_base64=image_base64,
                config=config,
            )
            if cached:
                logger.info(f"Using cached detection result for {model_id}")
                return cached

        # Run inference
        result = await self._run_detection(...)

        # Cache result
        if use_cache:
            await cache.set(
                task="detection",
                model_id=model_id,
                image_base64=image_base64,
                config=config,
                result=result,
                ttl_minutes=cache_ttl_minutes,
            )

        return result
```

---

### 5.3 HTTP Client Pool (1 saat)

**Dosya:** `apps/api/src/services/http_client.py`

```python
"""
Global HTTP client pool for efficient connection reuse.
"""

import httpx
from typing import Optional

from config import settings


class HTTPClientPool:
    """
    Singleton HTTP client with connection pooling.

    Benefits:
    - Connection reuse (no TCP handshake overhead)
    - Keep-alive connections
    - Configurable limits
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=120.0,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton
http_pool = HTTPClientPool()


# Use in services:
# from services.http_client import http_pool
# response = await http_pool.client.get(url)
```

---

## Phase 6: Advanced Features (2 hafta)

### 6.1 Workflow Versioning (1 gün)

**Dosya:** `infra/supabase/migrations/064_workflow_versioning.sql`

```sql
-- Workflow Versioning System
-- Migration 064

-- Version history table
CREATE TABLE IF NOT EXISTS wf_workflow_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,

    -- Version info
    version INTEGER NOT NULL,
    version_label VARCHAR(50),  -- e.g., "v1.0", "stable", "experimental"

    -- Snapshot of definition at this version
    definition JSONB NOT NULL,

    -- Change info
    change_summary TEXT,
    changed_by VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(workflow_id, version)
);

CREATE INDEX IF NOT EXISTS idx_wf_versions_workflow ON wf_workflow_versions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wf_versions_created ON wf_workflow_versions(created_at DESC);

-- Add current version to workflows
ALTER TABLE wf_workflows
ADD COLUMN IF NOT EXISTS current_version INTEGER DEFAULT 1;

-- Link executions to specific version
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS workflow_version INTEGER;

-- Function to create new version on workflow update
CREATE OR REPLACE FUNCTION create_workflow_version()
RETURNS TRIGGER AS $$
BEGIN
    -- Only version if definition changed
    IF OLD.definition IS DISTINCT FROM NEW.definition THEN
        -- Increment version
        NEW.current_version := OLD.current_version + 1;

        -- Create version record
        INSERT INTO wf_workflow_versions (
            workflow_id,
            version,
            definition,
            change_summary
        ) VALUES (
            NEW.id,
            NEW.current_version,
            NEW.definition,
            'Auto-versioned on update'
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS wf_workflows_versioning ON wf_workflows;
CREATE TRIGGER wf_workflows_versioning
    BEFORE UPDATE ON wf_workflows
    FOR EACH ROW EXECUTE FUNCTION create_workflow_version();

COMMENT ON TABLE wf_workflow_versions IS 'Version history for workflow definitions';
```

---

### 6.2 Sub-Workflows (2 gün)

**Dosya:** `apps/web/src/lib/workflow/blocks/subworkflow.ts`

```typescript
export const subWorkflowBlock: BlockDefinition = {
  type: "sub_workflow",
  displayName: "Sub-Workflow",
  description: "Execute another workflow",
  category: "logic",
  icon: "GitBranch",

  inputs: [
    { name: "input_data", type: "any", required: false, description: "Data to pass to sub-workflow" },
  ],

  outputs: [
    { name: "output_data", type: "any", description: "Output from sub-workflow" },
    { name: "execution_id", type: "string", description: "Sub-workflow execution ID" },
    { name: "duration_ms", type: "number", description: "Sub-workflow duration" },
  ],

  configFields: [
    {
      key: "workflow_id",
      type: "select",
      label: "Workflow",
      description: "Workflow to execute",
      required: true,
      // Dynamically populated from API
      optionsEndpoint: "/api/v1/workflows?status=active",
    },
    {
      key: "execution_mode",
      type: "select",
      label: "Execution Mode",
      default: "sync",
      options: [
        { value: "sync", label: "Wait for completion" },
        { value: "async", label: "Fire and continue" },
      ],
    },
    {
      key: "timeout_seconds",
      type: "number",
      label: "Timeout",
      default: 300,
      min: 10,
      max: 3600,
    },
    {
      key: "pass_image",
      type: "boolean",
      label: "Pass Current Image",
      description: "Pass the current workflow image to sub-workflow",
      default: true,
    },
  ],
};
```

**Dosya:** `apps/api/src/services/workflow/blocks/subworkflow_block.py`

```python
class SubWorkflowBlock(BaseBlock):
    """Execute another workflow as a sub-workflow."""

    block_type = "sub_workflow"
    display_name = "Sub-Workflow"

    async def execute(self, inputs, config, context):
        start_time = time.time()

        workflow_id = config.get("workflow_id")
        execution_mode = config.get("execution_mode", "sync")
        timeout = config.get("timeout_seconds", 300)
        pass_image = config.get("pass_image", True)

        # Get sub-workflow definition
        workflow = supabase_service.client.table("wf_workflows").select(
            "definition"
        ).eq("id", workflow_id).single().execute()

        if not workflow.data:
            return BlockResult(error=f"Sub-workflow not found: {workflow_id}")

        # Prepare inputs for sub-workflow
        sub_inputs = inputs.get("input_data", {})
        if pass_image:
            sub_inputs["image"] = context.inputs.get("image")
            sub_inputs["image_url"] = context.inputs.get("image_url")
            sub_inputs["image_base64"] = context.inputs.get("image_base64")

        # Execute sub-workflow
        engine = get_workflow_engine()

        if execution_mode == "sync":
            result = await asyncio.wait_for(
                engine.execute(
                    workflow=workflow.data["definition"],
                    inputs=sub_inputs,
                    workflow_id=workflow_id,
                ),
                timeout=timeout,
            )

            return BlockResult(
                outputs={
                    "output_data": result.get("outputs", {}),
                    "execution_id": result.get("execution_id"),
                    "duration_ms": result.get("duration_ms"),
                },
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
        else:
            # Async: create execution and continue
            execution = supabase_service.client.table("wf_executions").insert({
                "workflow_id": workflow_id,
                "status": "pending",
                "execution_mode": "async",
                "input_data": sub_inputs,
            }).execute()

            return BlockResult(
                outputs={
                    "execution_id": execution.data[0]["id"],
                    "status": "pending",
                },
                duration_ms=round((time.time() - start_time) * 1000, 2),
            )
```

---

### 6.3 Execution Replay (4 saat)

**Dosya:** `apps/api/src/api/v1/workflows/executions.py`

```python
@router.post("/executions/{execution_id}/replay", response_model=WorkflowRunResponse)
async def replay_execution(
    execution_id: str,
    from_node: Optional[str] = Query(None, description="Resume from this node"),
):
    """
    Replay an execution with its original inputs.

    Optionally resume from a specific node (uses cached results for prior nodes).
    """

    # Get original execution
    original = supabase_service.client.table("wf_executions").select(
        "workflow_id, input_data, node_metrics"
    ).eq("id", execution_id).single().execute()

    if not original.data:
        raise HTTPException(404, "Execution not found")

    # Get workflow
    workflow = supabase_service.client.table("wf_workflows").select(
        "definition"
    ).eq("id", original.data["workflow_id"]).single().execute()

    if not workflow.data:
        raise HTTPException(404, "Workflow not found")

    # Create new execution
    new_execution = supabase_service.client.table("wf_executions").insert({
        "workflow_id": original.data["workflow_id"],
        "status": "pending",
        "input_data": original.data["input_data"],
        "execution_mode": "sync",
        # Link to original for replay tracking
        "replayed_from": execution_id,
    }).execute()

    new_execution_id = new_execution.data[0]["id"]

    # Execute with optional resume point
    engine = get_workflow_engine()

    # If resuming, pass cached results for prior nodes
    cached_results = None
    if from_node:
        cached_results = _get_results_before_node(
            original.data["node_metrics"],
            from_node,
        )

    result = await engine.execute(
        workflow=workflow.data["definition"],
        inputs=original.data["input_data"],
        workflow_id=original.data["workflow_id"],
        execution_id=new_execution_id,
        cached_node_results=cached_results,
        resume_from_node=from_node,
    )

    # Update execution
    supabase_service.client.table("wf_executions").update({
        "status": "completed" if not result.get("error") else "failed",
        "output_data": result.get("outputs"),
        "duration_ms": result.get("duration_ms"),
        "node_metrics": result.get("metrics"),
    }).eq("id", new_execution_id).execute()

    return WorkflowRunResponse(
        execution_id=new_execution_id,
        status="completed",
        output_data=result.get("outputs"),
        duration_ms=result.get("duration_ms"),
    )


def _get_results_before_node(node_metrics: dict, target_node: str) -> dict:
    """Extract cached results for nodes before target."""
    # This would require storing outputs in node_metrics
    # For now, return empty (full re-execution)
    return {}
```

---

## 📊 Uygulama Takvimi

```
Hafta 1:
├── Phase 1: Temel Düzeltmeler
│   ├── Rate Limiting (2 saat)
│   ├── Config Sync - Condition (4 saat)
│   ├── Config Sync - ForEach (2 saat)
│   └── Config Sync - Collect (2 saat)
│
Hafta 2:
├── Phase 2: Async & Queue
│   ├── DB Migration (2 saat)
│   ├── Schema Updates (2 saat)
│   ├── Background Worker (8 saat)
│   └── API Updates (4 saat)
│
Hafta 3:
├── Phase 2 (devam) + Phase 3
│   ├── Worker Integration (4 saat)
│   ├── WebSocket Endpoint (8 saat)
│   └── Engine Progress (4 saat)
│
Hafta 4:
├── Phase 4: Engine İyileştirmeleri
│   ├── Conditional Branching (8 saat)
│   ├── Parallel ForEach (8 saat)
│   └── Node Testing (4 saat)
│
Hafta 5:
├── Phase 5: Caching
│   ├── Redis Cache (4 saat)
│   ├── Inference Integration (2 saat)
│   ├── HTTP Pool (2 saat)
│   └── Testing & Optimization (8 saat)
│
Hafta 6-7:
├── Phase 6: Advanced
│   ├── Versioning (8 saat)
│   ├── Sub-Workflows (16 saat)
│   └── Execution Replay (4 saat)
│
Hafta 8:
└── Testing & Polish
    ├── Integration Tests
    ├── Load Testing
    ├── Documentation
    └── Bug Fixes
```

---

## ✅ Checklist

### Phase 1
- [ ] slowapi rate limiting
- [ ] Redis bağlantısı (opsiyonel memory fallback)
- [ ] Condition block config sync
- [ ] ForEach block config sync (max_iterations)
- [ ] Collect block unique/unique_key

### Phase 2
- [ ] DB migration 063
- [ ] WorkflowRunRequest mode ekleme
- [ ] WorkflowWorker class
- [ ] main.py worker startup
- [ ] /run endpoint mode handling

### Phase 3
- [ ] WebSocket manager
- [ ] /executions/{id}/progress endpoint
- [ ] Engine progress callback
- [ ] Frontend hook (referans)

### Phase 4
- [ ] Branch tracking in engine
- [ ] _should_skip_node logic
- [ ] Parallel ForEach with semaphore
- [ ] /test/node endpoint
- [ ] /test/validate-config endpoint

### Phase 5
- [ ] InferenceCache class
- [ ] Redis entegrasyonu
- [ ] detect/classify/embed cache
- [ ] HTTPClientPool singleton

### Phase 6
- [ ] wf_workflow_versions table
- [ ] Auto-versioning trigger
- [ ] sub_workflow block
- [ ] /replay endpoint

---

## 🔧 Gerekli Paketler

```
# requirements.txt eklemeleri
slowapi>=0.1.9
redis>=5.0.0
websockets>=12.0
```

---

## 🎯 Başarı Kriterleri

1. **Rate Limiting**: 60 req/min genel, 20 req/min workflow run
2. **Async Mode**: 1000+ concurrent execution queue'da bekleyebilmeli
3. **Real-time**: <100ms WebSocket event latency
4. **Parallel ForEach**: 10x speedup for batch operations
5. **Cache Hit Rate**: >50% for repeated inference calls
6. **Versioning**: Her definition değişikliği tracked
