# Local Background Jobs Infrastructure - Implementation Plan

## Overview

Bu plan, Runpod'a ihtiyaç duymayan CPU-bound bulk operasyonlar için generic bir background job infrastructure'ı oluşturmayı amaçlar. Mevcut functionality bozulmadan, backward-compatible şekilde eklenecektir.

---

## Phase 1: Database Schema Update

### 1.1 Migration: Add new job types to jobs table

**Dosya:** `infra/supabase/migrations/XXX_local_job_types.sql`

```sql
-- Add new job types for local background processing
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;
ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN (
    -- Existing Runpod job types
    'video_processing',
    'augmentation',
    'training',
    'embedding_extraction',
    'matching',
    'roboflow_import',
    'od_annotation',
    'buybuddy_sync',
    -- NEW: Local background job types
    'local_bulk_add_to_dataset',
    'local_bulk_remove_from_dataset',
    'local_bulk_update_status',
    'local_bulk_delete_images',
    'local_export_dataset',
    'local_bulk_update_products',
    'local_recalculate_counts',
    'local_data_cleanup'
));

-- Add index for local jobs (for worker polling)
CREATE INDEX IF NOT EXISTS idx_jobs_local_pending
ON jobs(created_at)
WHERE type LIKE 'local_%' AND status = 'pending';

-- Add worker_id field for job locking (prevents duplicate processing)
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS worker_id TEXT;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS locked_at TIMESTAMPTZ;
```

**Etki:** Sadece yeni job tipleri ekleniyor, mevcut tipler ve veriler etkilenmiyor.

---

## Phase 2: Core Infrastructure

### 2.1 Directory Structure

```
apps/api/src/services/
└── local_jobs/
    ├── __init__.py           # Public API exports
    ├── worker.py             # Background worker (polling loop)
    ├── registry.py           # Job handler registry
    ├── base.py               # BaseJobHandler abstract class
    ├── utils.py              # Helper functions (batching, etc.)
    └── handlers/
        ├── __init__.py       # Auto-register handlers
        ├── bulk_add_to_dataset.py
        ├── bulk_remove_from_dataset.py
        ├── bulk_update_status.py
        ├── bulk_delete_images.py
        ├── export_dataset.py
        ├── bulk_update_products.py
        └── recalculate_counts.py
```

### 2.2 Base Handler Class

**Dosya:** `apps/api/src/services/local_jobs/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass

@dataclass
class JobProgress:
    """Progress update for a job."""
    progress: int  # 0-100
    current_step: str
    processed: int = 0
    total: int = 0
    errors: list[str] = None

class BaseJobHandler(ABC):
    """Abstract base class for local job handlers."""

    # Subclasses must define this
    job_type: str = None

    @abstractmethod
    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None]
    ) -> dict:
        """
        Execute the job.

        Args:
            job_id: Unique job identifier
            config: Job configuration from jobs.config
            update_progress: Callback to update progress

        Returns:
            Result dict to store in jobs.result

        Raises:
            Exception: Job will be marked as failed
        """
        pass

    async def on_cancel(self, job_id: str, config: dict) -> None:
        """Called when job is cancelled. Override for cleanup."""
        pass

    def validate_config(self, config: dict) -> Optional[str]:
        """Validate config before execution. Return error message or None."""
        return None
```

### 2.3 Job Registry

**Dosya:** `apps/api/src/services/local_jobs/registry.py`

```python
from typing import Type
from .base import BaseJobHandler

class JobRegistry:
    """Registry for local job handlers."""

    _handlers: dict[str, Type[BaseJobHandler]] = {}

    @classmethod
    def register(cls, handler_class: Type[BaseJobHandler]) -> Type[BaseJobHandler]:
        """Decorator to register a job handler."""
        if not handler_class.job_type:
            raise ValueError(f"Handler {handler_class.__name__} must define job_type")
        cls._handlers[handler_class.job_type] = handler_class
        return handler_class

    @classmethod
    def get_handler(cls, job_type: str) -> Type[BaseJobHandler] | None:
        """Get handler class for job type."""
        return cls._handlers.get(job_type)

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all registered job types."""
        return list(cls._handlers.keys())

# Singleton instance
job_registry = JobRegistry()
```

### 2.4 Background Worker

**Dosya:** `apps/api/src/services/local_jobs/worker.py`

```python
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from services.supabase import supabase_service
from .registry import job_registry
from .base import JobProgress

class LocalJobWorker:
    """Background worker that processes local jobs."""

    def __init__(self):
        self.worker_id = str(uuid.uuid4())[:8]
        self.running = False
        self.current_job_id: Optional[str] = None
        self.poll_interval = 2.0  # seconds
        self.max_job_duration = 3600  # 1 hour max

    async def start(self):
        """Start the worker loop."""
        self.running = True
        print(f"[LocalJobWorker-{self.worker_id}] Started")

        while self.running:
            try:
                job = await self._claim_next_job()
                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LocalJobWorker-{self.worker_id}] Error: {e}")
                await asyncio.sleep(self.poll_interval)

        print(f"[LocalJobWorker-{self.worker_id}] Stopped")

    async def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        # TODO: Wait for current job to finish or mark as interrupted

    async def _claim_next_job(self) -> Optional[dict]:
        """Atomically claim the next pending job."""
        # Get pending local jobs
        result = supabase_service.client.table("jobs")\
            .select("*")\
            .like("type", "local_%")\
            .eq("status", "pending")\
            .is_("worker_id", "null")\
            .order("created_at")\
            .limit(1)\
            .execute()

        if not result.data:
            return None

        job = result.data[0]

        # Try to claim it (optimistic locking)
        update_result = supabase_service.client.table("jobs")\
            .update({
                "status": "running",
                "worker_id": self.worker_id,
                "locked_at": datetime.now(timezone.utc).isoformat(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            })\
            .eq("id", job["id"])\
            .eq("status", "pending")\
            .is_("worker_id", "null")\
            .execute()

        if update_result.data:
            return update_result.data[0]
        return None  # Another worker claimed it

    async def _process_job(self, job: dict):
        """Process a single job."""
        job_id = job["id"]
        job_type = job["type"]
        config = job.get("config", {})

        self.current_job_id = job_id
        print(f"[LocalJobWorker-{self.worker_id}] Processing {job_type} job {job_id[:8]}...")

        handler_class = job_registry.get_handler(job_type)
        if not handler_class:
            await self._fail_job(job_id, f"Unknown job type: {job_type}")
            return

        handler = handler_class()

        # Validate config
        error = handler.validate_config(config)
        if error:
            await self._fail_job(job_id, f"Invalid config: {error}")
            return

        try:
            # Execute with progress callback
            result = await handler.execute(
                job_id=job_id,
                config=config,
                update_progress=lambda p: self._update_progress(job_id, p)
            )

            # Mark as completed
            await self._complete_job(job_id, result)
            print(f"[LocalJobWorker-{self.worker_id}] Completed job {job_id[:8]}")

        except asyncio.CancelledError:
            await self._fail_job(job_id, "Job was cancelled")
            raise
        except Exception as e:
            await self._fail_job(job_id, str(e))
            print(f"[LocalJobWorker-{self.worker_id}] Failed job {job_id[:8]}: {e}")
        finally:
            self.current_job_id = None

    def _update_progress(self, job_id: str, progress: JobProgress):
        """Update job progress in database."""
        supabase_service.client.table("jobs").update({
            "progress": progress.progress,
            "current_step": progress.current_step,
            "result": {
                "processed": progress.processed,
                "total": progress.total,
                "errors": progress.errors or [],
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()

    async def _complete_job(self, job_id: str, result: dict):
        """Mark job as completed."""
        supabase_service.client.table("jobs").update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()

    async def _fail_job(self, job_id: str, error: str):
        """Mark job as failed."""
        supabase_service.client.table("jobs").update({
            "status": "failed",
            "error": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()

# Singleton instance
local_job_worker = LocalJobWorker()
```

### 2.5 Public API

**Dosya:** `apps/api/src/services/local_jobs/__init__.py`

```python
from .base import BaseJobHandler, JobProgress
from .registry import job_registry, JobRegistry
from .worker import local_job_worker, LocalJobWorker

async def create_local_job(job_type: str, config: dict) -> dict:
    """
    Create a new local background job.

    Args:
        job_type: Type of job (must start with 'local_')
        config: Job configuration

    Returns:
        Created job record
    """
    from services.supabase import supabase_service

    if not job_type.startswith("local_"):
        raise ValueError(f"Local job type must start with 'local_': {job_type}")

    if not job_registry.get_handler(job_type):
        raise ValueError(f"Unknown job type: {job_type}")

    result = supabase_service.client.table("jobs").insert({
        "type": job_type,
        "status": "pending",
        "config": config,
        "progress": 0,
    }).execute()

    return result.data[0]

__all__ = [
    "BaseJobHandler",
    "JobProgress",
    "job_registry",
    "local_job_worker",
    "create_local_job",
]
```

### 2.6 Main.py Integration

**Dosya:** `apps/api/src/main.py` (değişiklikler)

```python
# Yeni import ekle
from services.local_jobs import local_job_worker

# Lifespan fonksiyonuna ekle
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ... existing startup code ...

    # Start local job worker
    worker_task = asyncio.create_task(local_job_worker.start())
    print("   Local job worker started")

    yield

    # Shutdown
    print(f"👋 Shutting down {settings.app_name}")

    # Stop local job worker gracefully
    await local_job_worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
```

---

## Phase 3: Job Handlers Implementation

### 3.1 Bulk Add to Dataset Handler

**Dosya:** `apps/api/src/services/local_jobs/handlers/bulk_add_to_dataset.py`

```python
from typing import Callable
from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry

@job_registry.register
class BulkAddToDatasetHandler(BaseJobHandler):
    """Handler for bulk adding images to dataset."""

    job_type = "local_bulk_add_to_dataset"
    BATCH_SIZE = 100

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None]
    ) -> dict:
        dataset_id = config["dataset_id"]
        filters = config.get("filters", {})

        # Get all matching image IDs with pagination
        image_ids = await self._get_all_image_ids(filters)
        total = len(image_ids)

        if total == 0:
            return {"added": 0, "skipped": 0, "total": 0, "message": "No images match filters"}

        # Get existing images in dataset
        existing_ids = await self._get_existing_ids(dataset_id)

        added = 0
        skipped = 0
        errors = []

        # Process in batches
        for i in range(0, total, self.BATCH_SIZE):
            batch = image_ids[i:i + self.BATCH_SIZE]

            # Filter out existing
            new_links = []
            for img_id in batch:
                if img_id in existing_ids:
                    skipped += 1
                else:
                    new_links.append({
                        "dataset_id": dataset_id,
                        "image_id": img_id,
                        "status": "pending"
                    })

            # Insert batch
            if new_links:
                try:
                    supabase_service.client.table("od_dataset_images")\
                        .insert(new_links).execute()
                    added += len(new_links)
                except Exception as e:
                    errors.append(f"Batch {i//self.BATCH_SIZE}: {str(e)}")

            # Update progress
            processed = min(i + self.BATCH_SIZE, total)
            update_progress(JobProgress(
                progress=int(processed / total * 100),
                current_step=f"Processing batch {i//self.BATCH_SIZE + 1}",
                processed=processed,
                total=total,
                errors=errors[-5:] if errors else None
            ))

        # Update dataset count
        await self._update_dataset_count(dataset_id)

        return {
            "added": added,
            "skipped": skipped,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Added {added} images, skipped {skipped}"
        }

    async def _get_all_image_ids(self, filters: dict) -> list[str]:
        """Get all image IDs matching filters with pagination."""
        all_ids = []
        page_size = 1000
        offset = 0

        while True:
            query = supabase_service.client.table("od_images").select("id")

            # Apply filters
            if filters.get("statuses"):
                query = query.in_("status", filters["statuses"].split(","))
            if filters.get("sources"):
                query = query.in_("source", filters["sources"].split(","))
            if filters.get("folders"):
                query = query.in_("folder", filters["folders"].split(","))
            if filters.get("search"):
                query = query.ilike("filename", f"%{filters['search']}%")

            result = query.range(offset, offset + page_size - 1).execute()

            if not result.data:
                break

            all_ids.extend([img["id"] for img in result.data])

            if len(result.data) < page_size:
                break

            offset += page_size

        return all_ids

    async def _get_existing_ids(self, dataset_id: str) -> set[str]:
        """Get existing image IDs in dataset."""
        result = supabase_service.client.table("od_dataset_images")\
            .select("image_id")\
            .eq("dataset_id", dataset_id)\
            .execute()
        return {r["image_id"] for r in (result.data or [])}

    async def _update_dataset_count(self, dataset_id: str):
        """Update dataset image count."""
        count_result = supabase_service.client.table("od_dataset_images")\
            .select("id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .execute()

        supabase_service.client.table("od_datasets").update({
            "image_count": count_result.count or 0
        }).eq("id", dataset_id).execute()
```

### 3.2 Diğer Handler'lar (Özet)

Her handler aynı pattern'i takip eder:

| Handler | Dosya | Mevcut Kod Kaynağı |
|---------|-------|-------------------|
| `BulkRemoveFromDatasetHandler` | `bulk_remove_from_dataset.py` | `od/datasets.py:313-366` |
| `BulkUpdateStatusHandler` | `bulk_update_status.py` | `od/datasets.py:516-656` |
| `BulkDeleteImagesHandler` | `bulk_delete_images.py` | `od/images.py:bulk_delete_by_filters` |
| `ExportDatasetHandler` | `export_dataset.py` | `od_export.py` |
| `BulkUpdateProductsHandler` | `bulk_update_products.py` | `product_bulk_update.py:439-510` |
| `RecalculateCountsHandler` | `recalculate_counts.py` | `roboflow_streaming.py:28-100` |

---

## Phase 4: API Endpoint Updates

### 4.1 OD Images - Bulk Add to Dataset

**Dosya:** `apps/api/src/api/v1/od/images.py`

Mevcut senkron endpoint'i koruyup, async versiyonu ekle:

```python
# MEVCUT endpoint'i KORUYORUZ (backward compatibility)
@router.post("/bulk/add-to-dataset-by-filters")
async def bulk_add_to_dataset_by_filters(dataset_id: str, filters: BulkFilterRequest):
    """Add all images matching filters to a dataset (sync - for small batches)."""
    # ... existing code stays the same ...

# YENİ async endpoint
@router.post("/bulk/add-to-dataset-by-filters/async")
async def bulk_add_to_dataset_by_filters_async(
    dataset_id: str,
    filters: BulkFilterRequest
):
    """Add all images matching filters to a dataset (async - for large batches)."""
    from services.local_jobs import create_local_job

    job = await create_local_job(
        job_type="local_bulk_add_to_dataset",
        config={
            "dataset_id": dataset_id,
            "filters": filters.model_dump(),
        }
    )

    return {
        "job_id": job["id"],
        "status": "pending",
        "message": "Job created. Poll /api/v1/jobs/{job_id} for progress."
    }
```

### 4.2 Diğer Endpoint'ler (Aynı Pattern)

Her bulk endpoint için:
1. Mevcut senkron endpoint'i **KORU** (backward compatibility)
2. Yeni `/async` endpoint ekle
3. Frontend'de büyük batch'ler için async'i kullan

---

## Phase 5: Frontend Updates

### 5.1 Job Progress Hook

**Dosya:** `apps/web/src/hooks/use-job-progress.ts`

```typescript
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";

interface JobProgress {
  id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  current_step?: string;
  result?: {
    processed?: number;
    total?: number;
    added?: number;
    skipped?: number;
    errors?: string[];
  };
  error?: string;
}

export function useJobProgress(jobId: string | null, options?: {
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
}) {
  return useQuery({
    queryKey: ["job-progress", jobId],
    queryFn: async () => {
      if (!jobId) return null;
      return apiClient.getJob(jobId) as Promise<JobProgress>;
    },
    enabled: !!jobId,
    refetchInterval: (data) => {
      if (!data) return false;
      if (data.status === "completed") {
        options?.onComplete?.(data.result);
        return false;
      }
      if (data.status === "failed") {
        options?.onError?.(data.error || "Job failed");
        return false;
      }
      if (data.status === "cancelled") {
        return false;
      }
      return 1000; // Poll every second while running
    },
  });
}
```

### 5.2 Job Progress Modal Component

**Dosya:** `apps/web/src/components/common/job-progress-modal.tsx`

```typescript
interface JobProgressModalProps {
  jobId: string | null;
  title: string;
  onClose: () => void;
  onComplete?: (result: any) => void;
}

export function JobProgressModal({ jobId, title, onClose, onComplete }: JobProgressModalProps) {
  const { data: job } = useJobProgress(jobId, { onComplete });

  // ... Progress bar, status, cancel button UI ...
}
```

### 5.3 OD Images Page Update

**Dosya:** `apps/web/src/app/od/images/page.tsx`

```typescript
// Mevcut handleAddToDataset'i güncelle
const handleAddToDataset = async () => {
  if (!selectedDatasetId) return;

  if (selectAllFilteredMode) {
    // Büyük batch: async job kullan
    const total = imagesData?.total || 0;

    if (total > 1000) {
      // Async job başlat
      const result = await apiClient.addFilteredImagesToODDatasetAsync(
        selectedDatasetId,
        { search: debouncedSearch, ...apiFilters }
      );
      setActiveJobId(result.job_id);
      setShowJobProgressModal(true);
    } else {
      // Küçük batch: sync kullan (mevcut davranış)
      addToDatasetByFiltersMutation.mutate({ datasetId: selectedDatasetId });
    }
  } else {
    // Manuel seçim: sync kullan (mevcut davranış)
    addToDatasetMutation.mutate({
      datasetId: selectedDatasetId,
      imageIds: Array.from(selectedImages),
    });
  }
};
```

---

## Phase 6: Migration of Existing Bulk Operations

### 6.1 Migration Priority

| # | Operation | Dosya | Öncelik | Risk |
|---|-----------|-------|---------|------|
| 1 | OD Bulk Add to Dataset | `od/images.py` | HIGH | LOW |
| 2 | OD Bulk Update Status | `od/datasets.py` | HIGH | LOW |
| 3 | OD Bulk Delete | `od/images.py` | MEDIUM | MEDIUM |
| 4 | OD Export | `od_export.py` | HIGH | MEDIUM |
| 5 | Product Bulk Update | `product_bulk_update.py` | MEDIUM | MEDIUM |
| 6 | Recalculate Counts | `roboflow_streaming.py` | LOW | LOW |

### 6.2 Migration Strategy

Her operasyon için:

1. **Handler yaz** - Mevcut kodu handler'a taşı
2. **Async endpoint ekle** - `/async` suffix ile
3. **Frontend güncelle** - Büyük batch'ler için async kullan
4. **Test et** - Hem sync hem async
5. **Monitor et** - Production'da izle

**Mevcut endpoint'leri KALDIR*MA*** - Backward compatibility için koru.

---

## Phase 7: Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_local_jobs.py
import pytest
from services.local_jobs import create_local_job, job_registry
from services.local_jobs.handlers.bulk_add_to_dataset import BulkAddToDatasetHandler

class TestLocalJobs:
    async def test_create_job(self):
        job = await create_local_job(
            job_type="local_bulk_add_to_dataset",
            config={"dataset_id": "test-id", "filters": {}}
        )
        assert job["status"] == "pending"
        assert job["type"] == "local_bulk_add_to_dataset"

    async def test_handler_validation(self):
        handler = BulkAddToDatasetHandler()
        error = handler.validate_config({})
        assert error == "dataset_id is required"

        error = handler.validate_config({"dataset_id": "123"})
        assert error is None
```

### 7.2 Integration Tests

```python
# tests/test_local_jobs_integration.py
async def test_bulk_add_to_dataset_full_flow():
    # 1. Create test dataset and images
    # 2. Create job
    # 3. Wait for completion
    # 4. Verify images added
    pass
```

---

## Phase 8: Rollout Plan

### Week 1: Infrastructure
- [ ] Database migration
- [ ] Core infrastructure (worker, registry, base)
- [ ] Unit tests

### Week 2: First Handler
- [ ] `BulkAddToDatasetHandler`
- [ ] Async endpoint
- [ ] Frontend integration
- [ ] Manual testing

### Week 3: More Handlers
- [ ] `BulkUpdateStatusHandler`
- [ ] `BulkDeleteImagesHandler`
- [ ] Integration tests

### Week 4: Remaining Handlers
- [ ] `ExportDatasetHandler`
- [ ] `BulkUpdateProductsHandler`
- [ ] `RecalculateCountsHandler`

### Week 5: Polish & Deploy
- [ ] Performance tuning
- [ ] Monitoring & alerts
- [ ] Documentation
- [ ] Production deploy

---

## Risk Mitigation

### 1. Worker Crash
**Risk:** Worker crash'lerse job'lar stuck kalır.
**Mitigation:**
- `locked_at` timestamp ile stale job detection
- Startup'ta interrupted job'ları mark et
- Health check endpoint

### 2. Database Overload
**Risk:** Çok fazla progress update DB'yi zorlar.
**Mitigation:**
- Progress update'leri throttle et (her 100 item'da bir)
- Batch insert/update kullan

### 3. Memory Issues
**Risk:** Çok büyük dataset'ler memory'yi patlatır.
**Mitigation:**
- Pagination ile ID'leri al
- Streaming/generator pattern kullan
- Max limit koy (100k)

### 4. Backward Compatibility
**Risk:** Mevcut frontend/API kullanımları bozulur.
**Mitigation:**
- Mevcut endpoint'leri KALDIRMA
- Yeni endpoint'leri `/async` suffix ile ekle
- Gradual migration

---

## Success Metrics

- [ ] All bulk operations complete without timeout
- [ ] Progress visible in UI for long-running jobs
- [ ] Job cancellation works
- [ ] No increase in error rates
- [ ] P95 latency for job creation < 200ms
