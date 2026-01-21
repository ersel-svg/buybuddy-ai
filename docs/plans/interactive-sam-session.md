# Interactive SAM with Session-Based Embedding Caching

## Problem Statement

Current interactive SAM implementation is slow (~1-3 seconds per click) because it:
1. Downloads the image from URL on every click
2. Computes the image embedding (Vision Transformer) on every click
3. Runs the mask decoder on every click

The image embedding computation takes ~1 second and is the bottleneck. Since the image doesn't change between clicks, we should compute this embedding ONCE and reuse it.

## Goal

Achieve Roboflow-like interactive segmentation experience:
- **Session start:** ~1-2 seconds (compute embedding once)
- **Each click:** ~50-200ms (only run mask decoder)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CURRENT FLOW                             │
│                                                                  │
│  Click → API → RunPod → Download → Embedding → Decoder → Return │
│                                     ↑                            │
│                                  SLOW (~1s)                      │
│                                                                  │
│  Total: 1-3 seconds per click                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         TARGET FLOW                              │
│                                                                  │
│  Open Image → API → RunPod → Download → Embedding → Cache       │
│                                                     ↓            │
│                                              session_id          │
│                                                                  │
│  Click → API → RunPod → Get Cached Embedding → Decoder → Return │
│                                                  ↑               │
│                                               FAST (~50ms)       │
│                                                                  │
│  Total: 50-200ms per click                                       │
└─────────────────────────────────────────────────────────────────┘
```

## SAM Architecture Reference

```
Image (1024x1024)
       │
       ▼
┌─────────────────┐
│  Image Encoder  │  ← Vision Transformer (ViT-H, 632M params)
│     (ViT-H)     │  ← ~1 second on GPU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Image Embedding │  ← 256 x 64 x 64 tensor
│   (cached)      │  ← THIS SHOULD BE CACHED!
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Point │ │  Box  │  ← User input (changes each click)
│Prompt │ │Prompt │
└───┬───┘ └───┬───┘
    └────┬────┘
         ▼
┌─────────────────┐
│  Mask Decoder   │  ← Lightweight transformer
│                 │  ← ~50ms on GPU
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Mask + Score  │
└─────────────────┘
```

## Implementation Plan

### Phase 1: Worker Changes

**File:** `workers/od-annotation/models/sam2.py`

#### 1.1 Add Session Storage

```python
from dataclasses import dataclass
from typing import Dict, Optional
import time
import threading

@dataclass
class SAMSession:
    """Stores pre-computed image embedding for a session."""
    session_id: str
    image_url: str
    image_width: int
    image_height: int
    image_embedding: torch.Tensor  # Pre-computed embedding
    created_at: float
    last_used_at: float

class SAM2Model(BaseSegmentationModel):
    # Session storage (in-memory cache)
    _sessions: Dict[str, SAMSession] = {}
    _session_lock = threading.Lock()

    # Session expiry (5 minutes of inactivity)
    SESSION_TIMEOUT_SECONDS = 300

    # Max concurrent sessions (memory management)
    MAX_SESSIONS = 10
```

#### 1.2 Add Session Management Methods

```python
def start_session(self, image_url: str) -> dict:
    """
    Start a new SAM session by pre-computing image embedding.

    Args:
        image_url: URL of the image

    Returns:
        dict with session_id
    """
    import uuid

    # Download image
    image = self.download_image(image_url)
    width, height = image.size

    # Process image for SAM
    inputs = self.processor(image, return_tensors="pt").to(self.device)

    # Compute image embedding (this is the slow part)
    with torch.no_grad():
        image_embedding = self.model.get_image_embeddings(inputs["pixel_values"])

    # Create session
    session_id = str(uuid.uuid4())
    session = SAMSession(
        session_id=session_id,
        image_url=image_url,
        image_width=width,
        image_height=height,
        image_embedding=image_embedding,
        created_at=time.time(),
        last_used_at=time.time(),
    )

    # Store session (with cleanup of old sessions)
    with self._session_lock:
        self._cleanup_expired_sessions()
        self._sessions[session_id] = session

    logger.info(f"Started SAM session {session_id} for image {width}x{height}")

    return {
        "session_id": session_id,
        "image_width": width,
        "image_height": height,
    }

def segment_with_session(
    self,
    session_id: str,
    point: tuple[float, float],
    label: int = 1,
    return_mask: bool = True,
) -> dict:
    """
    Segment using a pre-computed session (fast).

    Args:
        session_id: Session ID from start_session()
        point: (x, y) in normalized 0-1 coords
        label: 1 for foreground, 0 for background
        return_mask: Whether to return mask as base64

    Returns:
        dict with bbox, confidence, and optionally mask
    """
    # Get session
    with self._session_lock:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")
        session.last_used_at = time.time()

    width = session.image_width
    height = session.image_height

    # Convert normalized point to pixel coords
    point_x = int(point[0] * width)
    point_y = int(point[1] * height)

    # Prepare point prompt
    input_points = torch.tensor([[[point_x, point_y]]], device=self.device)
    input_labels = torch.tensor([[label]], device=self.device)

    # Run ONLY the mask decoder (fast!)
    with torch.no_grad():
        outputs = self.model(
            image_embeddings=session.image_embedding,
            input_points=input_points,
            input_labels=input_labels,
        )

    # Process output (same as before)
    masks = outputs.pred_masks.cpu()
    scores = outputs.iou_scores.cpu().float().numpy()[0][0]

    best_idx = np.argmax(scores)
    best_mask = masks[0][0][best_idx].numpy()
    best_score = float(scores[best_idx])

    # Convert to bbox
    bbox_abs = self.mask_to_bbox(best_mask)
    if bbox_abs is None:
        return {"bbox": {"x": 0, "y": 0, "width": 0, "height": 0}, "confidence": 0}

    x1, y1, x2, y2 = bbox_abs
    result = {
        "bbox": {
            "x": x1 / width,
            "y": y1 / height,
            "width": (x2 - x1) / width,
            "height": (y2 - y1) / height,
        },
        "confidence": best_score,
    }

    if return_mask:
        result["mask"] = self.mask_to_base64(best_mask)

    return result

def close_session(self, session_id: str) -> bool:
    """Close a session and free memory."""
    with self._session_lock:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Closed SAM session {session_id}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
    return False

def _cleanup_expired_sessions(self):
    """Remove expired sessions (called with lock held)."""
    now = time.time()
    expired = [
        sid for sid, session in self._sessions.items()
        if now - session.last_used_at > self.SESSION_TIMEOUT_SECONDS
    ]
    for sid in expired:
        del self._sessions[sid]
        logger.info(f"Expired SAM session {sid}")

    # Also remove oldest if over limit
    while len(self._sessions) >= self.MAX_SESSIONS:
        oldest = min(self._sessions.values(), key=lambda s: s.last_used_at)
        del self._sessions[oldest.session_id]
        logger.info(f"Evicted SAM session {oldest.session_id} (max sessions)")
```

#### 1.3 Update Handler

**File:** `workers/od-annotation/handler.py`

Add new task types:

```python
def handler(job):
    job_input = job["input"]
    task_type = job_input.get("task")

    # ... existing code ...

    if task_type == "segment_session_start":
        return handle_segment_session_start(job_input)
    elif task_type == "segment_session_point":
        return handle_segment_session_point(job_input)
    elif task_type == "segment_session_close":
        return handle_segment_session_close(job_input)

def handle_segment_session_start(job_input: dict) -> dict:
    """Start a new SAM session with pre-computed embedding."""
    model_name = job_input.get("model", "sam2")
    image_url = job_input["image_url"]

    model = get_model(model_name, MODEL_CACHE)
    result = model.start_session(image_url)

    return {"status": "success", **result}

def handle_segment_session_point(job_input: dict) -> dict:
    """Segment a point using existing session."""
    model_name = job_input.get("model", "sam2")
    session_id = job_input["session_id"]
    point = tuple(job_input["point"])
    label = job_input.get("label", 1)

    model = get_model(model_name, MODEL_CACHE)
    result = model.segment_with_session(session_id, point, label)

    return result

def handle_segment_session_close(job_input: dict) -> dict:
    """Close a SAM session."""
    model_name = job_input.get("model", "sam2")
    session_id = job_input["session_id"]

    model = get_model(model_name, MODEL_CACHE)
    success = model.close_session(session_id)

    return {"status": "success" if success else "not_found"}
```

### Phase 2: Backend API Changes

**File:** `apps/api/src/api/v1/od/ai.py`

Add new endpoints:

```python
@router.post("/segment/session/start")
async def start_segment_session(request: SegmentSessionStartRequest):
    """
    Start a SAM session by pre-computing image embedding.
    Call this when user opens an image for interactive annotation.
    """
    # Get image URL
    image_result = supabase_service.client.table("od_images").select(
        "id, image_url"
    ).eq("id", request.image_id).single().execute()

    if not image_result.data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_result.data["image_url"]

    # Call RunPod
    runpod_input = {
        "task": "segment_session_start",
        "model": request.model,
        "image_url": image_url,
    }

    result = await runpod_service.submit_job_sync(
        endpoint_type=EndpointType.OD_ANNOTATION,
        input_data=runpod_input,
        timeout=30,
    )

    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail="Failed to start session")

    output = result.get("output", {})
    return {
        "session_id": output.get("session_id"),
        "image_width": output.get("image_width"),
        "image_height": output.get("image_height"),
    }

@router.post("/segment/session/{session_id}/point")
async def segment_session_point(session_id: str, request: SegmentSessionPointRequest):
    """
    Segment a point using an existing session (fast).
    """
    runpod_input = {
        "task": "segment_session_point",
        "model": request.model,
        "session_id": session_id,
        "point": list(request.point),
        "label": request.label,
    }

    result = await runpod_service.submit_job_sync(
        endpoint_type=EndpointType.OD_ANNOTATION,
        input_data=runpod_input,
        timeout=10,  # Should be fast
    )

    if result.get("status") == "FAILED":
        raise HTTPException(status_code=500, detail="Segmentation failed")

    output = result.get("output", {})
    return AISegmentResponse(
        bbox=BBox(**output.get("bbox", {})),
        confidence=output.get("confidence", 0),
        mask=output.get("mask"),
    )

@router.delete("/segment/session/{session_id}")
async def close_segment_session(session_id: str):
    """Close a SAM session to free memory."""
    runpod_input = {
        "task": "segment_session_close",
        "model": "sam2",
        "session_id": session_id,
    }

    await runpod_service.submit_job_sync(
        endpoint_type=EndpointType.OD_ANNOTATION,
        input_data=runpod_input,
        timeout=5,
    )

    return {"status": "closed"}
```

**File:** `apps/api/src/schemas/od.py`

Add new schemas:

```python
class SegmentSessionStartRequest(BaseModel):
    image_id: str
    model: str = "sam2"

class SegmentSessionPointRequest(BaseModel):
    model: str = "sam2"
    point: tuple[float, float]
    label: int = 1
```

### Phase 3: Frontend Changes

**File:** `apps/web/src/app/od/annotate/[datasetId]/[imageId]/page.tsx`

```typescript
// New state for session
const [samSessionId, setSamSessionId] = useState<string | null>(null);
const [samSessionLoading, setSamSessionLoading] = useState(false);

// Start session when entering SAM mode or when image loads
useEffect(() => {
  if (samMode && !samSessionId && !samSessionLoading) {
    startSAMSession();
  }

  // Cleanup session when leaving
  return () => {
    if (samSessionId) {
      closeSAMSession(samSessionId);
    }
  };
}, [samMode, imageId]);

const startSAMSession = async () => {
  setSamSessionLoading(true);
  try {
    const result = await apiClient.startSegmentSession({
      image_id: imageId,
      model: "sam2",
    });
    setSamSessionId(result.session_id);
    toast.success("SAM ready! Click on objects to segment.");
  } catch (error) {
    toast.error("Failed to initialize SAM");
    console.error(error);
  } finally {
    setSamSessionLoading(false);
  }
};

const closeSAMSession = async (sessionId: string) => {
  try {
    await apiClient.closeSegmentSession(sessionId);
  } catch (error) {
    console.error("Failed to close SAM session:", error);
  }
};

// Update click handler to use session
const handleSAMClick = useCallback(async (normalizedX: number, normalizedY: number) => {
  if (!samSessionId || samLoading) return;

  setSamLoading(true);
  setSamPreview(null);

  try {
    // Use session endpoint (fast!)
    const result = await apiClient.segmentSessionPoint(samSessionId, {
      point: [normalizedX, normalizedY],
      label: 1,
    });

    const preview: AIPrediction = {
      bbox: result.bbox,
      label: "SAM segment",
      confidence: result.confidence,
      mask: result.mask,
    };

    setSamPreview(preview);
  } catch (error) {
    toast.error("Segmentation failed");
    console.error(error);
  } finally {
    setSamLoading(false);
  }
}, [samSessionId, samLoading]);
```

**File:** `apps/web/src/lib/api-client.ts`

Add new API methods:

```typescript
async startSegmentSession(params: {
  image_id: string;
  model?: string;
}): Promise<{ session_id: string; image_width: number; image_height: number }> {
  const response = await this.fetch('/api/v1/od/ai/segment/session/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return response.json();
}

async segmentSessionPoint(
  sessionId: string,
  params: { point: [number, number]; label?: number }
): Promise<AISegmentResponse> {
  const response = await this.fetch(`/api/v1/od/ai/segment/session/${sessionId}/point`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
  return response.json();
}

async closeSegmentSession(sessionId: string): Promise<void> {
  await this.fetch(`/api/v1/od/ai/segment/session/${sessionId}`, {
    method: 'DELETE',
  });
}
```

### Phase 4: Testing

1. **Unit tests for session management:**
   - Session creation
   - Session expiry
   - Max sessions eviction
   - Concurrent access

2. **Integration tests:**
   - Start session → segment point → close session
   - Session timeout behavior
   - Multiple concurrent sessions

3. **Performance tests:**
   - Measure session start time
   - Measure per-click latency
   - Memory usage with multiple sessions

### Expected Performance

| Operation | Current | With Sessions |
|-----------|---------|---------------|
| First click | 1-3s | 1-2s (session start) |
| Subsequent clicks | 1-3s | 50-200ms |
| Memory per session | N/A | ~500MB GPU |

## Considerations

### Memory Management
- Sessions hold GPU memory (~500MB each for SAM-ViT-H)
- Limit concurrent sessions (default: 10)
- Auto-expire after inactivity (default: 5 minutes)
- Clean up on worker shutdown

### Error Handling
- Session not found → return error, frontend should restart session
- Session expired → same as not found
- GPU OOM → evict oldest sessions, retry

### Scaling
- Sessions are per-worker (not shared between RunPod workers)
- For multi-worker setup, consider sticky sessions or external cache (Redis)

## Files to Modify

| File | Changes |
|------|---------|
| `workers/od-annotation/models/sam2.py` | Add session management, `start_session()`, `segment_with_session()`, `close_session()` |
| `workers/od-annotation/handler.py` | Add new task handlers |
| `apps/api/src/api/v1/od/ai.py` | Add session endpoints |
| `apps/api/src/schemas/od.py` | Add session schemas |
| `apps/web/src/app/od/annotate/[datasetId]/[imageId]/page.tsx` | Session lifecycle, updated click handler |
| `apps/web/src/lib/api-client.ts` | Add session API methods |

## Future Improvements

1. **Multi-point refinement:** Add/remove points to refine the mask
2. **Box prompt with session:** Draw box using cached embedding
3. **Persistent sessions:** Redis-based session storage for multi-worker
4. **WebSocket:** Real-time updates instead of polling
