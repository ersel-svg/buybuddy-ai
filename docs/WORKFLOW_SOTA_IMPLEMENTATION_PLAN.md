# Workflow SOTA Implementation Plan

## Executive Summary

Bu plan, mevcut 14 workflow node'unu SOTA seviyesine yükseltmeyi ve yeni node'lar eklemeyi (SAM3, Annotation Models) kapsar. Plan, mevcut sistemi bozmadan incremental olarak uygulanacak şekilde tasarlanmıştır.

---

## Phase 0: Current State Analysis

### Mevcut Node'lar (14 Adet)

| Node | Category | Durum | SOTA Seviyesi |
|------|----------|-------|---------------|
| `image_input` | Input | ✅ Çalışıyor | 🟡 Orta |
| `parameter_input` | Input | ✅ Çalışıyor | 🟢 Yeterli |
| `detection` | Model | ✅ Çalışıyor | 🟡 Orta |
| `classification` | Model | ✅ Çalışıyor | 🟡 Orta |
| `embedding` | Model | ✅ Çalışıyor | 🟡 Orta |
| `similarity_search` | Model | ✅ Çalışıyor | 🟢 İyi |
| `segmentation` | Model | 🔴 Placeholder | 🔴 Yok |
| `crop` | Transform | ✅ Çalışıyor | 🟡 Orta |
| `blur_region` | Transform | ✅ Çalışıyor | 🟢 Yeterli |
| `draw_boxes` | Transform | ✅ Çalışıyor | 🟢 Yeterli |
| `condition` | Logic | ✅ Çalışıyor | 🟢 Yeterli |
| `filter` | Logic | ✅ Çalışıyor | 🟡 Orta |
| `grid_builder` | Output | ✅ Çalışıyor | 🟢 İyi |
| `json_output` | Output | ✅ Çalışıyor | 🟢 Yeterli |

### Mevcut Entegrasyonlar

| Sistem | Entegrasyon | Durum |
|--------|-------------|-------|
| OD Training (RT-DETR, D-FINE) | `od_trained_models` tablosu | ✅ Hazır |
| CLS Training (ViT, ConvNeXt, etc.) | `cls_trained_models` tablosu | ✅ Hazır |
| Embedding Models (DINOv2, CLIP) | `trained_models` tablosu | ✅ Hazır |
| SAM3 Segmentation | `od-annotation` worker | 🟡 Ayrı sistemde |
| Florence2 Annotation | `od-annotation` worker | 🟡 Ayrı sistemde |
| Qdrant Vector DB | `similarity_search` block | ✅ Çalışıyor |

---

## Phase 1: SOTA Node Configurations (Breaking Change Yok)

### 1.1 Detection Node SOTA Update

**Dosya:** `apps/api/src/services/workflow/blocks/model_blocks.py`

**Mevcut Config:**
```python
{
    "model_id": "yolov8n",
    "model_source": "pretrained",
    "confidence": 0.5,
    "iou_threshold": 0.45
}
```

**SOTA Config (Geriye Uyumlu):**
```python
{
    # Existing (backward compatible)
    "model_id": "yolov8n",
    "model_source": "pretrained" | "trained",
    "confidence": 0.5,
    "iou_threshold": 0.45,

    # NEW SOTA additions (all optional with defaults)
    "inference": {
        "max_detections": 300,
        "class_filter": null,  # [0, 1, 5] = only these classes
        "agnostic_nms": false,
        "half_precision": true,  # FP16 for speed
        "multi_scale": false,  # TTA with multiple scales
    },
    "output": {
        "include_scores": true,
        "include_class_names": true,
        "coordinate_format": "xyxy",  # xyxy | xywh | cxcywh
        "normalize_coords": false,
    },
    "augmentation": {
        "tta_enabled": false,
        "tta_scales": [0.5, 1.0, 1.5],
        "tta_flip": true,
    }
}
```

**Entegrasyon:**
- `model_source: "trained"` → `od_trained_models` tablosundan model çek
- Checkpoint URL'den dinamik yükleme
- Class mapping'i trained model'den al

### 1.2 Classification Node SOTA Update

**Mevcut Config:**
```python
{
    "model_id": "clip-vit-b-32",
    "top_k": 5
}
```

**SOTA Config:**
```python
{
    # Existing
    "model_id": "clip-vit-b-32",
    "model_source": "pretrained" | "trained",
    "top_k": 5,

    # NEW SOTA additions
    "inference": {
        "temperature": 1.0,  # softmax temperature
        "threshold": null,  # min confidence for prediction
        "ensemble": false,  # use multiple crops
    },
    "tta": {
        "enabled": false,
        "transforms": ["hflip", "five_crop"],
        "merge_mode": "mean",  # mean | max
    },
    "multi_crop": {
        "enabled": false,
        "scales": [1.0, 0.875, 0.75],
    }
}
```

**Entegrasyon:**
- `model_source: "trained"` → `cls_trained_models` tablosundan model çek
- ViT, ConvNeXt, EfficientNet, Swin desteklenen modeller
- Class names trained model'den alınır

### 1.3 Embedding Node SOTA Update

**Mevcut Config:**
```python
{
    "model_id": "dinov2-base",
    "normalize": true
}
```

**SOTA Config:**
```python
{
    # Existing
    "model_id": "dinov2-base",
    "model_source": "pretrained" | "trained",
    "normalize": true,

    # NEW SOTA additions
    "inference": {
        "pooling": "cls",  # cls | mean | gem (GeM is SOTA)
        "layer": -1,  # which transformer layer
        "batch_size": 1,
    },
    "multi_scale": {
        "enabled": false,
        "scales": [0.5, 1.0, 1.5],
        "merge": "concat",  # concat | mean
    },
    "pca": {
        "enabled": false,
        "n_components": 256,  # dimensionality reduction
    }
}
```

**Entegrasyon:**
- `model_source: "trained"` → `trained_models` tablosundan fine-tuned model çek
- ArcFace/SupCon ile eğitilmiş modeller desteklenir

### 1.4 Similarity Search SOTA Update

**Mevcut Config:**
```python
{
    "collection": "products",
    "top_k": 5,
    "threshold": 0.7
}
```

**SOTA Config:**
```python
{
    # Existing
    "collection": "products",
    "top_k": 10,
    "threshold": 0.7,

    # NEW SOTA additions
    "search": {
        "strategy": "nearest",  # nearest | recommend | discover
        "with_payload": true,
        "with_vectors": false,
    },
    "hybrid": {
        "enabled": false,
        "sparse_collection": null,  # for BM25 sparse vectors
        "sparse_weight": 0.3,
        "dense_weight": 0.7,
    },
    "rerank": {
        "enabled": false,
        "model": "cross-encoder",  # cross-encoder reranking
        "top_n": 100,
    },
    "filter": {
        "must": [],  # required conditions
        "should": [],  # optional conditions
        "must_not": [],  # exclusions
    },
    "grouping": {
        "enabled": false,
        "group_by": "product_id",
        "group_size": 3,
    }
}
```

### 1.5 Crop Node SOTA Update

**SOTA Config:**
```python
{
    # Existing
    "padding": 0,
    "min_size": 32,

    # NEW SOTA additions
    "mode": "bbox",  # bbox | detection | mask
    "padding_type": "absolute",  # absolute | relative
    "multi_crop": {
        "strategy": "all",  # all | largest | highest_confidence | center
        "max_crops": 100,
        "min_area": 1024,  # 32x32 minimum
    },
    "resize": {
        "enabled": false,
        "target_size": [224, 224],
        "interpolation": "bicubic",
    },
    "context": {
        "include_context": false,
        "context_ratio": 1.5,  # 1.5x bbox size
    }
}
```

### 1.6 Filter Node SOTA Update

**SOTA Config:**
```python
{
    # Existing
    "field": "confidence",
    "operator": "greater_than",
    "value": 0.5,

    # NEW SOTA additions
    "filters": [
        {"field": "confidence", "op": "gt", "value": 0.5},
        {"field": "class", "op": "in", "value": ["person", "car"]},
        {"field": "area", "op": "between", "value": [1000, 100000]},
    ],
    "logic": "and",  # and | or

    "nms": {
        "enabled": false,
        "iou_threshold": 0.5,
        "class_agnostic": false,
    },
    "soft_nms": {
        "enabled": false,
        "sigma": 0.5,
        "method": "gaussian",  # gaussian | linear
    },
    "area_filter": {
        "min_area": null,
        "max_area": null,
        "min_relative_area": 0.001,  # % of image
    }
}
```

---

## Phase 2: New Nodes (Yeni Block'lar)

### 2.1 SAM3 Segmentation Node

**Dosya:** `apps/api/src/services/workflow/blocks/model_blocks.py`

**Yeni Block Class:**
```python
class SAM3SegmentationBlock(BaseBlock):
    """
    SAM3 (Segment Anything Model 3) ile görüntü segmentasyonu.

    Text prompt ile istenen objeleri segmentlere ayırır.
    Video segmentation modelinin single-frame adaptasyonu.
    """

    block_type = "sam3_segmentation"
    display_name = "SAM3 Segmentation"
    description = "Segment objects using text prompts with SAM3"
    category = "model"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
        {"name": "detections", "type": "detections", "required": False},  # Optional bbox hints
    ]

    output_ports = [
        {"name": "masks", "type": "masks", "description": "Segmentation masks"},
        {"name": "annotated_image", "type": "image", "description": "Image with mask overlay"},
        {"name": "mask_count", "type": "integer"},
    ]

    config_schema = {
        "prompt": {
            "type": "string",
            "description": "Text description of objects to segment (e.g., 'red bottle', 'all products')",
            "default": "",
        },
        "points": {
            "type": "array",
            "description": "Optional point prompts [[x, y], ...]",
            "default": None,
        },
        "use_detections": {
            "type": "boolean",
            "description": "Use input detections as box prompts",
            "default": False,
        },
        "mask_threshold": {
            "type": "number",
            "default": 0.0,
            "description": "Mask confidence threshold",
        },
        "min_area_ratio": {
            "type": "number",
            "default": 0.0005,  # 0.05% of image
            "description": "Minimum mask area as ratio of image",
        },
        "max_masks": {
            "type": "integer",
            "default": 100,
            "description": "Maximum number of masks to return",
        },
        "return_largest": {
            "type": "boolean",
            "default": False,
            "description": "Only return the largest mask per prompt",
        },
        "output_format": {
            "type": "string",
            "enum": ["binary", "rle", "polygon"],
            "default": "binary",
        }
    }
```

**Config:**
```python
{
    "prompt": "bottles on the shelf",  # Text prompt
    "points": null,  # Optional point prompts
    "use_detections": false,  # Use bbox from detection node
    "mask_threshold": 0.0,
    "min_area_ratio": 0.0005,
    "max_masks": 100,
    "return_largest": false,
    "output_format": "binary",  # binary | rle | polygon
}
```

### 2.2 Florence2 Annotation Node

**Yeni Block Class:**
```python
class Florence2AnnotationBlock(BaseBlock):
    """
    Florence2 ile otomatik obje detection/captioning.

    Zero-shot ve few-shot annotation için kullanılır.
    """

    block_type = "florence2_annotation"
    display_name = "Florence2 Auto-Annotate"
    description = "Auto-detect objects with Florence2 vision-language model"
    category = "model"

    input_ports = [
        {"name": "image", "type": "image", "required": True},
    ]

    output_ports = [
        {"name": "detections", "type": "detections"},
        {"name": "captions", "type": "array"},
        {"name": "annotated_image", "type": "image"},
    ]

    config_schema = {
        "task": {
            "type": "string",
            "enum": [
                "object_detection",
                "dense_region_caption",
                "region_proposal",
                "caption",
                "detailed_caption",
                "ocr",
                "ocr_with_region",
            ],
            "default": "object_detection",
        },
        "text_prompt": {
            "type": "string",
            "description": "Optional text prompt for guided detection",
            "default": None,
        },
        "confidence": {
            "type": "number",
            "default": 0.3,
        }
    }
```

**Config:**
```python
{
    "task": "object_detection",  # or "dense_region_caption", "ocr", etc.
    "text_prompt": null,  # Optional: "find all red bottles"
    "confidence": 0.3,
}
```

### 2.3 Transform Node (Enhanced)

**Yeni Block Class:**
```python
class TransformBlock(BaseBlock):
    """
    SOTA görüntü dönüşümleri.

    Training augmentasyonları da dahil tüm transform işlemleri.
    """

    block_type = "transform"
    display_name = "Image Transform"
    description = "Apply SOTA image transformations"
    category = "transform"

    config_schema = {
        "transforms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object"}
                }
            },
            "default": [],
        },
        "preset": {
            "type": "string",
            "enum": ["none", "imagenet_normalize", "clip_normalize", "detection_preprocess", "classification_preprocess"],
            "default": "none",
        }
    }
```

### 2.4 Batch Process Node

**Yeni Block Class:**
```python
class BatchProcessBlock(BaseBlock):
    """
    Birden fazla görüntüyü batch olarak işler.

    Parallel processing ve batch inference için.
    """

    block_type = "batch_process"
    display_name = "Batch Process"
    description = "Process multiple images in batch"
    category = "logic"

    input_ports = [
        {"name": "images", "type": "array", "required": True},
    ]

    output_ports = [
        {"name": "results", "type": "array"},
        {"name": "count", "type": "integer"},
    ]
```

---

## Phase 3: Trained Model Integration

### 3.1 Model Loader Service

**Yeni Dosya:** `apps/api/src/services/workflow/model_loader.py`

```python
"""
Trained model loader for workflow blocks.

Supports:
- od_trained_models (RT-DETR, D-FINE, YOLO-NAS)
- cls_trained_models (ViT, ConvNeXt, EfficientNet, Swin)
- trained_models (Fine-tuned embeddings)
"""

from typing import Optional, Any
import httpx
import torch
from functools import lru_cache

from services.supabase import supabase_service


class ModelLoader:
    """Singleton model loader with caching."""

    _instance = None
    _model_cache: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def load_detection_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> tuple[Any, dict]:
        """
        Load detection model from pretrained or trained source.

        Returns: (model, class_mapping)
        """
        cache_key = f"detection:{model_source}:{model_id}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if model_source == "pretrained":
            # Load from wf_pretrained_models
            result = supabase_service.client.table("wf_pretrained_models")\
                .select("*").eq("id", model_id).single().execute()

            if not result.data:
                raise ValueError(f"Pretrained model {model_id} not found")

            model_info = result.data
            model_path = model_info["model_path"]
            classes = model_info.get("classes", [])

            # Load with ultralytics or custom loader
            from ultralytics import YOLO
            model = YOLO(model_path)
            class_mapping = {i: c for i, c in enumerate(classes)}

        else:  # trained
            # Load from od_trained_models
            result = supabase_service.client.table("od_trained_models")\
                .select("*").eq("id", model_id).single().execute()

            if not result.data:
                raise ValueError(f"Trained model {model_id} not found")

            model_info = result.data
            checkpoint_url = model_info["checkpoint_url"]
            model_type = model_info["model_type"]  # rt-detr, d-fine, yolo-nas
            class_mapping = model_info.get("class_mapping", {})

            # Download checkpoint
            model = await self._load_from_url(checkpoint_url, model_type)

        self._model_cache[cache_key] = (model, class_mapping)
        return model, class_mapping

    async def load_classification_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> tuple[Any, dict]:
        """Load classification model."""
        cache_key = f"classification:{model_source}:{model_id}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if model_source == "pretrained":
            # Load pretrained CLIP, DINOv2, etc.
            result = supabase_service.client.table("wf_pretrained_models")\
                .select("*").eq("id", model_id).single().execute()

            model_info = result.data
            # ... load logic

        else:  # trained
            # Load from cls_trained_models
            result = supabase_service.client.table("cls_trained_models")\
                .select("*").eq("id", model_id).single().execute()

            if not result.data:
                raise ValueError(f"Trained model {model_id} not found")

            model_info = result.data
            checkpoint_url = model_info["checkpoint_url"]
            model_type = model_info["model_type"]  # vit, convnext, etc.
            class_mapping = model_info.get("class_mapping", {})

            model = await self._load_classification_checkpoint(
                checkpoint_url, model_type, len(class_mapping)
            )

        self._model_cache[cache_key] = (model, class_mapping)
        return model, class_mapping

    async def load_embedding_model(
        self,
        model_id: str,
        model_source: str = "pretrained"
    ) -> Any:
        """Load embedding model."""
        cache_key = f"embedding:{model_source}:{model_id}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        if model_source == "pretrained":
            # DINOv2, CLIP pretrained
            result = supabase_service.client.table("wf_pretrained_models")\
                .select("*").eq("id", model_id).single().execute()

            # ... load logic

        else:  # trained
            # Load from trained_models (fine-tuned embeddings)
            result = supabase_service.client.table("trained_models")\
                .select("*, training_run:training_runs(*), embedding_model:embedding_models(*)")\
                .eq("id", model_id).single().execute()

            # ... load logic

        self._model_cache[cache_key] = model
        return model

    async def _load_from_url(self, url: str, model_type: str) -> Any:
        """Download and load model from URL."""
        import tempfile
        import os

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                f.write(response.content)
                temp_path = f.name

        try:
            if model_type in ["rt-detr", "d-fine"]:
                # Load RT-DETR or D-FINE
                from models.rtdetr import RTDETR  # Custom loader
                model = RTDETR.load(temp_path)
            elif model_type == "yolo-nas":
                from super_gradients.training import models
                model = models.get("yolo_nas_s", checkpoint_path=temp_path)
            else:
                model = torch.load(temp_path, map_location="cpu")
        finally:
            os.unlink(temp_path)

        return model
```

### 3.2 Database Schema Updates

**Migration:** `migrations/051_workflow_model_integration.sql`

```sql
-- Add model_source column to track pretrained vs trained
ALTER TABLE wf_pretrained_models
ADD COLUMN IF NOT EXISTS model_category VARCHAR(50) DEFAULT 'general';

-- Create view for unified model access
CREATE OR REPLACE VIEW wf_all_models AS
SELECT
    id,
    name,
    'detection' as model_type,
    'pretrained' as source,
    model_path as checkpoint_url,
    classes::jsonb as class_mapping,
    class_count,
    is_active,
    true as is_default,
    created_at
FROM wf_pretrained_models
WHERE model_type = 'detection'

UNION ALL

SELECT
    id::text,
    name,
    'detection' as model_type,
    'trained' as source,
    checkpoint_url,
    class_mapping::jsonb,
    class_count,
    is_active,
    is_default,
    created_at
FROM od_trained_models
WHERE is_active = true OR is_default = true

UNION ALL

SELECT
    id::text,
    name,
    'classification' as model_type,
    'trained' as source,
    checkpoint_url,
    class_mapping::jsonb,
    class_count,
    is_active,
    is_default,
    created_at
FROM cls_trained_models
WHERE is_active = true OR is_default = true

UNION ALL

SELECT
    id::text,
    name,
    'embedding' as model_type,
    'trained' as source,
    null as checkpoint_url,
    null as class_mapping,
    null as class_count,
    is_active,
    is_default,
    created_at
FROM trained_models
WHERE is_active = true OR is_default = true;
```

---

## Phase 4: API Updates

### 4.1 Models API Enhancement

**Dosya:** `apps/api/src/api/v1/workflows/models.py`

```python
# Add new endpoint for flattened model list
@router.get("/list")
async def get_workflow_models_list(
    model_type: Optional[str] = Query(None),
    source: Optional[str] = Query(None),  # pretrained | trained
    include_inactive: bool = Query(False),
) -> dict:
    """
    Get flattened list of all models for UI dropdowns.

    Returns unified list from:
    - wf_pretrained_models
    - od_trained_models
    - cls_trained_models
    - trained_models (embeddings)
    """
    items = []

    # Fetch pretrained
    if source is None or source == "pretrained":
        pretrained = await _fetch_pretrained_models(model_type, include_inactive)
        items.extend(pretrained)

    # Fetch trained
    if source is None or source == "trained":
        trained = await _fetch_trained_models(model_type, include_inactive)
        items.extend(trained)

    return {
        "items": items,
        "total": len(items),
    }


async def _fetch_trained_models(model_type: Optional[str], include_inactive: bool) -> list:
    """Fetch trained models from all training tables."""
    items = []

    # OD trained models
    if model_type is None or model_type == "detection":
        query = supabase_service.client.table("od_trained_models").select("*")
        if not include_inactive:
            query = query.or_("is_active.eq.true,is_default.eq.true")
        result = query.execute()

        for m in result.data or []:
            items.append({
                "id": m["id"],
                "name": m["name"],
                "model_type": m.get("model_type", "rt-detr"),
                "category": "detection",
                "source": "trained",
                "is_active": m.get("is_active", False),
                "is_default": m.get("is_default", False),
                "metrics": {
                    "map": m.get("map"),
                    "map_50": m.get("map_50"),
                },
                "created_at": m.get("created_at"),
            })

    # CLS trained models
    if model_type is None or model_type == "classification":
        query = supabase_service.client.table("cls_trained_models").select("*")
        if not include_inactive:
            query = query.or_("is_active.eq.true,is_default.eq.true")
        result = query.execute()

        for m in result.data or []:
            items.append({
                "id": m["id"],
                "name": m["name"],
                "model_type": m.get("model_type", "vit"),
                "category": "classification",
                "source": "trained",
                "is_active": m.get("is_active", False),
                "is_default": m.get("is_default", False),
                "metrics": {
                    "accuracy": m.get("test_accuracy"),
                },
                "created_at": m.get("created_at"),
            })

    # Embedding trained models
    if model_type is None or model_type == "embedding":
        query = supabase_service.client.table("trained_models").select(
            "*, embedding_model:embedding_models(model_family, embedding_dim)"
        )
        if not include_inactive:
            query = query.or_("is_active.eq.true,is_default.eq.true")
        result = query.execute()

        for m in result.data or []:
            emb_info = m.get("embedding_model", {}) or {}
            items.append({
                "id": m["id"],
                "name": m["name"],
                "model_type": emb_info.get("model_family", "dinov2"),
                "category": "embedding",
                "source": "trained",
                "is_active": m.get("is_active", False),
                "is_default": m.get("is_default", False),
                "metrics": {
                    "recall_at_1": m.get("test_metrics", {}).get("recall_at_1"),
                },
                "embedding_dim": emb_info.get("embedding_dim"),
                "created_at": m.get("created_at"),
            })

    return items
```

### 4.2 Blocks API Enhancement

**Dosya:** `apps/api/src/services/workflow/__init__.py`

```python
# Update block metadata to include SOTA config schemas
BLOCK_METADATA = {
    "detection": {
        "type": "detection",
        "display_name": "Object Detection",
        "description": "Detect objects using YOLO, RT-DETR, or custom trained models",
        "category": "model",
        "input_ports": [
            {"name": "image", "type": "image", "required": True}
        ],
        "output_ports": [
            {"name": "detections", "type": "detections"},
            {"name": "annotated_image", "type": "image"},
            {"name": "count", "type": "integer"},
        ],
        "config_schema": {
            "model_id": {
                "type": "string",
                "description": "Model ID from pretrained or trained models",
                "required": True,
            },
            "model_source": {
                "type": "string",
                "enum": ["pretrained", "trained"],
                "default": "pretrained",
            },
            "confidence": {
                "type": "number",
                "default": 0.5,
                "min": 0,
                "max": 1,
            },
            "iou_threshold": {
                "type": "number",
                "default": 0.45,
                "min": 0,
                "max": 1,
            },
            # SOTA additions
            "max_detections": {
                "type": "integer",
                "default": 300,
            },
            "class_filter": {
                "type": "array",
                "items": {"type": "string"},
                "default": None,
            },
            "half_precision": {
                "type": "boolean",
                "default": True,
            },
        },
        "supports_trained_models": True,
        "trained_model_table": "od_trained_models",
    },
    # ... other blocks
}
```

---

## Phase 5: Frontend Updates

### 5.1 Block Config Panel

**Dosya:** `apps/web/src/app/workflows/[id]/components/BlockConfigPanel.tsx`

```typescript
// Model selector with pretrained/trained switch
interface ModelSelectorProps {
  category: "detection" | "classification" | "embedding";
  value: { model_id: string; model_source: string };
  onChange: (value: { model_id: string; model_source: string }) => void;
}

function ModelSelector({ category, value, onChange }: ModelSelectorProps) {
  const { data: models } = useQuery({
    queryKey: ["workflow-models-list", category],
    queryFn: () => apiClient.getWorkflowModelsList({ model_type: category }),
  });

  const pretrainedModels = models?.items.filter(m => m.source === "pretrained") || [];
  const trainedModels = models?.items.filter(m => m.source === "trained") || [];

  return (
    <div className="space-y-3">
      <Tabs value={value.model_source} onValueChange={(v) => onChange({ ...value, model_source: v })}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="pretrained">
            Pretrained ({pretrainedModels.length})
          </TabsTrigger>
          <TabsTrigger value="trained">
            Your Models ({trainedModels.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pretrained">
          <Select
            value={value.model_id}
            onValueChange={(v) => onChange({ ...value, model_id: v })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select model..." />
            </SelectTrigger>
            <SelectContent>
              {pretrainedModels.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex items-center gap-2">
                    <span>{model.name}</span>
                    <Badge variant="secondary" className="text-xs">
                      {model.model_type}
                    </Badge>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </TabsContent>

        <TabsContent value="trained">
          {trainedModels.length === 0 ? (
            <div className="text-center py-4 text-muted-foreground">
              <p>No trained models yet.</p>
              <Link href={`/${category}`} className="text-primary">
                Train your first model →
              </Link>
            </div>
          ) : (
            <Select
              value={value.model_id}
              onValueChange={(v) => onChange({ ...value, model_id: v })}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select model..." />
              </SelectTrigger>
              <SelectContent>
                {trainedModels.map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.is_default && <Star className="h-3 w-3 fill-yellow-500" />}
                      </div>
                      {model.metrics?.map && (
                        <span className="text-xs text-muted-foreground">
                          mAP: {(model.metrics.map * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

### 5.2 Node Handles Update

**Multiple Input/Output Ports:**

```typescript
function WorkflowNodeComponent({ data, selected }: { data: WorkflowNodeData; selected?: boolean }) {
  // Get port definitions from block metadata
  const inputPorts = data.input_ports || [{ name: "image", type: "image" }];
  const outputPorts = data.output_ports || [{ name: "output", type: "any" }];

  return (
    <div className={cn("workflow-node", selected && "selected")} style={{ borderColor: color }}>
      {/* Input Handles */}
      {inputPorts.map((port, index) => (
        <Handle
          key={`input-${port.name}`}
          type="target"
          position={Position.Left}
          id={port.name}
          style={{ top: `${((index + 1) / (inputPorts.length + 1)) * 100}%` }}
          className={cn(
            "!w-3 !h-3 !border-2 !border-white",
            port.type === "image" && "!bg-blue-500",
            port.type === "detections" && "!bg-orange-500",
            port.type === "embeddings" && "!bg-green-500",
          )}
        />
      ))}

      {/* Node Content */}
      <div className="node-content">
        {/* ... */}
      </div>

      {/* Output Handles */}
      {outputPorts.map((port, index) => (
        <Handle
          key={`output-${port.name}`}
          type="source"
          position={Position.Right}
          id={port.name}
          style={{ top: `${((index + 1) / (outputPorts.length + 1)) * 100}%` }}
          className={cn(
            "!w-3 !h-3 !border-2 !border-white",
            port.type === "image" && "!bg-blue-500",
            port.type === "detections" && "!bg-orange-500",
            port.type === "embeddings" && "!bg-green-500",
          )}
        />
      ))}
    </div>
  );
}
```

---

## Phase 6: Implementation Order

### Sprint 1: Foundation (Week 1)
- [ ] **1.1** Add SOTA config fields to existing blocks (backward compatible defaults)
- [ ] **1.2** Create `ModelLoader` service for trained model loading
- [ ] **1.3** Update `/models/list` API endpoint
- [ ] **1.4** Add frontend `ModelSelector` component

### Sprint 2: Detection & Classification (Week 2)
- [ ] **2.1** Update `DetectionBlock` with trained model support
- [ ] **2.2** Update `ClassificationBlock` with trained model support
- [ ] **2.3** Add model metrics display in UI
- [ ] **2.4** Test with actual trained models from OD/CLS systems

### Sprint 3: Embedding & Search (Week 3)
- [ ] **3.1** Update `EmbeddingBlock` with trained model support
- [ ] **3.2** Add GeM pooling and multi-scale options
- [ ] **3.3** Update `SimilaritySearchBlock` with hybrid search
- [ ] **3.4** Add reranking support

### Sprint 4: SAM3 & Florence2 (Week 4)
- [ ] **4.1** Port SAM3 from od-annotation worker to workflow block
- [ ] **4.2** Create `SAM3SegmentationBlock`
- [ ] **4.3** Port Florence2 and create `Florence2AnnotationBlock`
- [ ] **4.4** Add text prompt UI for SAM3

### Sprint 5: Transform & Logic (Week 5)
- [ ] **5.1** Create enhanced `TransformBlock`
- [ ] **5.2** Update `FilterBlock` with NMS options
- [ ] **5.3** Update `CropBlock` with context cropping
- [ ] **5.4** Add `BatchProcessBlock`

### Sprint 6: Testing & Polish (Week 6)
- [ ] **6.1** End-to-end pipeline testing
- [ ] **6.2** Performance optimization (model caching)
- [ ] **6.3** Error handling improvements
- [ ] **6.4** Documentation

---

## Complete Pipeline Examples

### Example 1: Product Detection → Classification → Search

```yaml
Pipeline: "Product Recognition"
Nodes:
  - id: input_1
    type: image_input
    config:
      source: url

  - id: detect_1
    type: detection
    config:
      model_id: "{trained_od_model_id}"
      model_source: trained
      confidence: 0.5

  - id: crop_1
    type: crop
    config:
      multi_crop:
        strategy: all
        max_crops: 50

  - id: classify_1
    type: classification
    config:
      model_id: "{trained_cls_model_id}"
      model_source: trained
      top_k: 3

  - id: embed_1
    type: embedding
    config:
      model_id: dinov2-base
      normalize: true

  - id: search_1
    type: similarity_search
    config:
      collection: products
      top_k: 5
      threshold: 0.7

  - id: grid_1
    type: grid_builder
    config:
      columns: 4

Edges:
  - input_1.image → detect_1.image
  - detect_1.detections → crop_1.detections
  - input_1.image → crop_1.image
  - crop_1.crops → classify_1.images
  - crop_1.crops → embed_1.images
  - embed_1.embeddings → search_1.query_vectors
  - search_1.results → grid_1.items
```

### Example 2: SAM3 Segmentation Pipeline

```yaml
Pipeline: "Shelf Segmentation"
Nodes:
  - id: input_1
    type: image_input

  - id: sam3_1
    type: sam3_segmentation
    config:
      prompt: "products on shelf"
      max_masks: 100
      output_format: binary

  - id: crop_1
    type: crop
    config:
      mode: mask  # Use masks instead of bboxes

  - id: embed_1
    type: embedding
    config:
      model_id: dinov2-base

  - id: search_1
    type: similarity_search
    config:
      collection: products
      top_k: 3

Edges:
  - input_1.image → sam3_1.image
  - sam3_1.masks → crop_1.masks
  - input_1.image → crop_1.image
  - crop_1.crops → embed_1.images
  - embed_1.embeddings → search_1.query_vectors
```

### Example 3: Auto-Annotation Pipeline

```yaml
Pipeline: "Auto Annotate Dataset"
Nodes:
  - id: input_1
    type: image_input

  - id: florence_1
    type: florence2_annotation
    config:
      task: object_detection
      confidence: 0.3

  - id: filter_1
    type: filter
    config:
      filters:
        - field: confidence
          op: gt
          value: 0.5
      nms:
        enabled: true
        iou_threshold: 0.5

  - id: sam3_1
    type: sam3_segmentation
    config:
      use_detections: true  # Use Florence2 boxes

  - id: output_1
    type: json_output
    config:
      format: coco  # COCO annotation format

Edges:
  - input_1.image → florence_1.image
  - florence_1.detections → filter_1.items
  - filter_1.filtered → sam3_1.detections
  - input_1.image → sam3_1.image
  - sam3_1.masks → output_1.data
```

---

## Risk Mitigation

### Breaking Changes Prevention
1. Tüm yeni config alanları **optional** ve **default değerli**
2. Eski workflow'lar aynen çalışmaya devam eder
3. Migration script'leri geriye dönük uyumlu

### Performance Considerations
1. Model caching ile tekrar yükleme önlenir
2. Lazy loading - model sadece gerektiğinde yüklenir
3. Batch processing desteği

### Error Handling
1. Model yükleme hataları graceful handle edilir
2. Trained model bulunamazsa pretrained fallback
3. Timeout ve retry mekanizmaları

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Trained model integration | OD, CLS, Embedding 100% |
| SAM3 node çalışır durumda | ✅ |
| Florence2 node çalışır durumda | ✅ |
| Mevcut pipeline'lar kırılmaz | 0 regression |
| Model load time (cached) | < 100ms |
| Model load time (first) | < 5s |
