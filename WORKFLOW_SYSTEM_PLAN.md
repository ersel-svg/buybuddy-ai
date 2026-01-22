# BuyBuddy AI - Perakende CV Workflow Sistemi

## Executive Summary

Bu doküman, perakende sektörüne özel bir Computer Vision workflow sistemi tasarımını içermektedir. Mevcut sistemdeki trained modelleri (OD, Classification, Embedding) birbirine bağlayarak hızlı test ve inference yapılabilecek bir pipeline builder oluşturulacaktır.

---

## 1. Mevcut Model Envanteri

### 1.1 Object Detection (`od_trained_models`)
- **Model Tipleri:** RF-DETR, RT-DETR, YOLO-NAS
- **Kullanım:** Ürün detection, raf yapısı, fiyat etiketi, insan tespiti
- **Output:** Bounding boxes, class labels, confidence scores

### 1.2 Classification (`cls_trained_models`)
- **Model Tipleri:** ViT, ConvNeXt, EfficientNet, Swin, DINOv2, CLIP
- **Kullanım:** Ürün kategorisi, marka tanıma, kalite değerlendirme
- **Output:** Class predictions, confidence scores

### 1.3 Embedding/Similarity (`trained_models` + `embedding_models`)
- **Model Tipleri:** DINOv2, DINOv3, CLIP
- **Kullanım:** SKU identification, ürün eşleştirme
- **Output:** Embedding vectors → Qdrant similarity search → Product match

---

## 2. Perakende Use Cases & Pipeline'ları

### 2.1 Temel Pipeline: Ürün Tanıma

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raf Görseli │───▶│   Product   │───▶│   Dynamic   │───▶│  Embedding  │
│   (Input)    │    │  Detection  │    │    Crop     │    │  Extraction │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ürün      │◀───│   Product   │◀───│   Qdrant    │◀───│  Similarity │
│   Bilgisi   │    │   Database  │    │   Search    │    │   Match     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.2 Privacy Pipeline: İnsan Blur

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raf Görseli │───▶│   Person    │───▶│  Condition  │───▶│    Blur     │
│   (Input)    │    │  Detection  │    │  (has_det?) │    │   Region    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.3 Fiyat Okuma Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raf Görseli │───▶│  Price Tag  │───▶│   Dynamic   │───▶│     OCR     │
│   (Input)    │    │  Detection  │    │    Crop     │    │   Extract   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                         ┌─────────────┐
                                                         │   Regex     │
                                                         │   Parse     │
                                                         └─────────────┘
```

### 2.4 Boş Raf Tespiti Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raf Görseli │───▶│   Shelf     │───▶│  Empty Area │───▶│  Condition  │
│   (Input)    │    │  Detection  │    │  Analysis   │    │   Alert     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2.5 Kombine Pipeline: Full Retail Analysis

```
                                    ┌─────────────┐
                              ┌────▶│   Person    │────▶ Blur ────┐
                              │     │  Detection  │               │
┌─────────────┐    ┌─────────┴─┐   └─────────────┘               │
│  Raf Görseli │───▶│  Parallel  │                                │
│   (Input)    │    │   Split    │   ┌─────────────┐              │    ┌─────────┐
└─────────────┘    └─────────┬─┘   │   Product   │              ├───▶│  Merge  │
                              │  ┌─▶│  Detection  │──▶ Crop ─┐   │    │ Results │
                              │  │  └─────────────┘          │   │    └─────────┘
                              │  │                           ▼   │
                              └──┤  ┌─────────────┐    ┌─────────┴─┐
                                 │  │  Price Tag  │    │ Embedding │
                                 └─▶│  Detection  │──▶ │ + Match   │
                                    └─────────────┘    └───────────┘
```

---

## 3. Workflow Block Kategorileri

### 3.1 Input Blocks

| Block | Açıklama | Outputs |
|-------|----------|---------|
| `image_input` | Tek görsel girişi | `image` |
| `batch_images` | Çoklu görsel girişi | `images[]` |
| `url_fetch` | URL'den görsel çekme | `image` |

### 3.2 Model Blocks - Detection

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `od_model` | Trained OD model inference | `model_id`, `confidence`, `nms_threshold` | `detections[]` |
| `yolo_pretrained` | YOLO pretrained (person, face, etc.) | `model_name`, `classes_filter` | `detections[]` |
| `face_detection` | Yüz tespiti (RetinaFace/YOLO) | `confidence` | `detections[]` |

### 3.3 Model Blocks - Classification

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `cls_model` | Trained CLS model inference | `model_id`, `top_k` | `predictions[]` |
| `clip_classify` | CLIP zero-shot classification | `labels[]` | `predictions[]` |

### 3.4 Model Blocks - Embedding

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `embedding_extract` | Embedding çıkarma | `model_id` | `embedding_vector` |
| `similarity_search` | Qdrant'ta arama | `collection`, `top_k`, `threshold` | `matches[]` |
| `product_lookup` | Product DB'de arama | `match_result` | `product_info` |

### 3.5 Transformation Blocks

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `dynamic_crop` | Detection'lardan crop | `padding`, `min_size` | `crops[]` |
| `resize` | Boyutlandırma | `width`, `height`, `keep_aspect` | `image` |
| `normalize` | Normalizasyon | `mean`, `std` | `image` |
| `grayscale` | Gri tonlama | - | `image` |
| `rotate` | Döndürme | `angle` | `image` |
| `flip` | Çevirme | `horizontal`, `vertical` | `image` |
| `pad` | Padding ekleme | `size`, `color` | `image` |
| `letterbox` | Letterbox resize | `target_size` | `image` |

### 3.6 Classical CV Blocks

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `blur` | Gaussian/Median blur | `kernel_size`, `type` | `image` |
| `threshold` | Binary threshold | `value`, `type`, `adaptive` | `image` |
| `edge_detection` | Canny edge | `low`, `high` | `image` |
| `contour_detection` | Kontur bulma | `mode`, `method` | `contours[]` |
| `morphology` | Dilate/Erode/Open/Close | `operation`, `kernel` | `image` |
| `color_filter` | HSV renk filtreleme | `lower`, `upper` | `mask` |
| `histogram_eq` | Histogram equalization | `clahe` | `image` |
| `perspective_correct` | Perspektif düzeltme | `points` | `image` |

### 3.7 OCR Blocks

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `ocr_extract` | Metin çıkarma (PaddleOCR) | `lang`, `det_only` | `text`, `boxes[]` |
| `barcode_read` | Barkod/QR okuma | `formats[]` | `codes[]` |
| `regex_extract` | Regex ile parse | `pattern`, `groups` | `matches[]` |
| `price_parse` | Fiyat çıkarma | `currency`, `format` | `price` |
| `date_parse` | Tarih çıkarma | `formats[]` | `date` |

### 3.8 Logic & Flow Control

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `condition` | Boolean condition | `expression` | `true_branch`, `false_branch` |
| `switch` | Multi-way branch | `cases[]` | `case_outputs[]` |
| `filter` | Detection/result filtreleme | `expression` | `filtered[]` |
| `loop` | Her detection için işlem | `block_sequence` | `results[]` |
| `parallel` | Paralel execution | `branches[]` | `branch_outputs[]` |
| `merge` | Sonuçları birleştir | `strategy` | `merged` |
| `aggregate` | İstatistik hesapla | `operation` | `result` |

### 3.9 Visualization & Annotation

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `draw_boxes` | Bounding box çiz | `color`, `thickness`, `labels` | `image` |
| `draw_masks` | Mask overlay | `alpha`, `colors` | `image` |
| `draw_text` | Text yazma | `position`, `font`, `size` | `image` |
| `blur_regions` | Bölge blur (privacy) | `detections`, `blur_type` | `image` |
| `pixelate_regions` | Bölge pixelate | `detections`, `block_size` | `image` |
| `highlight` | Bölge vurgula | `detections`, `color` | `image` |
| `crop_grid` | Crop'ları grid olarak göster | `cols` | `image` |

### 3.10 Output & Integration

| Block | Açıklama | Config | Outputs |
|-------|----------|--------|---------|
| `json_output` | JSON formatla | `schema` | `json` |
| `save_image` | Görseli kaydet | `path`, `format` | `url` |
| `webhook` | HTTP POST gönder | `url`, `headers` | `response` |
| `save_to_dataset` | Dataset'e ekle | `dataset_id` | `record_id` |

---

## 4. Workflow Definition Format

### 4.1 JSON Schema

```json
{
  "version": "1.0",
  "id": "uuid",
  "name": "Retail Product Recognition",
  "description": "Detect products, extract embeddings, match to database",

  "inputs": [
    {
      "name": "image",
      "type": "image",
      "required": true
    }
  ],

  "nodes": [
    {
      "id": "detect_products",
      "type": "od_model",
      "config": {
        "model_id": "uuid-of-trained-model",
        "confidence": 0.5,
        "nms_threshold": 0.4,
        "classes": ["product"]
      },
      "inputs": {
        "image": "$inputs.image"
      }
    },
    {
      "id": "crop_products",
      "type": "dynamic_crop",
      "config": {
        "padding": 0.05,
        "min_size": 32
      },
      "inputs": {
        "image": "$inputs.image",
        "detections": "$nodes.detect_products.detections"
      }
    },
    {
      "id": "extract_embeddings",
      "type": "embedding_extract",
      "config": {
        "model_id": "uuid-of-embedding-model"
      },
      "inputs": {
        "images": "$nodes.crop_products.crops"
      }
    },
    {
      "id": "search_products",
      "type": "similarity_search",
      "config": {
        "collection": "products",
        "top_k": 5,
        "threshold": 0.7
      },
      "inputs": {
        "embeddings": "$nodes.extract_embeddings.embeddings"
      }
    }
  ],

  "outputs": [
    {
      "name": "detections",
      "source": "$nodes.detect_products.detections"
    },
    {
      "name": "matches",
      "source": "$nodes.search_products.matches"
    }
  ]
}
```

### 4.2 Reference Syntax

| Pattern | Açıklama | Örnek |
|---------|----------|-------|
| `$inputs.X` | Workflow input | `$inputs.image` |
| `$nodes.X.Y` | Node output | `$nodes.detect_products.detections` |
| `$nodes.X.*` | Tüm node outputs | `$nodes.detect_products.*` |
| `$config.X` | Global config | `$config.threshold` |

---

## 5. Database Schema

### 5.1 Core Tables

```sql
-- ============================================
-- WORKFLOW DEFINITIONS
-- ============================================
CREATE TABLE IF NOT EXISTS wf_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Definition
    definition JSONB NOT NULL,
    version INTEGER DEFAULT 1,

    -- Status
    status VARCHAR(50) DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived')),

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    thumbnail_url TEXT,

    -- Stats
    run_count INTEGER DEFAULT 0,
    avg_duration_ms INTEGER,
    last_run_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wf_workflows_status ON wf_workflows(status);
CREATE INDEX idx_wf_workflows_tags ON wf_workflows USING GIN(tags);

-- ============================================
-- WORKFLOW EXECUTIONS
-- ============================================
CREATE TABLE IF NOT EXISTS wf_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,

    -- Status
    status VARCHAR(50) DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- I/O
    input_data JSONB,
    output_data JSONB,

    -- Per-node metrics
    node_metrics JSONB DEFAULT '{}',

    -- Error handling
    error_message TEXT,
    error_node_id VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wf_executions_workflow ON wf_executions(workflow_id);
CREATE INDEX idx_wf_executions_status ON wf_executions(status);
CREATE INDEX idx_wf_executions_created ON wf_executions(created_at DESC);

-- ============================================
-- PRETRAINED MODELS REGISTRY
-- (Default models for common tasks)
-- ============================================
CREATE TABLE IF NOT EXISTS wf_pretrained_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Model info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(50) NOT NULL
        CHECK (model_type IN ('detection', 'classification', 'embedding', 'ocr', 'face')),

    -- Source
    source VARCHAR(50) NOT NULL
        CHECK (source IN ('ultralytics', 'huggingface', 'custom', 'roboflow')),
    model_path TEXT NOT NULL,  -- HF model ID or path

    -- Capabilities
    task VARCHAR(100),  -- e.g., "person_detection", "face_detection"
    classes JSONB,      -- Available classes

    -- Config
    default_config JSONB DEFAULT '{}',
    input_size INTEGER[],

    -- Flags
    is_active BOOLEAN DEFAULT true,
    requires_gpu BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed default pretrained models
INSERT INTO wf_pretrained_models (name, description, model_type, source, model_path, task, classes, default_config) VALUES
    ('YOLOv8n Person', 'Person detection', 'detection', 'ultralytics', 'yolov8n.pt', 'person_detection', '["person"]', '{"confidence": 0.5, "classes": [0]}'),
    ('YOLOv8n Face', 'Face detection', 'face', 'ultralytics', 'yolov8n-face.pt', 'face_detection', '["face"]', '{"confidence": 0.5}'),
    ('PaddleOCR', 'Text recognition', 'ocr', 'custom', 'paddleocr', 'text_extraction', NULL, '{"lang": "en"}'),
    ('CLIP ViT-B/32', 'Zero-shot classification', 'classification', 'huggingface', 'openai/clip-vit-base-patch32', 'zero_shot_classification', NULL, '{}'),
    ('DINOv2 Base', 'Image embeddings', 'embedding', 'huggingface', 'facebook/dinov2-base', 'embedding_extraction', NULL, '{"image_size": 518}')
ON CONFLICT DO NOTHING;

-- ============================================
-- WORKFLOW TEMPLATES
-- (Pre-built workflows for common use cases)
-- ============================================
CREATE TABLE IF NOT EXISTS wf_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Template info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),  -- e.g., "retail", "privacy", "ocr"

    -- Definition (same format as workflows)
    definition JSONB NOT NULL,

    -- Preview
    thumbnail_url TEXT,
    example_input_url TEXT,
    example_output JSONB,

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    difficulty VARCHAR(20) CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),

    -- Stats
    use_count INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wf_templates_category ON wf_templates(category);
```

### 5.2 Storage Bucket

```sql
-- Workflow assets (test images, results)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'wf-assets',
    'wf-assets',
    true,
    52428800,  -- 50MB
    ARRAY['image/jpeg', 'image/png', 'image/webp', 'application/json']
)
ON CONFLICT (id) DO NOTHING;
```

---

## 6. API Design

### 6.1 Workflow CRUD

```
GET    /api/v1/workflows                     # List workflows
POST   /api/v1/workflows                     # Create workflow
GET    /api/v1/workflows/{id}                # Get workflow
PUT    /api/v1/workflows/{id}                # Update workflow
DELETE /api/v1/workflows/{id}                # Delete workflow
POST   /api/v1/workflows/{id}/duplicate      # Duplicate workflow
```

### 6.2 Workflow Execution

```
POST   /api/v1/workflows/{id}/run            # Execute workflow
GET    /api/v1/workflows/{id}/executions     # List executions
GET    /api/v1/executions/{id}               # Get execution result
```

### 6.3 Model Registry

```
GET    /api/v1/workflow-models               # List all available models
GET    /api/v1/workflow-models/trained       # Trained models (OD, CLS, Embed)
GET    /api/v1/workflow-models/pretrained    # Pretrained models
```

### 6.4 Templates

```
GET    /api/v1/workflow-templates            # List templates
GET    /api/v1/workflow-templates/{id}       # Get template
POST   /api/v1/workflow-templates/{id}/use   # Create workflow from template
```

### 6.5 Block Registry

```
GET    /api/v1/workflow-blocks               # List available blocks
GET    /api/v1/workflow-blocks/{type}        # Get block schema
```

---

## 7. Execution Engine

### 7.1 Architecture

```python
class WorkflowEngine:
    """Workflow execution engine"""

    def __init__(self):
        self.block_registry = BlockRegistry()
        self.model_cache = ModelCache()

    async def execute(self, workflow: dict, inputs: dict) -> ExecutionResult:
        # 1. Parse & validate workflow
        parsed = self.parse_workflow(workflow)

        # 2. Build execution graph (DAG)
        dag = self.build_dag(parsed)

        # 3. Execute nodes in topological order
        context = ExecutionContext(inputs)

        for node in dag.topological_sort():
            # Resolve inputs from context
            node_inputs = self.resolve_inputs(node.inputs, context)

            # Get block implementation
            block = self.block_registry.get(node.type)

            # Execute
            result = await block.execute(node_inputs, node.config)

            # Store in context
            context.set_node_output(node.id, result)

        # 4. Collect outputs
        return self.collect_outputs(parsed.outputs, context)
```

### 7.2 Block Implementation Example

```python
class DynamicCropBlock(Block):
    """Crop images based on detections"""

    input_schema = {
        "image": ImageType,
        "detections": List[Detection]
    }

    output_schema = {
        "crops": List[ImageType]
    }

    async def execute(self, inputs: dict, config: dict) -> dict:
        image = inputs["image"]
        detections = inputs["detections"]
        padding = config.get("padding", 0.0)

        crops = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Apply padding
            if padding > 0:
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - w * padding)
                y1 = max(0, y1 - h * padding)
                x2 = min(image.width, x2 + w * padding)
                y2 = min(image.height, y2 + h * padding)

            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        return {"crops": crops}
```

---

## 8. Frontend - Workflow Builder

### 8.1 UI Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [Save] [Run ▶] [Test Image] [Export]                     Workflow Name │
├─────────────┬───────────────────────────────────────┬───────────────────┤
│   BLOCKS    │              CANVAS                   │     SETTINGS      │
│             │                                       │                   │
│  ▼ Input    │    ┌─────┐      ┌─────┐              │  Node: crop_1     │
│    ○ Image  │    │ Det │─────▶│Crop │              │  ─────────────    │
│             │    └─────┘      └─────┘              │  Padding: 0.05    │
│  ▼ Detection│        │            │                │  Min Size: 32     │
│    ○ OD Mod │        ▼            ▼                │                   │
│    ○ Person │    ┌─────┐      ┌─────┐              │                   │
│    ○ Face   │    │Blur │      │Embed│              │                   │
│             │    └─────┘      └─────┘              │                   │
│  ▼ Transform│                     │                │                   │
│    ○ Crop   │                     ▼                │                   │
│    ○ Resize │                 ┌─────┐              │                   │
│    ○ Blur   │                 │Match│              │                   │
│             │                 └─────┘              │                   │
│  ▼ Logic    │                                      │                   │
│    ○ Filter │                                      │                   │
│    ○ Cond.  │                                      │                   │
│             │                                      │                   │
│  ▼ Output   │                                      │                   │
│    ○ JSON   │                                      │                   │
└─────────────┴───────────────────────────────────────┴───────────────────┘
│                           TEST PANEL                                    │
│  [Upload Image]  [URL Input]                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Input Image        │  Output Result                            │   │
│  │  [image preview]    │  [annotated image + JSON results]         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Page Structure

```
/workflows
├── /                           # Workflow list
├── /new                        # Create new (template seçimi)
├── /[id]                       # Workflow editor
├── /[id]/test                  # Test interface
├── /[id]/executions            # Execution history
└── /templates                  # Template gallery
```

### 8.3 Tech Stack

- **Canvas:** React Flow (xyflow)
- **State:** Zustand
- **UI:** shadcn/ui (mevcut)
- **Validation:** Zod

---

## 9. Pre-built Templates

### 9.1 Retail Product Recognition

```json
{
  "name": "Retail Product Recognition",
  "category": "retail",
  "description": "Detect products on shelf, extract embeddings, match to database",
  "nodes": ["od_model", "dynamic_crop", "embedding_extract", "similarity_search"]
}
```

### 9.2 Privacy Blur (Person/Face)

```json
{
  "name": "Privacy Blur",
  "category": "privacy",
  "description": "Detect and blur persons/faces in retail imagery",
  "nodes": ["yolo_pretrained(person)", "face_detection", "blur_regions"]
}
```

### 9.3 Price Tag Reader

```json
{
  "name": "Price Tag OCR",
  "category": "ocr",
  "description": "Detect price tags, extract text, parse price",
  "nodes": ["od_model(price_tag)", "dynamic_crop", "ocr_extract", "price_parse"]
}
```

### 9.4 Out-of-Stock Detection

```json
{
  "name": "Out-of-Stock Alert",
  "category": "retail",
  "description": "Detect empty shelf areas",
  "nodes": ["od_model(shelf)", "od_model(product)", "empty_area_analysis", "condition", "alert"]
}
```

### 9.5 Full Retail Analysis

```json
{
  "name": "Full Retail Analysis",
  "category": "retail",
  "description": "Combined pipeline: product detection, matching, person blur, price reading",
  "nodes": ["parallel", "od_model", "dynamic_crop", "embedding_extract", "similarity_search", "blur_regions", "ocr_extract", "merge"]
}
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (2 hafta)

- [ ] Database schema (migration 050)
- [ ] Block registry sistemi
- [ ] Temel execution engine
- [ ] API endpoints (CRUD)

### Phase 2: Core Blocks (2 hafta)

- [ ] Input blocks
- [ ] Model blocks (OD, CLS, Embedding integration)
- [ ] Dynamic crop
- [ ] Similarity search (Qdrant)

### Phase 3: CV & Logic Blocks (2 hafta)

- [ ] Blur/pixelate blocks
- [ ] OCR block (PaddleOCR)
- [ ] Barcode reader
- [ ] Condition/filter/loop blocks
- [ ] Classical CV blocks (threshold, contour, etc.)

### Phase 4: UI (2 hafta)

- [ ] Workflow builder (React Flow)
- [ ] Block palette
- [ ] Node configuration panel
- [ ] Test panel (image upload + run)

### Phase 5: Templates & Polish (1 hafta)

- [ ] Pre-built templates
- [ ] Execution history
- [ ] Error handling & logging
- [ ] Documentation

---

## 11. Örnek Kullanım Senaryosu

### Senaryo: Mağaza Raf Analizi

1. **Kullanıcı** yeni workflow oluşturur
2. **Blocks ekler:**
   - Image Input
   - OD Model (trained product detector)
   - Dynamic Crop
   - Embedding Extract (DINOv2)
   - Similarity Search (Qdrant)
   - Person Detection (YOLO pretrained)
   - Blur Regions
   - Draw Boxes
   - JSON Output

3. **Bağlantıları yapar** (drag & drop)

4. **Test eder:**
   - Test görseli upload eder
   - "Run" butonuna tıklar
   - Sonuçları görür:
     - Annotated image (boxes + blur)
     - JSON: detected products + matches

5. **Kaydeder** ve production'da kullanır

---

## 12. Sonuç

Bu plan, mevcut BuyBuddy AI sistemindeki trained modelleri (OD, Classification, Embedding) kullanarak perakende sektörüne özel bir workflow builder oluşturmayı hedefler.

**Temel Özellikler:**
- Mevcut modelleri kullanma (yeni training gerektirmez)
- Pretrained modeller (person, face, CLIP, etc.)
- Classical CV operasyonları
- OCR/barcode desteği
- Logic & branching
- Hızlı test interface
- Pre-built retail templates

**Kapsam Dışı (Bu Plan İçin):**
- Video/streaming
- VLM/LLM integration
- Real-time deployment
- Edge deployment
