# BuyBuddy AI - Workflow System MVP

## MVP Hedefi

Minimum viable product: **Perakende raf analizi için tam bir pipeline oluşturup test edebilmek.**

### Core Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETAIL SHELF ANALYSIS FLOW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Image Input                                                                │
│       │                                                                     │
│       ├──────────────┬──────────────┬──────────────┬──────────────┐        │
│       ▼              ▼              ▼              ▼              │        │
│  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐            │        │
│  │ Reyon  │    │ Shelf  │    │  Slot  │    │ Human  │            │        │
│  │ Detect │    │ Detect │    │ Detect │    │ Detect │            │        │
│  └────────┘    └────────┘    └───┬────┘    └───┬────┘            │        │
│                                  │             │                  │        │
│                                  │             ▼                  │        │
│                                  │        ┌────────┐              │        │
│                                  │        │Segment │              │        │
│                                  │        │ + Blur │              │        │
│                                  │        └───┬────┘              │        │
│                                  ▼             │                  │        │
│                             ┌────────┐         │                  │        │
│                             │  Crop  │         │                  │        │
│                             │ Slots  │         │                  │        │
│                             └───┬────┘         │                  │        │
│                                 │              │                  │        │
│                                 ▼              │                  │        │
│                            ┌────────┐          │                  │        │
│                            │Classify│          │                  │        │
│                            │Empty/  │          │                  │        │
│                            │ Full   │          │                  │        │
│                            └───┬────┘          │                  │        │
│                                │               │                  │        │
│                    ┌───────────┴───────────┐   │                  │        │
│                    ▼                       ▼   │                  │        │
│               ┌────────┐              ┌────────┐                  │        │
│               │  FULL  │              │ EMPTY  │                  │        │
│               │Embedding│             │→ Void  │                  │        │
│               └───┬────┘              └───┬────┘                  │        │
│                   │                       │                       │        │
│                   ▼                       │                       │        │
│              ┌────────┐                   │                       │        │
│              │Similarity                  │                       │        │
│              │ Match  │                   │                       │        │
│              └───┬────┘                   │                       │        │
│                  │                        │                       │        │
│                  ▼                        ▼                       ▼        │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                    MERGE & BUILD REALOGRAM                       │      │
│  │         Grid with Product IDs, Identifiers, Void Slots          │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Extensibility
Bu core flow üzerine sonradan eklenebilir:
- Price tag detection + OCR
- Promotion detection
- Damage detection
- Expiry date reading
- vs.

---

## 1. MVP Scope

### ✅ MVP'de Olacaklar

| Özellik | Açıklama |
|---------|----------|
| **Unified Model Picker** | Pretrained + Trained modelleri tek listede göster |
| **Multi-Detection** | Birden fazla detection modeli zincirleme kullanabilme |
| **Classification** | Boş/Dolu, kategori sınıflandırma |
| **Segmentation + Blur** | Human segmentation ve blur (privacy) |
| **Condition/Logic** | If-else branching (empty → void, full → embedding) |
| **Embedding + Match** | Qdrant similarity search |
| **Grid Builder** | Realogram/planogram output |
| **Visual Builder** | Drag-drop node editor |
| **Test Panel** | Görsel upload + run + sonuç görüntüleme |
| **Save/Load** | Workflow kaydetme ve yükleme |

### ❌ MVP'de Olmayacaklar (Sonraki fazlar)

- OCR / Barcode / Price reading
- Templates gallery
- Batch processing
- Webhook/external integrations
- Video processing

---

## 2. Unified Model System

### 2.1 Model Kaynakları

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED MODEL PICKER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📦 DETECTION MODELS                                            │
│  ├── 🏷️ Pretrained                                              │
│  │   ├── YOLOv8n (person, car, etc.) [80 classes]              │
│  │   ├── YOLOv8n-face                                          │
│  │   └── YOLOv11n                                              │
│  │                                                              │
│  └── 🎯 Your Trained Models                                     │
│      ├── Product Detector v2 (RF-DETR) - 96.2% mAP             │
│      ├── Shelf Structure (RT-DETR) - 94.1% mAP                 │
│      └── Price Tag Detector (YOLO-NAS) - 91.8% mAP             │
│                                                                 │
│  📦 CLASSIFICATION MODELS                                       │
│  ├── 🏷️ Pretrained                                              │
│  │   └── CLIP ViT-B/32 (zero-shot)                             │
│  │                                                              │
│  └── 🎯 Your Trained Models                                     │
│      ├── Brand Classifier (ConvNeXt) - 94.5% acc               │
│      └── Category Classifier (ViT) - 92.1% acc                 │
│                                                                 │
│  📦 EMBEDDING MODELS                                            │
│  ├── 🏷️ Pretrained                                              │
│  │   ├── DINOv2-base                                           │
│  │   ├── DINOv2-large                                          │
│  │   └── CLIP ViT-L/14                                         │
│  │                                                              │
│  └── 🎯 Your Trained Models                                     │
│      ├── Product Embedder v3 (DINOv2-ft) - 89.2% R@1           │
│      └── SKU Matcher (CLIP-ft) - 87.4% R@1                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Model Registry API

```typescript
// GET /api/v1/workflow-models
{
  "detection": {
    "pretrained": [
      { "id": "yolov8n", "name": "YOLOv8n", "source": "ultralytics", "classes": [...] },
      { "id": "yolov8n-face", "name": "YOLOv8n Face", "source": "ultralytics", "classes": ["face"] }
    ],
    "trained": [
      { "id": "uuid-1", "name": "Product Detector v2", "model_type": "rf-detr", "map": 0.962, "classes": [...] },
      { "id": "uuid-2", "name": "Shelf Structure", "model_type": "rt-detr", "map": 0.941, "classes": [...] }
    ]
  },
  "classification": {
    "pretrained": [...],
    "trained": [...]
  },
  "embedding": {
    "pretrained": [...],
    "trained": [...]
  }
}
```

---

## 3. MVP Blocks

### 3.1 Block Listesi

#### Input Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `image_input` | - | `image` | - |

#### Model Blocks - Detection
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `detection` | `image` | `detections[]`, `annotated_image`, `count` | `model_id`, `model_source`, `confidence`, `classes[]` |

#### Model Blocks - Segmentation
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `segmentation` | `image`, `detections` | `masks[]`, `masked_image` | `model_id` (SAM, YOLO-seg) |

#### Model Blocks - Classification
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `classification` | `image` veya `images[]` | `predictions[]` | `model_id`, `model_source`, `top_k` |

#### Model Blocks - Embedding
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `embedding` | `image` veya `images[]` | `embeddings[]` | `model_id`, `model_source` |
| `similarity_search` | `embeddings[]` | `matches[]` | `collection`, `top_k`, `threshold` |

#### Transform Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `crop` | `image`, `detections` | `crops[]`, `crop_metadata[]` | `padding`, `min_size` |
| `resize` | `image` | `image` | `width`, `height`, `keep_aspect` |

#### Classical CV Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `blur_region` | `image`, `detections` veya `masks` | `image` | `blur_type`, `intensity` |
| `gaussian_blur` | `image` | `image` | `kernel_size` |

#### Logic Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `condition` | `value` | `true_output`, `false_output` | `expression` |
| `filter` | `items[]` | `filtered[]`, `rejected[]` | `expression` |
| `for_each` | `items[]` | `results[]` | `block_ref` |

#### Visualization Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `draw_boxes` | `image`, `detections` | `image` | `colors`, `thickness`, `labels` |
| `draw_masks` | `image`, `masks` | `image` | `alpha`, `colors` |

#### Output/Merge Blocks
| Block | Inputs | Outputs | Config |
|-------|--------|---------|--------|
| `merge` | `inputs[]` | `merged` | `strategy` |
| `grid_builder` | `slots[]`, `matches[]`, `voids[]` | `grid`, `realogram` | `rows`, `cols` |
| `json_output` | `any` | `json` | `schema` |

### 3.2 Block Detayları

```typescript
// ============================================
// DETECTION BLOCK
// ============================================
interface DetectionBlock {
  type: "detection";
  name: string;  // e.g., "Slot Detector", "Human Detector"
  config: {
    model_id: string;        // UUID (trained) veya preset ID (pretrained)
    model_source: "trained" | "pretrained";
    confidence: number;      // 0-1, default 0.5
    nms_threshold: number;   // 0-1, default 0.4
    classes?: string[];      // Filter specific classes
  };
  inputs: {
    image: "$inputs.image" | "$nodes.X.image";
  };
  outputs: {
    detections: Detection[];
    annotated_image: Image;
    count: number;
  };
}

// ============================================
// CLASSIFICATION BLOCK
// ============================================
interface ClassificationBlock {
  type: "classification";
  config: {
    model_id: string;
    model_source: "trained" | "pretrained";
    top_k: number;           // default 1
  };
  inputs: {
    images: "$nodes.crop.crops";  // Array of cropped images
  };
  outputs: {
    predictions: Array<{
      class_name: string;
      confidence: number;
      top_k: Array<{class_name: string, confidence: number}>;
    }>;
  };
}

// ============================================
// SEGMENTATION BLOCK
// ============================================
interface SegmentationBlock {
  type: "segmentation";
  config: {
    model_id: string;        // SAM, YOLO-seg, etc.
    model_source: "trained" | "pretrained";
  };
  inputs: {
    image: "$inputs.image";
    detections?: "$nodes.X.detections";  // Optional: segment only detected regions
  };
  outputs: {
    masks: Mask[];           // Binary masks for each detection
    masked_image: Image;     // Image with masks applied
  };
}

// ============================================
// BLUR REGION BLOCK
// ============================================
interface BlurRegionBlock {
  type: "blur_region";
  config: {
    blur_type: "gaussian" | "pixelate" | "black";
    intensity: number;       // kernel size or pixel size
  };
  inputs: {
    image: "$inputs.image";
    regions: "$nodes.X.detections" | "$nodes.X.masks";
  };
  outputs: {
    image: Image;            // Image with blurred regions
  };
}

// ============================================
// EMBEDDING BLOCK
// ============================================
interface EmbeddingBlock {
  type: "embedding";
  config: {
    model_id: string;
    model_source: "trained" | "pretrained";
  };
  inputs: {
    images: "$nodes.crop.crops";  // Array of cropped images
  };
  outputs: {
    embeddings: number[][];  // Array of embedding vectors
  };
}

// ============================================
// SIMILARITY SEARCH BLOCK
// ============================================
interface SimilaritySearchBlock {
  type: "similarity_search";
  config: {
    collection: string;      // Qdrant collection name
    top_k: number;           // default 5
    threshold: number;       // minimum similarity, default 0.7
  };
  inputs: {
    embeddings: "$nodes.embedding.embeddings";
  };
  outputs: {
    matches: Array<{
      product_id: string;
      similarity: number;
      product_info: ProductInfo;
      identifiers: {         // All identifiers for this product
        barcode?: string;
        upc?: string;
        sku?: string;
      };
    }>;
  };
}

// ============================================
// CONDITION BLOCK (IF-ELSE)
// ============================================
interface ConditionBlock {
  type: "condition";
  config: {
    expression: string;      // e.g., "$input.class_name == 'full'"
  };
  inputs: {
    value: any;              // Value to evaluate
  };
  outputs: {
    true_output: any;        // Passes through if true
    false_output: any;       // Passes through if false
  };
}

// ============================================
// FILTER BLOCK
// ============================================
interface FilterBlock {
  type: "filter";
  config: {
    expression: string;      // e.g., "confidence > 0.8"
  };
  inputs: {
    items: any[];            // Array to filter
  };
  outputs: {
    passed: any[];           // Items that passed
    rejected: any[];         // Items that failed
  };
}

// ============================================
// FOR EACH BLOCK (Loop)
// ============================================
interface ForEachBlock {
  type: "for_each";
  config: {
    // Sub-workflow to execute for each item
    sub_nodes: Node[];
  };
  inputs: {
    items: any[];            // Array to iterate
  };
  outputs: {
    results: any[];          // Results from each iteration
  };
}

// ============================================
// GRID BUILDER BLOCK (Realogram)
// ============================================
interface GridBuilderBlock {
  type: "grid_builder";
  config: {
    structure_source: "auto" | "manual";  // Auto from shelf/slot detections
  };
  inputs: {
    shelves: "$nodes.shelf_detect.detections";
    slots: "$nodes.slot_detect.detections";
    matches: "$nodes.similarity.matches";
    voids: "$nodes.condition.false_output";
  };
  outputs: {
    grid: Array<Array<GridCell>>;  // 2D grid
    realogram: {
      rows: number;
      cols: number;
      cells: GridCell[];
      total_products: number;
      total_voids: number;
      identified_products: number;
    };
  };
}

interface GridCell {
  row: number;
  col: number;
  status: "product" | "void" | "unknown";
  product_id?: string;
  identifiers?: object;
  confidence?: number;
  bbox: BoundingBox;
}
```

---

## 4. Frontend Design

### 4.1 Ana Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ☰  BuyBuddy AI          Workflows    OD    Classification    ...    👤    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ WORKFLOW EDITOR ──────────────────────────────────────────────────────┐│
│  │                                                                         ││
│  │  ┌─────────┐  Untitled Workflow                    [Save] [Run ▶] [⚙️] ││
│  │  │ < Back  │                                                            ││
│  │  └─────────┘                                                            ││
│  │                                                                         ││
│  │  ┌─────────────┬─────────────────────────────────┬─────────────────┐   ││
│  │  │   BLOCKS    │           CANVAS                │    INSPECTOR    │   ││
│  │  │             │                                 │                 │   ││
│  │  │  🔍 Search  │     ┌───────┐                   │  Detection      │   ││
│  │  │             │     │ Image │                   │  ─────────────  │   ││
│  │  │  ▼ Input    │     │ Input │                   │                 │   ││
│  │  │    ○ Image  │     └───┬───┘                   │  Model          │   ││
│  │  │             │         │                       │  ┌───────────┐  │   ││
│  │  │  ▼ Models   │         ▼                       │  │ Product   │  │   ││
│  │  │    ○ Detect │     ┌───────┐    ┌───────┐     │  │ Detector  ▼│  │   ││
│  │  │    ○ Class. │     │Detect │───▶│ Crop  │     │  └───────────┘  │   ││
│  │  │    ○ Embed  │     └───┬───┘    └───┬───┘     │                 │   ││
│  │  │             │         │            │         │  Confidence     │   ││
│  │  │  ▼ Process  │         │            ▼         │  ┌───────────┐  │   ││
│  │  │    ○ Crop   │         │        ┌───────┐     │  │   0.50    │  │   ││
│  │  │             │         │        │ Embed │     │  └───────────┘  │   ││
│  │  │  ▼ Search   │         │        └───┬───┘     │                 │   ││
│  │  │    ○ Simil. │         │            │         │  Classes        │   ││
│  │  │             │         │            ▼         │  ☑ product      │   ││
│  │  │  ▼ Visual   │         │        ┌───────┐     │  ☐ shelf        │   ││
│  │  │    ○ Draw   │         │        │ Match │     │  ☐ price_tag    │   ││
│  │  │             │         │        └───────┘     │                 │   ││
│  │  │             │         │                      │                 │   ││
│  │  └─────────────┴─────────┴──────────────────────┴─────────────────┘   ││
│  │                                                                         ││
│  │  ┌─ TEST PANEL ───────────────────────────────────────────────────────┐││
│  │  │                                                                     │││
│  │  │  [📁 Upload Image]  [🔗 URL]  [📋 From Dataset]                    │││
│  │  │                                                                     │││
│  │  │  ┌─────────────────────────┐  ┌─────────────────────────────────┐  │││
│  │  │  │                         │  │  Results                        │  │││
│  │  │  │    [Uploaded Image]     │  │                                 │  │││
│  │  │  │                         │  │  ✓ 12 products detected         │  │││
│  │  │  │                         │  │  ✓ 12 embeddings extracted      │  │││
│  │  │  │    Click to upload      │  │  ✓ 10 products matched          │  │││
│  │  │  │    or drag & drop       │  │                                 │  │││
│  │  │  │                         │  │  ┌─ Matches ─────────────────┐  │  │││
│  │  │  │                         │  │  │ Coca-Cola 330ml   (0.94)  │  │  │││
│  │  │  │                         │  │  │ Fanta Orange 330ml (0.91) │  │  │││
│  │  │  │                         │  │  │ Sprite 330ml      (0.89)  │  │  │││
│  │  │  │                         │  │  │ ...                       │  │  │││
│  │  │  │                         │  │  └───────────────────────────┘  │  │││
│  │  │  └─────────────────────────┘  └─────────────────────────────────┘  │││
│  │  │                                                                     │││
│  │  └─────────────────────────────────────────────────────────────────────┘││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Model Picker Dialog

```
┌─────────────────────────────────────────────────────────────────┐
│  Select Detection Model                                    [X]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🔍 Search models...                                            │
│                                                                 │
│  ┌─ Your Trained Models ──────────────────────────────────────┐│
│  │                                                             ││
│  │  ◉ Product Detector v2                                      ││
│  │    RF-DETR · 96.2% mAP · 3 classes                         ││
│  │    Trained 2 days ago                                       ││
│  │                                                             ││
│  │  ○ Shelf Structure Detector                                 ││
│  │    RT-DETR · 94.1% mAP · 5 classes                         ││
│  │    Trained 1 week ago                                       ││
│  │                                                             ││
│  │  ○ Price Tag Detector                                       ││
│  │    YOLO-NAS · 91.8% mAP · 1 class                          ││
│  │    Trained 3 weeks ago                                      ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─ Pretrained Models ────────────────────────────────────────┐│
│  │                                                             ││
│  │  ○ YOLOv8n (COCO)                                          ││
│  │    General object detection · 80 classes                    ││
│  │    person, car, bottle, chair, ...                         ││
│  │                                                             ││
│  │  ○ YOLOv8n-face                                            ││
│  │    Face detection · 1 class                                 ││
│  │                                                             ││
│  │  ○ YOLOv11n (COCO)                                         ││
│  │    Latest YOLO · 80 classes                                 ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│                                              [Cancel]  [Select] │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Block Node Design

```
┌─────────────────────────────────┐
│ ● Detection                     │  ← Header with color indicator
├─────────────────────────────────┤
│                                 │
│  Model: Product Detector v2    │  ← Summary info
│  Conf: 0.50                    │
│                                 │
├─────────────────────────────────┤
│  ○ image          detections ○ │  ← Input/Output ports
│                annotated_img ○ │
│                       count ○  │
└─────────────────────────────────┘

Color coding:
- 🟦 Blue: Input blocks
- 🟩 Green: Model blocks
- 🟨 Yellow: Transform blocks
- 🟪 Purple: Search/Query blocks
- 🟧 Orange: Visualization blocks
```

### 4.4 Results Panel

```
┌─ Execution Results ─────────────────────────────────────────────┐
│                                                                 │
│  Status: ✓ Completed in 1.24s                                  │
│                                                                 │
│  ┌─ Node Metrics ─────────────────────────────────────────────┐│
│  │  detection     → 450ms  │ 12 detections                    ││
│  │  crop          → 23ms   │ 12 crops                         ││
│  │  embedding     → 680ms  │ 12 embeddings                    ││
│  │  similarity    → 89ms   │ 10 matches                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─ Tabs ─────────────────────────────────────────────────────┐│
│  │  [Annotated Image]  [Detections]  [Matches]  [Raw JSON]    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                             ││
│  │     [Annotated image with bounding boxes]                   ││
│  │                                                             ││
│  │                                                             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Matches (10 of 12 products identified):                       │
│  ┌──────┬────────────────────┬──────────┬───────────┐          │
│  │ Crop │ Product            │ Score    │ Barcode   │          │
│  ├──────┼────────────────────┼──────────┼───────────┤          │
│  │ [img]│ Coca-Cola 330ml    │ 94.2%    │ 54912345  │          │
│  │ [img]│ Fanta Orange 330ml │ 91.8%    │ 54912346  │          │
│  │ [img]│ Sprite 330ml       │ 89.3%    │ 54912347  │          │
│  │ [img]│ ⚠️ No match        │ -        │ -         │          │
│  │ ...  │                    │          │           │          │
│  └──────┴────────────────────┴──────────┴───────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Page Structure

```
/workflows
├── page.tsx                    # Workflow list
├── new/
│   └── page.tsx               # Create new workflow
├── [id]/
│   ├── page.tsx               # Workflow editor (main)
│   └── layout.tsx             # Editor layout
└── components/
    ├── workflow-list.tsx
    ├── workflow-editor/
    │   ├── index.tsx          # Main editor component
    │   ├── canvas.tsx         # React Flow canvas
    │   ├── block-palette.tsx  # Left sidebar blocks
    │   ├── inspector.tsx      # Right sidebar config
    │   ├── test-panel.tsx     # Bottom test area
    │   └── nodes/
    │       ├── base-node.tsx
    │       ├── input-node.tsx
    │       ├── detection-node.tsx
    │       ├── crop-node.tsx
    │       ├── embedding-node.tsx
    │       ├── similarity-node.tsx
    │       └── visualize-node.tsx
    ├── model-picker/
    │   ├── index.tsx
    │   ├── detection-picker.tsx
    │   ├── classification-picker.tsx
    │   └── embedding-picker.tsx
    └── results-panel/
        ├── index.tsx
        ├── annotated-image.tsx
        ├── detections-list.tsx
        └── matches-table.tsx
```

---

## 6. Database Schema (MVP)

```sql
-- Migration: 050_workflows_mvp.sql

-- ============================================
-- WORKFLOWS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS wf_workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Definition (JSON)
    definition JSONB NOT NULL,

    -- Status
    status VARCHAR(20) DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived')),

    -- Stats
    run_count INTEGER DEFAULT 0,
    last_run_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wf_workflows_status ON wf_workflows(status);

-- ============================================
-- EXECUTIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS wf_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,

    -- Status
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Data
    input_data JSONB,      -- Input image URL/path
    output_data JSONB,     -- Full results
    node_metrics JSONB,    -- Per-node timing

    -- Error
    error_message TEXT,
    error_node_id VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_wf_executions_workflow ON wf_executions(workflow_id);
CREATE INDEX idx_wf_executions_status ON wf_executions(status);

-- ============================================
-- PRETRAINED MODELS REGISTRY
-- ============================================
CREATE TABLE IF NOT EXISTS wf_pretrained_models (
    id VARCHAR(100) PRIMARY KEY,  -- e.g., "yolov8n", "dinov2-base"

    -- Info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(20) NOT NULL
        CHECK (model_type IN ('detection', 'classification', 'embedding')),

    -- Source
    source VARCHAR(50) NOT NULL,  -- ultralytics, huggingface, etc.
    model_path TEXT NOT NULL,

    -- Metadata
    classes JSONB,
    default_config JSONB DEFAULT '{}',

    -- Flags
    is_active BOOLEAN DEFAULT true,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed pretrained models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, default_config) VALUES
    ('yolov8n', 'YOLOv8n (COCO)', 'General object detection', 'detection', 'ultralytics', 'yolov8n.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]',
     '{"confidence": 0.5, "nms": 0.4}'),

    ('yolov8n-face', 'YOLOv8n Face', 'Face detection', 'detection', 'ultralytics', 'yolov8n-face.pt',
     '["face"]', '{"confidence": 0.5}'),

    ('yolov11n', 'YOLOv11n (COCO)', 'Latest YOLO model', 'detection', 'ultralytics', 'yolo11n.pt',
     NULL, '{"confidence": 0.5, "nms": 0.4}'),

    ('dinov2-base', 'DINOv2 Base', 'Image embeddings (768-dim)', 'embedding', 'huggingface', 'facebook/dinov2-base',
     NULL, '{"image_size": 518}'),

    ('dinov2-small', 'DINOv2 Small', 'Image embeddings (384-dim)', 'embedding', 'huggingface', 'facebook/dinov2-small',
     NULL, '{"image_size": 518}'),

    ('clip-vit-b-32', 'CLIP ViT-B/32', 'Vision-language embeddings', 'embedding', 'huggingface', 'openai/clip-vit-base-patch32',
     NULL, '{"image_size": 224}')
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- TRIGGERS
-- ============================================
CREATE OR REPLACE FUNCTION update_workflow_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS wf_workflows_updated_at ON wf_workflows;
CREATE TRIGGER wf_workflows_updated_at
    BEFORE UPDATE ON wf_workflows
    FOR EACH ROW EXECUTE FUNCTION update_workflow_updated_at();

-- ============================================
-- RLS
-- ============================================
ALTER TABLE wf_workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE wf_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE wf_pretrained_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all" ON wf_workflows FOR ALL USING (true);
CREATE POLICY "Allow all" ON wf_executions FOR ALL USING (true);
CREATE POLICY "Allow all" ON wf_pretrained_models FOR ALL USING (true);
```

---

## 7. API Endpoints (MVP)

### 7.1 Workflows

```
GET    /api/v1/workflows                # List all workflows
POST   /api/v1/workflows                # Create workflow
GET    /api/v1/workflows/{id}           # Get workflow
PUT    /api/v1/workflows/{id}           # Update workflow
DELETE /api/v1/workflows/{id}           # Delete workflow
```

### 7.2 Execution

```
POST   /api/v1/workflows/{id}/run       # Run workflow with image
GET    /api/v1/executions/{id}          # Get execution result
```

### 7.3 Models

```
GET    /api/v1/workflow-models          # Get all available models (unified)
```

**Response:**
```json
{
  "detection": {
    "pretrained": [
      {"id": "yolov8n", "name": "YOLOv8n", "classes": [...], "source": "ultralytics"}
    ],
    "trained": [
      {"id": "uuid", "name": "Product Detector", "model_type": "rf-detr", "map": 0.96, "classes": [...]}
    ]
  },
  "classification": {...},
  "embedding": {...}
}
```

---

## 8. Backend Execution Engine (MVP)

```python
# apps/api/src/services/workflow_engine.py

from typing import Dict, Any, List
import asyncio

class WorkflowEngine:
    """Simple synchronous workflow executor for MVP"""

    def __init__(self):
        self.blocks = {
            "image_input": ImageInputBlock(),
            "detection": DetectionBlock(),
            "crop": CropBlock(),
            "embedding": EmbeddingBlock(),
            "similarity_search": SimilaritySearchBlock(),
            "draw_boxes": DrawBoxesBlock(),
        }

    async def execute(
        self,
        workflow: dict,
        inputs: dict
    ) -> dict:
        """Execute workflow and return results"""

        context = {"inputs": inputs, "nodes": {}}
        metrics = {}

        # Get execution order (topological sort)
        nodes = self._get_execution_order(workflow["nodes"])

        for node in nodes:
            block = self.blocks[node["type"]]

            # Resolve inputs
            node_inputs = self._resolve_inputs(node["inputs"], context)

            # Execute block
            start = time.time()
            result = await block.execute(node_inputs, node.get("config", {}))
            duration = (time.time() - start) * 1000

            # Store results
            context["nodes"][node["id"]] = result
            metrics[node["id"]] = {"duration_ms": duration, **result.get("_metrics", {})}

        # Collect outputs
        outputs = {}
        for output in workflow.get("outputs", []):
            outputs[output["name"]] = self._resolve_ref(output["source"], context)

        return {
            "outputs": outputs,
            "metrics": metrics
        }


class DetectionBlock:
    """Object detection block - supports both pretrained and trained models"""

    async def execute(self, inputs: dict, config: dict) -> dict:
        image = inputs["image"]
        model_id = config["model_id"]
        model_source = config.get("model_source", "trained")
        confidence = config.get("confidence", 0.5)

        if model_source == "pretrained":
            # Load YOLO pretrained
            model = self._load_pretrained(model_id)
        else:
            # Load from od_trained_models
            model = await self._load_trained(model_id)

        # Run inference
        results = model.predict(image, conf=confidence)

        detections = self._parse_results(results)
        annotated = results.plot()

        return {
            "detections": detections,
            "annotated_image": annotated,
            "count": len(detections),
            "_metrics": {"detection_count": len(detections)}
        }


class EmbeddingBlock:
    """Extract embeddings - supports pretrained and trained models"""

    async def execute(self, inputs: dict, config: dict) -> dict:
        images = inputs["images"]  # List of cropped images
        model_id = config["model_id"]
        model_source = config.get("model_source", "trained")

        if model_source == "pretrained":
            model = self._load_pretrained(model_id)  # DINOv2, CLIP
        else:
            model = await self._load_trained(model_id)  # trained_models

        embeddings = []
        for img in images:
            emb = model.encode(img)
            embeddings.append(emb.tolist())

        return {
            "embeddings": embeddings,
            "_metrics": {"embedding_count": len(embeddings)}
        }


class SimilaritySearchBlock:
    """Search Qdrant for similar products"""

    def __init__(self):
        self.qdrant = QdrantClient(...)

    async def execute(self, inputs: dict, config: dict) -> dict:
        embeddings = inputs["embeddings"]
        collection = config["collection"]
        top_k = config.get("top_k", 5)
        threshold = config.get("threshold", 0.7)

        matches = []
        for emb in embeddings:
            results = self.qdrant.search(
                collection_name=collection,
                query_vector=emb,
                limit=top_k,
                score_threshold=threshold
            )

            if results:
                best = results[0]
                matches.append({
                    "product_id": best.payload["product_id"],
                    "similarity": best.score,
                    "product_info": best.payload
                })
            else:
                matches.append(None)

        return {
            "matches": matches,
            "_metrics": {"matched_count": sum(1 for m in matches if m)}
        }
```

---

## 9. Implementation Plan

### Week 1: Backend Foundation

- [ ] Migration 050 - Database schema (wf_workflows, wf_executions, wf_pretrained_models)
- [ ] API: workflow CRUD endpoints
- [ ] API: unified model list endpoint (pretrained + trained from existing tables)
- [ ] Basic workflow engine structure

### Week 2: Core Model Blocks

- [ ] Detection block (pretrained YOLO + trained OD models)
- [ ] Classification block (pretrained CLIP + trained CLS models)
- [ ] Embedding block (pretrained DINOv2 + trained embedding models)
- [ ] Similarity search block (Qdrant integration)
- [ ] Model caching system

### Week 3: Transform & CV Blocks

- [ ] Crop block (dynamic crop from detections)
- [ ] Segmentation block (SAM / YOLO-seg)
- [ ] Blur region block (gaussian, pixelate)
- [ ] Resize block
- [ ] Draw boxes / masks blocks

### Week 4: Logic Blocks

- [ ] Condition block (if-else branching)
- [ ] Filter block (filter arrays by expression)
- [ ] For-each block (iterate over detections)
- [ ] Grid builder block (realogram output)
- [ ] Merge block

### Week 5: Frontend - Editor Core

- [ ] Workflow list page
- [ ] React Flow canvas setup
- [ ] Block palette component (categorized)
- [ ] Base node components
- [ ] Edge/connection handling
- [ ] Zustand store for workflow state

### Week 6: Frontend - Node Components

- [ ] Input node
- [ ] Detection node
- [ ] Classification node
- [ ] Embedding node
- [ ] Similarity node
- [ ] Crop/Transform nodes
- [ ] Logic nodes (condition, filter)
- [ ] Output nodes

### Week 7: Frontend - Inspector & Model Picker

- [ ] Inspector panel (node configuration)
- [ ] Model picker dialog (unified: pretrained + trained)
- [ ] Expression builder for conditions
- [ ] Class selector for detection models

### Week 8: Frontend - Test Panel & Results

- [ ] Test panel (image upload, URL input)
- [ ] Execution trigger & loading states
- [ ] Results panel (tabs: image, detections, matches, JSON)
- [ ] Realogram/grid visualization
- [ ] Error handling & display

### Week 9: Integration & Polish

- [ ] End-to-end testing with retail flow
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] UI polish & responsive design
- [ ] Documentation

---

## 10. Tech Stack

### Backend
- FastAPI (existing)
- Ultralytics (YOLO, YOLO-seg)
- Transformers (DINOv2, CLIP, SAM)
- Qdrant (existing)
- PIL/OpenCV
- segment-anything (SAM)

### Frontend
- Next.js (existing)
- React Flow (@xyflow/react)
- Zustand (state management)
- shadcn/ui (existing)
- Tailwind (existing)
- Zod (validation)

---

## 11. Success Criteria

MVP başarılı sayılır eğer:

1. ✅ Kullanıcı yeni workflow oluşturabilir
2. ✅ Drag-drop ile block ekleyebilir (tüm kategoriler)
3. ✅ Birden fazla detection modeli zincirleme kullanabilir
4. ✅ Hem pretrained hem trained model seçebilir
5. ✅ Classification ile boş/dolu kontrolü yapabilir
6. ✅ Condition ile branching yapabilir (full → embedding, empty → void)
7. ✅ Human detection + segmentation + blur yapabilir
8. ✅ Embedding + similarity match ile ürün tanıyabilir
9. ✅ Realogram/grid output alabilir
10. ✅ Test görseli yükleyip full pipeline çalıştırabilir
11. ✅ Workflow'u kaydedip tekrar yükleyebilir

---

## 12. Example Workflow JSON (Full Retail Flow)

```json
{
  "version": "1.0",
  "name": "Full Retail Shelf Analysis",
  "nodes": [
    {"id": "input", "type": "image_input"},

    // Multi-layer detection
    {"id": "detect_reyon", "type": "detection", "config": {"model_id": "uuid-reyon", "model_source": "trained"}},
    {"id": "detect_shelf", "type": "detection", "config": {"model_id": "uuid-shelf", "model_source": "trained"}},
    {"id": "detect_slot", "type": "detection", "config": {"model_id": "uuid-slot", "model_source": "trained"}},
    {"id": "detect_human", "type": "detection", "config": {"model_id": "yolov8n", "model_source": "pretrained", "classes": ["person"]}},

    // Privacy
    {"id": "segment_human", "type": "segmentation", "config": {"model_id": "sam-base"}},
    {"id": "blur_human", "type": "blur_region", "config": {"blur_type": "gaussian", "intensity": 51}},

    // Crop & Classify
    {"id": "crop_slots", "type": "crop", "inputs": {"image": "$nodes.blur_human.image", "detections": "$nodes.detect_slot.detections"}},
    {"id": "classify_empty", "type": "classification", "config": {"model_id": "uuid-empty-full", "model_source": "trained"}},

    // Branch: Full vs Empty
    {"id": "filter_full", "type": "filter", "config": {"expression": "class_name == 'full'"}},

    // Embedding & Match (only for full slots)
    {"id": "embedding", "type": "embedding", "config": {"model_id": "uuid-embedder", "model_source": "trained"}},
    {"id": "match", "type": "similarity_search", "config": {"collection": "products", "top_k": 1, "threshold": 0.7}},

    // Build output
    {"id": "grid", "type": "grid_builder"},
    {"id": "visualize", "type": "draw_boxes"}
  ],
  "outputs": [
    {"name": "annotated_image", "source": "$nodes.visualize.image"},
    {"name": "realogram", "source": "$nodes.grid.realogram"},
    {"name": "matches", "source": "$nodes.match.matches"}
  ]
}
```

---

## 13. Sonraki Fazlar (Post-MVP)

### Phase 2: OCR & Barcode
- OCR block (PaddleOCR)
- Barcode/QR reader
- Price parsing
- Expiry date reading

### Phase 3: Templates & Gallery
- Pre-built retail templates
- Template gallery UI
- One-click template use

### Phase 4: Advanced Features
- Workflow versioning
- Execution history & comparison
- Webhook output
- Batch processing

### Phase 5: Production
- API deployment endpoints
- Performance optimization
- Model caching improvements
- Monitoring & logging
