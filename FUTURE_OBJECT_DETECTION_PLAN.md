# Object Detection & Annotation Platform - Future Plan

Bu dokuman, reyon (shelf) gorsellerinden object detection modeli egitmek icin gerekli annotation platformunun detayli planini icerir.

---

## 1. Genel Bakis

### 1.1 Hedef

Roboflow benzeri, AI-destekli bir annotation ve training platformu olusturmak:

- Reyon resimlerini yukleyebilme (manuel + open source dataset import)
- Ayni resim setini farkli annotation semalariyla kullanabilme
- AI-assisted annotation (SAM3, YOLO, GroundingDINO)
- Merkezi class yonetimi (rename, delete, merge)
- Object detection model egitimi (YOLOv8/v11, RT-DETR)
- Label Studio entegrasyonu (detayli annotation icin)

### 1.2 Mimari Genel Bakis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANA UYGULAMA                                    │
│                       (app.buybuddy.co)                                 │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Shelf Images │  │ OD Datasets  │  │ Class Mgmt   │  │ Training   │  │
│  │              │  │              │  │              │  │            │  │
│  │ - Upload     │  │ - Create     │  │ - Rename     │  │ - Config   │  │
│  │ - Import     │  │ - Annotate   │  │ - Delete     │  │ - Start    │  │
│  │ - List       │  │ - Export     │  │ - Merge      │  │ - Monitor  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Label Studio   │  │   Runpod GPU     │  │    Supabase      │
│  (Annotation UI) │  │   Workers        │  │  (DB + Storage)  │
│                  │  │                  │  │                  │
│ annotate.        │  │ - SAM3           │  │ - shelf_images   │
│ buybuddy.co      │  │ - YOLO predict   │  │ - annotations    │
│                  │  │ - Training       │  │ - classes        │
│                  │  │ - Export         │  │ - od_datasets    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 2. Veritabani Semasi

### 2.1 Yeni Tablolar

```sql
-- =====================================================
-- SHELF IMAGES (Reyon Resimleri)
-- =====================================================
CREATE TABLE shelf_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Temel bilgiler
    filename TEXT NOT NULL,
    image_url TEXT NOT NULL,
    thumbnail_url TEXT,

    -- Metadata
    width INTEGER,
    height INTEGER,
    file_size INTEGER,

    -- Kaynak bilgisi
    source TEXT NOT NULL DEFAULT 'upload',  -- 'upload', 'import', 'api'
    source_dataset TEXT,                     -- 'COCO', 'OpenImages', 'custom', etc.
    original_filename TEXT,

    -- Organizasyon
    folder TEXT,                             -- Klasor/grup ismi
    tags TEXT[],                             -- Etiketler

    -- Durum
    status TEXT DEFAULT 'pending',           -- 'pending', 'annotating', 'completed'

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- OD DATASETS (Object Detection Dataset'leri)
-- =====================================================
CREATE TABLE od_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Temel bilgiler
    name TEXT NOT NULL,
    description TEXT,

    -- Annotation tipi
    annotation_type TEXT NOT NULL,           -- 'bbox', 'polygon', 'segmentation'

    -- Istatistikler (cache)
    image_count INTEGER DEFAULT 0,
    annotation_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,

    -- Annotation durumu
    annotated_count INTEGER DEFAULT 0,       -- Tamamlanan resim sayisi
    pending_count INTEGER DEFAULT 0,         -- Bekleyen resim sayisi

    -- Label Studio entegrasyonu
    labelstudio_project_id INTEGER,

    -- Ayarlar
    default_classes TEXT[],                  -- Bu dataset icin varsayilan class'lar

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- OD DATASET IMAGES (Dataset-Image iliskisi)
-- =====================================================
CREATE TABLE od_dataset_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE CASCADE,
    shelf_image_id UUID NOT NULL REFERENCES shelf_images(id) ON DELETE CASCADE,

    -- Bu dataset icindeki annotation durumu
    annotation_status TEXT DEFAULT 'pending', -- 'pending', 'ai_predicted', 'in_progress', 'completed', 'skipped'

    -- Label Studio task ID (bu dataset icin)
    labelstudio_task_id INTEGER,

    -- Annotation sayisi (bu dataset icin)
    annotation_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(dataset_id, shelf_image_id)
);

-- =====================================================
-- OD CLASSES (Global Class Tanimlari)
-- =====================================================
CREATE TABLE od_classes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Class bilgileri
    name TEXT NOT NULL UNIQUE,               -- 'coca_cola_330ml', 'refrigerator', 'shelf'
    display_name TEXT,                       -- 'Coca Cola 330ml'

    -- Renk (annotation UI icin)
    color TEXT DEFAULT '#FF0000',

    -- Kategori (gruplama icin)
    category TEXT,                           -- 'product', 'fixture', 'zone'

    -- Alisaslar (eski isimler, import sirasinda eslestirme icin)
    aliases TEXT[],                          -- ['cooler', 'fridge'] -> 'refrigerator'

    -- Istatistikler
    usage_count INTEGER DEFAULT 0,           -- Kac annotation'da kullanildi

    -- Durum
    is_active BOOLEAN DEFAULT true,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- OD ANNOTATIONS (Annotation'lar)
-- =====================================================
CREATE TABLE od_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Iliskiler
    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE CASCADE,
    shelf_image_id UUID NOT NULL REFERENCES shelf_images(id) ON DELETE CASCADE,
    class_id UUID NOT NULL REFERENCES od_classes(id),

    -- Bounding Box (normalized 0-1)
    x FLOAT NOT NULL,                        -- Center X
    y FLOAT NOT NULL,                        -- Center Y
    width FLOAT NOT NULL,
    height FLOAT NOT NULL,

    -- Alternatif: Polygon (segmentation icin)
    polygon JSONB,                           -- [[x1,y1], [x2,y2], ...]

    -- Kaynak
    source TEXT DEFAULT 'manual',            -- 'manual', 'ai_sam3', 'ai_yolo', 'ai_grounding_dino', 'import'
    confidence FLOAT,                        -- AI prediction confidence

    -- Review durumu
    is_verified BOOLEAN DEFAULT false,
    verified_by TEXT,
    verified_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- OD TRAINING JOBS (Training Job'lari)
-- =====================================================
CREATE TABLE od_training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Iliski
    dataset_id UUID NOT NULL REFERENCES od_datasets(id),

    -- Model bilgileri
    model_type TEXT NOT NULL,                -- 'yolov8n', 'yolov8s', 'yolov8m', 'yolov11', 'rt-detr'
    base_model TEXT,                         -- Pre-trained model path

    -- Training config
    config JSONB NOT NULL,                   -- epochs, batch_size, imgsz, augmentation, etc.

    -- Split config
    train_split FLOAT DEFAULT 0.8,
    val_split FLOAT DEFAULT 0.15,
    test_split FLOAT DEFAULT 0.05,

    -- Durum
    status TEXT DEFAULT 'pending',           -- 'pending', 'preparing', 'training', 'completed', 'failed'
    progress FLOAT DEFAULT 0,
    current_epoch INTEGER,
    total_epochs INTEGER,

    -- Runpod
    runpod_job_id TEXT,

    -- Sonuclar
    metrics JSONB,                           -- mAP, precision, recall, etc.
    model_url TEXT,                          -- Trained model artifact URL

    -- Zamanlar
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- CLASS RENAME HISTORY (Audit log)
-- =====================================================
CREATE TABLE od_class_changes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    class_id UUID REFERENCES od_classes(id),

    action TEXT NOT NULL,                    -- 'rename', 'merge', 'delete'

    -- Detaylar
    old_value TEXT,
    new_value TEXT,
    affected_annotations INTEGER,

    performed_by TEXT,
    performed_at TIMESTAMP DEFAULT NOW()
);

-- =====================================================
-- INDEXES
-- =====================================================
CREATE INDEX idx_shelf_images_source ON shelf_images(source);
CREATE INDEX idx_shelf_images_status ON shelf_images(status);
CREATE INDEX idx_od_dataset_images_dataset ON od_dataset_images(dataset_id);
CREATE INDEX idx_od_dataset_images_status ON od_dataset_images(annotation_status);
CREATE INDEX idx_od_annotations_dataset ON od_annotations(dataset_id);
CREATE INDEX idx_od_annotations_class ON od_annotations(class_id);
CREATE INDEX idx_od_annotations_image ON od_annotations(shelf_image_id);
CREATE INDEX idx_od_classes_name ON od_classes(name);
CREATE INDEX idx_od_classes_category ON od_classes(category);
```

---

## 3. API Endpoints

### 3.1 Shelf Images API

```
# Resim Yonetimi
GET    /api/v1/shelf-images                  # Liste (pagination, filter)
POST   /api/v1/shelf-images                  # Tekli upload
POST   /api/v1/shelf-images/bulk             # Coklu upload
DELETE /api/v1/shelf-images/:id              # Sil

# Import
POST   /api/v1/shelf-images/import/folder    # Klasorden import
POST   /api/v1/shelf-images/import/coco      # COCO dataset import
POST   /api/v1/shelf-images/import/yolo      # YOLO dataset import
POST   /api/v1/shelf-images/import/url       # URL'den import
```

### 3.2 OD Datasets API

```
# Dataset Yonetimi
GET    /api/v1/od-datasets                   # Liste
POST   /api/v1/od-datasets                   # Olustur
GET    /api/v1/od-datasets/:id               # Detay
PUT    /api/v1/od-datasets/:id               # Guncelle
DELETE /api/v1/od-datasets/:id               # Sil

# Dataset Images
GET    /api/v1/od-datasets/:id/images        # Dataset'teki resimler
POST   /api/v1/od-datasets/:id/images        # Resimleri ekle
DELETE /api/v1/od-datasets/:id/images/:imgId # Resmi cikar

# Annotations
GET    /api/v1/od-datasets/:id/annotations   # Tum annotation'lar
POST   /api/v1/od-datasets/:id/annotations   # Annotation ekle
PUT    /api/v1/od-datasets/:id/annotations/:annId  # Guncelle
DELETE /api/v1/od-datasets/:id/annotations/:annId  # Sil

# AI Annotation
POST   /api/v1/od-datasets/:id/ai-annotate   # AI ile bulk annotate
POST   /api/v1/od-datasets/:id/ai-annotate/image/:imgId  # Tek resim

# Export
GET    /api/v1/od-datasets/:id/export/yolo   # YOLO format export
GET    /api/v1/od-datasets/:id/export/coco   # COCO format export

# Label Studio Sync
POST   /api/v1/od-datasets/:id/sync-labelstudio  # LS'ye gonder
POST   /api/v1/od-datasets/:id/import-labelstudio # LS'den al
```

### 3.3 Classes API

```
# Class Yonetimi
GET    /api/v1/od-classes                    # Liste
POST   /api/v1/od-classes                    # Olustur
PUT    /api/v1/od-classes/:id                # Guncelle (rename dahil)
DELETE /api/v1/od-classes/:id                # Sil (tum annotation'lari da siler)

# Bulk Islemler
POST   /api/v1/od-classes/merge              # Iki class'i birlestir
POST   /api/v1/od-classes/bulk-rename        # Toplu isim degistir

# Import/Export
GET    /api/v1/od-classes/export             # Class listesini export et
POST   /api/v1/od-classes/import             # Class listesi import et
```

### 3.4 Training API

```
# Training Jobs
GET    /api/v1/od-training                   # Job listesi
POST   /api/v1/od-training                   # Yeni training baslat
GET    /api/v1/od-training/:id               # Job detay
DELETE /api/v1/od-training/:id               # Iptal et

# Webhooks
POST   /api/v1/webhooks/od-training          # Runpod callback
```

---

## 4. Frontend Sayfalari

### 4.1 Shelf Images Sayfasi (`/shelf-images`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📷 Shelf Images                                     [+ Upload] [Import]│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Filters: [Source ▼] [Status ▼] [Folder ▼]    Search: [____________]   │
│                                                                         │
│  ┌─── Grid View ─────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ☐ Select All (245 images)                            [Actions ▼] │ │
│  │                                                                    │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │  │ 🖼️      │  │ 🖼️      │  │ 🖼️      │  │ 🖼️      │          │ │
│  │  │          │  │          │  │          │  │          │          │ │
│  │  │          │  │          │  │          │  │          │          │ │
│  │  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤          │ │
│  │  │ upload   │  │ COCO     │  │ upload   │  │ OpenImg  │          │ │
│  │  │ 1920x1080│  │ 640x480  │  │ 1280x720 │  │ 1024x768 │          │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Showing 1-24 of 245                                    [< 1 2 3 ... >] │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Import Modal

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📥 Import Images                                                   [X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── Import Source ───────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ◉ Local Files / Folder                                         │   │
│  │    [Choose Files] [Choose Folder]                                │   │
│  │                                                                  │   │
│  │  ○ URL (zip or image)                                           │   │
│  │    [________________________________________________]            │   │
│  │                                                                  │   │
│  │  ○ COCO Dataset                                                 │   │
│  │    Images folder: [Choose...]                                   │   │
│  │    Annotations JSON: [Choose...]                                │   │
│  │    ☐ Import annotations too                                     │   │
│  │    ☐ Auto-create classes from categories                        │   │
│  │                                                                  │   │
│  │  ○ YOLO Dataset                                                 │   │
│  │    Dataset folder: [Choose...]                                  │   │
│  │    ☐ Import annotations too                                     │   │
│  │    ☐ Auto-create classes from data.yaml                         │   │
│  │                                                                  │   │
│  │  ○ Roboflow Export                                              │   │
│  │    [Paste Roboflow download link...]                            │   │
│  │                                                                  │   │
│  │  ○ Open Images Dataset                                          │   │
│  │    Class filter: [refrigerator, shelf, ...]                     │   │
│  │    Max images: [1000]                                           │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─── Options ─────────────────────────────────────────────────────┐   │
│  │  Folder name: [imported_coco_2024]                              │   │
│  │  ☐ Add to existing dataset: [Select dataset ▼]                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                                        [Cancel] [🚀 Start Import]       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 OD Datasets Sayfasi (`/od-datasets`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📦 Object Detection Datasets                              [+ New Dataset]│
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │                 │  │                 │  │                 │         │
│  │ 🧊 Refrigerator │  │ 📦 Product      │  │ 📏 Shelf        │         │
│  │    Detection    │  │    Detection    │  │    Detection    │         │
│  │                 │  │                 │  │                 │         │
│  │ ────────────────│  │ ────────────────│  │ ────────────────│         │
│  │                 │  │                 │  │                 │         │
│  │ 245 images      │  │ 245 images      │  │ 180 images      │         │
│  │ 3 classes       │  │ 847 classes     │  │ 5 classes       │         │
│  │                 │  │                 │  │                 │         │
│  │ ████████░░ 89%  │  │ ████░░░░░░ 45% │  │ ██████████ 100% │         │
│  │ annotated       │  │ annotated       │  │ annotated       │         │
│  │                 │  │                 │  │                 │         │
│  │ [Open]          │  │ [Open]          │  │ [Open]          │         │
│  │                 │  │                 │  │                 │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Dataset Detay Sayfasi (`/od-datasets/:id`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📦 Product Detection                              [Edit] [Delete]      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── Actions ───────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  [🤖 AI Auto-Annotate]  [✏️ Open in Label Studio]                 │ │
│  │                                                                    │ │
│  │  [📤 Export ▼]  [🏋️ Train Model]  [+ Add Images]                  │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── Stats ─────────────────────────────────────────────────────────┐ │
│  │  Total: 245 │ Annotated: 110 │ AI Predicted: 80 │ Pending: 55     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── Classes (847) ──────────────────────────────── [Manage Classes] ┐ │
│  │                                                                    │ │
│  │  🔴 coca_cola_330ml (145)   🔵 fanta_500ml (89)   🟢 sprite (67)   │ │
│  │  🟡 pepsi_500ml (56)        🟣 water_500ml (48)   ... [+844 more]  │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── Images ────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  Filter: [All ▼] [Status ▼]                    [□ Grid] [≡ List]  │ │
│  │                                                                    │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │  │ 🖼️      │  │ 🖼️      │  │ 🖼️      │  │ 🖼️      │          │ │
│  │  │ ┌─┐┌─┐   │  │          │  │ ┌─┐ ┌─┐  │  │ ┌─┐      │          │ │
│  │  │ └─┘└─┘   │  │          │  │ └─┘ └─┘  │  │ └─┘      │          │ │
│  │  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤          │ │
│  │  │ ✅ Done  │  │ ⏳ Empty │  │ 🤖 AI    │  │ 🤖 AI    │          │ │
│  │  │ 12 boxes │  │ 0 boxes  │  │ 8 boxes  │  │ 3 boxes  │          │ │
│  │  │ [Edit]   │  │ [Start]  │  │ [Review] │  │ [Review] │          │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.5 Class Management Sayfasi (`/od-classes`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🏷️ Class Management                                    [+ New Class]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Search: [________________]     Category: [All ▼]     [Import] [Export] │
│                                                                         │
│  ┌─── Classes ───────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  ☐ │ Name                  │ Display      │ Category │ Usage │    │ │
│  │  ──┼───────────────────────┼──────────────┼──────────┼───────┼──  │ │
│  │  ☐ │ 🔴 coca_cola_330ml    │ Coca Cola..  │ product  │ 1,245 │ ⋮  │ │
│  │  ☐ │ 🔵 fanta_500ml        │ Fanta 500ml  │ product  │   892 │ ⋮  │ │
│  │  ☐ │ 🟢 refrigerator       │ Refrigerator │ fixture  │   156 │ ⋮  │ │
│  │  ☐ │ 🟡 cooler             │ Cooler       │ fixture  │    45 │ ⋮  │ │
│  │                                                                    │ │
│  │  ────────────────────────────────────────────────────────────────  │ │
│  │  [Merge Selected]  [Delete Selected]  [Change Category]            │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  ✏️ Edit Class: cooler                                              [X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Name:         [refrigerator________]   ← "cooler" -> "refrigerator"   │
│  Display Name: [Refrigerator_________]                                  │
│  Category:     [fixture ▼]                                             │
│  Color:        [🟢 #00FF00] [Pick]                                     │
│                                                                         │
│  Aliases:      [cooler, fridge, buzdolabi]                             │
│                (eski isimler, import sirasinda otomatik eslestirme)     │
│                                                                         │
│  ⚠️ Warning: This will rename 45 annotations across 3 datasets.        │
│                                                                         │
│                                              [Cancel] [💾 Save Changes] │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.6 AI Auto-Annotate Modal

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🤖 AI Auto-Annotate                                                [X] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─── Model Selection ─────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ◉ SAM3 (Segment Anything 3)                                    │   │
│  │    Otomatik olarak tum objeleri tespit eder                      │   │
│  │    ┌─────────────────────────────────────────────────────────┐  │   │
│  │    │  Points per side:  [32 ▼]                               │  │   │
│  │    │  Min mask area:    [500 ▼] px                           │  │   │
│  │    │  Confidence:       [0.85 ▼]                             │  │   │
│  │    │  ☐ Convert masks to bboxes                              │  │   │
│  │    └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ○ YOLO-World (Open Vocabulary)                                 │   │
│  │    Text prompt ile obje tespiti                                  │   │
│  │    ┌─────────────────────────────────────────────────────────┐  │   │
│  │    │  Prompts: [refrigerator, shelf, product, bottle, can]   │  │   │
│  │    │  Confidence: [0.5 ▼]                                    │  │   │
│  │    └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ○ Grounding DINO                                               │   │
│  │    Daha hassas text-based detection                             │   │
│  │    ┌─────────────────────────────────────────────────────────┐  │   │
│  │    │  Prompts: [___________________________________________] │  │   │
│  │    │  Box threshold: [0.35 ▼]  Text threshold: [0.25 ▼]     │  │   │
│  │    └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │  ○ Custom Model (Onceki egittigim model)                        │   │
│  │    ┌─────────────────────────────────────────────────────────┐  │   │
│  │    │  Model: [shelf_detector_v2.pt ▼]                        │  │   │
│  │    │  Confidence: [0.5 ▼]                                    │  │   │
│  │    └─────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─── Apply To ────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  ◉ All images (245)                                             │   │
│  │  ○ Only unannotated (156)                                        │   │
│  │  ○ Selected images (12)                                          │   │
│  │                                                                  │   │
│  │  ☐ Overwrite existing AI predictions                            │   │
│  │  ☐ Skip images with manual annotations                          │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                                     [Cancel] [🚀 Start Auto-Annotation] │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Label Studio Entegrasyonu

### 5.1 Deployment

```yaml
# docker-compose.labelstudio.yml
version: '3.8'

services:
  labelstudio:
    image: heartexlabs/label-studio:latest
    ports:
      - "8080:8080"
    environment:
      - LABEL_STUDIO_HOST=https://annotate.buybuddy.co
      - DJANGO_DB=postgresql
      - POSTGRE_HOST=your-db-host
      - POSTGRE_PORT=5432
      - POSTGRE_NAME=labelstudio
      - POSTGRE_USER=labelstudio
      - POSTGRE_PASSWORD=${LS_DB_PASSWORD}
      - LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
    volumes:
      - labelstudio-data:/label-studio/data
    restart: always

  ml-backend:
    build: ./labelstudio-ml-backend
    ports:
      - "9090:9090"
    environment:
      - MODEL_PATH=/models/sam3.pt
      - DEVICE=cuda
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: always

volumes:
  labelstudio-data:
```

### 5.2 Annotation Template (Object Detection)

```xml
<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <!-- Dynamic labels from API -->
    <Label value="refrigerator" background="#FF0000"/>
    <Label value="shelf" background="#00FF00"/>
    <Label value="product" background="#0000FF"/>
  </RectangleLabels>

  <TextArea name="notes" toName="image"
            placeholder="Optional notes..."
            maxSubmissions="1"/>
</View>
```

### 5.3 ML Backend (SAM3 Pre-annotation)

```python
# labelstudio-ml-backend/model.py
from label_studio_ml import LabelStudioMLBase
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

class SAM3Backend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # SAM3 model yukle
        self.sam = sam_model_registry["vit_h"](checkpoint="/models/sam3.pt")
        self.sam.to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            image_url = task['data']['image']
            image = self._load_image(image_url)

            # SAM3 ile mask'lari bul
            masks = self.mask_generator.generate(image)

            # Mask'lari bbox'a cevir
            regions = []
            for mask in masks:
                bbox = mask['bbox']  # [x, y, w, h]
                x, y, w, h = bbox

                # Normalize (0-100)
                img_h, img_w = image.shape[:2]
                regions.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": (x / img_w) * 100,
                        "y": (y / img_h) * 100,
                        "width": (w / img_w) * 100,
                        "height": (h / img_h) * 100,
                        "rectanglelabels": ["product"],  # Default label
                    },
                    "score": mask['predicted_iou']
                })

            predictions.append({
                "result": regions,
                "score": np.mean([m['predicted_iou'] for m in masks]) if masks else 0
            })

        return predictions

    def _load_image(self, url):
        import requests
        from PIL import Image
        import io

        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return np.array(image)
```

### 5.4 Sync Workflow

```python
# api/src/services/labelstudio_sync.py

class LabelStudioSync:
    def __init__(self):
        self.ls_client = LabelStudio(
            url="https://annotate.buybuddy.co",
            api_key=os.environ["LABELSTUDIO_API_KEY"]
        )
        self.supabase = get_supabase_client()

    async def sync_dataset_to_labelstudio(self, dataset_id: str):
        """Dataset'i Label Studio'ya gonder"""

        # 1. Dataset bilgilerini al
        dataset = await self.supabase.table("od_datasets").select("*").eq("id", dataset_id).single()

        # 2. Label Studio projesi olustur/guncelle
        if dataset.labelstudio_project_id:
            project = self.ls_client.get_project(dataset.labelstudio_project_id)
        else:
            # Class'lari al
            classes = await self._get_dataset_classes(dataset_id)

            # Labeling config olustur
            labeling_config = self._generate_labeling_config(classes)

            project = self.ls_client.create_project(
                title=dataset.name,
                description=dataset.description,
                label_config=labeling_config
            )

            # Project ID'yi kaydet
            await self.supabase.table("od_datasets").update({
                "labelstudio_project_id": project.id
            }).eq("id", dataset_id)

        # 3. Resimleri task olarak ekle
        images = await self.supabase.table("od_dataset_images") \
            .select("*, shelf_images(*)") \
            .eq("dataset_id", dataset_id) \
            .is_("labelstudio_task_id", None)

        tasks = []
        for img in images:
            task_data = {
                "image": img.shelf_images.image_url,
                "shelf_image_id": img.shelf_image_id,
                "dataset_id": dataset_id
            }

            # Mevcut annotation'lari ekle (pre-annotation)
            annotations = await self._get_image_annotations(dataset_id, img.shelf_image_id)
            if annotations:
                task_data["predictions"] = [{
                    "result": self._convert_to_ls_format(annotations)
                }]

            tasks.append(task_data)

        # Bulk import
        imported = project.import_tasks(tasks)

        # Task ID'leri kaydet
        for i, task in enumerate(imported):
            await self.supabase.table("od_dataset_images").update({
                "labelstudio_task_id": task.id
            }).eq("id", images[i].id)

    async def import_from_labelstudio(self, dataset_id: str):
        """Label Studio'dan annotation'lari al"""

        dataset = await self.supabase.table("od_datasets").select("*").eq("id", dataset_id).single()

        if not dataset.labelstudio_project_id:
            raise ValueError("Dataset is not synced with Label Studio")

        project = self.ls_client.get_project(dataset.labelstudio_project_id)

        # Completed task'lari al
        tasks = project.get_tasks(filters={"completed": True})

        for task in tasks:
            if not task.annotations:
                continue

            shelf_image_id = task.data.get("shelf_image_id")

            # Mevcut annotation'lari sil
            await self.supabase.table("od_annotations") \
                .delete() \
                .eq("dataset_id", dataset_id) \
                .eq("shelf_image_id", shelf_image_id)

            # Yeni annotation'lari ekle
            for annotation in task.annotations:
                for region in annotation.result:
                    if region["type"] == "rectanglelabels":
                        value = region["value"]
                        label = value["rectanglelabels"][0]

                        # Class ID'yi bul veya olustur
                        class_id = await self._get_or_create_class(label)

                        await self.supabase.table("od_annotations").insert({
                            "dataset_id": dataset_id,
                            "shelf_image_id": shelf_image_id,
                            "class_id": class_id,
                            "x": value["x"] / 100,
                            "y": value["y"] / 100,
                            "width": value["width"] / 100,
                            "height": value["height"] / 100,
                            "source": "labelstudio",
                            "is_verified": True
                        })

            # Status guncelle
            await self.supabase.table("od_dataset_images").update({
                "annotation_status": "completed"
            }).eq("dataset_id", dataset_id).eq("shelf_image_id", shelf_image_id)
```

---

## 6. Runpod Workers

### 6.1 AI Annotation Worker

```python
# workers/ai-annotation/handler.py

import runpod
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO
import torch

# Model'leri yukle (cold start optimization)
sam = None
yolo_world = None
grounding_dino = None

def load_models():
    global sam, yolo_world, grounding_dino

    if sam is None:
        sam_checkpoint = "/models/sam3_vit_h.pt"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to("cuda")

    if yolo_world is None:
        yolo_world = YOLO("yolov8x-worldv2.pt")

    # Grounding DINO is loaded on demand due to size

def handler(job):
    load_models()

    job_input = job["input"]
    model_type = job_input["model_type"]  # 'sam3', 'yolo_world', 'grounding_dino', 'custom'
    images = job_input["images"]  # [{id, url}, ...]
    config = job_input.get("config", {})

    results = []

    for img_data in images:
        image_id = img_data["id"]
        image_url = img_data["url"]

        # Download image
        image = download_image(image_url)

        if model_type == "sam3":
            predictions = run_sam3(image, config)
        elif model_type == "yolo_world":
            predictions = run_yolo_world(image, config)
        elif model_type == "grounding_dino":
            predictions = run_grounding_dino(image, config)
        elif model_type == "custom":
            predictions = run_custom_model(image, config)

        results.append({
            "image_id": image_id,
            "predictions": predictions
        })

    return {"results": results}

def run_sam3(image, config):
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=config.get("points_per_side", 32),
        pred_iou_thresh=config.get("confidence", 0.86),
        min_mask_region_area=config.get("min_area", 500),
    )

    masks = mask_generator.generate(image)

    predictions = []
    for mask in masks:
        x, y, w, h = mask["bbox"]
        img_h, img_w = image.shape[:2]

        predictions.append({
            "x": x / img_w,
            "y": y / img_h,
            "width": w / img_w,
            "height": h / img_h,
            "confidence": mask["predicted_iou"],
            "class": "object",  # SAM doesn't classify
            "mask": mask["segmentation"].tolist() if config.get("include_masks") else None
        })

    return predictions

def run_yolo_world(image, config):
    prompts = config.get("prompts", ["object"])
    confidence = config.get("confidence", 0.5)

    yolo_world.set_classes(prompts)
    results = yolo_world(image, conf=confidence)

    predictions = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0].tolist()
        img_h, img_w = image.shape[:2]

        predictions.append({
            "x": x1 / img_w,
            "y": y1 / img_h,
            "width": (x2 - x1) / img_w,
            "height": (y2 - y1) / img_h,
            "confidence": r.conf[0].item(),
            "class": prompts[int(r.cls[0])]
        })

    return predictions

runpod.serverless.start({"handler": handler})
```

### 6.2 Object Detection Training Worker

```python
# workers/od-training/handler.py

import runpod
from ultralytics import YOLO
import os
import yaml
from supabase import create_client

def handler(job):
    job_input = job["input"]

    dataset_id = job_input["dataset_id"]
    model_type = job_input["model_type"]  # 'yolov8n', 'yolov8s', etc.
    config = job_input["config"]
    callback_url = job_input["callback_url"]

    # 1. Dataset'i indir ve YOLO formatina cevir
    dataset_path = prepare_dataset(dataset_id, config)

    # 2. Model olustur
    model = YOLO(f"{model_type}.pt")

    # 3. Training
    results = model.train(
        data=f"{dataset_path}/data.yaml",
        epochs=config.get("epochs", 100),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        device="cuda",
        project="/outputs",
        name=dataset_id,
        exist_ok=True,

        # Augmentation
        augment=config.get("augment", True),
        mosaic=config.get("mosaic", 1.0),
        mixup=config.get("mixup", 0.1),

        # Callbacks for progress
        callbacks={
            "on_train_epoch_end": lambda trainer: report_progress(
                callback_url,
                trainer.epoch,
                config.get("epochs", 100),
                trainer.metrics
            )
        }
    )

    # 4. Sonuclari kaydet
    best_model_path = f"/outputs/{dataset_id}/weights/best.pt"

    # Upload to Supabase Storage
    model_url = upload_model(best_model_path, dataset_id)

    return {
        "success": True,
        "model_url": model_url,
        "metrics": {
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"]
        }
    }

def prepare_dataset(dataset_id, config):
    """Supabase'den veriyi cekip YOLO formatina cevir"""

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

    # Annotation'lari al
    annotations = supabase.table("od_annotations") \
        .select("*, shelf_images(*), od_classes(*)") \
        .eq("dataset_id", dataset_id) \
        .execute().data

    # Class mapping olustur
    classes = list(set([a["od_classes"]["name"] for a in annotations]))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Dataset folder olustur
    dataset_path = f"/data/{dataset_id}"
    os.makedirs(f"{dataset_path}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_path}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_path}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_path}/labels/val", exist_ok=True)

    # Train/val split
    train_split = config.get("train_split", 0.8)

    # Group by image
    images = {}
    for ann in annotations:
        img_id = ann["shelf_image_id"]
        if img_id not in images:
            images[img_id] = {
                "url": ann["shelf_images"]["image_url"],
                "annotations": []
            }
        images[img_id]["annotations"].append(ann)

    # Process each image
    image_list = list(images.items())
    train_count = int(len(image_list) * train_split)

    for i, (img_id, data) in enumerate(image_list):
        split = "train" if i < train_count else "val"

        # Download image
        img_path = download_and_save_image(
            data["url"],
            f"{dataset_path}/images/{split}/{img_id}.jpg"
        )

        # Create label file
        label_path = f"{dataset_path}/labels/{split}/{img_id}.txt"
        with open(label_path, "w") as f:
            for ann in data["annotations"]:
                class_idx = class_to_idx[ann["od_classes"]["name"]]
                # YOLO format: class x_center y_center width height
                x_center = ann["x"] + ann["width"] / 2
                y_center = ann["y"] + ann["height"] / 2
                f.write(f"{class_idx} {x_center} {y_center} {ann['width']} {ann['height']}\n")

    # Create data.yaml
    data_yaml = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)}
    }

    with open(f"{dataset_path}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return dataset_path

runpod.serverless.start({"handler": handler})
```

---

## 7. Class Yonetimi Detaylari

### 7.1 Class Rename Islemi

```python
# api/src/services/class_manager.py

class ClassManager:
    def __init__(self, supabase):
        self.supabase = supabase

    async def rename_class(self, class_id: str, new_name: str, user_id: str):
        """Class ismini degistir ve tum annotation'lari guncelle"""

        # 1. Mevcut class'i al
        old_class = await self.supabase.table("od_classes") \
            .select("*") \
            .eq("id", class_id) \
            .single()

        old_name = old_class["name"]

        # 2. Yeni isim zaten var mi kontrol et
        existing = await self.supabase.table("od_classes") \
            .select("id") \
            .eq("name", new_name) \
            .maybeSingle()

        if existing:
            raise ValueError(f"Class '{new_name}' already exists. Use merge instead.")

        # 3. Kac annotation etkilenecek
        affected = await self.supabase.table("od_annotations") \
            .select("id", count="exact") \
            .eq("class_id", class_id)

        # 4. Class'i guncelle
        await self.supabase.table("od_classes").update({
            "name": new_name,
            "aliases": old_class.get("aliases", []) + [old_name],  # Eski ismi alias olarak ekle
            "updated_at": "now()"
        }).eq("id", class_id)

        # 5. Audit log
        await self.supabase.table("od_class_changes").insert({
            "class_id": class_id,
            "action": "rename",
            "old_value": old_name,
            "new_value": new_name,
            "affected_annotations": affected.count,
            "performed_by": user_id
        })

        # 6. Label Studio projelerini guncelle (eger varsa)
        await self._update_labelstudio_labels(old_name, new_name)

        return {
            "success": True,
            "old_name": old_name,
            "new_name": new_name,
            "affected_annotations": affected.count
        }

    async def merge_classes(self, source_ids: list[str], target_id: str, user_id: str):
        """Birden fazla class'i tek bir class'a birlestir"""

        target_class = await self.supabase.table("od_classes") \
            .select("*") \
            .eq("id", target_id) \
            .single()

        total_affected = 0
        merged_names = []

        for source_id in source_ids:
            if source_id == target_id:
                continue

            source_class = await self.supabase.table("od_classes") \
                .select("*") \
                .eq("id", source_id) \
                .single()

            merged_names.append(source_class["name"])

            # Annotation'lari hedef class'a tasi
            result = await self.supabase.table("od_annotations") \
                .update({"class_id": target_id}) \
                .eq("class_id", source_id)

            total_affected += len(result.data)

            # Kaynak class'i sil
            await self.supabase.table("od_classes") \
                .delete() \
                .eq("id", source_id)

        # Hedef class'in alias'larina ekle
        new_aliases = target_class.get("aliases", []) + merged_names
        await self.supabase.table("od_classes").update({
            "aliases": new_aliases,
            "usage_count": target_class["usage_count"] + total_affected
        }).eq("id", target_id)

        # Audit log
        await self.supabase.table("od_class_changes").insert({
            "class_id": target_id,
            "action": "merge",
            "old_value": ",".join(merged_names),
            "new_value": target_class["name"],
            "affected_annotations": total_affected,
            "performed_by": user_id
        })

        return {
            "success": True,
            "merged": merged_names,
            "target": target_class["name"],
            "affected_annotations": total_affected
        }

    async def delete_class(self, class_id: str, user_id: str, delete_annotations: bool = True):
        """Class'i sil, opsiyonel olarak annotation'lari da sil"""

        class_data = await self.supabase.table("od_classes") \
            .select("*") \
            .eq("id", class_id) \
            .single()

        affected = await self.supabase.table("od_annotations") \
            .select("id", count="exact") \
            .eq("class_id", class_id)

        if delete_annotations:
            # Annotation'lari sil
            await self.supabase.table("od_annotations") \
                .delete() \
                .eq("class_id", class_id)
        else:
            # Annotation'lari "unknown" class'a tasi
            unknown_class = await self._get_or_create_unknown_class()
            await self.supabase.table("od_annotations") \
                .update({"class_id": unknown_class["id"]}) \
                .eq("class_id", class_id)

        # Class'i sil
        await self.supabase.table("od_classes") \
            .delete() \
            .eq("id", class_id)

        # Audit log
        await self.supabase.table("od_class_changes").insert({
            "class_id": class_id,
            "action": "delete",
            "old_value": class_data["name"],
            "new_value": None,
            "affected_annotations": affected.count,
            "performed_by": user_id
        })

        return {
            "success": True,
            "deleted_class": class_data["name"],
            "affected_annotations": affected.count,
            "annotations_deleted": delete_annotations
        }
```

---

## 8. Import/Export

### 8.1 COCO Import

```python
# api/src/services/dataset_import.py

async def import_coco_dataset(
    images_folder: str,
    annotations_file: str,
    target_dataset_id: str = None,
    import_annotations: bool = True,
    auto_create_classes: bool = True
):
    """COCO formatindaki dataset'i import et"""

    with open(annotations_file) as f:
        coco_data = json.load(f)

    # Category -> Class mapping
    class_mapping = {}
    if auto_create_classes:
        for cat in coco_data["categories"]:
            existing = await supabase.table("od_classes") \
                .select("id") \
                .eq("name", cat["name"]) \
                .maybeSingle()

            if existing:
                class_mapping[cat["id"]] = existing["id"]
            else:
                new_class = await supabase.table("od_classes").insert({
                    "name": cat["name"],
                    "display_name": cat["name"].replace("_", " ").title(),
                    "category": "imported"
                }).execute()
                class_mapping[cat["id"]] = new_class.data[0]["id"]

    # Images
    image_mapping = {}  # coco_image_id -> shelf_image_id

    for img in coco_data["images"]:
        # Upload image to Supabase Storage
        image_path = os.path.join(images_folder, img["file_name"])
        image_url = await upload_to_storage(image_path, f"shelf-images/{img['file_name']}")

        # Create shelf_image record
        shelf_image = await supabase.table("shelf_images").insert({
            "filename": img["file_name"],
            "image_url": image_url,
            "width": img["width"],
            "height": img["height"],
            "source": "import",
            "source_dataset": "COCO"
        }).execute()

        image_mapping[img["id"]] = shelf_image.data[0]["id"]

        # Add to dataset if specified
        if target_dataset_id:
            await supabase.table("od_dataset_images").insert({
                "dataset_id": target_dataset_id,
                "shelf_image_id": shelf_image.data[0]["id"]
            })

    # Annotations
    if import_annotations and target_dataset_id:
        for ann in coco_data["annotations"]:
            shelf_image_id = image_mapping.get(ann["image_id"])
            class_id = class_mapping.get(ann["category_id"])

            if not shelf_image_id or not class_id:
                continue

            # COCO bbox: [x, y, width, height] (absolute pixels)
            # Convert to normalized
            img_info = next(i for i in coco_data["images"] if i["id"] == ann["image_id"])
            x, y, w, h = ann["bbox"]

            await supabase.table("od_annotations").insert({
                "dataset_id": target_dataset_id,
                "shelf_image_id": shelf_image_id,
                "class_id": class_id,
                "x": x / img_info["width"],
                "y": y / img_info["height"],
                "width": w / img_info["width"],
                "height": h / img_info["height"],
                "source": "import"
            })

    return {
        "images_imported": len(image_mapping),
        "classes_created": len([c for c in class_mapping.values() if c]),
        "annotations_imported": len(coco_data["annotations"]) if import_annotations else 0
    }
```

### 8.2 YOLO Export

```python
# api/src/services/dataset_export.py

async def export_to_yolo(dataset_id: str) -> str:
    """Dataset'i YOLO formatinda export et"""

    # Get dataset info
    dataset = await supabase.table("od_datasets") \
        .select("*") \
        .eq("id", dataset_id) \
        .single()

    # Get all annotations with class info
    annotations = await supabase.table("od_annotations") \
        .select("*, shelf_images(*), od_classes(*)") \
        .eq("dataset_id", dataset_id) \
        .execute()

    # Create temp directory
    export_dir = f"/tmp/export_{dataset_id}"
    os.makedirs(f"{export_dir}/images/train", exist_ok=True)
    os.makedirs(f"{export_dir}/images/val", exist_ok=True)
    os.makedirs(f"{export_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{export_dir}/labels/val", exist_ok=True)

    # Get unique classes
    classes = list(set([a["od_classes"]["name"] for a in annotations.data]))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Group annotations by image
    images = {}
    for ann in annotations.data:
        img_id = ann["shelf_image_id"]
        if img_id not in images:
            images[img_id] = {
                "url": ann["shelf_images"]["image_url"],
                "filename": ann["shelf_images"]["filename"],
                "annotations": []
            }
        images[img_id]["annotations"].append(ann)

    # Split and export
    image_list = list(images.items())
    random.shuffle(image_list)
    train_count = int(len(image_list) * 0.8)

    for i, (img_id, data) in enumerate(image_list):
        split = "train" if i < train_count else "val"

        # Copy image
        img_response = requests.get(data["url"])
        img_path = f"{export_dir}/images/{split}/{data['filename']}"
        with open(img_path, "wb") as f:
            f.write(img_response.content)

        # Create label file
        label_filename = os.path.splitext(data["filename"])[0] + ".txt"
        label_path = f"{export_dir}/labels/{split}/{label_filename}"

        with open(label_path, "w") as f:
            for ann in data["annotations"]:
                class_idx = class_to_idx[ann["od_classes"]["name"]]
                # YOLO format: class x_center y_center width height
                x_center = ann["x"] + ann["width"] / 2
                y_center = ann["y"] + ann["height"] / 2
                f.write(f"{class_idx} {x_center} {y_center} {ann['width']} {ann['height']}\n")

    # Create data.yaml
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes
    }

    with open(f"{export_dir}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # Zip and upload
    zip_path = f"/tmp/{dataset_id}.zip"
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", export_dir)

    # Upload to storage
    export_url = await upload_to_storage(zip_path, f"exports/{dataset_id}.zip")

    # Cleanup
    shutil.rmtree(export_dir)
    os.remove(zip_path)

    return export_url
```

---

## 9. Uygulama Adimlari

### Phase 1: Temel Altyapi
- [ ] Veritabani tablolarini olustur
- [ ] Shelf Images CRUD API
- [ ] Basic frontend: image upload, list, delete
- [ ] Supabase Storage entegrasyonu

### Phase 2: Dataset Yonetimi
- [ ] OD Datasets CRUD API
- [ ] Dataset-Image iliskisi
- [ ] Dataset detay sayfasi
- [ ] Image selection ve ekleme

### Phase 3: Class Yonetimi
- [ ] Classes CRUD API
- [ ] Rename, merge, delete islemleri
- [ ] Class management UI
- [ ] Audit logging

### Phase 4: Annotation Altyapisi
- [ ] Annotations CRUD API
- [ ] Basic in-app annotation viewer (read-only)
- [ ] Annotation statistics

### Phase 5: Label Studio Entegrasyonu
- [ ] Label Studio deployment (Docker)
- [ ] Sync service (dataset -> LS project)
- [ ] Import service (LS -> database)
- [ ] ML Backend (SAM3 pre-annotation)

### Phase 6: AI Auto-Annotation
- [ ] Runpod AI Annotation Worker
- [ ] SAM3 integration
- [ ] YOLO-World integration
- [ ] Grounding DINO integration
- [ ] Progress tracking UI

### Phase 7: Import/Export
- [ ] COCO import
- [ ] YOLO import
- [ ] YOLO export
- [ ] COCO export
- [ ] Roboflow import

### Phase 8: Training Pipeline
- [ ] Training job API
- [ ] Runpod Training Worker
- [ ] Progress monitoring
- [ ] Model artifact storage
- [ ] Metrics display

### Phase 9: Polish & Optimization
- [ ] Bulk operations optimization
- [ ] Caching layer
- [ ] Error handling
- [ ] Documentation

---

## 10. Notlar

### Onemli Kararlar

1. **Label Studio vs Custom UI**: Baslangicta Label Studio kullan, ileride ihtiyaca gore custom annotation UI ekle.

2. **Class Yonetimi Merkezi**: Tum class'lar tek tabloda, dataset'ler arasi paylasimli.

3. **Annotation Storage**: Annotation'lar database'de, Label Studio sadece editing UI.

4. **AI Models**: SAM3 primary, YOLO-World/GroundingDINO secondary options.

### Performans Notlari

- Bulk import: Batch processing, progress tracking
- AI annotation: Runpod serverless, parallel processing
- Export: Async job, notification when ready

### Guvenlik

- Supabase RLS policies
- API authentication
- Label Studio SSO integration
