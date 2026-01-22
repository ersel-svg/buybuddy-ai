# Classification System Implementation Plan

> **Created:** 2026-01-21
> **Status:** Approved
> **Author:** Claude (AI Assistant)

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Database Schema](#database-schema)
4. [API Endpoints](#api-endpoints)
5. [Frontend Pages](#frontend-pages)
6. [Labeling System](#labeling-system)
7. [Import from Existing Sources](#import-from-existing-sources)
8. [Training Worker](#training-worker)
9. [Reusable Components](#reusable-components)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

### Goals
- Add a SOTA-level image classification system to BuyBuddy AI platform
- Mirror the Object Detection (OD) system's UX patterns
- Enable importing images from existing sources (Products, Cutouts, OD Images)
- Provide both single-image labeling and bulk labeling capabilities
- Support single-label and multi-label classification tasks

### Key Features
- **Image Management:** Upload, URL import, labeled dataset import (Roboflow-style)
- **Import Sources:** Products, Cutouts, OD Images with auto-labeling
- **Labeling:** Single-image labeling page + bulk edit in grid view
- **Training:** 6-step wizard with SOTA models and augmentation
- **Models:** Confusion matrix, per-class metrics, ONNX export

---

## System Architecture

### Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           IMAGE SOURCES                 │
                    └─────────────────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
   ┌─────────┐   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Upload  │   │  URL    │    │Products │    │ Cutouts │    │OD Images│
   │ (drag&  │   │ Import  │    │ Import  │    │ Import  │    │ Import  │
   │  drop)  │   │         │    │ (w/label)│   │(w/label)│    │         │
   └────┬────┘   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
        │              │               │               │              │
        └──────────────┴───────────────┼───────────────┴──────────────┘
                                       ▼
                          ┌────────────────────────┐
                          │     cls_images         │
                          │  (Classification       │
                          │   Image Library)       │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │    cls_datasets        │
                          │  + cls_dataset_images  │
                          └───────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │   Labeling   │  │  Bulk Edit   │  │  AI Auto-    │
           │   Page (1x1) │  │  (Grid)      │  │  Label       │
           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    ▼
                          ┌────────────────────────┐
                          │     cls_labels         │
                          │  (Image-Class mapping) │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  cls_dataset_versions  │
                          │  (Training Snapshot)   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  cls_training_runs     │
                          │  (RunPod Worker)       │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  cls_trained_models    │
                          │  (Checkpoint + Metrics)│
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │     Predictions        │
                          │  (Inference API)       │
                          └────────────────────────┘
```

### Sidebar Navigation (Updated)

```
┌─────────────────────┐
│  🏠 Dashboard       │
├─────────────────────┤
│  MATCHING           │
│  ├─ 📹 Videos       │
│  ├─ 📦 Products     │
│  ├─ 🖼️ Cutouts      │
│  ├─ 🧬 Embeddings   │
│  └─ 🔗 Matching     │
├─────────────────────┤
│  EMBEDDING TRAINING │
│  ├─ 📁 Datasets     │
│  ├─ 🔺 Triplets     │
│  ├─ 🎯 Training     │
│  └─ ✨ Augmentation │
├─────────────────────┤
│  OBJECT DETECTION   │
│  ├─ 🖼️ Images       │
│  ├─ 📁 Datasets     │
│  ├─ ✏️ Annotate     │
│  └─ 🎯 Training     │
├─────────────────────┤
│  CLASSIFICATION ⭐   │  ← NEW
│  ├─ 🖼️ Images       │
│  ├─ 🏷️ Classes      │
│  ├─ 📁 Datasets     │
│  ├─ ✏️ Labeling     │
│  ├─ 🎯 Training     │
│  └─ 🤖 Models       │
├─────────────────────┤
│  OPERATIONAL        │
│  └─ 📋 Scan Requests│
├─────────────────────┤
│  ⚙️ Settings        │
└─────────────────────┘
```

---

## Database Schema

### Migration File: `040_classification.sql`

```sql
-- ============================================
-- CLASSIFICATION IMAGES (Same as OD)
-- ============================================
CREATE TABLE cls_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users,

    -- File info
    filename TEXT NOT NULL,              -- UUID.ext
    original_filename TEXT,
    image_url TEXT NOT NULL,             -- Supabase public URL
    storage_path TEXT,                   -- Bucket path

    -- Dimensions
    width INTEGER,
    height INTEGER,
    file_size_bytes BIGINT,

    -- Organization
    source TEXT DEFAULT 'upload' CHECK (source IN (
        'upload', 'url_import', 'products_import',
        'cutouts_import', 'od_import', 'dataset_import'
    )),
    folder TEXT,
    tags TEXT[] DEFAULT '{}',

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'labeled', 'review', 'completed', 'skipped'
    )),

    -- Duplicate detection (same as OD)
    file_hash TEXT,                      -- SHA256
    phash TEXT,                          -- Perceptual hash

    -- Source references (when imported)
    source_type TEXT,                    -- 'product', 'cutout', 'od_image'
    source_id UUID,                      -- Original ID

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cls_images_user ON cls_images(user_id);
CREATE INDEX idx_cls_images_status ON cls_images(status);
CREATE INDEX idx_cls_images_source ON cls_images(source);
CREATE INDEX idx_cls_images_folder ON cls_images(folder);
CREATE INDEX idx_cls_images_file_hash ON cls_images(file_hash);
CREATE INDEX idx_cls_images_phash ON cls_images(phash);

-- ============================================
-- CLASSIFICATION CLASSES
-- ============================================
CREATE TABLE cls_classes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users,

    name TEXT NOT NULL,
    display_name TEXT,
    description TEXT,
    color TEXT DEFAULT '#3B82F6',

    -- Hierarchy (optional)
    parent_class_id UUID REFERENCES cls_classes(id),

    -- Stats (denormalized)
    image_count INTEGER DEFAULT 0,

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(user_id, name)
);

-- ============================================
-- CLASSIFICATION DATASETS
-- ============================================
CREATE TABLE cls_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users,

    name TEXT NOT NULL,
    description TEXT,

    -- Task type
    task_type TEXT DEFAULT 'single_label' CHECK (task_type IN (
        'single_label', 'multi_label'
    )),

    -- Stats (denormalized)
    image_count INTEGER DEFAULT 0,
    labeled_image_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,

    -- Split ratios
    split_ratios JSONB DEFAULT '{"train": 0.8, "val": 0.1, "test": 0.1}',

    -- Preprocessing config
    preprocessing JSONB DEFAULT '{"image_size": 224, "normalize": true}',

    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- DATASET-IMAGES (Many-to-Many, same as OD)
-- ============================================
CREATE TABLE cls_dataset_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES cls_datasets(id) ON DELETE CASCADE,
    image_id UUID REFERENCES cls_images(id) ON DELETE CASCADE,

    -- Status per dataset
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'labeled', 'review', 'completed', 'skipped'
    )),

    -- Split
    split TEXT CHECK (split IN ('train', 'val', 'test', 'unassigned')),

    added_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dataset_id, image_id)
);

CREATE INDEX idx_cls_dataset_images_dataset ON cls_dataset_images(dataset_id);
CREATE INDEX idx_cls_dataset_images_image ON cls_dataset_images(image_id);
CREATE INDEX idx_cls_dataset_images_status ON cls_dataset_images(dataset_id, status);
CREATE INDEX idx_cls_dataset_images_split ON cls_dataset_images(dataset_id, split);

-- ============================================
-- CLASSIFICATIONS (Labels - instead of annotations)
-- ============================================
CREATE TABLE cls_labels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES cls_datasets(id) ON DELETE CASCADE,
    image_id UUID REFERENCES cls_images(id) ON DELETE CASCADE,
    class_id UUID REFERENCES cls_classes(id) ON DELETE CASCADE,

    -- For multi-label: multiple rows per image
    -- For single-label: UNIQUE(dataset_id, image_id)

    -- Confidence (for AI-generated)
    confidence REAL,

    -- AI vs manual
    is_ai_generated BOOLEAN DEFAULT false,
    ai_model TEXT,

    -- Review status
    is_reviewed BOOLEAN DEFAULT false,
    reviewed_by UUID REFERENCES auth.users,
    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cls_labels_dataset ON cls_labels(dataset_id);
CREATE INDEX idx_cls_labels_image ON cls_labels(dataset_id, image_id);
CREATE INDEX idx_cls_labels_class ON cls_labels(class_id);

-- ============================================
-- DATASET VERSIONS (Snapshots for training)
-- ============================================
CREATE TABLE cls_dataset_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES cls_datasets(id) ON DELETE CASCADE,

    version_number INTEGER NOT NULL,

    -- Snapshot stats
    image_count INTEGER NOT NULL,
    labeled_image_count INTEGER NOT NULL,
    class_count INTEGER NOT NULL,

    -- Class mapping (for training)
    class_mapping JSONB NOT NULL,        -- {class_id: index}
    class_names JSONB NOT NULL,          -- ["class1", "class2", ...]

    -- Split counts
    split_counts JSONB NOT NULL,         -- {train: N, val: N, test: N}

    -- Image IDs per split
    train_image_ids UUID[] NOT NULL,
    val_image_ids UUID[] NOT NULL,
    test_image_ids UUID[] NOT NULL,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dataset_id, version_number)
);

-- ============================================
-- TRAINING RUNS
-- ============================================
CREATE TABLE cls_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users,

    name TEXT NOT NULL,
    description TEXT,

    -- Dataset
    dataset_id UUID REFERENCES cls_datasets(id),
    dataset_version_id UUID REFERENCES cls_dataset_versions(id),

    -- Model config
    model_type TEXT NOT NULL CHECK (model_type IN (
        'vit', 'convnext', 'efficientnet', 'swin', 'dinov2', 'clip'
    )),
    model_size TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (task_type IN ('single_label', 'multi_label')),
    num_classes INTEGER NOT NULL,

    -- Full config
    config JSONB NOT NULL,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'queued', 'training',
        'completed', 'failed', 'cancelled'
    )),

    -- Progress
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,

    -- Best metrics
    best_accuracy REAL,
    best_f1 REAL,
    best_top5_accuracy REAL,

    -- History
    metrics_history JSONB DEFAULT '[]',

    -- RunPod
    runpod_job_id TEXT,
    error_message TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- TRAINED MODELS
-- ============================================
CREATE TABLE cls_trained_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users,
    training_run_id UUID REFERENCES cls_training_runs(id),

    name TEXT NOT NULL,
    description TEXT,

    -- Model info
    model_type TEXT NOT NULL,
    model_size TEXT NOT NULL,
    task_type TEXT NOT NULL,

    -- Checkpoints
    checkpoint_url TEXT,
    onnx_url TEXT,
    torchscript_url TEXT,

    -- Class info
    num_classes INTEGER NOT NULL,
    class_names JSONB NOT NULL,
    class_mapping JSONB NOT NULL,

    -- Metrics
    accuracy REAL,
    f1_score REAL,
    top5_accuracy REAL,
    precision_macro REAL,
    recall_macro REAL,

    -- Detailed metrics
    confusion_matrix JSONB,
    per_class_metrics JSONB,

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trigger: Single default per model_type
CREATE OR REPLACE FUNCTION ensure_single_default_cls_model()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_default = true THEN
        UPDATE cls_trained_models
        SET is_default = false
        WHERE model_type = NEW.model_type
          AND user_id = NEW.user_id
          AND id != NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_single_default_cls_model
BEFORE INSERT OR UPDATE ON cls_trained_models
FOR EACH ROW EXECUTE FUNCTION ensure_single_default_cls_model();

-- ============================================
-- RPC FUNCTIONS
-- ============================================

-- Filter options for images
CREATE OR REPLACE FUNCTION get_cls_image_filter_options(p_user_id UUID)
RETURNS JSON AS $$
BEGIN
    RETURN json_build_object(
        'statuses', (SELECT COALESCE(json_agg(DISTINCT status), '[]') FROM cls_images WHERE user_id = p_user_id),
        'sources', (SELECT COALESCE(json_agg(DISTINCT source), '[]') FROM cls_images WHERE user_id = p_user_id),
        'folders', (SELECT COALESCE(json_agg(DISTINCT folder), '[]') FROM cls_images WHERE user_id = p_user_id AND folder IS NOT NULL),
        'total_count', (SELECT COUNT(*) FROM cls_images WHERE user_id = p_user_id)
    );
END;
$$ LANGUAGE plpgsql;

-- Update dataset stats
CREATE OR REPLACE FUNCTION update_cls_dataset_stats(p_dataset_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE cls_datasets SET
        image_count = (SELECT COUNT(*) FROM cls_dataset_images WHERE dataset_id = p_dataset_id),
        labeled_image_count = (SELECT COUNT(DISTINCT image_id) FROM cls_labels WHERE dataset_id = p_dataset_id),
        class_count = (SELECT COUNT(DISTINCT class_id) FROM cls_labels WHERE dataset_id = p_dataset_id),
        updated_at = NOW()
    WHERE id = p_dataset_id;
END;
$$ LANGUAGE plpgsql;
```

---

## API Endpoints

### Complete API Structure

```
/api/v1/classification/
│
├── /health                          GET     Health check
├── /stats                           GET     Dashboard stats
│
├── /images                          # ═══ SAME AS OD/IMAGES ═══
│   ├── GET     /                    List images (filters, pagination)
│   ├── POST    /                    Upload single image
│   ├── POST    /bulk                Bulk upload
│   ├── GET     /{id}                Get image details
│   ├── PATCH   /{id}                Update image (folder, tags, status)
│   ├── DELETE  /{id}                Delete image
│   │
│   ├── /import
│   │   ├── POST /url                Import from URLs
│   │   ├── POST /preview            Preview dataset import (ZIP)
│   │   ├── POST /labeled            Import labeled dataset (folder structure)
│   │   ├── POST /products           Import from Products  ← NEW
│   │   ├── POST /cutouts            Import from Cutouts   ← NEW
│   │   └── POST /od-images          Import from OD Images ← NEW
│   │
│   ├── /bulk
│   │   ├── POST /tags               Add/remove/replace tags
│   │   ├── POST /move               Move to folder
│   │   ├── POST /add-to-dataset     Add to dataset
│   │   ├── POST /delete             Bulk delete by IDs
│   │   └── POST /delete-by-filters  Delete by filters
│   │
│   ├── /duplicates
│   │   ├── POST /check              Check single image
│   │   └── GET  /                   List duplicate groups
│   │
│   └── /filters
│       └── GET /options             Get filter options
│
├── /classes
│   ├── GET     /                    List classes
│   ├── POST    /                    Create class
│   ├── GET     /{id}                Get class
│   ├── PATCH   /{id}                Update class
│   ├── DELETE  /{id}                Delete class
│   ├── POST    /bulk                Bulk create classes
│   ├── POST    /merge               Merge classes
│   └── GET     /hierarchy           Get class tree
│
├── /datasets
│   ├── GET     /                    List datasets
│   ├── POST    /                    Create dataset
│   ├── GET     /{id}                Get dataset with stats
│   ├── PATCH   /{id}                Update dataset
│   ├── DELETE  /{id}                Delete dataset
│   │
│   ├── /{id}/images
│   │   ├── GET     /                List images in dataset
│   │   ├── POST    /add             Add images to dataset
│   │   ├── POST    /remove          Remove images from dataset
│   │   └── POST    /add-by-filters  Add by filters
│   │
│   ├── /{id}/labels                 # ═══ LABELING ═══
│   │   ├── GET     /                List all labels in dataset
│   │   ├── GET     /{image_id}      Get labels for image
│   │   ├── POST    /{image_id}      Set label(s) for image
│   │   ├── DELETE  /{image_id}      Clear labels for image
│   │   ├── POST    /bulk            Bulk set labels
│   │   └── POST    /bulk-clear      Bulk clear labels
│   │
│   ├── /{id}/split
│   │   ├── POST    /auto            Auto-split (stratified)
│   │   ├── POST    /manual          Manual split assignment
│   │   └── GET     /stats           Get split statistics
│   │
│   ├── /{id}/versions
│   │   ├── GET     /                List versions
│   │   ├── POST    /                Create new version (snapshot)
│   │   └── GET     /{version_id}    Get version details
│   │
│   ├── /{id}/health                 GET     Dataset health check
│   └── /{id}/export                 GET     Export dataset
│
├── /labeling                        # ═══ ANNOTATION PAGE API ═══
│   ├── GET     /queue/{dataset_id}              Get labeling queue
│   ├── GET     /image/{dataset_id}/{image_id}   Get image for labeling
│   ├── POST    /image/{dataset_id}/{image_id}   Save label
│   ├── POST    /skip/{dataset_id}/{image_id}    Skip image
│   └── GET     /progress/{dataset_id}           Get labeling progress
│
├── /training
│   ├── GET     /                    List training runs
│   ├── POST    /                    Start training
│   ├── GET     /{id}                Get training details
│   ├── POST    /{id}/cancel         Cancel training
│   ├── DELETE  /{id}                Delete training run
│   ├── GET     /{id}/metrics        Get metrics history
│   ├── GET     /{id}/checkpoints    List checkpoints
│   ├── GET     /presets             Get augmentation presets
│   ├── GET     /model-configs       Get supported models
│   └── POST    /webhook             RunPod webhook
│
├── /models
│   ├── GET     /                    List trained models
│   ├── GET     /{id}                Get model details
│   ├── PATCH   /{id}                Update model
│   ├── DELETE  /{id}                Delete model
│   ├── POST    /{id}/activate       Activate model
│   ├── POST    /{id}/deactivate     Deactivate model
│   ├── POST    /{id}/set-default    Set as default
│   ├── GET     /{id}/download       Download checkpoint
│   ├── POST    /{id}/export-onnx    Export to ONNX
│   └── GET     /default/{type}      Get default model
│
└── /predict
    ├── POST    /                    Single image prediction
    ├── POST    /batch               Batch prediction
    ├── POST    /url                 Predict from URL
    └── POST    /explain             Grad-CAM visualization
```

### Import Request Schemas

```python
class ImportFromProductsRequest(BaseModel):
    """Import images from Products."""
    product_ids: Optional[list[str]] = None      # Specific products
    filters: Optional[ProductFilters] = None     # Or filter-based

    # Label strategy
    label_source: Literal[
        "category",      # Use product category
        "brand",         # Use brand name
        "product_name",  # Use product name (fine-grained)
        "manual"         # Import unlabeled
    ] = "category"

    # Image types to import
    image_types: list[Literal["synthetic", "real", "augmented"]] = ["synthetic", "real"]
    max_frames_per_product: int = 5

    # Options
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None  # Auto-add to dataset


class ImportFromCutoutsRequest(BaseModel):
    """Import images from Cutouts."""
    cutout_ids: Optional[list[str]] = None
    filters: Optional[CutoutFilters] = None

    # Label strategy
    label_source: Literal[
        "matched_product_category",  # Use matched product's category
        "matched_product_brand",     # Use matched product's brand
        "manual"                     # Import unlabeled
    ] = "matched_product_category"

    # Options
    only_matched: bool = True  # Only import matched cutouts
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None


class ImportFromODRequest(BaseModel):
    """Import images from Object Detection."""
    od_image_ids: Optional[list[str]] = None
    filters: Optional[ODImageFilters] = None

    # Options
    skip_duplicates: bool = True
    dataset_id: Optional[str] = None
    # Labels will be manual (OD has different annotation type)


class ImportLabeledDatasetRequest(BaseModel):
    """Import labeled dataset from ZIP (folder structure)."""
    # ZIP structure:
    # dataset.zip/
    #   ├── class1/
    #   │   ├── img1.jpg
    #   │   └── img2.jpg
    #   ├── class2/
    #   │   └── img3.jpg

    dataset_id: str
    class_mapping: list[ClassMapping]
    skip_duplicates: bool = True


class ClassMapping(BaseModel):
    source_name: str          # Folder name in ZIP
    target_class_id: Optional[str] = None
    create_new: bool = False
    color: Optional[str] = None
    skip: bool = False
```

---

## Frontend Pages

### Page Structure

```
/src/app/classification/
│
├── page.tsx                              # Dashboard
│
├── images/
│   └── page.tsx                          # Image Library (same as OD)
│
├── classes/
│   └── page.tsx                          # Class Management
│
├── datasets/
│   ├── page.tsx                          # Dataset List
│   ├── new/
│   │   └── page.tsx                      # Create Dataset
│   └── [id]/
│       ├── page.tsx                      # Dataset Detail (image grid + labels)
│       ├── upload/
│       │   └── page.tsx                  # Upload to Dataset
│       └── import/
│           └── page.tsx                  # Import Wizard
│
├── labeling/                             # ═══ ANNOTATION EQUIVALENT ═══
│   └── [datasetId]/
│       ├── page.tsx                      # Labeling Queue Entry
│       └── [imageId]/
│           └── page.tsx                  # Single Image Labeling Page
│
├── training/
│   ├── page.tsx                          # Training Runs List
│   ├── new/
│   │   └── page.tsx                      # Training Wizard (6-step)
│   └── [id]/
│       └── page.tsx                      # Training Detail (metrics)
│
└── models/
    ├── page.tsx                          # Trained Models List
    └── [id]/
        └── page.tsx                      # Model Detail (confusion matrix)
```

### Images Page (Same as OD)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION > IMAGES                                    [+ Upload]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ [🔍 Search...]  [Status ▼] [Source ▼] [Folder ▼] [Clear All]    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  View: [▦ Grid ✓] [☰ List]     Showing 2,450 images     [↻ Refresh]   │
│                                                                         │
│  ☐ Select All (48)                     Sort: [Date Added ▼]            │
│                                                                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │┌────────┐│┌────────┐│┌────────┐│┌────────┐│┌────────┐│┌────────┐│  │
│  ││  📷    │││  📷    │││  📷    │││  📷    │││  📷    │││  📷    ││  │
│  ││        │││        │││        │││        │││        │││        ││  │
│  │└────────┘│└────────┘│└────────┘│└────────┘│└────────┘│└────────┘│  │
│  │☐ pending │☐ pending │☐ labeled │☐ labeled │☐ pending │☐ complete│  │
│  │📁 upload │📁 products│📁 cutouts│📁 upload │📁 url    │📁 upload │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘  │
│                                                                         │
│  ◀ Prev  [1] [2] [3] ... [51]  Next ▶        48 per page               │
│                                                                         │
│  BULK ACTIONS (12 selected):                                            │
│  [📁 Add to Dataset] [🏷️ Set Folder] [🏷️ Add Tags] [🗑️ Delete]       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Import Modal (OD + New Tabs)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  IMPORT IMAGES                                                     [X]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┐              │
│  │Upload  │  URL   │Labeled │Products│Cutouts │OD Images│              │
│  │   ✓    │        │Dataset │        │        │         │              │
│  └────────┴────────┴────────┴────────┴────────┴────────┘              │
│                                                                         │
│  [Tab content based on selection]                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Import from Products Tab

```
┌─────────────────────────────────────────────────────────────────────────┐
│  IMPORT FROM PRODUCTS                                                   │
│                                                                         │
│  Label Strategy:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ● Use Category as Label   (e.g., "Beverages", "Snacks")         │   │
│  │ ○ Use Brand as Label      (e.g., "Coca-Cola", "Pepsi")          │   │
│  │ ○ Use Product Name        (fine-grained, many classes)          │   │
│  │ ○ Import Unlabeled        (label manually later)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Image Types:                                                           │
│  ☑️ Synthetic frames (360° rotating)                                   │
│  ☑️ Real matched images                                                │
│  ☐ Augmented images                                                    │
│                                                                         │
│  Max frames per product: [5━━━●━━━━━━━━━━━━] 5                        │
│                                                                         │
│  Filter Products:                                                       │
│  [Status: matched ▼] [Category: All ▼] [Brand: ... ▼]                 │
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗ │
│  ║  PREVIEW                                                          ║ │
│  ║  📦 1,234 products matching filters                               ║ │
│  ║  📸 ~6,170 images (5 frames × 1,234 products)                     ║ │
│  ║  🏷️ 15 unique categories will become classes                      ║ │
│  ╚═══════════════════════════════════════════════════════════════════╝ │
│                                                                         │
│  [Cancel]                                         [Import 6,170 Images] │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Labeling System

### Labeling Page (Annotation Equivalent)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ← Back to Dataset                          Dataset: Product Categories v2      │
│                                             Progress: 1,847 / 2,450 (75.4%)     │
├──────────────────────────────────┬──────────────────────────────────────────────┤
│                                  │                                              │
│                                  │  CLASSES                        [+ New]     │
│                                  │  ────────────────────────────────────────    │
│                                  │  🔍 Search classes...                        │
│                                  │                                              │
│     ┌───────────────────────┐    │  ┌────────────────────────────────────────┐ │
│     │                       │    │  │ ● Beverages            (421)     [1]  │ │
│     │                       │    │  │ ○ Snacks               (356)     [2]  │ │
│     │                       │    │  │ ○ Dairy                (298)     [3]  │ │
│     │                       │    │  │ ○ Bakery               (245)     [4]  │ │
│     │      📷 IMAGE         │    │  │ ○ Frozen Foods         (189)     [5]  │ │
│     │                       │    │  │ ○ Canned Goods         (156)     [6]  │ │
│     │      (512 x 512)      │    │  │ ○ Condiments           (98)      [7]  │ │
│     │                       │    │  │ ○ Breakfast            (67)      [8]  │ │
│     │                       │    │  │ ○ Organic              (17)      [9]  │ │
│     │                       │    │  └────────────────────────────────────────┘ │
│     │                       │    │                                              │
│     └───────────────────────┘    │  CURRENT LABEL                              │
│                                  │  ┌────────────────────────────────────────┐ │
│  ┌──────────────────────────┐    │  │        ✓ Beverages                     │ │
│  │ img_1847.jpg             │    │  │  Confidence: ████████░░ 85% (AI)       │ │
│  │ 512 × 512 px             │    │  └────────────────────────────────────────┘ │
│  └──────────────────────────┘    │                                              │
│                                  │  [Clear Label]  [Skip]  [✓ Save & Next]     │
│  ┌──────────────────────────┐    │                                              │
│  │ ◀ Prev  [1847/2450]  ▶  │    │  KEYBOARD SHORTCUTS                [?]      │
│  │     ← →  keyboard nav    │    │  1-9: Select class | ←→: Navigate          │
│  └──────────────────────────┘    │  Enter: Save & Next | S: Skip | C: Clear   │
├──────────────────────────────────┴──────────────────────────────────────────────┤
│  [◀ Previous]                                                    [Next ▶]      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-9` | Select class 1-9 |
| `0` | Select class 10 |
| `←` / `A` | Previous image |
| `→` / `D` | Next image |
| `Enter` | Save & Next |
| `S` | Skip image |
| `C` | Clear label |
| `R` | Toggle review mode |
| `?` | Show shortcuts panel |
| `Esc` | Back to dataset |

### Queue Modes

```typescript
type QueueMode =
  | "all"            // All images in order
  | "unlabeled"      // Only unlabeled images
  | "review"         // Only AI-labeled for review
  | "random"         // Random unlabeled
  | "low_confidence" // AI labels with low confidence
```

### Bulk Labeling (Dataset Detail Page)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BULK ACTIONS (15 selected):                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Set Label: [Beverages        ▼]  [Apply to 15 images]          │   │
│  │                                                                  │   │
│  │ Set Split: [○ Train  ○ Val  ○ Test]  [Apply]                   │   │
│  │                                                                  │   │
│  │ [Clear Labels] [Remove from Dataset] [🗑️ Delete Images]        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Import from Existing Sources

### Source Options

| Source | Auto-Label Options | Notes |
|--------|-------------------|-------|
| **Products** | Category, Brand, Product Name, Manual | Includes synthetic/real/augmented frames |
| **Cutouts** | Matched Product Category, Matched Product Brand, Manual | Only matched cutouts by default |
| **OD Images** | Manual only | Different annotation type |

### Import Flow

1. **Select Source Tab**
2. **Configure Label Strategy** (if applicable)
3. **Apply Filters** (optional)
4. **Preview Results** (count, classes)
5. **Select Target Dataset** (optional)
6. **Import**

---

## Training Worker

### Project Structure

```
/workers/classification-training/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── handler.py              # RunPod handler
│   ├── config.py               # Model configs & presets
│   ├── trainer.py              # Main training loop
│   ├── dataset.py              # PyTorch dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py          # Model factory
│   │   ├── vit.py              # Vision Transformer
│   │   ├── convnext.py         # ConvNeXt v2
│   │   ├── efficientnet.py     # EfficientNet v2
│   │   ├── swin.py             # Swin Transformer v2
│   │   └── heads.py            # Classification heads
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── cross_entropy.py    # Label smoothing CE
│   │   ├── focal.py            # Focal loss
│   │   └── bce.py              # Multi-label BCE
│   ├── augmentations/
│   │   └── __init__.py         # → imports from /libs/augmentation
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── accuracy.py
│   │   ├── f1.py
│   │   └── confusion.py
│   └── utils/
│       ├── checkpoint.py
│       ├── ema.py
│       └── scheduler.py
└── tests/
```

### Supported Models

| Model | Sizes | Params | Use Case |
|-------|-------|--------|----------|
| **ViT** | tiny, small, base, large | 5M-300M | General purpose |
| **ConvNeXt v2** | atto, femto, pico, nano, tiny, base | 3M-89M | Production |
| **EfficientNet v2** | s, m, l | 21M-120M | Mobile/Edge |
| **Swin v2** | tiny, small, base | 28M-88M | Hierarchical |
| **DINOv2** | small, base, large | 22M-300M | Transfer learning |
| **CLIP** | ViT-B/16, ViT-L/14 | 86M-300M | Zero-shot capable |

### Training Config

```python
class ClassificationTrainingConfig(BaseModel):
    # Model
    model_type: Literal["vit", "convnext", "efficientnet", "swin", "dinov2"]
    model_size: str
    pretrained: bool = True
    freeze_backbone_epochs: int = 0

    # Task
    task_type: Literal["single_label", "multi_label"] = "single_label"
    num_classes: int

    # Training
    epochs: int = 100
    batch_size: int = 32

    # Optimizer
    optimizer: Literal["adamw", "sgd", "lamb"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05

    # SOTA Features
    use_ema: bool = True
    ema_decay: float = 0.9999
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # LR Schedule
    lr_scheduler: Literal["cosine", "step", "plateau", "one_cycle"] = "cosine"
    warmup_epochs: int = 5
    llrd_decay: float = 0.9  # Layer-wise LR Decay

    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    drop_path_rate: float = 0.1

    # Augmentation
    augmentation_preset: Literal["sota", "heavy", "medium", "light", "none"] = "sota"
    image_size: int = 224

    # Class Imbalance
    class_weights: Literal["balanced", "sqrt", "none"] = "balanced"
    focal_loss_gamma: float = 0.0

    # Early Stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_f1"
```

### Augmentation Presets

```python
CLASSIFICATION_PRESETS = {
    "sota": {
        "name": "SOTA (Recommended)",
        "description": "RandAugment + MixUp + CutMix + Label Smoothing",
        "training_time_factor": 1.3,
        "accuracy_boost": "+2-4%",
    },
    "heavy": {
        "name": "Heavy (Small Datasets)",
        "description": "TrivialAugmentWide + Strong regularization",
        "training_time_factor": 1.8,
        "accuracy_boost": "+4-6%",
    },
    "medium": {
        "name": "Medium (Balanced)",
        "training_time_factor": 1.2,
        "accuracy_boost": "+1-2%",
    },
    "light": {
        "name": "Light (Large Datasets)",
        "training_time_factor": 1.05,
        "accuracy_boost": "+0.5-1%",
    },
    "none": {
        "name": "None (Baseline)",
        "training_time_factor": 1.0,
        "accuracy_boost": "Baseline",
    }
}
```

---

## Reusable Components

### From OD System

| OD Component | Classification Equivalent | Reuse Level |
|--------------|---------------------------|-------------|
| `/od/images/page.tsx` | `/classification/images/page.tsx` | 95% |
| `import-modal.tsx` | Same + 3 new tabs | 80% |
| `/od/annotate/` | `/classification/labeling/` | 40% |
| `/od/datasets/` | `/classification/datasets/` | 90% |
| `/od/training/` | `/classification/training/` | 85% |
| `WizardStepper` | Same | 100% |
| `DatasetStatsCard` | Same | 100% |
| `StatusBadge` | Same | 100% |
| `MetricsChart` | Same + confusion matrix | 90% |

### Shared Libraries

```
/libs/augmentation/                    # NEW: Shared library
├── __init__.py
├── pipeline.py                        # From OD
├── presets.py                         # From OD
└── transforms/
    ├── geometric.py
    ├── color.py
    └── quality.py
```

---

## Implementation Roadmap

| Phase | Task | Files | Duration |
|-------|------|-------|----------|
| **1** | Database Migration | `040_classification.sql` | 1 day |
| **2** | Images API (copy from OD) | `api/v1/classification/images.py` | 1 day |
| **3** | Import Sources | `images.py` + import handlers | 2 days |
| **4** | Classes API | `api/v1/classification/classes.py` | 0.5 day |
| **5** | Datasets API | `api/v1/classification/datasets.py` | 1 day |
| **6** | Labels/Labeling API | `api/v1/classification/labeling.py` | 1 day |
| **7** | Frontend: Images Page | `/classification/images/page.tsx` | 1 day |
| **8** | Frontend: Import Modal | `import-modal.tsx` + new tabs | 2 days |
| **9** | Frontend: Classes Page | `/classification/classes/page.tsx` | 0.5 day |
| **10** | Frontend: Datasets Page | `/classification/datasets/` | 1 day |
| **11** | Frontend: Labeling Page | `/classification/labeling/` | 2 days |
| **12** | Training Worker | `/workers/classification-training/` | 3 days |
| **13** | Training API | `api/v1/classification/training.py` | 1 day |
| **14** | Frontend: Training Wizard | `/classification/training/new/` | 2 days |
| **15** | Frontend: Training Detail | `/classification/training/[id]/` | 1 day |
| **16** | Models API | `api/v1/classification/models.py` | 0.5 day |
| **17** | Frontend: Models Page | `/classification/models/` | 1 day |
| **18** | Predictions API | `api/v1/classification/predict.py` | 1 day |
| **19** | Testing & Polish | Tests, bug fixes | 2 days |

**Total: ~24 working days (~5 weeks)**

---

## Summary

### Key Features
- ✅ OD upload features (identical)
- ✅ Import from Products/Cutouts/OD with auto-labeling
- ✅ Single-image labeling page (keyboard shortcuts)
- ✅ Bulk labeling from grid
- ✅ SOTA training (augmentation reuse from OD)
- ✅ Confusion matrix + per-class metrics

### Tables
- `cls_images` - Image library
- `cls_classes` - Class definitions
- `cls_datasets` - Dataset metadata
- `cls_dataset_images` - Many-to-many
- `cls_labels` - Image-class assignments
- `cls_dataset_versions` - Training snapshots
- `cls_training_runs` - Training jobs
- `cls_trained_models` - Model registry

### API Modules
- `/classification/images` - Image management
- `/classification/classes` - Class management
- `/classification/datasets` - Dataset management
- `/classification/labeling` - Labeling workflow
- `/classification/training` - Training management
- `/classification/models` - Model management
- `/classification/predict` - Inference

### Frontend Pages
- `/classification/images` - Image library
- `/classification/classes` - Class management
- `/classification/datasets` - Dataset list & detail
- `/classification/labeling` - Labeling interface
- `/classification/training` - Training wizard & monitoring
- `/classification/models` - Model gallery
