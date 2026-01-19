-- Buybuddy AI Platform - Object Detection Module
-- Migration 030: Object Detection Tables
--
-- This migration creates all tables for the OD annotation platform:
-- 1. od_images - Shelf/reyon images for annotation
-- 2. od_classes - Detection class definitions
-- 3. od_datasets - Detection datasets
-- 4. od_dataset_images - Dataset-image relationships
-- 5. od_annotations - Bounding box annotations
-- 6. od_dataset_versions - Dataset snapshots for training
-- 7. od_training_runs - Detection model training
-- 8. od_trained_models - Trained detection models
-- 9. od_class_changes - Audit log for class changes

-- ============================================
-- OD_IMAGES TABLE
-- Shelf/reyon images for object detection
-- ============================================
CREATE TABLE IF NOT EXISTS od_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- File info
    filename TEXT NOT NULL,
    original_filename TEXT,
    image_url TEXT NOT NULL,
    thumbnail_url TEXT,

    -- Dimensions
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size_bytes BIGINT,
    mime_type TEXT DEFAULT 'image/jpeg',

    -- Source tracking
    source TEXT NOT NULL DEFAULT 'upload' CHECK (source IN ('upload', 'buybuddy_sync', 'import')),
    buybuddy_image_id TEXT,  -- For synced images from BuyBuddy API
    buybuddy_evaluation_id TEXT,

    -- Organization
    folder TEXT,
    tags TEXT[],

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'annotating', 'completed', 'skipped')),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_images_status ON od_images(status);
CREATE INDEX IF NOT EXISTS idx_od_images_source ON od_images(source);
CREATE INDEX IF NOT EXISTS idx_od_images_folder ON od_images(folder);
CREATE INDEX IF NOT EXISTS idx_od_images_created ON od_images(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_od_images_buybuddy_id ON od_images(buybuddy_image_id) WHERE buybuddy_image_id IS NOT NULL;

COMMENT ON TABLE od_images IS 'Shelf/reyon images for object detection annotation';
COMMENT ON COLUMN od_images.buybuddy_image_id IS 'Original image ID from BuyBuddy API for synced images';

-- ============================================
-- OD_CLASSES TABLE
-- Detection class definitions (global)
-- ============================================
CREATE TABLE IF NOT EXISTS od_classes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Class info
    name TEXT NOT NULL UNIQUE,  -- Internal name (e.g., "shelf", "bay", "product")
    display_name TEXT,          -- User-friendly name
    description TEXT,

    -- Visual
    color TEXT NOT NULL DEFAULT '#3B82F6',  -- Hex color for annotation display

    -- Organization
    category TEXT,              -- e.g., "structure", "product", "label"
    parent_class_id UUID REFERENCES od_classes(id) ON DELETE SET NULL,  -- For hierarchical classes

    -- Aliases for import compatibility
    aliases TEXT[],

    -- Stats (denormalized for performance)
    annotation_count INTEGER DEFAULT 0,

    -- Flags
    is_active BOOLEAN DEFAULT true,
    is_system BOOLEAN DEFAULT false,  -- System classes cannot be deleted

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_classes_name ON od_classes(name);
CREATE INDEX IF NOT EXISTS idx_od_classes_category ON od_classes(category);
CREATE INDEX IF NOT EXISTS idx_od_classes_active ON od_classes(is_active) WHERE is_active = true;

COMMENT ON TABLE od_classes IS 'Detection class definitions for object detection';
COMMENT ON COLUMN od_classes.aliases IS 'Alternative names for COCO/YOLO import compatibility';

-- ============================================
-- OD_DATASETS TABLE
-- Detection datasets
-- ============================================
CREATE TABLE IF NOT EXISTS od_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Dataset info
    name TEXT NOT NULL,
    description TEXT,

    -- Annotation type
    annotation_type TEXT NOT NULL DEFAULT 'bbox' CHECK (annotation_type IN ('bbox', 'polygon', 'segmentation')),

    -- Stats (denormalized)
    image_count INTEGER DEFAULT 0,
    annotated_image_count INTEGER DEFAULT 0,
    annotation_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,

    -- Versioning
    version INTEGER DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_datasets_name ON od_datasets(name);
CREATE INDEX IF NOT EXISTS idx_od_datasets_created ON od_datasets(created_at DESC);

COMMENT ON TABLE od_datasets IS 'Detection datasets for organizing images and annotations';

-- ============================================
-- OD_DATASET_IMAGES TABLE
-- Many-to-many relationship between datasets and images
-- ============================================
CREATE TABLE IF NOT EXISTS od_dataset_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES od_images(id) ON DELETE CASCADE,

    -- Per-dataset image status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'annotating', 'completed', 'skipped')),

    -- Annotation count for this image in this dataset
    annotation_count INTEGER DEFAULT 0,

    -- Assignment
    assigned_to TEXT,  -- User email if assigned

    -- Split assignment (for training)
    split TEXT CHECK (split IN ('train', 'val', 'test')),

    -- Timestamps
    added_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    UNIQUE(dataset_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_od_dataset_images_dataset ON od_dataset_images(dataset_id);
CREATE INDEX IF NOT EXISTS idx_od_dataset_images_image ON od_dataset_images(image_id);
CREATE INDEX IF NOT EXISTS idx_od_dataset_images_status ON od_dataset_images(dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_od_dataset_images_split ON od_dataset_images(dataset_id, split);

COMMENT ON TABLE od_dataset_images IS 'Links images to datasets with per-dataset annotation status';

-- ============================================
-- OD_ANNOTATIONS TABLE
-- Bounding box annotations
-- ============================================
CREATE TABLE IF NOT EXISTS od_annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- References
    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES od_images(id) ON DELETE CASCADE,
    class_id UUID NOT NULL REFERENCES od_classes(id) ON DELETE RESTRICT,

    -- Bounding box (normalized 0-1 coordinates)
    bbox_x REAL NOT NULL CHECK (bbox_x >= 0 AND bbox_x <= 1),
    bbox_y REAL NOT NULL CHECK (bbox_y >= 0 AND bbox_y <= 1),
    bbox_width REAL NOT NULL CHECK (bbox_width > 0 AND bbox_width <= 1),
    bbox_height REAL NOT NULL CHECK (bbox_height > 0 AND bbox_height <= 1),

    -- Optional polygon (for polygon annotations)
    polygon JSONB,  -- Array of {x, y} points, normalized 0-1

    -- Optional segmentation mask
    mask_url TEXT,
    mask_rle TEXT,  -- Run-length encoded mask

    -- AI prediction metadata
    is_ai_generated BOOLEAN DEFAULT false,
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    ai_model TEXT,

    -- Review status
    is_reviewed BOOLEAN DEFAULT false,
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,

    -- Attributes (for additional annotation data)
    attributes JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT
);

CREATE INDEX IF NOT EXISTS idx_od_annotations_dataset ON od_annotations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_od_annotations_image ON od_annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_od_annotations_class ON od_annotations(class_id);
CREATE INDEX IF NOT EXISTS idx_od_annotations_dataset_image ON od_annotations(dataset_id, image_id);
CREATE INDEX IF NOT EXISTS idx_od_annotations_ai ON od_annotations(is_ai_generated) WHERE is_ai_generated = true;

COMMENT ON TABLE od_annotations IS 'Bounding box and polygon annotations for object detection';
COMMENT ON COLUMN od_annotations.bbox_x IS 'Top-left X coordinate, normalized 0-1';
COMMENT ON COLUMN od_annotations.bbox_y IS 'Top-left Y coordinate, normalized 0-1';
COMMENT ON COLUMN od_annotations.polygon IS 'Array of {x, y} points for polygon annotation, normalized 0-1';

-- ============================================
-- OD_DATASET_VERSIONS TABLE
-- Frozen snapshots for training
-- ============================================
CREATE TABLE IF NOT EXISTS od_dataset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE CASCADE,

    -- Version info
    version_number INTEGER NOT NULL,
    name TEXT,
    description TEXT,

    -- Stats at time of snapshot
    image_count INTEGER NOT NULL,
    annotation_count INTEGER NOT NULL,
    class_count INTEGER NOT NULL,

    -- Class mapping (frozen at version time)
    class_mapping JSONB NOT NULL,  -- {class_id: {name, color, index}}

    -- Split configuration
    split_config JSONB,  -- {train_ratio, val_ratio, test_ratio, seed}
    train_image_ids TEXT[],
    val_image_ids TEXT[],
    test_image_ids TEXT[],

    -- Export info
    export_format TEXT,
    export_url TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dataset_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_od_versions_dataset ON od_dataset_versions(dataset_id);

COMMENT ON TABLE od_dataset_versions IS 'Frozen dataset snapshots for reproducible training';
COMMENT ON COLUMN od_dataset_versions.class_mapping IS 'Frozen class definitions with index mapping for training';

-- ============================================
-- OD_TRAINING_RUNS TABLE
-- Detection model training runs
-- ============================================
CREATE TABLE IF NOT EXISTS od_training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Run info
    name TEXT NOT NULL,
    description TEXT,

    -- Data source
    dataset_id UUID NOT NULL REFERENCES od_datasets(id) ON DELETE SET NULL,
    dataset_version_id UUID REFERENCES od_dataset_versions(id) ON DELETE SET NULL,

    -- Model configuration
    model_type TEXT NOT NULL CHECK (model_type IN ('rf-detr', 'rt-detr', 'yolo-nas')),
    model_size TEXT DEFAULT 'medium' CHECK (model_size IN ('small', 'medium', 'large')),

    -- Training configuration
    config JSONB NOT NULL DEFAULT '{}',

    -- Progress
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'running', 'completed', 'failed', 'cancelled'
    )),
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,

    -- Best metrics
    best_map REAL,
    best_map_50 REAL,
    best_map_75 REAL,
    best_epoch INTEGER,

    -- Metrics history
    metrics_history JSONB,  -- Array of {epoch, train_loss, val_loss, map, map_50, ...}

    -- RunPod integration
    runpod_job_id TEXT,
    runpod_endpoint_id TEXT,

    -- Error handling
    error_message TEXT,
    error_traceback TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_training_status ON od_training_runs(status);
CREATE INDEX IF NOT EXISTS idx_od_training_dataset ON od_training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_od_training_created ON od_training_runs(created_at DESC);

COMMENT ON TABLE od_training_runs IS 'Object detection model training runs';

-- ============================================
-- OD_TRAINED_MODELS TABLE
-- Trained detection models
-- ============================================
CREATE TABLE IF NOT EXISTS od_trained_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Source
    training_run_id UUID REFERENCES od_training_runs(id) ON DELETE SET NULL,

    -- Model info
    name TEXT NOT NULL,
    description TEXT,
    model_type TEXT NOT NULL,
    model_size TEXT,

    -- Checkpoints
    checkpoint_url TEXT NOT NULL,
    onnx_url TEXT,
    tensorrt_url TEXT,

    -- Metrics
    map REAL,
    map_50 REAL,
    map_75 REAL,

    -- Class info
    class_count INTEGER NOT NULL,
    class_mapping JSONB NOT NULL,  -- {index: {id, name, color}}

    -- Flags
    is_active BOOLEAN DEFAULT false,  -- Active for auto-annotation
    is_default BOOLEAN DEFAULT false,

    -- File info
    file_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_models_active ON od_trained_models(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_od_models_type ON od_trained_models(model_type);

COMMENT ON TABLE od_trained_models IS 'Trained object detection models';

-- ============================================
-- OD_CLASS_CHANGES TABLE
-- Audit log for class changes
-- ============================================
CREATE TABLE IF NOT EXISTS od_class_changes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Change info
    change_type TEXT NOT NULL CHECK (change_type IN ('create', 'rename', 'merge', 'delete')),

    -- Class references
    class_id UUID,  -- May be null for deleted classes
    old_name TEXT,
    new_name TEXT,

    -- For merge operations
    source_class_ids UUID[],
    target_class_id UUID,

    -- Affected annotations
    affected_annotation_count INTEGER,

    -- User info
    changed_by TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_od_class_changes_class ON od_class_changes(class_id);
CREATE INDEX IF NOT EXISTS idx_od_class_changes_created ON od_class_changes(created_at DESC);

COMMENT ON TABLE od_class_changes IS 'Audit log for class changes (rename, merge, delete)';

-- ============================================
-- TRIGGERS - Update timestamps
-- ============================================
DROP TRIGGER IF EXISTS od_images_updated_at ON od_images;
CREATE TRIGGER od_images_updated_at
    BEFORE UPDATE ON od_images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS od_classes_updated_at ON od_classes;
CREATE TRIGGER od_classes_updated_at
    BEFORE UPDATE ON od_classes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS od_datasets_updated_at ON od_datasets;
CREATE TRIGGER od_datasets_updated_at
    BEFORE UPDATE ON od_datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS od_annotations_updated_at ON od_annotations;
CREATE TRIGGER od_annotations_updated_at
    BEFORE UPDATE ON od_annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS od_training_runs_updated_at ON od_training_runs;
CREATE TRIGGER od_training_runs_updated_at
    BEFORE UPDATE ON od_training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS od_trained_models_updated_at ON od_trained_models;
CREATE TRIGGER od_trained_models_updated_at
    BEFORE UPDATE ON od_trained_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE od_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_classes ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_dataset_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_dataset_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_trained_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE od_class_changes ENABLE ROW LEVEL SECURITY;

-- Allow all for authenticated (internal tool)
CREATE POLICY "Allow all for authenticated" ON od_images FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_classes FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_datasets FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_dataset_images FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_annotations FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_dataset_versions FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_training_runs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_trained_models FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON od_class_changes FOR ALL USING (true);

-- ============================================
-- SEED DEFAULT CLASSES
-- Common detection classes
-- ============================================
INSERT INTO od_classes (name, display_name, color, category, is_system) VALUES
    ('shelf', 'Shelf', '#3B82F6', 'structure', true),
    ('bay', 'Bay', '#10B981', 'structure', true),
    ('slot', 'Slot', '#F59E0B', 'structure', true),
    ('product', 'Product', '#EF4444', 'product', true),
    ('price_tag', 'Price Tag', '#8B5CF6', 'label', true),
    ('promotion', 'Promotion', '#EC4899', 'label', true),
    ('empty_slot', 'Empty Slot', '#6B7280', 'structure', true)
ON CONFLICT (name) DO NOTHING;

-- ============================================
-- STORAGE BUCKETS (run manually in Supabase Dashboard)
-- ============================================
-- Create bucket: "od-images" (public)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('od-images', 'od-images', true);

-- Create bucket: "od-models" (private)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('od-models', 'od-models', false);

-- Create bucket: "od-exports" (private)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('od-exports', 'od-exports', false);
