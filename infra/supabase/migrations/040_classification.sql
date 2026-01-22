-- Buybuddy AI Platform - Classification Module
-- Migration 040: Image Classification Tables
--
-- This migration creates all tables for the Classification platform:
-- 1. cls_images - Images for classification
-- 2. cls_classes - Classification class definitions
-- 3. cls_datasets - Classification datasets
-- 4. cls_dataset_images - Dataset-image relationships
-- 5. cls_labels - Image-class assignments (labels)
-- 6. cls_dataset_versions - Dataset snapshots for training
-- 7. cls_training_runs - Classification model training
-- 8. cls_trained_models - Trained classification models

-- ============================================
-- CLS_IMAGES TABLE
-- Images for image classification
-- ============================================
CREATE TABLE IF NOT EXISTS cls_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- File info
    filename TEXT NOT NULL,
    original_filename TEXT,
    image_url TEXT NOT NULL,
    storage_path TEXT,
    thumbnail_url TEXT,

    -- Dimensions
    width INTEGER,
    height INTEGER,
    file_size_bytes BIGINT,
    mime_type TEXT DEFAULT 'image/jpeg',

    -- Source tracking
    source TEXT NOT NULL DEFAULT 'upload' CHECK (source IN (
        'upload', 'url_import', 'products_import',
        'cutouts_import', 'od_import', 'dataset_import'
    )),

    -- Source references (when imported from other modules)
    source_type TEXT,  -- 'product', 'cutout', 'od_image'
    source_id UUID,    -- Original ID from source

    -- Organization
    folder TEXT,
    tags TEXT[] DEFAULT '{}',

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'labeled', 'review', 'completed', 'skipped'
    )),

    -- Duplicate detection (same as OD)
    file_hash TEXT,   -- SHA256
    phash TEXT,       -- Perceptual hash

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_images_status ON cls_images(status);
CREATE INDEX IF NOT EXISTS idx_cls_images_source ON cls_images(source);
CREATE INDEX IF NOT EXISTS idx_cls_images_folder ON cls_images(folder);
CREATE INDEX IF NOT EXISTS idx_cls_images_created ON cls_images(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cls_images_file_hash ON cls_images(file_hash) WHERE file_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cls_images_phash ON cls_images(phash) WHERE phash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cls_images_source_ref ON cls_images(source_type, source_id) WHERE source_type IS NOT NULL;

COMMENT ON TABLE cls_images IS 'Images for image classification';
COMMENT ON COLUMN cls_images.source_type IS 'Type of source when imported (product, cutout, od_image)';
COMMENT ON COLUMN cls_images.source_id IS 'Original ID from source module';
COMMENT ON COLUMN cls_images.file_hash IS 'SHA256 hash for exact duplicate detection';
COMMENT ON COLUMN cls_images.phash IS 'Perceptual hash for similar image detection';

-- ============================================
-- CLS_CLASSES TABLE
-- Classification class definitions
-- ============================================
CREATE TABLE IF NOT EXISTS cls_classes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Class info
    name TEXT NOT NULL UNIQUE,  -- Internal name (e.g., "beverages", "snacks")
    display_name TEXT,          -- User-friendly name
    description TEXT,

    -- Visual
    color TEXT NOT NULL DEFAULT '#3B82F6',  -- Hex color for display

    -- Organization
    parent_class_id UUID REFERENCES cls_classes(id) ON DELETE SET NULL,  -- For hierarchical classes

    -- Stats (denormalized for performance)
    image_count INTEGER DEFAULT 0,

    -- Flags
    is_active BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_classes_name ON cls_classes(name);
CREATE INDEX IF NOT EXISTS idx_cls_classes_active ON cls_classes(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_cls_classes_parent ON cls_classes(parent_class_id) WHERE parent_class_id IS NOT NULL;

COMMENT ON TABLE cls_classes IS 'Classification class definitions';
COMMENT ON COLUMN cls_classes.parent_class_id IS 'Parent class for hierarchical classification';

-- ============================================
-- CLS_DATASETS TABLE
-- Classification datasets
-- ============================================
CREATE TABLE IF NOT EXISTS cls_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Dataset info
    name TEXT NOT NULL,
    description TEXT,

    -- Task type
    task_type TEXT NOT NULL DEFAULT 'single_label' CHECK (task_type IN (
        'single_label', 'multi_label'
    )),

    -- Stats (denormalized)
    image_count INTEGER DEFAULT 0,
    labeled_image_count INTEGER DEFAULT 0,
    class_count INTEGER DEFAULT 0,

    -- Split ratios (default 80/10/10)
    split_ratios JSONB DEFAULT '{"train": 0.8, "val": 0.1, "test": 0.1}',

    -- Preprocessing config
    preprocessing JSONB DEFAULT '{"image_size": 224, "normalize": true}',

    -- Versioning
    version INTEGER DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_datasets_name ON cls_datasets(name);
CREATE INDEX IF NOT EXISTS idx_cls_datasets_task_type ON cls_datasets(task_type);
CREATE INDEX IF NOT EXISTS idx_cls_datasets_created ON cls_datasets(created_at DESC);

COMMENT ON TABLE cls_datasets IS 'Classification datasets for organizing images and labels';
COMMENT ON COLUMN cls_datasets.task_type IS 'single_label: one class per image, multi_label: multiple classes per image';
COMMENT ON COLUMN cls_datasets.split_ratios IS 'Train/val/test split ratios, must sum to 1.0';

-- ============================================
-- CLS_DATASET_IMAGES TABLE
-- Many-to-many relationship between datasets and images
-- ============================================
CREATE TABLE IF NOT EXISTS cls_dataset_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES cls_datasets(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES cls_images(id) ON DELETE CASCADE,

    -- Per-dataset image status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'labeled', 'review', 'completed', 'skipped'
    )),

    -- Split assignment (for training)
    split TEXT CHECK (split IN ('train', 'val', 'test', 'unassigned')),

    -- Timestamps
    added_at TIMESTAMPTZ DEFAULT NOW(),
    labeled_at TIMESTAMPTZ,

    UNIQUE(dataset_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_cls_dataset_images_dataset ON cls_dataset_images(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cls_dataset_images_image ON cls_dataset_images(image_id);
CREATE INDEX IF NOT EXISTS idx_cls_dataset_images_status ON cls_dataset_images(dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_cls_dataset_images_split ON cls_dataset_images(dataset_id, split);

COMMENT ON TABLE cls_dataset_images IS 'Links images to datasets with per-dataset labeling status';

-- ============================================
-- CLS_LABELS TABLE
-- Image-class assignments (labels)
-- For single-label: one row per image
-- For multi-label: multiple rows per image
-- ============================================
CREATE TABLE IF NOT EXISTS cls_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- References
    dataset_id UUID NOT NULL REFERENCES cls_datasets(id) ON DELETE CASCADE,
    image_id UUID NOT NULL REFERENCES cls_images(id) ON DELETE CASCADE,
    class_id UUID NOT NULL REFERENCES cls_classes(id) ON DELETE RESTRICT,

    -- AI prediction metadata
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),
    is_ai_generated BOOLEAN DEFAULT false,
    ai_model TEXT,

    -- Review status
    is_reviewed BOOLEAN DEFAULT false,
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_labels_dataset ON cls_labels(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cls_labels_image ON cls_labels(image_id);
CREATE INDEX IF NOT EXISTS idx_cls_labels_class ON cls_labels(class_id);
CREATE INDEX IF NOT EXISTS idx_cls_labels_dataset_image ON cls_labels(dataset_id, image_id);
CREATE INDEX IF NOT EXISTS idx_cls_labels_ai ON cls_labels(is_ai_generated) WHERE is_ai_generated = true;
CREATE INDEX IF NOT EXISTS idx_cls_labels_review ON cls_labels(is_reviewed) WHERE is_reviewed = false;

-- Unique constraint for single-label datasets (enforced at application level)
-- For multi-label, multiple rows per image are allowed
-- CREATE UNIQUE INDEX idx_cls_labels_single ON cls_labels(dataset_id, image_id)
--     WHERE (SELECT task_type FROM cls_datasets WHERE id = dataset_id) = 'single_label';

COMMENT ON TABLE cls_labels IS 'Image-class assignments (labels) for classification';
COMMENT ON COLUMN cls_labels.confidence IS 'AI prediction confidence score (0-1)';

-- ============================================
-- CLS_DATASET_VERSIONS TABLE
-- Frozen snapshots for training
-- ============================================
CREATE TABLE IF NOT EXISTS cls_dataset_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES cls_datasets(id) ON DELETE CASCADE,

    -- Version info
    version_number INTEGER NOT NULL,
    name TEXT,
    description TEXT,

    -- Stats at time of snapshot
    image_count INTEGER NOT NULL,
    labeled_image_count INTEGER NOT NULL,
    class_count INTEGER NOT NULL,

    -- Class mapping (frozen at version time)
    class_mapping JSONB NOT NULL,  -- {class_id: index}
    class_names JSONB NOT NULL,    -- ["class1", "class2", ...]

    -- Split counts
    split_counts JSONB NOT NULL,   -- {train: N, val: N, test: N}

    -- Image IDs per split
    train_image_ids UUID[] NOT NULL,
    val_image_ids UUID[] NOT NULL,
    test_image_ids UUID[] NOT NULL,

    -- Export info
    export_format TEXT,
    export_url TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(dataset_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_cls_versions_dataset ON cls_dataset_versions(dataset_id);

COMMENT ON TABLE cls_dataset_versions IS 'Frozen dataset snapshots for reproducible training';
COMMENT ON COLUMN cls_dataset_versions.class_mapping IS 'Frozen class definitions with index mapping for training';
COMMENT ON COLUMN cls_dataset_versions.class_names IS 'Ordered list of class names matching index mapping';

-- ============================================
-- CLS_TRAINING_RUNS TABLE
-- Classification model training runs
-- ============================================
CREATE TABLE IF NOT EXISTS cls_training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Run info
    name TEXT NOT NULL,
    description TEXT,

    -- Data source
    dataset_id UUID NOT NULL REFERENCES cls_datasets(id) ON DELETE SET NULL,
    dataset_version_id UUID REFERENCES cls_dataset_versions(id) ON DELETE SET NULL,

    -- Task type
    task_type TEXT NOT NULL CHECK (task_type IN ('single_label', 'multi_label')),
    num_classes INTEGER NOT NULL,

    -- Model configuration
    model_type TEXT NOT NULL CHECK (model_type IN (
        'vit', 'convnext', 'efficientnet', 'swin', 'dinov2', 'clip'
    )),
    model_size TEXT NOT NULL,

    -- Training configuration
    config JSONB NOT NULL DEFAULT '{}',

    -- Progress
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'queued', 'training', 'completed', 'failed', 'cancelled'
    )),
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,

    -- Best metrics
    best_accuracy REAL,
    best_f1 REAL,
    best_top5_accuracy REAL,
    best_epoch INTEGER,

    -- Metrics history
    metrics_history JSONB DEFAULT '[]',  -- Array of {epoch, train_loss, val_loss, accuracy, f1, ...}

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

CREATE INDEX IF NOT EXISTS idx_cls_training_status ON cls_training_runs(status);
CREATE INDEX IF NOT EXISTS idx_cls_training_dataset ON cls_training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_cls_training_model_type ON cls_training_runs(model_type);
CREATE INDEX IF NOT EXISTS idx_cls_training_created ON cls_training_runs(created_at DESC);

COMMENT ON TABLE cls_training_runs IS 'Classification model training runs';
COMMENT ON COLUMN cls_training_runs.model_type IS 'Model architecture: vit, convnext, efficientnet, swin, dinov2, clip';

-- ============================================
-- CLS_TRAINED_MODELS TABLE
-- Trained classification models
-- ============================================
CREATE TABLE IF NOT EXISTS cls_trained_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Source
    training_run_id UUID REFERENCES cls_training_runs(id) ON DELETE SET NULL,

    -- Model info
    name TEXT NOT NULL,
    description TEXT,
    model_type TEXT NOT NULL,
    model_size TEXT,
    task_type TEXT NOT NULL,

    -- Checkpoints
    checkpoint_url TEXT NOT NULL,
    onnx_url TEXT,
    torchscript_url TEXT,

    -- Class info
    num_classes INTEGER NOT NULL,
    class_names JSONB NOT NULL,    -- ["class1", "class2", ...]
    class_mapping JSONB NOT NULL,  -- {index: {id, name, color}}

    -- Metrics
    accuracy REAL,
    f1_score REAL,
    top5_accuracy REAL,
    precision_macro REAL,
    recall_macro REAL,

    -- Detailed metrics
    confusion_matrix JSONB,    -- 2D array
    per_class_metrics JSONB,   -- {class_name: {precision, recall, f1, support}}

    -- Flags
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,

    -- File info
    file_size_bytes BIGINT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cls_models_active ON cls_trained_models(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_cls_models_default ON cls_trained_models(is_default) WHERE is_default = true;
CREATE INDEX IF NOT EXISTS idx_cls_models_type ON cls_trained_models(model_type);

COMMENT ON TABLE cls_trained_models IS 'Trained classification models';
COMMENT ON COLUMN cls_trained_models.confusion_matrix IS '2D array of confusion matrix values';
COMMENT ON COLUMN cls_trained_models.per_class_metrics IS 'Per-class precision, recall, f1, support';

-- ============================================
-- TRIGGERS - Ensure single default model per type
-- ============================================
CREATE OR REPLACE FUNCTION ensure_single_default_cls_model()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_default = true THEN
        UPDATE cls_trained_models
        SET is_default = false
        WHERE model_type = NEW.model_type
          AND id != NEW.id
          AND is_default = true;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cls_models_single_default ON cls_trained_models;
CREATE TRIGGER cls_models_single_default
    BEFORE INSERT OR UPDATE ON cls_trained_models
    FOR EACH ROW EXECUTE FUNCTION ensure_single_default_cls_model();

-- ============================================
-- TRIGGERS - Update timestamps
-- ============================================
DROP TRIGGER IF EXISTS cls_images_updated_at ON cls_images;
CREATE TRIGGER cls_images_updated_at
    BEFORE UPDATE ON cls_images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS cls_classes_updated_at ON cls_classes;
CREATE TRIGGER cls_classes_updated_at
    BEFORE UPDATE ON cls_classes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS cls_datasets_updated_at ON cls_datasets;
CREATE TRIGGER cls_datasets_updated_at
    BEFORE UPDATE ON cls_datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS cls_labels_updated_at ON cls_labels;
CREATE TRIGGER cls_labels_updated_at
    BEFORE UPDATE ON cls_labels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS cls_training_runs_updated_at ON cls_training_runs;
CREATE TRIGGER cls_training_runs_updated_at
    BEFORE UPDATE ON cls_training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS cls_trained_models_updated_at ON cls_trained_models;
CREATE TRIGGER cls_trained_models_updated_at
    BEFORE UPDATE ON cls_trained_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- RPC FUNCTIONS
-- ============================================

-- Get filter options for images
CREATE OR REPLACE FUNCTION get_cls_image_filter_options()
RETURNS JSON AS $$
BEGIN
    RETURN json_build_object(
        'statuses', (SELECT COALESCE(json_agg(DISTINCT status), '[]') FROM cls_images),
        'sources', (SELECT COALESCE(json_agg(DISTINCT source), '[]') FROM cls_images),
        'folders', (SELECT COALESCE(json_agg(DISTINCT folder), '[]') FROM cls_images WHERE folder IS NOT NULL),
        'total_count', (SELECT COUNT(*) FROM cls_images)
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

-- Update class image count
CREATE OR REPLACE FUNCTION update_cls_class_image_count(p_class_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE cls_classes SET
        image_count = (SELECT COUNT(DISTINCT image_id) FROM cls_labels WHERE class_id = p_class_id),
        updated_at = NOW()
    WHERE id = p_class_id;
END;
$$ LANGUAGE plpgsql;

-- Get labeling progress for a dataset
CREATE OR REPLACE FUNCTION get_cls_labeling_progress(p_dataset_id UUID)
RETURNS JSON AS $$
DECLARE
    total_count INTEGER;
    labeled_count INTEGER;
    pending_count INTEGER;
    review_count INTEGER;
    completed_count INTEGER;
    skipped_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id;
    SELECT COUNT(*) INTO labeled_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id AND status = 'labeled';
    SELECT COUNT(*) INTO pending_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id AND status = 'pending';
    SELECT COUNT(*) INTO review_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id AND status = 'review';
    SELECT COUNT(*) INTO completed_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id AND status = 'completed';
    SELECT COUNT(*) INTO skipped_count FROM cls_dataset_images WHERE dataset_id = p_dataset_id AND status = 'skipped';

    RETURN json_build_object(
        'total', total_count,
        'labeled', labeled_count,
        'pending', pending_count,
        'review', review_count,
        'completed', completed_count,
        'skipped', skipped_count,
        'progress_pct', CASE WHEN total_count > 0 THEN ROUND((labeled_count + completed_count)::NUMERIC / total_count * 100, 1) ELSE 0 END
    );
END;
$$ LANGUAGE plpgsql;

-- Get class distribution for a dataset
CREATE OR REPLACE FUNCTION get_cls_class_distribution(p_dataset_id UUID)
RETURNS JSON AS $$
BEGIN
    RETURN (
        SELECT COALESCE(json_agg(row_to_json(t)), '[]')
        FROM (
            SELECT
                c.id,
                c.name,
                c.display_name,
                c.color,
                COUNT(l.id) as image_count
            FROM cls_classes c
            LEFT JOIN cls_labels l ON l.class_id = c.id AND l.dataset_id = p_dataset_id
            WHERE c.is_active = true
            GROUP BY c.id, c.name, c.display_name, c.color
            ORDER BY image_count DESC
        ) t
    );
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE cls_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_classes ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_dataset_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_labels ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_dataset_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE cls_trained_models ENABLE ROW LEVEL SECURITY;

-- Allow all for authenticated (internal tool)
CREATE POLICY "Allow all for authenticated" ON cls_images FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_classes FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_datasets FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_dataset_images FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_labels FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_dataset_versions FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_training_runs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cls_trained_models FOR ALL USING (true);

-- ============================================
-- STORAGE BUCKETS (run manually in Supabase Dashboard)
-- ============================================
-- Create bucket: "cls-images" (public)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('cls-images', 'cls-images', true);

-- Create bucket: "cls-models" (private)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('cls-models', 'cls-models', false);

-- Create bucket: "cls-exports" (private)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('cls-exports', 'cls-exports', false);
