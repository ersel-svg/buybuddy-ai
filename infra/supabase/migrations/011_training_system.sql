-- Buybuddy AI Platform - Training System
-- Migration 011: Add unified model support, training runs, checkpoints, and evaluation
--
-- This migration adds:
-- 1. Extended embedding_models with new model families (DINOv2, DINOv3, CLIP)
-- 2. Training configuration system
-- 3. Training runs with product_id-based split tracking
-- 4. Checkpoint management
-- 5. Trained model registry
-- 6. Model evaluation results

-- ============================================
-- EXTEND EMBEDDING_MODELS TABLE
-- Add support for new model families
-- ============================================

-- Add new columns for model metadata
ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS model_family TEXT;

ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS hf_model_id TEXT;

ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS is_pretrained BOOLEAN DEFAULT true;

ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS is_default BOOLEAN DEFAULT false;

ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS base_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL;

-- Add product/cutout collection separation
ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS product_collection TEXT;

ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS cutout_collection TEXT;

-- Drop the old constraint if exists and create new one
ALTER TABLE embedding_models DROP CONSTRAINT IF EXISTS embedding_models_model_type_check;
ALTER TABLE embedding_models ADD CONSTRAINT embedding_models_model_type_check
    CHECK (model_type IN (
        'dinov2-small', 'dinov2-base', 'dinov2-large',
        'dinov3-small', 'dinov3-base', 'dinov3-large',
        'clip-vit-b-16', 'clip-vit-b-32', 'clip-vit-l-14',
        'custom'
    ));

COMMENT ON COLUMN embedding_models.model_family IS 'Model family: dinov2, dinov3, clip, custom';
COMMENT ON COLUMN embedding_models.hf_model_id IS 'HuggingFace model ID for loading';
COMMENT ON COLUMN embedding_models.is_pretrained IS 'Whether this is a pretrained model (vs fine-tuned)';
COMMENT ON COLUMN embedding_models.is_default IS 'Default models cannot be deleted';
COMMENT ON COLUMN embedding_models.base_model_id IS 'Reference to base model for fine-tuned models';
COMMENT ON COLUMN embedding_models.product_collection IS 'Qdrant collection for product embeddings';
COMMENT ON COLUMN embedding_models.cutout_collection IS 'Qdrant collection for cutout embeddings';

-- ============================================
-- TRAINING CONFIGS TABLE
-- User-saved training configurations
-- ============================================
CREATE TABLE IF NOT EXISTS training_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Config info
    name TEXT NOT NULL,
    description TEXT,

    -- Base model reference
    base_model_type TEXT NOT NULL,  -- e.g., "dinov3-base"

    -- Full merged config
    config JSONB NOT NULL,

    -- Flags
    is_default BOOLEAN DEFAULT false,

    -- Ownership
    created_by TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_configs_base_model ON training_configs(base_model_type);

COMMENT ON TABLE training_configs IS 'User-saved training configurations with model presets and overrides';

-- ============================================
-- TRAINING RUNS TABLE
-- Tracks training jobs with product_id-based splits
-- ============================================
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Run info
    name TEXT NOT NULL,
    description TEXT,

    -- Base model
    base_model_type TEXT NOT NULL,  -- e.g., "dinov3-base"

    -- Data source
    data_source TEXT NOT NULL CHECK (data_source IN ('all_products', 'matched_products', 'dataset', 'selected')),
    dataset_id UUID,  -- Reference to datasets table if applicable

    -- Split configuration (product_id based, NOT UPC based)
    split_config JSONB NOT NULL,         -- {train_ratio, val_ratio, test_ratio, seed}
    train_product_ids TEXT[] NOT NULL,   -- Product IDs in train set
    val_product_ids TEXT[] NOT NULL,     -- Product IDs in val set
    test_product_ids TEXT[] NOT NULL,    -- Product IDs in test set (NEVER used in training)

    -- Counts
    train_product_count INTEGER,
    val_product_count INTEGER,
    test_product_count INTEGER,
    train_image_count INTEGER,
    val_image_count INTEGER,
    test_image_count INTEGER,
    num_classes INTEGER,                  -- Number of unique product_ids (classes)

    -- Training configuration
    training_config JSONB NOT NULL,       -- Full config (preset + overrides)
    config_id UUID REFERENCES training_configs(id) ON DELETE SET NULL,

    -- RunPod integration
    runpod_job_id TEXT,
    runpod_endpoint_id TEXT,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'running', 'completed', 'failed', 'cancelled'
    )),

    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,

    -- Best metrics during training (on validation set)
    best_val_loss REAL,
    best_val_recall_at_1 REAL,
    best_val_recall_at_5 REAL,
    best_epoch INTEGER,

    -- Error info
    error_message TEXT,
    error_traceback TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_base_model ON training_runs(base_model_type);
CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs(created_at DESC);

COMMENT ON TABLE training_runs IS 'Training runs with product_id-based data splits to prevent data leakage';
COMMENT ON COLUMN training_runs.train_product_ids IS 'Product IDs in training set - all frames of these products used for training';
COMMENT ON COLUMN training_runs.test_product_ids IS 'Product IDs in test set - NEVER used during training, only for final evaluation';
COMMENT ON COLUMN training_runs.num_classes IS 'Number of unique product_ids (each product_id is a class)';

-- ============================================
-- TRAINING CHECKPOINTS TABLE
-- Model checkpoints during training
-- ============================================
CREATE TABLE IF NOT EXISTS training_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent run
    training_run_id UUID NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,

    -- Checkpoint info
    epoch INTEGER NOT NULL,
    step INTEGER,                         -- Global step (optional)

    -- Storage
    checkpoint_url TEXT NOT NULL,         -- S3/GCS URL
    file_size_bytes BIGINT,

    -- Metrics at this checkpoint
    train_loss REAL,
    val_loss REAL,
    val_recall_at_1 REAL,
    val_recall_at_5 REAL,
    val_recall_at_10 REAL,
    val_map REAL,

    -- Flags
    is_best BOOLEAN DEFAULT false,        -- Best validation loss
    is_final BOOLEAN DEFAULT false,       -- Last epoch

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(training_run_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON training_checkpoints(training_run_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_best ON training_checkpoints(training_run_id) WHERE is_best = true;

COMMENT ON TABLE training_checkpoints IS 'Model checkpoints saved during training with metrics';

-- ============================================
-- TRAINED MODELS TABLE
-- Registered models from training checkpoints
-- ============================================
CREATE TABLE IF NOT EXISTS trained_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Source
    training_run_id UUID NOT NULL REFERENCES training_runs(id),
    checkpoint_id UUID NOT NULL REFERENCES training_checkpoints(id),

    -- Model info
    name TEXT NOT NULL,
    description TEXT,

    -- Link to embedding_models for use in extraction
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,

    -- Test evaluation (evaluated on held-out test set)
    test_evaluated BOOLEAN DEFAULT false,
    test_metrics JSONB,                   -- {recall@1, recall@5, mAP, ...}
    test_evaluated_at TIMESTAMPTZ,

    -- Cross-domain metrics
    cross_domain_metrics JSONB,           -- {real_to_synth, synth_to_real}

    -- Identifier mapping (product_id -> identifiers)
    identifier_mapping_url TEXT,          -- S3 URL to JSON mapping file

    -- Flags
    is_default BOOLEAN DEFAULT false,     -- Default models cannot be deleted
    is_active BOOLEAN DEFAULT false,      -- Currently active for extraction

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trained_models_run ON trained_models(training_run_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_active ON trained_models(is_active) WHERE is_active = true;

COMMENT ON TABLE trained_models IS 'Registered trained models ready for deployment';
COMMENT ON COLUMN trained_models.identifier_mapping_url IS 'URL to JSON file mapping product_id to identifiers (barcode, short_code, upc)';

-- ============================================
-- MODEL EVALUATIONS TABLE
-- Detailed evaluation results
-- ============================================
CREATE TABLE IF NOT EXISTS model_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent model
    trained_model_id UUID NOT NULL REFERENCES trained_models(id) ON DELETE CASCADE,

    -- Evaluation config
    eval_config JSONB,                    -- {metrics, include_cross_domain, ...}

    -- Overall metrics
    overall_metrics JSONB NOT NULL,       -- {recall@1, recall@5, recall@10, mAP}

    -- Cross-domain metrics
    real_to_synthetic JSONB,              -- {recall@1, recall@5, ...}
    synthetic_to_real JSONB,              -- {recall@1, recall@5, ...}

    -- Per-category breakdown
    per_category_metrics JSONB,           -- {category: {recall@1, ...}}

    -- Hard cases analysis
    worst_product_ids JSONB,              -- [{product_id, recall@1, ...}]
    most_confused_pairs JSONB,            -- [{product_id_1, product_id_2, similarity}]

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_model ON model_evaluations(trained_model_id);

COMMENT ON TABLE model_evaluations IS 'Detailed evaluation results for trained models';
COMMENT ON COLUMN model_evaluations.worst_product_ids IS 'Products with worst recall - candidates for more training data';
COMMENT ON COLUMN model_evaluations.most_confused_pairs IS 'Product pairs that are often confused - may need visual review';

-- ============================================
-- UPDATE TRIGGERS
-- ============================================
DROP TRIGGER IF EXISTS training_configs_updated_at ON training_configs;
CREATE TRIGGER training_configs_updated_at
    BEFORE UPDATE ON training_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS training_runs_updated_at ON training_runs;
CREATE TRIGGER training_runs_updated_at
    BEFORE UPDATE ON training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trained_models_updated_at ON trained_models;
CREATE TRIGGER trained_models_updated_at
    BEFORE UPDATE ON trained_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE training_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE trained_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_evaluations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON training_configs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON training_runs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON training_checkpoints FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON trained_models FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON model_evaluations FOR ALL USING (true);

-- ============================================
-- UPDATE EXISTING DINOV2-BASE MODEL
-- Set is_default and model_family
-- ============================================
UPDATE embedding_models
SET
    model_family = 'dinov2',
    hf_model_id = 'facebook/dinov2-base',
    is_default = true,
    is_pretrained = true
WHERE model_type = 'dinov2-base';

-- ============================================
-- SEED DEFAULT MODELS
-- Insert all supported pretrained models
-- ============================================

-- DINOv2 Small
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv2 Small', 'dinov2-small', 'dinov2', 'facebook/dinov2-small', 384,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- DINOv2 Large
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv2 Large', 'dinov2-large', 'dinov2', 'facebook/dinov2-large', 1024,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- DINOv3 Small
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Small', 'dinov3-small', 'dinov3', 'facebook/dinov3-vits16-pretrain-lvd1689m', 384,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- DINOv3 Base
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Base', 'dinov3-base', 'dinov3', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 768,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- DINOv3 Large
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Large', 'dinov3-large', 'dinov3', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 1024,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- CLIP ViT-B/16
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-B/16', 'clip-vit-b-16', 'clip', 'openai/clip-vit-base-patch16', 512,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- CLIP ViT-B/32
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-B/32', 'clip-vit-b-32', 'clip', 'openai/clip-vit-base-patch32', 512,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- CLIP ViT-L/14
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-L/14', 'clip-vit-l-14', 'clip', 'openai/clip-vit-large-patch14', 768,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get active training runs count
CREATE OR REPLACE FUNCTION get_active_training_runs_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM training_runs
        WHERE status IN ('pending', 'preparing', 'running')
    );
END;
$$ LANGUAGE plpgsql;

-- Function to validate no product_id overlap in splits
CREATE OR REPLACE FUNCTION validate_training_split(
    train_ids TEXT[],
    val_ids TEXT[],
    test_ids TEXT[]
) RETURNS BOOLEAN AS $$
DECLARE
    overlap_count INTEGER;
BEGIN
    -- Check train vs val
    SELECT COUNT(*) INTO overlap_count
    FROM unnest(train_ids) t
    WHERE t = ANY(val_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    -- Check train vs test
    SELECT COUNT(*) INTO overlap_count
    FROM unnest(train_ids) t
    WHERE t = ANY(test_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    -- Check val vs test
    SELECT COUNT(*) INTO overlap_count
    FROM unnest(val_ids) t
    WHERE t = ANY(test_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    RETURN true;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION validate_training_split IS 'Validates that product_ids do not overlap between train/val/test splits';
