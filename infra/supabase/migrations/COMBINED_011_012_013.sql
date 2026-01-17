-- ============================================
-- COMBINED MIGRATION: 011 + 012 + 013
-- Training System + Label Config + Identifier Mapping
-- Run this in Supabase SQL Editor
-- ============================================

-- ============================================
-- MIGRATION 011: TRAINING SYSTEM
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

-- Training Configs Table
CREATE TABLE IF NOT EXISTS training_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    base_model_type TEXT NOT NULL,
    config JSONB NOT NULL,
    is_default BOOLEAN DEFAULT false,
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_configs_base_model ON training_configs(base_model_type);

-- Training Runs Table
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    base_model_type TEXT NOT NULL,
    data_source TEXT NOT NULL CHECK (data_source IN ('all_products', 'matched_products', 'dataset', 'selected')),
    dataset_id UUID,
    split_config JSONB NOT NULL,
    train_product_ids TEXT[] NOT NULL,
    val_product_ids TEXT[] NOT NULL,
    test_product_ids TEXT[] NOT NULL,
    train_product_count INTEGER,
    val_product_count INTEGER,
    test_product_count INTEGER,
    train_image_count INTEGER,
    val_image_count INTEGER,
    test_image_count INTEGER,
    num_classes INTEGER,
    training_config JSONB NOT NULL,
    config_id UUID REFERENCES training_configs(id) ON DELETE SET NULL,
    runpod_job_id TEXT,
    runpod_endpoint_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'preparing', 'running', 'completed', 'failed', 'cancelled'
    )),
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER NOT NULL,
    best_val_loss REAL,
    best_val_recall_at_1 REAL,
    best_val_recall_at_5 REAL,
    best_epoch INTEGER,
    error_message TEXT,
    error_traceback TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_base_model ON training_runs(base_model_type);
CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs(created_at DESC);

-- Training Checkpoints Table
CREATE TABLE IF NOT EXISTS training_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    step INTEGER,
    checkpoint_url TEXT NOT NULL,
    file_size_bytes BIGINT,
    train_loss REAL,
    val_loss REAL,
    val_recall_at_1 REAL,
    val_recall_at_5 REAL,
    val_recall_at_10 REAL,
    val_map REAL,
    is_best BOOLEAN DEFAULT false,
    is_final BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(training_run_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON training_checkpoints(training_run_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_best ON training_checkpoints(training_run_id) WHERE is_best = true;

-- Trained Models Table
CREATE TABLE IF NOT EXISTS trained_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES training_runs(id),
    checkpoint_id UUID NOT NULL REFERENCES training_checkpoints(id),
    name TEXT NOT NULL,
    description TEXT,
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,
    test_evaluated BOOLEAN DEFAULT false,
    test_metrics JSONB,
    test_evaluated_at TIMESTAMPTZ,
    cross_domain_metrics JSONB,
    identifier_mapping_url TEXT,
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trained_models_run ON trained_models(training_run_id);
CREATE INDEX IF NOT EXISTS idx_trained_models_active ON trained_models(is_active) WHERE is_active = true;

-- Model Evaluations Table
CREATE TABLE IF NOT EXISTS model_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trained_model_id UUID NOT NULL REFERENCES trained_models(id) ON DELETE CASCADE,
    eval_config JSONB,
    overall_metrics JSONB NOT NULL,
    real_to_synthetic JSONB,
    synthetic_to_real JSONB,
    per_category_metrics JSONB,
    worst_product_ids JSONB,
    most_confused_pairs JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evaluations_model ON model_evaluations(trained_model_id);

-- Update Triggers
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

-- RLS Policies
ALTER TABLE training_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE trained_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_evaluations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all for authenticated" ON training_configs;
DROP POLICY IF EXISTS "Allow all for authenticated" ON training_runs;
DROP POLICY IF EXISTS "Allow all for authenticated" ON training_checkpoints;
DROP POLICY IF EXISTS "Allow all for authenticated" ON trained_models;
DROP POLICY IF EXISTS "Allow all for authenticated" ON model_evaluations;

CREATE POLICY "Allow all for authenticated" ON training_configs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON training_runs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON training_checkpoints FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON trained_models FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON model_evaluations FOR ALL USING (true);

-- Update existing DINOv2-base model
UPDATE embedding_models
SET
    model_family = 'dinov2',
    hf_model_id = 'facebook/dinov2-base',
    is_default = true,
    is_pretrained = true
WHERE model_type = 'dinov2-base';

-- Seed Default Models
INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv2 Small', 'dinov2-small', 'dinov2', 'facebook/dinov2-small', 384,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv2 Large', 'dinov2-large', 'dinov2', 'facebook/dinov2-large', 1024,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Small', 'dinov3-small', 'dinov3', 'facebook/dinov3-vits16-pretrain-lvd1689m', 384,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Base', 'dinov3-base', 'dinov3', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 768,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'DINOv3 Large', 'dinov3-large', 'dinov3', 'facebook/dinov3-vitl16-pretrain-lvd1689m', 1024,
    true, true, false,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-B/16', 'clip-vit-b-16', 'clip', 'openai/clip-vit-base-patch16', 512,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-B/32', 'clip-vit-b-32', 'clip', 'openai/clip-vit-base-patch32', 512,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

INSERT INTO embedding_models (
    name, model_type, model_family, hf_model_id, embedding_dim,
    is_pretrained, is_default, is_matching_active, config
) VALUES (
    'CLIP ViT-L/14', 'clip-vit-l-14', 'clip', 'openai/clip-vit-large-patch14', 768,
    true, true, false,
    '{"image_size": 224, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- Helper Functions
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

CREATE OR REPLACE FUNCTION validate_training_split(
    train_ids TEXT[],
    val_ids TEXT[],
    test_ids TEXT[]
) RETURNS BOOLEAN AS $$
DECLARE
    overlap_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO overlap_count
    FROM unnest(train_ids) t
    WHERE t = ANY(val_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    SELECT COUNT(*) INTO overlap_count
    FROM unnest(train_ids) t
    WHERE t = ANY(test_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    SELECT COUNT(*) INTO overlap_count
    FROM unnest(val_ids) t
    WHERE t = ANY(test_ids);
    IF overlap_count > 0 THEN
        RETURN false;
    END IF;

    RETURN true;
END;
$$ LANGUAGE plpgsql;


-- ============================================
-- MIGRATION 012: LABEL CONFIG
-- ============================================

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS label_config JSONB DEFAULT '{"label_field": "product_id", "min_samples_per_class": 2}'::jsonb;

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS label_mapping JSONB;

UPDATE training_runs
SET label_config = '{"label_field": "product_id", "min_samples_per_class": 1}'::jsonb
WHERE label_config IS NULL;


-- ============================================
-- MIGRATION 013: IDENTIFIER MAPPING
-- ============================================

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS identifier_mapping JSONB;

-- Done!
SELECT 'All migrations completed successfully!' as status;
