-- Buybuddy AI Platform - Triplet Mining System
-- Migration 014: Add triplet mining tables and training enhancements
--
-- This migration adds:
-- 1. Triplet mining runs table
-- 2. Mined triplets table
-- 3. Training run enhancements for SOTA training
-- 4. Matching feedback table for active learning

-- ============================================
-- TRIPLET MINING RUNS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS triplet_mining_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Run info
    name TEXT NOT NULL,
    description TEXT,

    -- Source
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    embedding_model_id UUID REFERENCES embedding_models(id) NOT NULL,
    collection_name TEXT NOT NULL,

    -- Mining configuration
    hard_negative_threshold REAL DEFAULT 0.7,
    positive_threshold REAL DEFAULT 0.9,
    max_triplets_per_anchor INTEGER DEFAULT 10,
    include_cross_domain BOOLEAN DEFAULT true,

    -- Statistics
    total_anchors INTEGER,
    total_triplets INTEGER,
    hard_triplets INTEGER,
    semi_hard_triplets INTEGER,
    easy_triplets INTEGER,
    cross_domain_triplets INTEGER,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled'
    )),
    error_message TEXT,

    -- Output
    output_url TEXT,

    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_triplet_mining_runs_status ON triplet_mining_runs(status);
CREATE INDEX IF NOT EXISTS idx_triplet_mining_runs_model ON triplet_mining_runs(embedding_model_id);
CREATE INDEX IF NOT EXISTS idx_triplet_mining_runs_created ON triplet_mining_runs(created_at DESC);

COMMENT ON TABLE triplet_mining_runs IS 'Triplet mining job runs for finding hard negatives';
COMMENT ON COLUMN triplet_mining_runs.hard_negative_threshold IS 'Similarity threshold above which negatives are considered hard';
COMMENT ON COLUMN triplet_mining_runs.include_cross_domain IS 'Include synthetic-to-real cross-domain triplets';

-- ============================================
-- MINED TRIPLETS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS mined_triplets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mining_run_id UUID NOT NULL REFERENCES triplet_mining_runs(id) ON DELETE CASCADE,

    -- Triplet components (product IDs)
    anchor_product_id TEXT NOT NULL,
    positive_product_id TEXT NOT NULL,
    negative_product_id TEXT NOT NULL,

    -- Frame indices
    anchor_frame_idx INTEGER DEFAULT 0,
    positive_frame_idx INTEGER DEFAULT 0,
    negative_frame_idx INTEGER DEFAULT 0,

    -- Similarities
    anchor_positive_sim REAL NOT NULL,
    anchor_negative_sim REAL NOT NULL,

    -- Computed margin (positive_sim - negative_sim)
    margin REAL GENERATED ALWAYS AS (anchor_positive_sim - anchor_negative_sim) STORED,

    -- Classification
    difficulty TEXT CHECK (difficulty IN ('hard', 'semi_hard', 'easy')),

    -- Domain info
    is_cross_domain BOOLEAN DEFAULT false,
    anchor_domain TEXT CHECK (anchor_domain IN ('synthetic', 'real', 'unknown')),
    positive_domain TEXT CHECK (positive_domain IN ('synthetic', 'real', 'unknown')),
    negative_domain TEXT CHECK (negative_domain IN ('synthetic', 'real', 'unknown')),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mined_triplets_run ON mined_triplets(mining_run_id);
CREATE INDEX IF NOT EXISTS idx_mined_triplets_difficulty ON mined_triplets(mining_run_id, difficulty);
CREATE INDEX IF NOT EXISTS idx_mined_triplets_cross_domain ON mined_triplets(mining_run_id, is_cross_domain);
CREATE INDEX IF NOT EXISTS idx_mined_triplets_anchor ON mined_triplets(anchor_product_id);

COMMENT ON TABLE mined_triplets IS 'Individual mined triplets with similarity scores and difficulty classification';
COMMENT ON COLUMN mined_triplets.margin IS 'Difference between positive and negative similarity (higher = easier)';
COMMENT ON COLUMN mined_triplets.is_cross_domain IS 'True if anchor is synthetic and negative is real';

-- ============================================
-- TRAINING RUNS ENHANCEMENTS
-- ============================================

-- Link training runs to triplet mining
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS triplet_mining_run_id UUID REFERENCES triplet_mining_runs(id) ON DELETE SET NULL;

-- SOTA training configuration
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS sota_config JSONB DEFAULT '{
    "loss": {
        "loss_type": "arcface",
        "arcface_weight": 1.0,
        "triplet_weight": 0.0,
        "domain_weight": 0.0,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "triplet_margin": 0.3
    },
    "sampling": {
        "strategy": "balanced",
        "products_per_batch": 8,
        "samples_per_product": 4,
        "synthetic_ratio": 0.5
    },
    "curriculum": {
        "enabled": false,
        "warmup_epochs": 2,
        "easy_epochs": 5,
        "hard_epochs": 10,
        "finetune_epochs": 3
    },
    "domain_adaptation": {
        "enabled": false,
        "use_mixup": false,
        "use_adversarial": false,
        "cross_domain_triplet_ratio": 0.3
    },
    "early_stopping": {
        "enabled": true,
        "patience": 5,
        "min_delta": 0.001
    }
}'::jsonb;

COMMENT ON COLUMN training_runs.triplet_mining_run_id IS 'Reference to pre-mined triplets for this training run';
COMMENT ON COLUMN training_runs.sota_config IS 'SOTA training configuration: loss weights, sampling strategy, curriculum, domain adaptation';

-- ============================================
-- MATCHING FEEDBACK TABLE (Active Learning)
-- ============================================

CREATE TABLE IF NOT EXISTS matching_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- What was matched
    cutout_id UUID,
    cutout_image_url TEXT,

    -- Prediction details
    predicted_product_id TEXT,
    predicted_similarity REAL,
    model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,
    collection_name TEXT,

    -- User feedback
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('correct', 'wrong', 'uncertain')),
    correct_product_id TEXT,  -- If wrong, what's the correct product?

    -- Metadata
    user_id TEXT,
    feedback_source TEXT DEFAULT 'web',  -- 'web', 'api', 'review', 'auto'
    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matching_feedback_type ON matching_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_matching_feedback_model ON matching_feedback(model_id);
CREATE INDEX IF NOT EXISTS idx_matching_feedback_wrong ON matching_feedback(feedback_type)
    WHERE feedback_type = 'wrong';
CREATE INDEX IF NOT EXISTS idx_matching_feedback_created ON matching_feedback(created_at DESC);

COMMENT ON TABLE matching_feedback IS 'User feedback on matching results for active learning';
COMMENT ON COLUMN matching_feedback.correct_product_id IS 'The correct product ID if prediction was wrong';

-- ============================================
-- UPDATE TRIGGERS
-- ============================================

DROP TRIGGER IF EXISTS triplet_mining_runs_updated_at ON triplet_mining_runs;
CREATE TRIGGER triplet_mining_runs_updated_at
    BEFORE UPDATE ON triplet_mining_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- RLS POLICIES
-- ============================================

ALTER TABLE triplet_mining_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE mined_triplets ENABLE ROW LEVEL SECURITY;
ALTER TABLE matching_feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow all for authenticated" ON triplet_mining_runs;
DROP POLICY IF EXISTS "Allow all for authenticated" ON mined_triplets;
DROP POLICY IF EXISTS "Allow all for authenticated" ON matching_feedback;

CREATE POLICY "Allow all for authenticated" ON triplet_mining_runs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON mined_triplets FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON matching_feedback FOR ALL USING (true);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Get triplet statistics for a mining run
CREATE OR REPLACE FUNCTION get_triplet_mining_stats(run_id UUID)
RETURNS TABLE (
    total_triplets BIGINT,
    hard_count BIGINT,
    semi_hard_count BIGINT,
    easy_count BIGINT,
    cross_domain_count BIGINT,
    avg_margin REAL,
    min_margin REAL,
    max_margin REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_triplets,
        COUNT(*) FILTER (WHERE difficulty = 'hard')::BIGINT as hard_count,
        COUNT(*) FILTER (WHERE difficulty = 'semi_hard')::BIGINT as semi_hard_count,
        COUNT(*) FILTER (WHERE difficulty = 'easy')::BIGINT as easy_count,
        COUNT(*) FILTER (WHERE is_cross_domain = true)::BIGINT as cross_domain_count,
        AVG(margin)::REAL as avg_margin,
        MIN(margin)::REAL as min_margin,
        MAX(margin)::REAL as max_margin
    FROM mined_triplets
    WHERE mining_run_id = run_id;
END;
$$ LANGUAGE plpgsql;

-- Get hard examples from feedback for training
CREATE OR REPLACE FUNCTION get_hard_examples_from_feedback(
    p_model_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 1000
)
RETURNS TABLE (
    cutout_image_url TEXT,
    correct_product_id TEXT,
    wrong_product_id TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        mf.cutout_image_url,
        mf.correct_product_id,
        mf.predicted_product_id as wrong_product_id
    FROM matching_feedback mf
    WHERE mf.feedback_type = 'wrong'
      AND mf.correct_product_id IS NOT NULL
      AND (p_model_id IS NULL OR mf.model_id = p_model_id)
    ORDER BY mf.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_triplet_mining_stats IS 'Get statistics for a triplet mining run';
COMMENT ON FUNCTION get_hard_examples_from_feedback IS 'Get hard training examples from user feedback';

-- Done!
SELECT 'Migration 014: Triplet Mining System completed successfully!' as status;
