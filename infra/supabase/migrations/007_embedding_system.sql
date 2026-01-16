-- Buybuddy AI Platform - Embedding & Matching System
-- Migration 007: Add tables for embedding extraction, cutout sync, and export

-- ============================================
-- EMBEDDING MODELS TABLE
-- Tracks different embedding models (DINOv2, custom fine-tuned, etc.)
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Model info
    name TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('dinov2-base', 'dinov2-large', 'custom')),
    model_path TEXT,                          -- HuggingFace ID or description
    checkpoint_url TEXT,                      -- S3 path for custom models
    embedding_dim INTEGER NOT NULL,           -- 768, 1024, 256, etc.
    config JSONB DEFAULT '{}',                -- Additional model config

    -- Qdrant info
    qdrant_collection TEXT,                   -- Collection name in Qdrant
    qdrant_vector_count INTEGER DEFAULT 0,    -- Number of vectors in collection

    -- Status flags
    is_matching_active BOOLEAN DEFAULT false, -- Active model for matching UI

    -- Lineage
    training_job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Ensure only one model is active for matching at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_models_active
    ON embedding_models(is_matching_active)
    WHERE is_matching_active = true;

CREATE INDEX IF NOT EXISTS idx_embedding_models_type ON embedding_models(model_type);

-- ============================================
-- CUTOUT IMAGES TABLE
-- Real shelf photos from BuyBuddy API
-- ============================================
CREATE TABLE IF NOT EXISTS cutout_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- External reference
    external_id INTEGER UNIQUE NOT NULL,      -- BuyBuddy API ID
    image_url TEXT NOT NULL,
    predicted_upc TEXT,                       -- AI prediction (unreliable hint)

    -- Qdrant embedding reference
    qdrant_point_id UUID,                     -- Point ID in Qdrant
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,
    has_embedding BOOLEAN DEFAULT false,

    -- Match result (links to product, NOT to UPC)
    matched_product_id UUID REFERENCES products(id) ON DELETE SET NULL,
    match_similarity REAL,
    matched_by TEXT,                          -- User ID who made the match
    matched_at TIMESTAMPTZ,

    -- Timestamps
    synced_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_cutout_external ON cutout_images(external_id);
CREATE INDEX IF NOT EXISTS idx_cutout_unmatched ON cutout_images(id)
    WHERE matched_product_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_cutout_no_embedding ON cutout_images(id)
    WHERE has_embedding = false;
CREATE INDEX IF NOT EXISTS idx_cutout_predicted_upc ON cutout_images(predicted_upc);
CREATE INDEX IF NOT EXISTS idx_cutout_matched_product ON cutout_images(matched_product_id);
CREATE INDEX IF NOT EXISTS idx_cutout_synced ON cutout_images(synced_at DESC);

-- ============================================
-- EMBEDDING JOBS TABLE
-- Tracks embedding extraction jobs
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Job type
    job_type TEXT NOT NULL CHECK (job_type IN (
        'full',                   -- Full extraction - all images
        'incremental'             -- Only images without embeddings
    )),

    -- Source type
    source TEXT NOT NULL DEFAULT 'cutouts' CHECK (source IN (
        'cutouts',                -- Cutout images only
        'products',               -- Product synthetic frames only
        'both'                    -- Both cutouts and products
    )),

    -- Model reference
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,

    -- Progress tracking
    status TEXT DEFAULT 'pending' CHECK (status IN (
        'pending', 'queued', 'running', 'completed', 'failed', 'cancelled'
    )),
    total_images INTEGER DEFAULT 0,
    processed_images INTEGER DEFAULT 0,

    -- RunPod integration
    runpod_job_id TEXT,

    -- Error info
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_model ON embedding_jobs(embedding_model_id);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_created ON embedding_jobs(created_at DESC);

-- ============================================
-- EMBEDDING EXPORTS TABLE
-- Tracks export requests for other teams
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_exports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Model reference
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,

    -- Export config
    format TEXT NOT NULL CHECK (format IN ('json', 'numpy', 'faiss', 'qdrant_snapshot')),

    -- Status and result
    file_url TEXT,                            -- Storage URL
    file_size_bytes BIGINT,
    vector_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_exports_model ON embedding_exports(embedding_model_id);

-- ============================================
-- UPDATE TRIGGERS
-- ============================================
-- Add updated_at trigger for embedding_models
DROP TRIGGER IF EXISTS embedding_models_updated_at ON embedding_models;
CREATE TRIGGER embedding_models_updated_at
    BEFORE UPDATE ON embedding_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE embedding_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE cutout_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_exports ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON embedding_models FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON cutout_images FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON embedding_jobs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON embedding_exports FOR ALL USING (true);

-- ============================================
-- SEED DEFAULT MODEL
-- Insert DINOv2-base as the default matching model
-- ============================================
INSERT INTO embedding_models (
    name,
    model_type,
    model_path,
    embedding_dim,
    qdrant_collection,
    is_matching_active,
    config
) VALUES (
    'DINOv2 Base',
    'dinov2-base',
    'facebook/dinov2-base',
    768,
    'embeddings_default',
    true,
    '{"image_size": 518, "normalize": true}'::jsonb
) ON CONFLICT DO NOTHING;

-- ============================================
-- STORAGE BUCKET FOR EXPORTS
-- Run in Supabase Dashboard > Storage
-- ============================================
-- INSERT INTO storage.buckets (id, name, public) VALUES ('exports', 'exports', false);
