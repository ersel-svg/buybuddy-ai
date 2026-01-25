-- Buybuddy AI Platform - Embedding Collections Table
-- Migration 061: Create embedding_collections table and add source_config to embedding_jobs
-- Required for SOTA embedding extraction pattern (worker fetches from DB)

-- ============================================
-- CREATE EMBEDDING_COLLECTIONS TABLE
-- Worker writes collection metadata here after extraction
-- ============================================

CREATE TABLE IF NOT EXISTS embedding_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Collection identification
    name TEXT UNIQUE NOT NULL,              -- Qdrant collection name (e.g., "products_dinov2_base")

    -- Collection metadata
    collection_type TEXT,                   -- matching, training, evaluation, production
    source_type TEXT,                       -- all, matched, dataset, cutouts, products

    -- Model reference
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,

    -- Configuration
    image_types TEXT[] DEFAULT '{}',        -- Array of image types: ["synthetic", "real", "augmented", "cutout"]

    -- Statistics
    vector_count INTEGER DEFAULT 0,         -- Number of vectors in collection

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_sync_at TIMESTAMPTZ,               -- Last time collection was updated
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_embedding_collections_model
    ON embedding_collections(embedding_model_id);
CREATE INDEX IF NOT EXISTS idx_embedding_collections_type
    ON embedding_collections(collection_type);
CREATE INDEX IF NOT EXISTS idx_embedding_collections_name
    ON embedding_collections(name);

-- ============================================
-- ADD SOURCE_CONFIG TO EMBEDDING_JOBS
-- Stores extraction configuration for SOTA pattern
-- ============================================

-- Add source_config column (JSONB for flexible configuration)
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS source_config JSONB DEFAULT '{}';

-- Add purpose column if not exists
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS purpose TEXT DEFAULT 'matching';

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE embedding_collections IS
    'Tracks Qdrant embedding collections and their metadata. Worker writes here after extraction.';

COMMENT ON COLUMN embedding_collections.name IS
    'Unique Qdrant collection name (e.g., products_dinov2_base, cutouts_matching_v2)';

COMMENT ON COLUMN embedding_collections.collection_type IS
    'Purpose of collection: matching (product search), training (triplet mining), evaluation (model testing), production (inference)';

COMMENT ON COLUMN embedding_collections.source_type IS
    'Source of images: all (everything), matched (products with cutouts), dataset (specific dataset), cutouts, products';

COMMENT ON COLUMN embedding_collections.image_types IS
    'Types of images included: synthetic (rendered), real (photos), augmented (transformed), cutout (store photos)';

COMMENT ON COLUMN embedding_jobs.source_config IS
    'SOTA extraction config: {type, filters, frame_selection, max_frames}. Worker uses this to fetch from DB.';

COMMENT ON COLUMN embedding_jobs.purpose IS
    'Job purpose: matching (product search), training (model training), evaluation (model eval), production (inference)';
