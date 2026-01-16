-- Buybuddy AI Platform - Embedding System Enhancements
-- Migration 010: Add collection metadata, job purpose, and training support

-- ============================================
-- EMBEDDING COLLECTIONS METADATA TABLE
-- Tracks collection source, config, and stats
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Collection info
    name TEXT UNIQUE NOT NULL,
    collection_type TEXT NOT NULL CHECK (collection_type IN ('products', 'cutouts', 'training', 'evaluation')),

    -- Source tracking
    source_type TEXT CHECK (source_type IN ('all', 'selected', 'dataset', 'matched', 'filter', 'new')),
    source_dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    source_product_ids TEXT[],  -- If selected products
    source_filter JSONB,        -- If filter-based

    -- Model info
    embedding_model_id UUID REFERENCES embedding_models(id) ON DELETE SET NULL,
    vector_count INTEGER DEFAULT 0,

    -- Config
    image_types TEXT[] DEFAULT ARRAY['synthetic'],  -- synthetic, real, augmented
    frame_selection TEXT DEFAULT 'first',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_sync_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_embedding_collections_type ON embedding_collections(collection_type);
CREATE INDEX IF NOT EXISTS idx_embedding_collections_model ON embedding_collections(embedding_model_id);
CREATE INDEX IF NOT EXISTS idx_embedding_collections_name ON embedding_collections(name);

-- ============================================
-- EXTEND EMBEDDING_JOBS TABLE
-- Add purpose, image_types, and output config
-- ============================================

-- Add purpose field for job categorization
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS purpose TEXT DEFAULT 'matching'
    CHECK (purpose IN ('matching', 'training', 'evaluation'));

-- Add image types for training jobs
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS image_types TEXT[] DEFAULT ARRAY['synthetic'];

-- Add output target for training jobs
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS output_target TEXT DEFAULT 'qdrant'
    CHECK (output_target IN ('qdrant', 'file'));

-- Add file URL for file exports
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS output_file_url TEXT;

-- Add source config for detailed tracking
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS source_config JSONB DEFAULT '{}';

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON TABLE embedding_collections IS 'Metadata for Qdrant collections including source, config, and stats';
COMMENT ON COLUMN embedding_collections.collection_type IS 'Type: products, cutouts, training, evaluation';
COMMENT ON COLUMN embedding_collections.source_type IS 'How products were selected: all, selected, dataset, matched, filter, new';
COMMENT ON COLUMN embedding_collections.source_product_ids IS 'Product IDs if source_type is selected';
COMMENT ON COLUMN embedding_collections.image_types IS 'Image types included: synthetic, real, augmented';

COMMENT ON COLUMN embedding_jobs.purpose IS 'Job purpose: matching, training, evaluation';
COMMENT ON COLUMN embedding_jobs.image_types IS 'Image types to extract: synthetic, real, augmented';
COMMENT ON COLUMN embedding_jobs.output_target IS 'Output destination: qdrant or file';
COMMENT ON COLUMN embedding_jobs.output_file_url IS 'S3 URL if output_target is file';
COMMENT ON COLUMN embedding_jobs.source_config IS 'JSON config for source selection';

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE embedding_collections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON embedding_collections FOR ALL USING (true);

-- ============================================
-- UPDATE TRIGGERS
-- ============================================
DROP TRIGGER IF EXISTS embedding_collections_updated_at ON embedding_collections;
CREATE TRIGGER embedding_collections_updated_at
    BEFORE UPDATE ON embedding_collections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- HELPER FUNCTION: Get matched products count
-- ============================================
CREATE OR REPLACE FUNCTION get_matched_products_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(DISTINCT matched_product_id)
        FROM cutout_images
        WHERE matched_product_id IS NOT NULL
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_matched_products_count() IS 'Returns count of products that have at least one matched cutout';
