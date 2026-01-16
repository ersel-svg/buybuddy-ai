-- Buybuddy AI Platform - Embedding Collections Enhancement
-- Migration 009: Add separate collection support for products and cutouts

-- ============================================
-- ADD COLLECTION COLUMNS TO EMBEDDING_MODELS
-- Allows separate collections for products vs cutouts
-- ============================================

-- Add product-specific collection
ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS product_collection TEXT;

-- Add cutout-specific collection
ALTER TABLE embedding_models
ADD COLUMN IF NOT EXISTS cutout_collection TEXT;

-- Migrate existing data: copy qdrant_collection to both if set
UPDATE embedding_models
SET product_collection = qdrant_collection,
    cutout_collection = qdrant_collection
WHERE qdrant_collection IS NOT NULL
  AND product_collection IS NULL
  AND cutout_collection IS NULL;

-- ============================================
-- ADD EXTRACTION CONFIG TO EMBEDDING_JOBS
-- Stores advanced extraction configuration
-- ============================================

-- Add extraction config JSON column
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS extraction_config JSONB DEFAULT '{}';

-- Add collection names array (for tracking which collections were created/updated)
ALTER TABLE embedding_jobs
ADD COLUMN IF NOT EXISTS collection_names TEXT[];

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON COLUMN embedding_models.product_collection IS 'Qdrant collection name for product embeddings (e.g., products_dinov2_base)';
COMMENT ON COLUMN embedding_models.cutout_collection IS 'Qdrant collection name for cutout embeddings (e.g., cutouts_dinov2_base)';
COMMENT ON COLUMN embedding_jobs.extraction_config IS 'JSON config for extraction: frame_selection, max_frames, separate_collections, etc.';
COMMENT ON COLUMN embedding_jobs.collection_names IS 'Array of collection names created/updated by this job';
