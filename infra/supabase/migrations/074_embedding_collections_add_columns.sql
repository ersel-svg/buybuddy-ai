-- Migration: Add missing columns to embedding_collections table
-- Description: Add product_collection, cutout_collection, frame_selection, source_dataset_id
-- Date: 2026-02-02

-- Add product_collection column (references the product embedding collection name)
ALTER TABLE embedding_collections
ADD COLUMN IF NOT EXISTS product_collection TEXT;

-- Add cutout_collection column (references the cutout embedding collection name)
ALTER TABLE embedding_collections
ADD COLUMN IF NOT EXISTS cutout_collection TEXT;

-- Add frame_selection column (first, all, key_frames, interval)
ALTER TABLE embedding_collections
ADD COLUMN IF NOT EXISTS frame_selection TEXT DEFAULT 'first';

-- Add source_dataset_id column (for dataset-filtered extractions)
ALTER TABLE embedding_collections
ADD COLUMN IF NOT EXISTS source_dataset_id UUID;

-- Comments
COMMENT ON COLUMN embedding_collections.product_collection IS
    'Name of the related product embedding collection in Qdrant';

COMMENT ON COLUMN embedding_collections.cutout_collection IS
    'Name of the related cutout embedding collection in Qdrant';

COMMENT ON COLUMN embedding_collections.frame_selection IS
    'Frame selection strategy: first, all, key_frames, interval';

COMMENT ON COLUMN embedding_collections.source_dataset_id IS
    'Dataset ID if extraction was filtered by a specific dataset';
