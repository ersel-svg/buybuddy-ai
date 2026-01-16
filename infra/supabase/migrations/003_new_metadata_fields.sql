-- Add new metadata fields from enhanced Gemini prompt
-- Run this in Supabase SQL Editor

-- ============================================
-- NEW PRODUCT METADATA FIELDS
-- ============================================

-- Brand info
ALTER TABLE products ADD COLUMN IF NOT EXISTS manufacturer_country TEXT;

-- Specifications
-- pack_configuration is JSONB because it contains {type: string, item_count: number}
ALTER TABLE products ADD COLUMN IF NOT EXISTS pack_configuration JSONB;
ALTER TABLE products ADD COLUMN IF NOT EXISTS identifiers JSONB;

-- Marketing
ALTER TABLE products ADD COLUMN IF NOT EXISTS marketing_description TEXT;

-- Extraction metadata
ALTER TABLE products ADD COLUMN IF NOT EXISTS issues_detected TEXT[];

-- ============================================
-- INDEXES (optional, add if needed for queries)
-- ============================================
-- CREATE INDEX IF NOT EXISTS idx_products_manufacturer ON products(manufacturer_country);
-- CREATE INDEX IF NOT EXISTS idx_products_pack_config ON products USING GIN(pack_configuration);
