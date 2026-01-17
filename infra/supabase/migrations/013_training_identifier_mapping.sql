-- Buybuddy AI Platform - Training Identifier Mapping
-- Migration 013: Add identifier_mapping column to training_runs
--
-- This migration adds:
-- 1. identifier_mapping column to store product_id -> identifiers mapping directly in the run

-- ============================================
-- ADD IDENTIFIER MAPPING COLUMN
-- ============================================

-- Add identifier_mapping column to training_runs
-- This stores the mapping of product_id to identifiers (barcode, brand, category, etc.)
-- Used for inference after training to map predictions back to product identifiers
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS identifier_mapping JSONB;

-- Comment
COMMENT ON COLUMN training_runs.identifier_mapping IS 'Mapping from product_id to product identifiers (barcode, brand_name, category, etc.) for inference';
