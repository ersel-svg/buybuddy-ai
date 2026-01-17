-- Buybuddy AI Platform - Training Label Configuration
-- Migration 012: Add flexible label field configuration to training runs
--
-- This migration adds:
-- 1. label_config column for storing label field configuration
-- 2. label_mapping column for storing label-to-index mapping

-- ============================================
-- ADD LABEL CONFIGURATION COLUMNS
-- ============================================

-- Add label_config column (stores label field settings)
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS label_config JSONB DEFAULT '{"label_field": "product_id", "min_samples_per_class": 2}'::jsonb;

-- Add label_mapping column (stores label -> class index mapping)
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS label_mapping JSONB;

-- Comments
COMMENT ON COLUMN training_runs.label_config IS 'Label field configuration: {label_field: product_id|category|brand_name|custom, min_samples_per_class, custom_mapping}';
COMMENT ON COLUMN training_runs.label_mapping IS 'Mapping from label values to class indices for the training run';

-- ============================================
-- UPDATE EXISTING RUNS (set default label_config)
-- ============================================

-- Set default label_config for existing runs that don't have it
UPDATE training_runs
SET label_config = '{"label_field": "product_id", "min_samples_per_class": 1}'::jsonb
WHERE label_config IS NULL;
