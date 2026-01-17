-- Migration 021: Add image configuration columns to training_runs
-- Supports multi-image-type training (synthetic, real, augmented)

-- Add image_config column for storing frame selection and image type config
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS image_config JSONB;

-- Add image_stats column for storing counts by image type
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS image_stats JSONB;

-- Add total_images column for quick reference
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS total_images INTEGER;

-- Add SOTA config column
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS sota_config JSONB;

COMMENT ON COLUMN training_runs.image_config IS 'Image configuration: {image_types, frame_selection, max_frames_per_type, include_matched_cutouts}';
COMMENT ON COLUMN training_runs.image_stats IS 'Image counts by type: {synthetic, real, augmented, cutout}';
COMMENT ON COLUMN training_runs.total_images IS 'Total number of training images';
COMMENT ON COLUMN training_runs.sota_config IS 'SOTA training configuration if enabled';
