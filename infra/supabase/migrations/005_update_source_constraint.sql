-- Migration: Update source constraint to allow augmentation worker sources
-- The augmentation worker uses more specific source types like aug_syn_light, aug_syn_heavy, etc.

-- ============================================
-- 1. Drop the old constraint
-- ============================================

ALTER TABLE product_images DROP CONSTRAINT IF EXISTS product_images_source_check;

-- ============================================
-- 2. Add new constraint with expanded values
-- ============================================

ALTER TABLE product_images ADD CONSTRAINT product_images_source_check
    CHECK (source IN (
        'video_frame',      -- From video segmentation pipeline
        'matching',         -- From real image matching
        'augmentation',     -- Generic augmentation
        'aug_synthetic',    -- Augmented from synthetic frames
        'aug_syn_light',    -- Light augmentation on synthetic
        'aug_syn_heavy',    -- Heavy augmentation on synthetic
        'aug_real'          -- Augmented from real images
    ));

-- ============================================
-- Done!
-- ============================================
