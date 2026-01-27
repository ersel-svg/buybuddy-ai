-- ============================================
-- CUTOUT IMAGES - NEW FIELDS FROM BUYBUDDY API
-- Adds merchant info, position, and annotated UPC
-- ============================================

-- Add new columns to cutout_images table
ALTER TABLE cutout_images ADD COLUMN IF NOT EXISTS merchant TEXT;
ALTER TABLE cutout_images ADD COLUMN IF NOT EXISTS merchant_id INTEGER;
ALTER TABLE cutout_images ADD COLUMN IF NOT EXISTS row_index TEXT;
ALTER TABLE cutout_images ADD COLUMN IF NOT EXISTS column_index TEXT;
ALTER TABLE cutout_images ADD COLUMN IF NOT EXISTS annotated_upc TEXT;

-- Performance indexes for filtering
CREATE INDEX IF NOT EXISTS idx_cutout_merchant_id ON cutout_images(merchant_id);
CREATE INDEX IF NOT EXISTS idx_cutout_annotated_upc ON cutout_images(annotated_upc);

-- Composite index for merchant + annotated_upc queries (accuracy analysis)
CREATE INDEX IF NOT EXISTS idx_cutout_merchant_annotated ON cutout_images(merchant_id, annotated_upc)
    WHERE annotated_upc IS NOT NULL AND annotated_upc != '';
