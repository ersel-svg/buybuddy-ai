-- Migration: Consolidate all product images into product_images table
-- This migrates data from product_real_images to the unified product_images table

-- ============================================
-- 1. Migrate product_real_images to product_images
-- ============================================

-- Insert real images from product_real_images into product_images
INSERT INTO product_images (product_id, image_path, image_url, image_type, source, created_at)
SELECT
    product_id,
    COALESCE(image_path, image_url) as image_path,
    image_url,
    'real' as image_type,
    'matching' as source,
    created_at
FROM product_real_images
ON CONFLICT DO NOTHING;

-- ============================================
-- 2. Add indexes if not exist
-- ============================================

-- Index for filtering by product and type
CREATE INDEX IF NOT EXISTS idx_product_images_product_type
ON product_images(product_id, image_type);

-- Index for ordering by frame_index
CREATE INDEX IF NOT EXISTS idx_product_images_frame_index
ON product_images(product_id, frame_index);

-- ============================================
-- 3. Enable RLS on product_images if not already
-- ============================================

ALTER TABLE product_images ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (internal tool)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'product_images'
        AND policyname = 'Allow all for authenticated'
    ) THEN
        CREATE POLICY "Allow all for authenticated" ON product_images FOR ALL USING (true);
    END IF;
END $$;

-- ============================================
-- 4. Verify migration
-- ============================================

-- Count records in both tables
DO $$
DECLARE
    old_count INTEGER;
    new_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO old_count FROM product_real_images;
    SELECT COUNT(*) INTO new_count FROM product_images WHERE image_type = 'real';

    RAISE NOTICE 'Migration result: % records in product_real_images, % real records in product_images',
        old_count, new_count;
END $$;

-- ============================================
-- 5. OPTIONAL: Drop old table after verification
-- Uncomment these lines after confirming migration is successful
-- ============================================

-- DROP TABLE IF EXISTS product_real_images;

-- ============================================
-- 6. Create helper function for frame counts
-- ============================================

CREATE OR REPLACE FUNCTION get_product_frame_counts(p_product_id UUID)
RETURNS TABLE(synthetic_count BIGINT, real_count BIGINT, augmented_count BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) FILTER (WHERE image_type = 'synthetic') as synthetic_count,
        COUNT(*) FILTER (WHERE image_type = 'real') as real_count,
        COUNT(*) FILTER (WHERE image_type = 'augmented') as augmented_count
    FROM product_images
    WHERE product_id = p_product_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Done!
-- ============================================
