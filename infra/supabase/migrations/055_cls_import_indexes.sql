-- ============================================
-- CLS Import Visual Picker - Index Optimizations
-- ============================================
-- Optimizes queries for the visual image picker UI
-- that allows users to browse and select images from
-- Products and Cutouts to import into CLS datasets.

-- ============================================
-- Product Images Browse Optimization
-- ============================================

-- Composite index for filtering by image_type (most common filter)
CREATE INDEX IF NOT EXISTS idx_product_images_type_product 
ON product_images(image_type, product_id);

-- Index for product_id lookups (for JOIN with products table)
CREATE INDEX IF NOT EXISTS idx_product_images_product_id 
ON product_images(product_id);

-- ============================================
-- Cutout Images Browse Optimization
-- ============================================

-- Index for matched product lookups (JOIN optimization)
CREATE INDEX IF NOT EXISTS idx_cutout_images_matched_product 
ON cutout_images(matched_product_id) 
WHERE matched_product_id IS NOT NULL;

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_cutout_images_synced_at 
ON cutout_images(synced_at DESC);

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_cutout_images_embedding_matched 
ON cutout_images(has_embedding, matched_product_id);

-- ============================================
-- Products Table Optimization (for JOIN queries)
-- ============================================

-- Index for category filtering (used in visual picker)
CREATE INDEX IF NOT EXISTS idx_products_category 
ON products(category) 
WHERE category IS NOT NULL;

-- Index for brand filtering (used in visual picker)
CREATE INDEX IF NOT EXISTS idx_products_brand_name 
ON products(brand_name) 
WHERE brand_name IS NOT NULL;

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_products_status_category 
ON products(status, category);

-- ============================================
-- Comments
-- ============================================

COMMENT ON INDEX idx_product_images_type_product IS 
'Optimizes product images browse queries filtered by image_type';

COMMENT ON INDEX idx_cutout_images_matched_product IS 
'Optimizes cutout browse queries that filter by matched products';

COMMENT ON INDEX idx_cutout_images_synced_at IS 
'Optimizes cutout browse queries with date range filters';
