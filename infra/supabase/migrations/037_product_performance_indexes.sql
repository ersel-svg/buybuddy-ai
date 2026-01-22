-- Migration: Add performance indexes for Products table
-- These indexes optimize filtering, sorting, and search operations

-- ===========================================
-- Single Column Indexes (for frequently filtered columns)
-- ===========================================

-- Sub brand filtering
CREATE INDEX IF NOT EXISTS idx_products_sub_brand
ON products(sub_brand)
WHERE sub_brand IS NOT NULL;

-- Product name filtering
CREATE INDEX IF NOT EXISTS idx_products_product_name
ON products(product_name)
WHERE product_name IS NOT NULL;

-- Variant/Flavor filtering
CREATE INDEX IF NOT EXISTS idx_products_variant_flavor
ON products(variant_flavor)
WHERE variant_flavor IS NOT NULL;

-- Container type filtering
CREATE INDEX IF NOT EXISTS idx_products_container_type
ON products(container_type)
WHERE container_type IS NOT NULL;

-- Net quantity filtering
CREATE INDEX IF NOT EXISTS idx_products_net_quantity
ON products(net_quantity)
WHERE net_quantity IS NOT NULL;

-- Manufacturer country filtering
CREATE INDEX IF NOT EXISTS idx_products_manufacturer_country
ON products(manufacturer_country)
WHERE manufacturer_country IS NOT NULL;

-- ===========================================
-- GIN Indexes (for array and JSONB fields)
-- ===========================================

-- Claims array - for overlap queries (claims && ARRAY['claim1', 'claim2'])
CREATE INDEX IF NOT EXISTS idx_products_claims_gin
ON products USING GIN(claims);

-- Issues detected array - for overlap and containment queries
CREATE INDEX IF NOT EXISTS idx_products_issues_gin
ON products USING GIN(issues_detected);

-- Pack configuration JSONB - for type queries
CREATE INDEX IF NOT EXISTS idx_products_pack_config_gin
ON products USING GIN(pack_configuration);

-- Nutrition facts JSONB - for has/doesn't have queries
CREATE INDEX IF NOT EXISTS idx_products_nutrition_gin
ON products USING GIN(nutrition_facts);

-- ===========================================
-- Composite Indexes (for common filter combinations)
-- ===========================================

-- Status + Category (very common filter combination)
CREATE INDEX IF NOT EXISTS idx_products_status_category
ON products(status, category);

-- Brand + Category (for brand filtering within category)
CREATE INDEX IF NOT EXISTS idx_products_brand_category
ON products(brand_name, category);

-- Status + Brand (for status filtering by brand)
CREATE INDEX IF NOT EXISTS idx_products_status_brand
ON products(status, brand_name);

-- Created at + Status (for recent products by status)
CREATE INDEX IF NOT EXISTS idx_products_created_status
ON products(created_at DESC, status);

-- ===========================================
-- Full-Text Search Index
-- ===========================================

-- Combined search on barcode, product_name, brand_name
CREATE INDEX IF NOT EXISTS idx_products_search_fts
ON products USING GIN(
    to_tsvector('english',
        COALESCE(barcode, '') || ' ' ||
        COALESCE(product_name, '') || ' ' ||
        COALESCE(brand_name, '')
    )
);

-- ===========================================
-- Partial Indexes (for common boolean conditions)
-- ===========================================

-- Products with video
CREATE INDEX IF NOT EXISTS idx_products_has_video
ON products(id)
WHERE video_url IS NOT NULL;

-- Products with primary image
CREATE INDEX IF NOT EXISTS idx_products_has_image
ON products(id)
WHERE primary_image_url IS NOT NULL;

-- Products with nutrition facts
CREATE INDEX IF NOT EXISTS idx_products_has_nutrition
ON products(id)
WHERE nutrition_facts IS NOT NULL AND nutrition_facts != '{}'::jsonb;

-- Products with issues (text[] array)
CREATE INDEX IF NOT EXISTS idx_products_has_issues
ON products(id)
WHERE issues_detected IS NOT NULL AND array_length(issues_detected, 1) > 0;

-- Products with grounding prompt
CREATE INDEX IF NOT EXISTS idx_products_has_prompt
ON products(id)
WHERE grounding_prompt IS NOT NULL;

-- Products with marketing description
CREATE INDEX IF NOT EXISTS idx_products_has_description
ON products(id)
WHERE marketing_description IS NOT NULL;

-- ===========================================
-- Sorting Indexes
-- ===========================================

-- Updated at descending (for "recently modified")
CREATE INDEX IF NOT EXISTS idx_products_updated_at_desc
ON products(updated_at DESC NULLS LAST);

-- Frame count (for sorting by frame count)
CREATE INDEX IF NOT EXISTS idx_products_frame_count
ON products(frame_count);

-- Visibility score (for sorting by score)
CREATE INDEX IF NOT EXISTS idx_products_visibility_score
ON products(visibility_score)
WHERE visibility_score IS NOT NULL;

-- ===========================================
-- Dataset Operations Index
-- ===========================================

-- Composite index for dataset_products lookups
CREATE INDEX IF NOT EXISTS idx_dataset_products_dataset_product
ON dataset_products(dataset_id, product_id);

-- ===========================================
-- Comments
-- ===========================================

COMMENT ON INDEX idx_products_claims_gin IS 'GIN index for claims array overlap queries';
COMMENT ON INDEX idx_products_search_fts IS 'Full-text search index combining barcode, product_name, and brand_name';
COMMENT ON INDEX idx_products_status_category IS 'Composite index for common status+category filter combination';
