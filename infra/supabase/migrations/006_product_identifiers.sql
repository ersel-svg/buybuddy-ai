-- Migration: Add product_identifiers table and custom_fields column
-- Supports multiple identifier types with primary designation
-- Version: 006
-- Date: 2025-01-14

-- ============================================
-- 1. CREATE PRODUCT IDENTIFIERS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS product_identifiers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,

    -- Identifier type and value
    identifier_type TEXT NOT NULL CHECK (
        identifier_type IN ('barcode', 'short_code', 'sku', 'upc', 'ean', 'custom')
    ),
    identifier_value TEXT NOT NULL,

    -- Custom type label (only for 'custom' type)
    custom_label TEXT,

    -- Primary flag (only one per product should be primary)
    is_primary BOOLEAN DEFAULT false,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure unique identifier values per product per type
    UNIQUE(product_id, identifier_type, identifier_value)
);

-- ============================================
-- 2. INDEXES FOR EFFICIENT LOOKUPS
-- ============================================

-- Index for product lookups
CREATE INDEX IF NOT EXISTS idx_product_identifiers_product
ON product_identifiers(product_id);

-- Index for searching by identifier type and value
CREATE INDEX IF NOT EXISTS idx_product_identifiers_type_value
ON product_identifiers(identifier_type, identifier_value);

-- Index for searching by identifier value only (for global search)
CREATE INDEX IF NOT EXISTS idx_product_identifiers_value
ON product_identifiers(identifier_value);

-- Partial index for primary identifiers
CREATE INDEX IF NOT EXISTS idx_product_identifiers_primary
ON product_identifiers(product_id) WHERE is_primary = true;

-- ============================================
-- 3. ADD CUSTOM FIELDS COLUMN TO PRODUCTS
-- ============================================

ALTER TABLE products ADD COLUMN IF NOT EXISTS custom_fields JSONB DEFAULT '{}';

-- GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_products_custom_fields
ON products USING GIN(custom_fields);

-- ============================================
-- 4. MIGRATE EXISTING DATA
-- ============================================

-- Migrate existing barcode column to product_identifiers table
-- Set as primary identifier
INSERT INTO product_identifiers (product_id, identifier_type, identifier_value, is_primary)
SELECT
    id as product_id,
    'barcode' as identifier_type,
    barcode as identifier_value,
    true as is_primary
FROM products
WHERE barcode IS NOT NULL AND barcode != ''
ON CONFLICT (product_id, identifier_type, identifier_value) DO NOTHING;

-- Migrate existing identifiers.sku_model_code to product_identifiers
INSERT INTO product_identifiers (product_id, identifier_type, identifier_value, is_primary)
SELECT
    id as product_id,
    'sku' as identifier_type,
    identifiers->>'sku_model_code' as identifier_value,
    false as is_primary
FROM products
WHERE identifiers->>'sku_model_code' IS NOT NULL
  AND identifiers->>'sku_model_code' != ''
ON CONFLICT (product_id, identifier_type, identifier_value) DO NOTHING;

-- Migrate existing identifiers.barcode (secondary) to product_identifiers
-- Only if different from primary barcode
INSERT INTO product_identifiers (product_id, identifier_type, identifier_value, is_primary)
SELECT
    id as product_id,
    'barcode' as identifier_type,
    identifiers->>'barcode' as identifier_value,
    false as is_primary
FROM products
WHERE identifiers->>'barcode' IS NOT NULL
  AND identifiers->>'barcode' != ''
  AND (barcode IS NULL OR identifiers->>'barcode' != barcode)
ON CONFLICT (product_id, identifier_type, identifier_value) DO NOTHING;

-- ============================================
-- 5. TRIGGER: ENFORCE SINGLE PRIMARY PER PRODUCT
-- ============================================

-- Function to enforce single primary identifier per product
CREATE OR REPLACE FUNCTION enforce_single_primary_identifier()
RETURNS TRIGGER AS $$
BEGIN
    -- If setting this as primary, unset others for the same product
    IF NEW.is_primary = true THEN
        UPDATE product_identifiers
        SET is_primary = false, updated_at = NOW()
        WHERE product_id = NEW.product_id
          AND id != NEW.id
          AND is_primary = true;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS trg_enforce_single_primary ON product_identifiers;

-- Create trigger
CREATE TRIGGER trg_enforce_single_primary
    BEFORE INSERT OR UPDATE ON product_identifiers
    FOR EACH ROW
    WHEN (NEW.is_primary = true)
    EXECUTE FUNCTION enforce_single_primary_identifier();

-- ============================================
-- 6. TRIGGER: UPDATE TIMESTAMP
-- ============================================

-- Use existing update_updated_at function if available, otherwise create
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS product_identifiers_updated_at ON product_identifiers;

-- Create trigger for updated_at
CREATE TRIGGER product_identifiers_updated_at
    BEFORE UPDATE ON product_identifiers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================
-- 7. ROW LEVEL SECURITY
-- ============================================

ALTER TABLE product_identifiers ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users
DROP POLICY IF EXISTS "Allow all for authenticated" ON product_identifiers;
CREATE POLICY "Allow all for authenticated" ON product_identifiers
FOR ALL USING (true);

-- ============================================
-- 8. HELPER FUNCTIONS
-- ============================================

-- Get primary identifier for a product
CREATE OR REPLACE FUNCTION get_primary_identifier(p_product_id UUID)
RETURNS TABLE(identifier_type TEXT, identifier_value TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT pi.identifier_type, pi.identifier_value
    FROM product_identifiers pi
    WHERE pi.product_id = p_product_id AND pi.is_primary = true
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Search products by any identifier value (partial match)
CREATE OR REPLACE FUNCTION search_products_by_identifier(p_value TEXT)
RETURNS TABLE(product_id UUID, identifier_type TEXT, identifier_value TEXT) AS $$
BEGIN
    RETURN QUERY
    SELECT pi.product_id, pi.identifier_type, pi.identifier_value
    FROM product_identifiers pi
    WHERE pi.identifier_value ILIKE '%' || p_value || '%';
END;
$$ LANGUAGE plpgsql;

-- Get all identifiers for a product
CREATE OR REPLACE FUNCTION get_product_identifiers(p_product_id UUID)
RETURNS TABLE(
    id UUID,
    identifier_type TEXT,
    identifier_value TEXT,
    custom_label TEXT,
    is_primary BOOLEAN,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pi.id,
        pi.identifier_type,
        pi.identifier_value,
        pi.custom_label,
        pi.is_primary,
        pi.created_at
    FROM product_identifiers pi
    WHERE pi.product_id = p_product_id
    ORDER BY pi.is_primary DESC, pi.created_at ASC;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 9. COMMENTS
-- ============================================

COMMENT ON TABLE product_identifiers IS 'Stores multiple identifiers (barcode, sku, upc, etc.) for each product';
COMMENT ON COLUMN product_identifiers.identifier_type IS 'Type of identifier: barcode, short_code, sku, upc, ean, or custom';
COMMENT ON COLUMN product_identifiers.custom_label IS 'Custom label for identifiers of type "custom"';
COMMENT ON COLUMN product_identifiers.is_primary IS 'Whether this is the primary identifier for the product (only one per product)';
COMMENT ON COLUMN products.custom_fields IS 'Arbitrary key-value pairs for user-defined product attributes';
