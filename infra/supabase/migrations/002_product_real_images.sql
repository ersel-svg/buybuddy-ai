-- Product Real Images Table
-- Stores real product images from retail datasets for matching

-- ============================================
-- PRODUCT REAL IMAGES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS product_real_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,

    -- Image info
    image_url TEXT NOT NULL,
    image_path TEXT,  -- For local files
    source TEXT,  -- Where the image came from (e.g., 'retail_dataset', 'manual_upload')

    -- Metadata
    similarity FLOAT,  -- If matched via FAISS
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_real_images_product ON product_real_images(product_id);
CREATE INDEX IF NOT EXISTS idx_real_images_created ON product_real_images(created_at DESC);

-- Enable RLS
ALTER TABLE product_real_images ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users
CREATE POLICY "Allow all for authenticated" ON product_real_images FOR ALL USING (true);
