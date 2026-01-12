-- Buybuddy AI Platform - Initial Database Schema
-- Run this in Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- PRODUCTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identifiers
    barcode TEXT UNIQUE NOT NULL,
    video_id INTEGER,
    video_url TEXT,

    -- Metadata (from Gemini)
    brand_name TEXT,
    sub_brand TEXT,
    product_name TEXT,
    variant_flavor TEXT,
    category TEXT,
    container_type TEXT,
    net_quantity TEXT,
    nutrition_facts JSONB,
    claims TEXT[],
    grounding_prompt TEXT,
    visibility_score INTEGER,

    -- Frames
    frame_count INTEGER DEFAULT 0,
    frames_path TEXT,
    primary_image_url TEXT,

    -- Status
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'needs_matching', 'ready', 'rejected')),

    -- Optimistic locking
    version INTEGER DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_products_status ON products(status);
CREATE INDEX IF NOT EXISTS idx_products_barcode ON products(barcode);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand_name);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_created ON products(created_at DESC);

-- ============================================
-- VIDEOS TABLE (from Buybuddy API sync)
-- ============================================
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    barcode TEXT NOT NULL,
    video_url TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    product_id UUID REFERENCES products(id) ON DELETE SET NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_barcode ON videos(barcode);

-- ============================================
-- PRODUCT IMAGES
-- ============================================
CREATE TABLE IF NOT EXISTS product_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,

    image_path TEXT NOT NULL,
    image_url TEXT,
    image_type TEXT NOT NULL CHECK (image_type IN ('synthetic', 'real', 'augmented')),
    source TEXT NOT NULL CHECK (source IN ('video_frame', 'matching', 'augmentation')),

    -- For ordering frames
    frame_index INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_images_product ON product_images(product_id);
CREATE INDEX IF NOT EXISTS idx_images_type ON product_images(image_type);

-- ============================================
-- JOBS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL CHECK (type IN ('video_processing', 'augmentation', 'training', 'embedding_extraction', 'matching')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled')),

    -- Config
    config JSONB NOT NULL DEFAULT '{}',

    -- Progress
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    current_step TEXT,

    -- Results
    result JSONB,
    error TEXT,

    -- Runpod
    runpod_job_id TEXT,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);

-- ============================================
-- DATASETS
-- ============================================
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    product_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_products (
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (dataset_id, product_id)
);

CREATE INDEX IF NOT EXISTS idx_dataset_products_dataset ON dataset_products(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_products_product ON dataset_products(product_id);

-- ============================================
-- TRAINING JOBS (extends jobs)
-- ============================================
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,

    -- Config
    epochs INTEGER DEFAULT 30,
    epochs_completed INTEGER DEFAULT 0,
    batch_size INTEGER DEFAULT 32,
    learning_rate FLOAT DEFAULT 0.0001,

    -- Results
    final_loss FLOAT,
    best_accuracy FLOAT,
    checkpoint_url TEXT,

    -- Metrics history
    metrics JSONB
);

-- ============================================
-- MODEL ARTIFACTS
-- ============================================
CREATE TABLE IF NOT EXISTS model_artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_job_id UUID REFERENCES training_jobs(id) ON DELETE SET NULL,

    name TEXT NOT NULL,
    version TEXT NOT NULL,
    checkpoint_url TEXT NOT NULL,

    embedding_dim INTEGER NOT NULL DEFAULT 512,
    num_classes INTEGER,

    -- Metrics
    final_loss FLOAT,
    accuracy FLOAT,

    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_active ON model_artifacts(is_active) WHERE is_active = true;

-- ============================================
-- EMBEDDING INDEXES
-- ============================================
CREATE TABLE IF NOT EXISTS embedding_indexes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    model_artifact_id UUID REFERENCES model_artifacts(id) ON DELETE SET NULL,

    vector_count INTEGER DEFAULT 0,
    index_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- PRODUCT MATCHES (for matching UI)
-- ============================================
CREATE TABLE IF NOT EXISTS product_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,

    candidate_image_path TEXT NOT NULL,
    candidate_image_url TEXT,
    similarity FLOAT NOT NULL,

    is_approved BOOLEAN,
    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matches_product ON product_matches(product_id);
CREATE INDEX IF NOT EXISTS idx_matches_approved ON product_matches(is_approved);

-- ============================================
-- RESOURCE LOCKS (for multi-user editing)
-- ============================================
CREATE TABLE IF NOT EXISTS resource_locks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type TEXT NOT NULL CHECK (resource_type IN ('product', 'dataset')),
    resource_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    user_email TEXT,
    locked_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    UNIQUE(resource_type, resource_id)
);

CREATE INDEX IF NOT EXISTS idx_locks_resource ON resource_locks(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_locks_expires ON resource_locks(expires_at);

-- ============================================
-- TRIGGERS
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY['products', 'videos', 'jobs', 'datasets', 'embedding_indexes']
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS %I_updated_at ON %I;
            CREATE TRIGGER %I_updated_at
                BEFORE UPDATE ON %I
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        ', t, t, t, t);
    END LOOP;
END $$;

-- ============================================
-- RLS POLICIES (Row Level Security)
-- ============================================
-- Enable RLS on all tables
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_indexes ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated users (internal tool)
CREATE POLICY "Allow all for authenticated" ON products FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON videos FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON jobs FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON datasets FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON model_artifacts FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON embedding_indexes FOR ALL USING (true);

-- ============================================
-- STORAGE BUCKETS
-- Run these in Supabase Dashboard > Storage
-- ============================================
-- 1. Create bucket: "frames" (public)
-- 2. Create bucket: "models" (private)
-- 3. Create bucket: "augmented-images" (public)
-- 4. Create bucket: "real-images" (public)

-- Example bucket policy (run in SQL):
-- INSERT INTO storage.buckets (id, name, public) VALUES ('frames', 'frames', true);
-- INSERT INTO storage.buckets (id, name, public) VALUES ('models', 'models', false);
-- INSERT INTO storage.buckets (id, name, public) VALUES ('augmented-images', 'augmented-images', true);
-- INSERT INTO storage.buckets (id, name, public) VALUES ('real-images', 'real-images', true);
