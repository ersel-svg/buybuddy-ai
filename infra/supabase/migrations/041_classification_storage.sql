-- Buybuddy AI Platform - Classification Module Storage
-- Migration 041: Storage Buckets for Classification
--
-- This migration creates storage buckets for the Classification platform:
-- 1. cls-images - Public bucket for classification images
-- 2. cls-models - Private bucket for trained model checkpoints
-- 3. cls-exports - Private bucket for dataset exports

-- ============================================
-- STORAGE BUCKETS
-- ============================================

-- Create cls-images bucket (public - images need to be viewable)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'cls-images',
    'cls-images',
    true,
    52428800,  -- 50MB max file size
    ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/gif']
)
ON CONFLICT (id) DO UPDATE SET
    public = true,
    file_size_limit = 52428800,
    allowed_mime_types = ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

-- Create cls-models bucket (private - model files are sensitive)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'cls-models',
    'cls-models',
    false,
    5368709120,  -- 5GB max file size (models can be large)
    ARRAY['application/octet-stream', 'application/zip', 'application/x-tar']
)
ON CONFLICT (id) DO UPDATE SET
    public = false,
    file_size_limit = 5368709120,
    allowed_mime_types = ARRAY['application/octet-stream', 'application/zip', 'application/x-tar'];

-- Create cls-exports bucket (private - dataset exports)
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'cls-exports',
    'cls-exports',
    false,
    10737418240,  -- 10GB max file size (exports can be large)
    ARRAY['application/zip', 'application/x-tar', 'application/gzip']
)
ON CONFLICT (id) DO UPDATE SET
    public = false,
    file_size_limit = 10737418240,
    allowed_mime_types = ARRAY['application/zip', 'application/x-tar', 'application/gzip'];

-- ============================================
-- STORAGE POLICIES - cls-images (public bucket)
-- ============================================

-- Allow public read access to cls-images
DROP POLICY IF EXISTS "Public read access for cls-images" ON storage.objects;
CREATE POLICY "Public read access for cls-images"
ON storage.objects FOR SELECT
USING (bucket_id = 'cls-images');

-- Allow authenticated users to upload to cls-images
DROP POLICY IF EXISTS "Authenticated upload to cls-images" ON storage.objects;
CREATE POLICY "Authenticated upload to cls-images"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'cls-images');

-- Allow authenticated users to update their uploads in cls-images
DROP POLICY IF EXISTS "Authenticated update cls-images" ON storage.objects;
CREATE POLICY "Authenticated update cls-images"
ON storage.objects FOR UPDATE
USING (bucket_id = 'cls-images');

-- Allow authenticated users to delete from cls-images
DROP POLICY IF EXISTS "Authenticated delete from cls-images" ON storage.objects;
CREATE POLICY "Authenticated delete from cls-images"
ON storage.objects FOR DELETE
USING (bucket_id = 'cls-images');

-- ============================================
-- STORAGE POLICIES - cls-models (private bucket)
-- ============================================

-- Allow authenticated read access to cls-models
DROP POLICY IF EXISTS "Authenticated read cls-models" ON storage.objects;
CREATE POLICY "Authenticated read cls-models"
ON storage.objects FOR SELECT
USING (bucket_id = 'cls-models');

-- Allow authenticated upload to cls-models
DROP POLICY IF EXISTS "Authenticated upload to cls-models" ON storage.objects;
CREATE POLICY "Authenticated upload to cls-models"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'cls-models');

-- Allow authenticated update in cls-models
DROP POLICY IF EXISTS "Authenticated update cls-models" ON storage.objects;
CREATE POLICY "Authenticated update cls-models"
ON storage.objects FOR UPDATE
USING (bucket_id = 'cls-models');

-- Allow authenticated delete from cls-models
DROP POLICY IF EXISTS "Authenticated delete from cls-models" ON storage.objects;
CREATE POLICY "Authenticated delete from cls-models"
ON storage.objects FOR DELETE
USING (bucket_id = 'cls-models');

-- ============================================
-- STORAGE POLICIES - cls-exports (private bucket)
-- ============================================

-- Allow authenticated read access to cls-exports
DROP POLICY IF EXISTS "Authenticated read cls-exports" ON storage.objects;
CREATE POLICY "Authenticated read cls-exports"
ON storage.objects FOR SELECT
USING (bucket_id = 'cls-exports');

-- Allow authenticated upload to cls-exports
DROP POLICY IF EXISTS "Authenticated upload to cls-exports" ON storage.objects;
CREATE POLICY "Authenticated upload to cls-exports"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'cls-exports');

-- Allow authenticated update in cls-exports
DROP POLICY IF EXISTS "Authenticated update cls-exports" ON storage.objects;
CREATE POLICY "Authenticated update cls-exports"
ON storage.objects FOR UPDATE
USING (bucket_id = 'cls-exports');

-- Allow authenticated delete from cls-exports
DROP POLICY IF EXISTS "Authenticated delete from cls-exports" ON storage.objects;
CREATE POLICY "Authenticated delete from cls-exports"
ON storage.objects FOR DELETE
USING (bucket_id = 'cls-exports');

-- ============================================
-- HELPER FUNCTIONS FOR STORAGE
-- ============================================

-- Generate signed URL for model download (for private bucket)
CREATE OR REPLACE FUNCTION get_cls_model_download_url(p_model_id UUID, p_expires_in INTEGER DEFAULT 3600)
RETURNS TEXT AS $$
DECLARE
    v_checkpoint_url TEXT;
    v_storage_path TEXT;
BEGIN
    SELECT checkpoint_url INTO v_checkpoint_url
    FROM cls_trained_models
    WHERE id = p_model_id;

    IF v_checkpoint_url IS NULL THEN
        RETURN NULL;
    END IF;

    -- If it's already a full URL (external), return as-is
    IF v_checkpoint_url LIKE 'http%' THEN
        RETURN v_checkpoint_url;
    END IF;

    -- Otherwise, it's a storage path - generate signed URL
    -- Note: This would need to be implemented via the API
    RETURN v_checkpoint_url;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_cls_model_download_url IS 'Get download URL for a trained model checkpoint';
