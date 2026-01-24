-- Migration: Local Background Jobs Infrastructure
-- Description: Add support for local (non-Runpod) background job processing
-- Date: 2024-01-24

-- ============================================
-- 1. Add new local job types to jobs table
-- ============================================

-- Drop existing constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;

-- Add new constraint with local job types
ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN (
    -- Existing Runpod job types
    'video_processing',
    'augmentation',
    'training',
    'embedding_extraction',
    'matching',
    'roboflow_import',
    'od_annotation',
    'od_training',
    'cls_annotation',
    'cls_training',
    'buybuddy_sync',
    -- NEW: Local background job types (CPU-bound, no GPU needed)
    'local_bulk_add_to_dataset',
    'local_bulk_remove_from_dataset',
    'local_bulk_update_status',
    'local_bulk_delete_images',
    'local_export_dataset',
    'local_bulk_update_products',
    'local_recalculate_counts',
    'local_data_cleanup'
));

-- ============================================
-- 2. Add worker fields for job locking
-- ============================================

-- Worker ID field - identifies which worker is processing the job
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS worker_id TEXT;

-- Lock timestamp - when the job was claimed by a worker
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS locked_at TIMESTAMPTZ;

-- ============================================
-- 3. Add indexes for efficient polling
-- ============================================

-- Index for local job worker polling (pending local jobs)
CREATE INDEX IF NOT EXISTS idx_jobs_local_pending
ON jobs(created_at ASC)
WHERE type LIKE 'local_%' AND status = 'pending' AND worker_id IS NULL;

-- Index for finding stale jobs (locked but not progressing)
CREATE INDEX IF NOT EXISTS idx_jobs_stale_detection
ON jobs(locked_at)
WHERE status = 'running' AND worker_id IS NOT NULL;

-- ============================================
-- 4. Add comments for documentation
-- ============================================

COMMENT ON COLUMN jobs.worker_id IS 'ID of the worker processing this job (for local jobs)';
COMMENT ON COLUMN jobs.locked_at IS 'Timestamp when job was claimed by a worker';
