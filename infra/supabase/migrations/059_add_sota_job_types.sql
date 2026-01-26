-- Migration: Add SOTA job types for product operations
-- Description: Add local_bulk_delete_products and local_bulk_product_matcher job types
-- Date: 2026-01-26

-- Drop existing constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;

-- Add new constraint with all job types including new SOTA types
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
    -- Existing local background job types
    'local_bulk_add_to_dataset',
    'local_bulk_remove_from_dataset',
    'local_bulk_update_status',
    'local_bulk_delete_images',
    'local_export_dataset',
    'local_bulk_update_products',
    'local_recalculate_counts',
    'local_data_cleanup',
    -- NEW: SOTA product operation types
    'local_bulk_delete_products',
    'local_bulk_product_matcher'
));
