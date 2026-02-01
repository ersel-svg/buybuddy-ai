-- Migration: Add local_od_sync_counts job type
-- Description: Allows OD resync jobs in local background worker
-- Date: 2026-01-30

-- Drop existing constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;

-- Add new constraint with all job types
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
    -- OD local background job types
    'local_bulk_add_to_dataset',
    'local_bulk_remove_from_dataset',
    'local_bulk_update_status',
    'local_bulk_delete_images',
    'local_export_dataset',
    'local_class_merge',
    'local_od_sync_counts',
    -- Product local background job types
    'local_bulk_update_products',
    'local_bulk_delete_products',
    'local_bulk_product_matcher',
    'local_bulk_add_products_to_dataset',
    -- Classification local background job types
    'local_cls_bulk_delete_images',
    'local_cls_bulk_add_to_dataset',
    'local_cls_bulk_remove_from_dataset',
    'local_cls_bulk_set_labels',
    'local_cls_bulk_clear_labels',
    'local_cls_bulk_update_tags',
    -- Cutout local background job types
    'local_cutout_sync',
    -- System job types
    'local_recalculate_counts',
    'local_data_cleanup'
));
