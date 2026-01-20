-- Migration: Add roboflow_import to jobs type constraint
-- This allows using the jobs table for Roboflow dataset import tracking

-- Drop the existing constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;

-- Add new constraint with roboflow_import type
ALTER TABLE jobs ADD CONSTRAINT jobs_type_check
    CHECK (type IN ('video_processing', 'augmentation', 'training', 'embedding_extraction', 'matching', 'roboflow_import'));
