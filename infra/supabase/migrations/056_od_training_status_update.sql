-- Migration 056: Update OD Training Status Constraint
--
-- Adds new status values for the training workflow:
-- - 'started': Job has started on RunPod
-- - 'downloading': Downloading dataset from Supabase storage
--
-- Full status flow: pending -> started -> downloading -> training -> completed/failed/cancelled

-- Drop existing status constraint
ALTER TABLE od_training_runs DROP CONSTRAINT IF EXISTS od_training_runs_status_check;

-- Add updated status constraint with new values
ALTER TABLE od_training_runs ADD CONSTRAINT od_training_runs_status_check
    CHECK (status IN (
        'pending',      -- Initial state, waiting to be picked up
        'started',      -- Job started on RunPod (NEW)
        'downloading',  -- Downloading dataset (NEW)
        'preparing',    -- Preparing training environment
        'queued',       -- Queued for training
        'training',     -- Training in progress
        'running',      -- Alternative to training (legacy)
        'completed',    -- Training completed successfully
        'failed',       -- Training failed with error
        'cancelled'     -- Training cancelled by user or system
    ));

COMMENT ON COLUMN od_training_runs.status IS 'Training status: pending -> started -> downloading -> training -> completed/failed/cancelled';
