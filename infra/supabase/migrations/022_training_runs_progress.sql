-- Migration 022: Add progress tracking columns to training_runs
-- The RunPod training worker needs to report progress and messages

-- Add progress column (0.0 to 1.0)
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS progress REAL DEFAULT 0.0;

-- Add message column for status messages
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS message TEXT;

COMMENT ON COLUMN training_runs.progress IS 'Training progress 0.0 to 1.0';
COMMENT ON COLUMN training_runs.message IS 'Current status message from training worker';

-- Add metrics column for storing training metrics during run
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS metrics JSONB;

COMMENT ON COLUMN training_runs.metrics IS 'Current training metrics (loss, accuracy, etc.)';
