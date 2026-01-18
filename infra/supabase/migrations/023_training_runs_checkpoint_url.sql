-- Add checkpoint_url column to training_runs table
-- This stores the Supabase Storage URL of the trained model checkpoint

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS checkpoint_url TEXT;

COMMENT ON COLUMN training_runs.checkpoint_url IS 'Supabase Storage URL of the best checkpoint file';
