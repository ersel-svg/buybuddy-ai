-- Migration 060: Fix training_metrics_history foreign key for multi-table support
-- The original FK only references training_runs table, but we need to support:
-- - training_runs (embedding)
-- - cls_training_runs (classification)
-- - od_training_runs (object detection)
--
-- Solution: Drop the restrictive FK and allow all training types to write metrics
-- Integrity is maintained by worker code, not DB constraints

-- ============================================
-- DROP EXISTING FOREIGN KEY CONSTRAINT
-- ============================================
-- The constraint name follows Postgres naming convention: {table}_{column}_fkey
ALTER TABLE training_metrics_history
DROP CONSTRAINT IF EXISTS training_metrics_history_training_run_id_fkey;

-- Also try the auto-generated name pattern
DO $$
BEGIN
    -- Try to drop if exists with different naming patterns
    BEGIN
        ALTER TABLE training_metrics_history
        DROP CONSTRAINT training_metrics_history_training_run_id_fkey;
    EXCEPTION WHEN undefined_object THEN
        -- Constraint doesn't exist, that's fine
        NULL;
    END;
END $$;

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON COLUMN training_metrics_history.training_run_id IS
'References training_runs.id, cls_training_runs.id, or od_training_runs.id depending on training_type. No FK constraint to allow unified storage.';

COMMENT ON COLUMN training_metrics_history.training_type IS
'Training type: embedding (training_runs), cls (cls_training_runs), od (od_training_runs)';
