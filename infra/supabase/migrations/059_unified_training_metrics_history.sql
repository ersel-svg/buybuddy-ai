-- Migration 059: Extend training_metrics_history for all training types (CLS, OD, Embedding)
-- This allows using a single table for all training metrics with type-specific columns

-- ============================================
-- ADD TRAINING TYPE COLUMN
-- Existing rows default to 'embedding'
-- ============================================
ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS training_type VARCHAR(20) DEFAULT 'embedding';

-- Update existing rows to have explicit training_type
UPDATE training_metrics_history
SET training_type = 'embedding'
WHERE training_type IS NULL;

-- Make it NOT NULL after setting defaults
ALTER TABLE training_metrics_history
ALTER COLUMN training_type SET NOT NULL;

-- ============================================
-- ADD CLS-SPECIFIC COLUMNS
-- ============================================
-- val_accuracy already exists in original schema
ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS val_f1 REAL;

ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS val_top5_accuracy REAL;

-- ============================================
-- ADD OD-SPECIFIC COLUMNS
-- ============================================
ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS map REAL;

ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS map50 REAL;

ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS map75 REAL;

-- ============================================
-- ADD FLEXIBLE EXTRA METRICS JSONB
-- For any additional metrics not in fixed columns
-- ============================================
ALTER TABLE training_metrics_history
ADD COLUMN IF NOT EXISTS extra_metrics JSONB DEFAULT '{}';

-- ============================================
-- CREATE NEW UNIQUE INDEX FOR ALL TRAINING TYPES
-- This allows same training_run_id with different training_types
-- ============================================
-- Drop old unique constraint if exists (it only covers training_run_id, epoch)
-- Must drop CONSTRAINT not INDEX since it's a UNIQUE constraint
ALTER TABLE training_metrics_history
DROP CONSTRAINT IF EXISTS training_metrics_history_training_run_id_epoch_key;

-- Create new unique index including training_type
CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_run_type_epoch
ON training_metrics_history(training_run_id, training_type, epoch);

-- ============================================
-- ADD INDEXES FOR EFFICIENT QUERIES
-- ============================================
CREATE INDEX IF NOT EXISTS idx_metrics_training_type
ON training_metrics_history(training_type);

CREATE INDEX IF NOT EXISTS idx_metrics_run_type
ON training_metrics_history(training_run_id, training_type);

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON COLUMN training_metrics_history.training_type IS 'Type of training: embedding, cls, od';
COMMENT ON COLUMN training_metrics_history.val_f1 IS 'Validation F1 score (CLS)';
COMMENT ON COLUMN training_metrics_history.map IS 'Mean Average Precision (OD)';
COMMENT ON COLUMN training_metrics_history.map50 IS 'mAP at IoU=0.50 (OD)';
COMMENT ON COLUMN training_metrics_history.map75 IS 'mAP at IoU=0.75 (OD)';
COMMENT ON COLUMN training_metrics_history.extra_metrics IS 'Additional metrics as JSONB for flexibility';
