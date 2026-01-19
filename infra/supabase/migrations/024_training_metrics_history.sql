-- Migration 024: Add training_metrics_history table for detailed tracking
-- Stores per-epoch metrics for charts and analysis

-- ============================================
-- TRAINING METRICS HISTORY TABLE
-- Per-epoch metrics for progress tracking
-- ============================================
CREATE TABLE IF NOT EXISTS training_metrics_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Parent run
    training_run_id UUID NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,

    -- Epoch info
    epoch INTEGER NOT NULL,

    -- Training metrics
    train_loss REAL,
    arcface_loss REAL,
    triplet_loss REAL,
    domain_loss REAL,

    -- Validation metrics
    val_loss REAL,
    val_accuracy REAL,
    val_recall_at_1 REAL,
    val_recall_at_5 REAL,
    val_recall_at_10 REAL,

    -- Learning rate
    learning_rate REAL,

    -- Timing
    epoch_duration_seconds REAL,

    -- Curriculum phase (if using curriculum learning)
    curriculum_phase TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(training_run_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_metrics_history_run ON training_metrics_history(training_run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_history_epoch ON training_metrics_history(training_run_id, epoch);

COMMENT ON TABLE training_metrics_history IS 'Per-epoch training metrics for charts and progress tracking';

-- ============================================
-- ADD CURRENT_EPOCH TO TRAINING_RUNS IF MISSING
-- ============================================
ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS current_epoch INTEGER DEFAULT 0;

ALTER TABLE training_runs
ADD COLUMN IF NOT EXISTS total_epochs INTEGER;

-- ============================================
-- ENABLE REALTIME FOR LIVE UPDATES
-- ============================================
-- Enable realtime for training_runs table
ALTER PUBLICATION supabase_realtime ADD TABLE training_runs;

-- Enable realtime for metrics history (optional, for live charts)
ALTER PUBLICATION supabase_realtime ADD TABLE training_metrics_history;

-- ============================================
-- RLS POLICY
-- ============================================
ALTER TABLE training_metrics_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON training_metrics_history FOR ALL USING (true);

-- ============================================
-- HELPER FUNCTION: Get training progress summary
-- ============================================
CREATE OR REPLACE FUNCTION get_training_progress(run_id UUID)
RETURNS TABLE (
    current_epoch INTEGER,
    total_epochs INTEGER,
    progress REAL,
    status TEXT,
    latest_train_loss REAL,
    latest_val_loss REAL,
    best_val_loss REAL,
    best_recall_at_1 REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        tr.current_epoch,
        tr.total_epochs,
        tr.progress,
        tr.status,
        (SELECT train_loss FROM training_metrics_history
         WHERE training_run_id = run_id
         ORDER BY epoch DESC LIMIT 1),
        (SELECT val_loss FROM training_metrics_history
         WHERE training_run_id = run_id
         ORDER BY epoch DESC LIMIT 1),
        tr.best_val_loss,
        tr.best_val_recall_at_1
    FROM training_runs tr
    WHERE tr.id = run_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_training_progress IS 'Get summary of training progress for a run';
