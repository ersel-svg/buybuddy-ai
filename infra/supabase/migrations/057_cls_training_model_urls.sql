-- Buybuddy AI Platform - CLS Training Model URLs
-- Migration 057: Add model_url and onnx_url columns to cls_training_runs
--
-- The CLS Training Worker needs to store the trained model URLs
-- directly in the training run record for easy access.

-- ============================================
-- ADD MODEL URL COLUMNS TO CLS_TRAINING_RUNS
-- ============================================

-- Add model_url column (storage path to the trained .pt model)
ALTER TABLE cls_training_runs
ADD COLUMN IF NOT EXISTS model_url TEXT;

-- Add onnx_url column (storage path to the exported ONNX model)
ALTER TABLE cls_training_runs
ADD COLUMN IF NOT EXISTS onnx_url TEXT;

COMMENT ON COLUMN cls_training_runs.model_url IS 'Storage path to the trained PyTorch model (e.g., cls-models/{id}/best_model.pt)';
COMMENT ON COLUMN cls_training_runs.onnx_url IS 'Storage path to the exported ONNX model (e.g., cls-models/{id}/model.onnx)';

-- ============================================
-- INDEX FOR COMPLETED RUNS WITH MODELS
-- ============================================

-- Index for finding completed training runs that have models
CREATE INDEX IF NOT EXISTS idx_cls_training_with_model
ON cls_training_runs(status, model_url)
WHERE status = 'completed' AND model_url IS NOT NULL;
