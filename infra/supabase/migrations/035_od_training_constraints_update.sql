-- Migration 035: Update OD Training Constraints
--
-- Updates the od_training_runs table constraints to support:
-- - New model types: d-fine
-- - Short model size codes: s, m, l, x

-- Drop existing constraints
ALTER TABLE od_training_runs DROP CONSTRAINT IF EXISTS od_training_runs_model_type_check;
ALTER TABLE od_training_runs DROP CONSTRAINT IF EXISTS od_training_runs_model_size_check;

-- Add updated model_type constraint (supports rt-detr and d-fine)
ALTER TABLE od_training_runs ADD CONSTRAINT od_training_runs_model_type_check
    CHECK (model_type IN ('rt-detr', 'd-fine', 'rf-detr', 'yolo-nas'));

-- Add updated model_size constraint (supports both full names and short codes)
ALTER TABLE od_training_runs ADD CONSTRAINT od_training_runs_model_size_check
    CHECK (model_size IN ('s', 'm', 'l', 'x', 'small', 'medium', 'large', 'xlarge'));

-- Update od_trained_models constraints as well
ALTER TABLE od_trained_models DROP CONSTRAINT IF EXISTS od_trained_models_model_type_check;
ALTER TABLE od_trained_models ADD CONSTRAINT od_trained_models_model_type_check
    CHECK (model_type IN ('rt-detr', 'd-fine', 'rf-detr', 'yolo-nas'));
