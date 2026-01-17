-- Migration: Remove unsupported CLIP models (clip-vit-b-16, clip-vit-b-32)
-- Reason: These models require torch 2.6+ due to safetensors compatibility issues
-- Remaining models: 7 (DINOv2 x3, DINOv3 x3, CLIP-ViT-L/14)

-- 1. Delete the unsupported model records from embedding_models
DELETE FROM public.embedding_models
WHERE model_type IN ('clip-vit-b-16', 'clip-vit-b-32');

-- 2. Update the check constraint on training_runs.base_model_type
ALTER TABLE public.training_runs
DROP CONSTRAINT IF EXISTS training_runs_base_model_type_check;

ALTER TABLE public.training_runs
ADD CONSTRAINT training_runs_base_model_type_check
CHECK (base_model_type IN (
    'dinov2-small', 'dinov2-base', 'dinov2-large',
    'dinov3-small', 'dinov3-base', 'dinov3-large',
    'clip-vit-l-14',
    'custom'
));

-- 3. Update any embedding_models check constraint if exists
-- (The original migration may not have a check constraint on model_type column)
DO $$
BEGIN
    -- Check if the constraint exists before trying to drop it
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_name = 'embedding_models' AND constraint_name LIKE '%model_type%check%'
    ) THEN
        ALTER TABLE public.embedding_models
        DROP CONSTRAINT embedding_models_model_type_check;
    END IF;
END $$;

-- Add comment for documentation
COMMENT ON TABLE public.embedding_models IS
'Embedding models registry. Supported models: dinov2-small, dinov2-base, dinov2-large, dinov3-small, dinov3-base, dinov3-large, clip-vit-l-14, custom';
