-- Migration: Add Roboflow trained models support
-- This table stores metadata for trained models imported from Roboflow
-- These models are "closed-vocabulary" - they detect fixed classes without text prompts

CREATE TABLE IF NOT EXISTS od_roboflow_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Model identity
  name VARCHAR(255) NOT NULL,
  display_name VARCHAR(255) NOT NULL,
  description TEXT,

  -- Source tracking (optional - for provenance)
  roboflow_workspace VARCHAR(255),
  roboflow_project VARCHAR(255),
  roboflow_version INTEGER,

  -- Model architecture: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'rf-detr', 'rt-detr'
  architecture VARCHAR(50) NOT NULL,

  -- Fixed classes (closed-vocabulary) - e.g., ["cooler", "object", "shelf", "void"]
  classes JSONB NOT NULL,
  class_count INTEGER NOT NULL,

  -- Performance metrics
  map FLOAT,          -- Mean Average Precision
  map_50 FLOAT,       -- mAP at IoU=0.5
  precision_score FLOAT,
  recall_score FLOAT,

  -- Storage - Supabase Storage URL for the .pt checkpoint file
  checkpoint_url TEXT NOT NULL,
  file_size_bytes BIGINT,

  -- Status
  is_active BOOLEAN DEFAULT false,  -- Only active models shown in AI panel
  is_default BOOLEAN DEFAULT false, -- Default model for new users

  -- Metadata
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for quick lookup of active models
CREATE INDEX IF NOT EXISTS idx_od_roboflow_models_active
  ON od_roboflow_models(is_active) WHERE is_active = true;

-- Index for architecture-based filtering
CREATE INDEX IF NOT EXISTS idx_od_roboflow_models_architecture
  ON od_roboflow_models(architecture);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_od_roboflow_models_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER od_roboflow_models_updated_at
  BEFORE UPDATE ON od_roboflow_models
  FOR EACH ROW
  EXECUTE FUNCTION update_od_roboflow_models_updated_at();

-- Comment for documentation
COMMENT ON TABLE od_roboflow_models IS 'Stores trained models imported from Roboflow for use in AI annotation. These are closed-vocabulary models that detect fixed classes.';
COMMENT ON COLUMN od_roboflow_models.classes IS 'JSON array of class names this model can detect, e.g., ["cooler", "object", "shelf", "void"]';
COMMENT ON COLUMN od_roboflow_models.checkpoint_url IS 'Supabase Storage public URL for the PyTorch .pt checkpoint file';
