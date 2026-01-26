-- Migration: Add Workflow Inference Jobs Table
-- Date: 2026-01-26
-- Purpose: Track async GPU inference jobs for workflow nodes

-- Create wf_inference_jobs table
CREATE TABLE IF NOT EXISTS wf_inference_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Workflow context
    execution_id UUID NOT NULL REFERENCES wf_executions(id) ON DELETE CASCADE,
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,  -- Node ID in workflow definition

    -- Job details
    task TEXT NOT NULL CHECK (task IN ('detection', 'classification', 'embedding', 'segmentation')),
    model_id TEXT NOT NULL,
    model_source TEXT NOT NULL DEFAULT 'pretrained' CHECK (model_source IN ('pretrained', 'trained')),
    runpod_job_id TEXT,

    -- Status tracking
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled')),
    progress INT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),

    -- Timing
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INT,

    -- Results (lightweight - large data goes to wf_executions.output_data)
    result JSONB,
    error_message TEXT,

    -- Metadata (worker info, GPU stats, etc.)
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_execution ON wf_inference_jobs(execution_id);
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_workflow ON wf_inference_jobs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_status ON wf_inference_jobs(status);
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_runpod ON wf_inference_jobs(runpod_job_id) WHERE runpod_job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_submitted ON wf_inference_jobs(submitted_at DESC);
CREATE INDEX IF NOT EXISTS idx_wf_inference_jobs_node ON wf_inference_jobs(execution_id, node_id);

-- Add columns to wf_executions for job tracking
ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS inference_jobs_pending INT DEFAULT 0;
ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS inference_jobs_completed INT DEFAULT 0;
ALTER TABLE wf_executions ADD COLUMN IF NOT EXISTS inference_jobs_failed INT DEFAULT 0;

-- Add trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_wf_inference_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER wf_inference_jobs_updated_at
    BEFORE UPDATE ON wf_inference_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_wf_inference_jobs_updated_at();

-- Comments for documentation
COMMENT ON TABLE wf_inference_jobs IS 'Tracks async GPU inference jobs for workflow model nodes';
COMMENT ON COLUMN wf_inference_jobs.execution_id IS 'Parent workflow execution';
COMMENT ON COLUMN wf_inference_jobs.node_id IS 'Node ID from workflow definition (detection_1, embed_1, etc.)';
COMMENT ON COLUMN wf_inference_jobs.task IS 'Inference task type: detection, classification, embedding, segmentation';
COMMENT ON COLUMN wf_inference_jobs.model_id IS 'Model identifier (yolo11n, dinov2-base, vit-base, etc.)';
COMMENT ON COLUMN wf_inference_jobs.model_source IS 'Model source: pretrained or trained (custom)';
COMMENT ON COLUMN wf_inference_jobs.runpod_job_id IS 'RunPod serverless job ID for tracking';
COMMENT ON COLUMN wf_inference_jobs.status IS 'Job status: pending → queued → running → completed/failed';
COMMENT ON COLUMN wf_inference_jobs.result IS 'Inference result (detections, classifications, embeddings)';
COMMENT ON COLUMN wf_inference_jobs.metadata IS 'Worker metadata: GPU model, cache hit, timing breakdown';

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON wf_inference_jobs TO api_user;
-- GRANT SELECT, UPDATE ON wf_inference_jobs TO worker_user;
