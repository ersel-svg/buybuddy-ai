-- Workflow Async Execution Support
-- Migration 063: Add async execution capabilities for production scalability
--
-- This migration adds:
-- 1. Execution mode (sync/async/background)
-- 2. Priority-based queue ordering
-- 3. Webhook callbacks for async completion
-- 4. Retry tracking
-- 5. Real-time progress tracking
-- 6. Workflow versioning support

-- ============================================
-- EXECUTION MODE & QUEUE SUPPORT
-- ============================================

-- Execution mode: how the workflow should be executed
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(20) DEFAULT 'sync'
    CHECK (execution_mode IN ('sync', 'async', 'background'));

COMMENT ON COLUMN wf_executions.execution_mode IS 'sync=wait for result, async=return ID immediately, background=fire-and-forget with webhook';

-- Priority for queue ordering (1=lowest, 10=highest)
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 5
    CHECK (priority BETWEEN 1 AND 10);

COMMENT ON COLUMN wf_executions.priority IS 'Queue priority: 1=lowest, 10=highest';

-- Webhook URL for async completion notification
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS callback_url TEXT;

COMMENT ON COLUMN wf_executions.callback_url IS 'Webhook URL to call when execution completes';

-- Timeout for execution (in seconds)
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS timeout_seconds INTEGER DEFAULT 300;

-- ============================================
-- RETRY TRACKING
-- ============================================

-- Retry count tracking
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;

ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS max_retries INTEGER DEFAULT 3;

ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS last_retry_at TIMESTAMPTZ;

COMMENT ON COLUMN wf_executions.retry_count IS 'Number of retry attempts made';
COMMENT ON COLUMN wf_executions.max_retries IS 'Maximum retry attempts allowed';

-- ============================================
-- REAL-TIME PROGRESS TRACKING
-- ============================================

-- Progress tracking for WebSocket updates
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS progress JSONB DEFAULT '{
    "current_node": null,
    "completed_nodes": [],
    "total_nodes": 0,
    "percent": 0
}'::jsonb;

COMMENT ON COLUMN wf_executions.progress IS 'Real-time progress info for WebSocket streaming';

-- ============================================
-- WORKFLOW VERSIONING
-- ============================================

-- Add current version to workflows
ALTER TABLE wf_workflows
ADD COLUMN IF NOT EXISTS current_version INTEGER DEFAULT 1;

COMMENT ON COLUMN wf_workflows.current_version IS 'Current version number, auto-incremented on definition change';

-- Link executions to specific workflow version
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS workflow_version INTEGER;

COMMENT ON COLUMN wf_executions.workflow_version IS 'Workflow version at time of execution';

-- Replay tracking - link to original execution
ALTER TABLE wf_executions
ADD COLUMN IF NOT EXISTS replayed_from UUID REFERENCES wf_executions(id) ON DELETE SET NULL;

COMMENT ON COLUMN wf_executions.replayed_from IS 'If this is a replay, points to original execution';

-- ============================================
-- VERSION HISTORY TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS wf_workflow_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,

    -- Version info
    version INTEGER NOT NULL,
    version_label VARCHAR(50),  -- e.g., "v1.0", "stable", "experimental"

    -- Snapshot of definition at this version
    definition JSONB NOT NULL,

    -- Change info
    change_summary TEXT,
    changed_by VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(workflow_id, version)
);

CREATE INDEX IF NOT EXISTS idx_wf_versions_workflow ON wf_workflow_versions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wf_versions_created ON wf_workflow_versions(created_at DESC);

COMMENT ON TABLE wf_workflow_versions IS 'Version history for workflow definitions';

-- ============================================
-- INDEXES FOR QUEUE PROCESSING
-- ============================================

-- Index for efficient queue polling
CREATE INDEX IF NOT EXISTS idx_wf_executions_queue
ON wf_executions(status, priority DESC, created_at ASC)
WHERE status IN ('pending', 'running');

-- Index for async mode filtering
CREATE INDEX IF NOT EXISTS idx_wf_executions_mode
ON wf_executions(execution_mode)
WHERE execution_mode IN ('async', 'background');

-- Index for retry processing
CREATE INDEX IF NOT EXISTS idx_wf_executions_retry
ON wf_executions(status, retry_count, last_retry_at)
WHERE status = 'pending' AND retry_count > 0;

-- ============================================
-- AUTO-VERSIONING TRIGGER
-- ============================================

-- Function to create new version on workflow update
CREATE OR REPLACE FUNCTION create_workflow_version()
RETURNS TRIGGER AS $$
BEGIN
    -- Only version if definition changed
    IF OLD.definition IS DISTINCT FROM NEW.definition THEN
        -- Increment version
        NEW.current_version := COALESCE(OLD.current_version, 0) + 1;

        -- Create version record
        INSERT INTO wf_workflow_versions (
            workflow_id,
            version,
            definition,
            change_summary
        ) VALUES (
            NEW.id,
            NEW.current_version,
            NEW.definition,
            'Auto-versioned on update'
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if any
DROP TRIGGER IF EXISTS wf_workflows_versioning ON wf_workflows;

-- Create versioning trigger
CREATE TRIGGER wf_workflows_versioning
    BEFORE UPDATE ON wf_workflows
    FOR EACH ROW EXECUTE FUNCTION create_workflow_version();

-- ============================================
-- RLS POLICIES
-- ============================================

ALTER TABLE wf_workflow_versions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON wf_workflow_versions FOR ALL USING (true);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get next pending execution from queue
CREATE OR REPLACE FUNCTION claim_next_workflow_execution(worker_id TEXT)
RETURNS TABLE (
    execution_id UUID,
    workflow_id UUID,
    input_data JSONB,
    priority INTEGER,
    retry_count INTEGER,
    max_retries INTEGER
) AS $$
DECLARE
    claimed_id UUID;
BEGIN
    -- Atomically claim the highest priority pending execution
    UPDATE wf_executions e
    SET
        status = 'running',
        started_at = NOW()
    WHERE e.id = (
        SELECT id FROM wf_executions
        WHERE status = 'pending'
        AND execution_mode IN ('async', 'background')
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING e.id INTO claimed_id;

    IF claimed_id IS NULL THEN
        RETURN;
    END IF;

    RETURN QUERY
    SELECT
        e.id,
        e.workflow_id,
        e.input_data,
        e.priority,
        e.retry_count,
        e.max_retries
    FROM wf_executions e
    WHERE e.id = claimed_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claim_next_workflow_execution IS 'Atomically claim and return the next pending workflow execution';

-- Function to mark stale executions as failed
CREATE OR REPLACE FUNCTION recover_stale_workflow_executions(stale_minutes INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    recovered_count INTEGER;
BEGIN
    UPDATE wf_executions
    SET
        status = 'pending',
        error_message = 'Recovered from stale state'
    WHERE status = 'running'
    AND started_at < NOW() - (stale_minutes || ' minutes')::INTERVAL;

    GET DIAGNOSTICS recovered_count = ROW_COUNT;
    RETURN recovered_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION recover_stale_workflow_executions IS 'Reset stale running executions back to pending';

-- Function to get queue statistics
CREATE OR REPLACE FUNCTION get_workflow_queue_stats()
RETURNS TABLE (
    pending_count BIGINT,
    running_count BIGINT,
    avg_wait_seconds NUMERIC,
    oldest_pending_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) FILTER (WHERE status = 'pending'),
        COUNT(*) FILTER (WHERE status = 'running'),
        EXTRACT(EPOCH FROM AVG(NOW() - created_at) FILTER (WHERE status = 'pending'))::NUMERIC,
        MIN(created_at) FILTER (WHERE status = 'pending')
    FROM wf_executions
    WHERE status IN ('pending', 'running')
    AND execution_mode IN ('async', 'background');
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_workflow_queue_stats IS 'Get statistics about the workflow execution queue';
