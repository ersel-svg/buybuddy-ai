-- Buybuddy AI Platform - Workflow System MVP
-- Migration 050: Workflow tables for visual pipeline builder
--
-- This migration creates:
-- 1. wf_workflows - Workflow definitions with JSON nodes/edges
-- 2. wf_executions - Execution history and results
-- 3. wf_pretrained_models - Registry of pretrained models for workflows

-- ============================================
-- WF_WORKFLOWS TABLE
-- Main workflow definitions
-- ============================================
CREATE TABLE IF NOT EXISTS wf_workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Definition (JSON) - contains nodes, edges, viewport, etc.
    definition JSONB NOT NULL DEFAULT '{
        "version": "1.0",
        "nodes": [],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1}
    }'::jsonb,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived')),

    -- Stats (denormalized for performance)
    run_count INTEGER DEFAULT 0,
    last_run_at TIMESTAMPTZ,
    avg_duration_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wf_workflows_status ON wf_workflows(status);
CREATE INDEX IF NOT EXISTS idx_wf_workflows_created ON wf_workflows(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wf_workflows_name ON wf_workflows(name);

COMMENT ON TABLE wf_workflows IS 'Visual workflow definitions for CV pipelines';
COMMENT ON COLUMN wf_workflows.definition IS 'JSON containing nodes, edges, and React Flow viewport config';

-- ============================================
-- WF_EXECUTIONS TABLE
-- Workflow execution history
-- ============================================
CREATE TABLE IF NOT EXISTS wf_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES wf_workflows(id) ON DELETE CASCADE,

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,

    -- Input data
    input_data JSONB,        -- {image_url, image_base64, parameters}

    -- Output data
    output_data JSONB,       -- Full results from all output nodes

    -- Per-node metrics
    node_metrics JSONB,      -- {node_id: {duration_ms, output_count, ...}}

    -- Error handling
    error_message TEXT,
    error_node_id VARCHAR(100),
    error_traceback TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wf_executions_workflow ON wf_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wf_executions_status ON wf_executions(status);
CREATE INDEX IF NOT EXISTS idx_wf_executions_created ON wf_executions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wf_executions_workflow_status ON wf_executions(workflow_id, status);

COMMENT ON TABLE wf_executions IS 'Workflow execution history with timing and results';
COMMENT ON COLUMN wf_executions.node_metrics IS 'Per-node execution metrics for debugging';

-- ============================================
-- WF_PRETRAINED_MODELS TABLE
-- Registry of available pretrained models
-- ============================================
CREATE TABLE IF NOT EXISTS wf_pretrained_models (
    id VARCHAR(100) PRIMARY KEY,  -- e.g., "yolov8n", "dinov2-base"

    -- Model info
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Model type
    model_type VARCHAR(30) NOT NULL
        CHECK (model_type IN ('detection', 'classification', 'embedding', 'segmentation')),

    -- Source info
    source VARCHAR(50) NOT NULL,      -- ultralytics, huggingface, custom
    model_path TEXT NOT NULL,         -- Path or HF model ID

    -- Class info (for detection/classification)
    classes JSONB,                    -- Array of class names
    class_count INTEGER,

    -- Default configuration
    default_config JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Model metadata
    embedding_dim INTEGER,            -- For embedding models
    input_size INTEGER,               -- Expected input image size

    -- Flags
    is_active BOOLEAN DEFAULT true,
    is_downloadable BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wf_pretrained_type ON wf_pretrained_models(model_type);
CREATE INDEX IF NOT EXISTS idx_wf_pretrained_active ON wf_pretrained_models(is_active) WHERE is_active = true;

COMMENT ON TABLE wf_pretrained_models IS 'Registry of pretrained models available for workflows';
COMMENT ON COLUMN wf_pretrained_models.source IS 'Model provider: ultralytics, huggingface, custom';

-- ============================================
-- SEED PRETRAINED MODELS
-- ============================================

-- Detection Models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, classes, class_count, default_config, input_size)
VALUES
    -- YOLOv8 Models
    ('yolov8n', 'YOLOv8 Nano', 'Fast general object detection (COCO)', 'detection', 'ultralytics', 'yolov8n.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640),

    ('yolov8s', 'YOLOv8 Small', 'Balanced speed/accuracy detection (COCO)', 'detection', 'ultralytics', 'yolov8s.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640),

    ('yolov8n-face', 'YOLOv8 Face', 'Face detection model', 'detection', 'ultralytics', 'yolov8n-face.pt',
     '["face"]'::jsonb, 1, '{"confidence": 0.5, "nms_threshold": 0.5}'::jsonb, 640),

    -- YOLOv11 Models
    ('yolo11n', 'YOLO11 Nano', 'Latest YOLO - fast detection (COCO)', 'detection', 'ultralytics', 'yolo11n.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640),

    ('yolo11s', 'YOLO11 Small', 'Latest YOLO - balanced detection (COCO)', 'detection', 'ultralytics', 'yolo11s.pt',
     '["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]'::jsonb,
     80, '{"confidence": 0.5, "nms_threshold": 0.4}'::jsonb, 640)

ON CONFLICT (id) DO NOTHING;

-- Segmentation Models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('yolov8n-seg', 'YOLOv8 Nano Seg', 'Instance segmentation (COCO)', 'segmentation', 'ultralytics', 'yolov8n-seg.pt',
     '{"confidence": 0.5}'::jsonb, 640),

    ('yolo11n-seg', 'YOLO11 Nano Seg', 'Latest instance segmentation (COCO)', 'segmentation', 'ultralytics', 'yolo11n-seg.pt',
     '{"confidence": 0.5}'::jsonb, 640),

    ('sam-base', 'SAM Base', 'Segment Anything Model - Base', 'segmentation', 'huggingface', 'facebook/sam-vit-base',
     '{"points_per_side": 32}'::jsonb, 1024),

    ('sam-large', 'SAM Large', 'Segment Anything Model - Large', 'segmentation', 'huggingface', 'facebook/sam-vit-large',
     '{"points_per_side": 32}'::jsonb, 1024)

ON CONFLICT (id) DO NOTHING;

-- Embedding Models
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, embedding_dim, input_size)
VALUES
    ('dinov2-small', 'DINOv2 Small', 'Self-supervised image embeddings (384-dim)', 'embedding', 'huggingface', 'facebook/dinov2-small',
     '{"normalize": true}'::jsonb, 384, 518),

    ('dinov2-base', 'DINOv2 Base', 'Self-supervised image embeddings (768-dim)', 'embedding', 'huggingface', 'facebook/dinov2-base',
     '{"normalize": true}'::jsonb, 768, 518),

    ('dinov2-large', 'DINOv2 Large', 'Self-supervised image embeddings (1024-dim)', 'embedding', 'huggingface', 'facebook/dinov2-large',
     '{"normalize": true}'::jsonb, 1024, 518),

    ('clip-vit-b-32', 'CLIP ViT-B/32', 'Vision-language embeddings (512-dim)', 'embedding', 'huggingface', 'openai/clip-vit-base-patch32',
     '{"normalize": true}'::jsonb, 512, 224),

    ('clip-vit-b-16', 'CLIP ViT-B/16', 'Vision-language embeddings - higher res (512-dim)', 'embedding', 'huggingface', 'openai/clip-vit-base-patch16',
     '{"normalize": true}'::jsonb, 512, 224),

    ('clip-vit-l-14', 'CLIP ViT-L/14', 'Large CLIP embeddings (768-dim)', 'embedding', 'huggingface', 'openai/clip-vit-large-patch14',
     '{"normalize": true}'::jsonb, 768, 224)

ON CONFLICT (id) DO NOTHING;

-- Classification Models (zero-shot)
INSERT INTO wf_pretrained_models (id, name, description, model_type, source, model_path, default_config, input_size)
VALUES
    ('clip-classifier', 'CLIP Zero-Shot', 'Zero-shot classification with CLIP', 'classification', 'huggingface', 'openai/clip-vit-base-patch32',
     '{"top_k": 3}'::jsonb, 224)

ON CONFLICT (id) DO NOTHING;

-- ============================================
-- TRIGGERS
-- ============================================
DROP TRIGGER IF EXISTS wf_workflows_updated_at ON wf_workflows;
CREATE TRIGGER wf_workflows_updated_at
    BEFORE UPDATE ON wf_workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS wf_pretrained_models_updated_at ON wf_pretrained_models;
CREATE TRIGGER wf_pretrained_models_updated_at
    BEFORE UPDATE ON wf_pretrained_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Function to update workflow stats after execution
CREATE OR REPLACE FUNCTION update_workflow_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND OLD.status != 'completed' THEN
        UPDATE wf_workflows
        SET
            run_count = run_count + 1,
            last_run_at = NEW.completed_at,
            avg_duration_ms = COALESCE(
                (avg_duration_ms * (run_count) + NEW.duration_ms) / (run_count + 1),
                NEW.duration_ms
            )
        WHERE id = NEW.workflow_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS wf_executions_update_stats ON wf_executions;
CREATE TRIGGER wf_executions_update_stats
    AFTER UPDATE ON wf_executions
    FOR EACH ROW EXECUTE FUNCTION update_workflow_stats();

-- ============================================
-- RLS POLICIES
-- ============================================
ALTER TABLE wf_workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE wf_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE wf_pretrained_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for authenticated" ON wf_workflows FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON wf_executions FOR ALL USING (true);
CREATE POLICY "Allow all for authenticated" ON wf_pretrained_models FOR ALL USING (true);

-- ============================================
-- HELPER FUNCTIONS
-- ============================================

-- Function to get active workflow count
CREATE OR REPLACE FUNCTION get_active_workflow_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*)
        FROM wf_workflows
        WHERE status = 'active'
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get workflow execution stats
CREATE OR REPLACE FUNCTION get_workflow_execution_stats(workflow_uuid UUID)
RETURNS TABLE (
    total_runs INTEGER,
    successful_runs INTEGER,
    failed_runs INTEGER,
    avg_duration_ms NUMERIC,
    last_run_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_runs,
        COUNT(*) FILTER (WHERE status = 'completed')::INTEGER as successful_runs,
        COUNT(*) FILTER (WHERE status = 'failed')::INTEGER as failed_runs,
        AVG(duration_ms)::NUMERIC as avg_duration_ms,
        MAX(completed_at) as last_run_at
    FROM wf_executions
    WHERE workflow_id = workflow_uuid;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_workflow_execution_stats IS 'Get execution statistics for a workflow';

-- ============================================
-- VIEWS
-- ============================================

-- View for workflows with stats
CREATE OR REPLACE VIEW wf_workflows_with_stats AS
SELECT
    w.*,
    COALESCE(e.total_runs, 0) as total_executions,
    COALESCE(e.successful_runs, 0) as successful_executions,
    COALESCE(e.failed_runs, 0) as failed_executions,
    e.last_execution_at
FROM wf_workflows w
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) as total_runs,
        COUNT(*) FILTER (WHERE status = 'completed') as successful_runs,
        COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
        MAX(completed_at) as last_execution_at
    FROM wf_executions
    WHERE workflow_id = w.id
) e ON true;

COMMENT ON VIEW wf_workflows_with_stats IS 'Workflows with execution statistics';

-- View for available models (pretrained + trained combined)
CREATE OR REPLACE VIEW wf_all_models AS
-- Pretrained models
SELECT
    id::TEXT as id,
    name,
    description,
    model_type,
    'pretrained' as model_source,
    source as provider,
    model_path,
    classes,
    class_count,
    default_config,
    embedding_dim,
    input_size,
    NULL::REAL as accuracy,
    NULL::REAL as map,
    NULL::REAL as recall_at_1,
    is_active,
    created_at
FROM wf_pretrained_models
WHERE is_active = true

UNION ALL

-- Trained OD models
SELECT
    od.id::TEXT,
    od.name,
    od.description,
    'detection' as model_type,
    'trained' as model_source,
    od.model_type as provider,  -- rf-detr, rt-detr, etc.
    od.checkpoint_url as model_path,
    od.class_mapping as classes,
    od.class_count,
    '{}'::jsonb as default_config,
    NULL::INTEGER as embedding_dim,
    NULL::INTEGER as input_size,
    NULL::REAL as accuracy,
    od.map,
    NULL::REAL as recall_at_1,
    od.is_active,
    od.created_at
FROM od_trained_models od
WHERE od.is_active = true OR od.is_default = true

UNION ALL

-- Trained classification models
SELECT
    cls.id::TEXT,
    cls.name,
    cls.description,
    'classification' as model_type,
    'trained' as model_source,
    cls.model_type as provider,  -- vit, convnext, etc.
    cls.checkpoint_url as model_path,
    cls.class_mapping as classes,
    cls.num_classes as class_count,
    '{}'::jsonb as default_config,
    NULL::INTEGER as embedding_dim,
    NULL::INTEGER as input_size,
    cls.accuracy,
    NULL::REAL as map,
    NULL::REAL as recall_at_1,
    cls.is_active,
    cls.created_at
FROM cls_trained_models cls
WHERE cls.is_active = true OR cls.is_default = true

UNION ALL

-- Trained embedding models
SELECT
    tm.id::TEXT,
    tm.name,
    tm.description,
    'embedding' as model_type,
    'trained' as model_source,
    em.model_family as provider,
    tc.checkpoint_url as model_path,
    NULL::jsonb as classes,
    tr.num_classes as class_count,
    em.config as default_config,
    em.embedding_dim,
    (em.config->>'image_size')::INTEGER as input_size,
    NULL as accuracy,
    NULL as map,
    (tm.test_metrics->>'recall_at_1')::REAL as recall_at_1,
    tm.is_active,
    tm.created_at
FROM trained_models tm
JOIN training_runs tr ON tm.training_run_id = tr.id
JOIN training_checkpoints tc ON tm.checkpoint_id = tc.id
LEFT JOIN embedding_models em ON tm.embedding_model_id = em.id
WHERE tm.is_active = true OR tm.is_default = true;

COMMENT ON VIEW wf_all_models IS 'Unified view of all models (pretrained + trained) available for workflows';
