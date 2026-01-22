-- Migration: Add optimized stats function for OD dashboard
-- This replaces multiple separate queries with a single optimized SQL function

CREATE OR REPLACE FUNCTION get_od_stats()
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
SELECT jsonb_build_object(
    'total_images', (SELECT COUNT(*) FROM od_images),
    'total_datasets', (SELECT COUNT(*) FROM od_datasets),
    'total_annotations', (SELECT COUNT(*) FROM od_annotations),
    'total_classes', (SELECT COUNT(*) FROM od_classes WHERE is_active = true),
    'total_models', (SELECT COUNT(*) FROM od_trained_models),
    'images_by_status', (
        SELECT COALESCE(
            jsonb_object_agg(
                COALESCE(status, 'pending'),
                cnt
            ),
            '{}'::jsonb
        )
        FROM (
            SELECT COALESCE(status, 'pending') as status, COUNT(*) as cnt
            FROM od_images
            GROUP BY COALESCE(status, 'pending')
        ) t
    ),
    'recent_datasets', (
        SELECT COALESCE(jsonb_agg(row_to_json(d.*) ORDER BY d.created_at DESC), '[]'::jsonb)
        FROM (
            SELECT *
            FROM od_datasets
            ORDER BY created_at DESC
            LIMIT 5
        ) d
    )
);
$$;

-- Add comment for documentation
COMMENT ON FUNCTION get_od_stats() IS
'Returns aggregated OD dashboard statistics in a single query.
Replaces 6+ separate queries with one optimized SQL function.
Returns: total_images, total_datasets, total_annotations, total_classes, total_models, images_by_status, recent_datasets';

-- Create index for faster status aggregation if not exists
CREATE INDEX IF NOT EXISTS idx_od_images_status ON od_images(status);
