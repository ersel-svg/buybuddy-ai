-- Migration: Add optimized filter options function for OD images
-- This replaces fetching all rows and counting in Python with SQL aggregation

CREATE OR REPLACE FUNCTION get_od_image_filter_options()
RETURNS jsonb
LANGUAGE sql
STABLE
AS $$
SELECT jsonb_build_object(
    'status', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'value', status,
                'label', INITCAP(REPLACE(status, '_', ' ')),
                'count', cnt
            ) ORDER BY cnt DESC
        ), '[]'::jsonb)
        FROM (
            SELECT COALESCE(status, 'pending') as status, COUNT(*) as cnt
            FROM od_images
            GROUP BY COALESCE(status, 'pending')
        ) t
    ),
    'source', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'value', source,
                'label', CASE source
                    WHEN 'upload' THEN 'Upload'
                    WHEN 'buybuddy_sync' THEN 'BuyBuddy Sync'
                    WHEN 'import' THEN 'Import'
                    WHEN 'url' THEN 'URL Import'
                    ELSE INITCAP(source)
                END,
                'count', cnt
            ) ORDER BY cnt DESC
        ), '[]'::jsonb)
        FROM (
            SELECT COALESCE(source, 'upload') as source, COUNT(*) as cnt
            FROM od_images
            GROUP BY COALESCE(source, 'upload')
        ) t
    ),
    'folder', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'value', folder,
                'label', folder,
                'count', cnt
            ) ORDER BY folder
        ), '[]'::jsonb)
        FROM (
            SELECT folder, COUNT(*) as cnt
            FROM od_images
            WHERE folder IS NOT NULL
            GROUP BY folder
        ) t
    ),
    'merchant', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'value', merchant_id::text,
                'label', merchant_name,
                'count', cnt
            ) ORDER BY merchant_name
        ), '[]'::jsonb)
        FROM (
            SELECT
                merchant_id,
                COALESCE(MAX(merchant_name), 'Merchant ' || merchant_id::text) as merchant_name,
                COUNT(*) as cnt
            FROM od_images
            WHERE merchant_id IS NOT NULL
            GROUP BY merchant_id
        ) t
    ),
    'store', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'value', store_id::text,
                'label', store_name,
                'count', cnt
            ) ORDER BY store_name
        ), '[]'::jsonb)
        FROM (
            SELECT
                store_id,
                COALESCE(MAX(store_name), 'Store ' || store_id::text) as store_name,
                COUNT(*) as cnt
            FROM od_images
            WHERE store_id IS NOT NULL
            GROUP BY store_id
        ) t
    ),
    'merchants', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'id', merchant_id,
                'name', merchant_name
            ) ORDER BY merchant_name
        ), '[]'::jsonb)
        FROM (
            SELECT DISTINCT
                merchant_id,
                COALESCE(MAX(merchant_name), 'Merchant ' || merchant_id::text) as merchant_name
            FROM od_images
            WHERE merchant_id IS NOT NULL
            GROUP BY merchant_id
        ) t
    ),
    'stores', (
        SELECT COALESCE(jsonb_agg(
            jsonb_build_object(
                'id', store_id,
                'name', store_name
            ) ORDER BY store_name
        ), '[]'::jsonb)
        FROM (
            SELECT DISTINCT
                store_id,
                COALESCE(MAX(store_name), 'Store ' || store_id::text) as store_name
            FROM od_images
            WHERE store_id IS NOT NULL
            GROUP BY store_id
        ) t
    )
);
$$;

-- Add comment for documentation
COMMENT ON FUNCTION get_od_image_filter_options() IS
'Returns aggregated filter options with counts for the OD images filter drawer.
Uses SQL aggregation instead of fetching all rows, significantly improving performance for large datasets.';
