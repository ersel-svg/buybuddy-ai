-- Migration: Fix RPC function for batch annotation count updates
-- Description:
--   1. Preserve completed/skipped statuses
--   2. Update ALL images (not just those with annotations)
--   3. Set status to 'pending' when annotation count is 0
-- Date: 2026-01-30

-- Drop and recreate the function with fixes
CREATE OR REPLACE FUNCTION update_annotation_counts_batch(
    p_dataset_id UUID,
    p_image_ids UUID[],
    p_class_ids UUID[]
)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    updated_images INTEGER := 0;
    updated_classes INTEGER := 0;
    dataset_count INTEGER := 0;
    annotated_count INTEGER := 0;
BEGIN
    -- 1. Batch update all image annotation counts
    -- Uses LEFT JOIN to include images with 0 annotations
    -- Preserves completed/skipped statuses
    WITH image_counts AS (
        SELECT
            di.image_id,
            COALESCE(ann.cnt, 0) as cnt
        FROM od_dataset_images di
        LEFT JOIN (
            SELECT image_id, COUNT(*) as cnt
            FROM od_annotations
            WHERE dataset_id = p_dataset_id
              AND image_id = ANY(p_image_ids)
            GROUP BY image_id
        ) ann ON di.image_id = ann.image_id
        WHERE di.dataset_id = p_dataset_id
          AND di.image_id = ANY(p_image_ids)
    )
    UPDATE od_dataset_images di
    SET
        annotation_count = ic.cnt,
        status = CASE
            -- Preserve completed and skipped statuses
            WHEN di.status IN ('completed', 'skipped') THEN di.status
            -- Set to annotated if has annotations
            WHEN ic.cnt > 0 THEN 'annotated'
            -- Otherwise set to pending
            ELSE 'pending'
        END,
        last_annotated_at = CASE
            WHEN ic.cnt > 0 THEN NOW()
            ELSE di.last_annotated_at
        END
    FROM image_counts ic
    WHERE di.dataset_id = p_dataset_id
      AND di.image_id = ic.image_id;

    GET DIAGNOSTICS updated_images = ROW_COUNT;

    -- 2. Batch update all class annotation counts
    -- Recalculate from actual annotation data
    WITH class_counts AS (
        SELECT class_id, COUNT(*) as cnt
        FROM od_annotations
        WHERE dataset_id = p_dataset_id
          AND class_id = ANY(p_class_ids)
        GROUP BY class_id
    )
    UPDATE od_classes c
    SET annotation_count = COALESCE(cc.cnt, 0)
    FROM class_counts cc
    WHERE c.id = cc.class_id;

    GET DIAGNOSTICS updated_classes = ROW_COUNT;

    -- Also reset classes with 0 annotations (not in the join above)
    UPDATE od_classes c
    SET annotation_count = 0
    WHERE c.id = ANY(p_class_ids)
      AND c.dataset_id = p_dataset_id
      AND NOT EXISTS (
          SELECT 1 FROM od_annotations a
          WHERE a.class_id = c.id AND a.dataset_id = p_dataset_id
      );

    -- 3. Update dataset totals
    SELECT COUNT(*) INTO dataset_count
    FROM od_annotations
    WHERE dataset_id = p_dataset_id;

    -- Count images with status 'annotated' or 'completed' (both mean they have been worked on)
    SELECT COUNT(*) INTO annotated_count
    FROM od_dataset_images
    WHERE dataset_id = p_dataset_id
      AND status IN ('annotated', 'completed');

    UPDATE od_datasets
    SET
        annotation_count = dataset_count,
        annotated_image_count = annotated_count
    WHERE id = p_dataset_id;

    RETURN json_build_object(
        'updated_images', updated_images,
        'updated_classes', updated_classes,
        'dataset_annotation_count', dataset_count,
        'annotated_image_count', annotated_count
    );
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION update_annotation_counts_batch(UUID, UUID[], UUID[]) TO authenticated;
GRANT EXECUTE ON FUNCTION update_annotation_counts_batch(UUID, UUID[], UUID[]) TO service_role;
