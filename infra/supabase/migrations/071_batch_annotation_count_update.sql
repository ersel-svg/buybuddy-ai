-- Migration: Add RPC function for batch annotation count updates
-- Description: Efficiently update annotation counts for multiple images at once
-- Prevents system lockup when processing large annotation batches (67K+ images)
-- Date: 2026-01-27

-- Function to update annotation counts for multiple images in a single query
-- Instead of 67K individual queries, this runs 2-3 queries total
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
    -- 1. Batch update all image annotation counts in ONE query
    WITH image_counts AS (
        SELECT image_id, COUNT(*) as cnt
        FROM od_annotations
        WHERE dataset_id = p_dataset_id
          AND image_id = ANY(p_image_ids)
        GROUP BY image_id
    )
    UPDATE od_dataset_images di
    SET
        annotation_count = COALESCE(ic.cnt, 0),
        status = CASE WHEN COALESCE(ic.cnt, 0) > 0 THEN 'annotated' ELSE di.status END,
        last_annotated_at = NOW()
    FROM image_counts ic
    WHERE di.dataset_id = p_dataset_id
      AND di.image_id = ic.image_id;

    GET DIAGNOSTICS updated_images = ROW_COUNT;

    -- 2. Batch update all class annotation counts in ONE query
    WITH class_counts AS (
        SELECT class_id, COUNT(*) as cnt
        FROM od_annotations
        WHERE dataset_id = p_dataset_id
          AND class_id = ANY(p_class_ids)
        GROUP BY class_id
    )
    UPDATE od_classes c
    SET annotation_count = cc.cnt
    FROM class_counts cc
    WHERE c.id = cc.class_id;

    GET DIAGNOSTICS updated_classes = ROW_COUNT;

    -- 3. Update dataset totals
    SELECT COUNT(*) INTO dataset_count
    FROM od_annotations
    WHERE dataset_id = p_dataset_id;

    SELECT COUNT(*) INTO annotated_count
    FROM od_dataset_images
    WHERE dataset_id = p_dataset_id
      AND annotation_count > 0;

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
