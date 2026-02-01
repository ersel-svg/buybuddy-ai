-- Migration: Add RPC for OD resync (counts + statuses)
-- Description: Fast, set-based resync to avoid per-image queries
-- Date: 2026-01-30

CREATE OR REPLACE FUNCTION resync_od_annotation_counts(p_dataset_id UUID)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    updated_images INTEGER := 0;
    updated_classes INTEGER := 0;
    dataset_count INTEGER := 0;
    annotated_count INTEGER := 0;
    total_images INTEGER := 0;
BEGIN
    -- Total images in dataset
    SELECT COUNT(*) INTO total_images
    FROM od_dataset_images
    WHERE dataset_id = p_dataset_id;

    -- Update images that have annotations
    WITH image_counts AS (
        SELECT image_id, COUNT(*) as cnt
        FROM od_annotations
        WHERE dataset_id = p_dataset_id
        GROUP BY image_id
    )
    UPDATE od_dataset_images di
    SET
        annotation_count = COALESCE(ic.cnt, 0),
        last_annotated_at = CASE WHEN COALESCE(ic.cnt, 0) > 0 THEN NOW() ELSE di.last_annotated_at END,
        status = CASE
            WHEN di.status IN ('completed', 'skipped') THEN di.status
            WHEN COALESCE(ic.cnt, 0) > 0 THEN 'annotated'
            ELSE 'pending'
        END
    FROM image_counts ic
    WHERE di.dataset_id = p_dataset_id
      AND di.image_id = ic.image_id;

    GET DIAGNOSTICS updated_images = ROW_COUNT;

    -- Update images with zero annotations
    UPDATE od_dataset_images di
    SET
        annotation_count = 0,
        status = CASE
            WHEN di.status IN ('completed', 'skipped') THEN di.status
            ELSE 'pending'
        END
    WHERE di.dataset_id = p_dataset_id
      AND NOT EXISTS (
          SELECT 1 FROM od_annotations a
          WHERE a.dataset_id = p_dataset_id AND a.image_id = di.image_id
      );

    -- Update class annotation counts
    WITH class_counts AS (
        SELECT class_id, COUNT(*) as cnt
        FROM od_annotations
        WHERE dataset_id = p_dataset_id
        GROUP BY class_id
    )
    UPDATE od_classes c
    SET annotation_count = cc.cnt
    FROM class_counts cc
    WHERE c.id = cc.class_id;

    GET DIAGNOSTICS updated_classes = ROW_COUNT;

    -- Update dataset totals
    SELECT COUNT(*) INTO dataset_count
    FROM od_annotations
    WHERE dataset_id = p_dataset_id;

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
        'annotated_image_count', annotated_count,
        'total_images', total_images
    );
END;
$$;

GRANT EXECUTE ON FUNCTION resync_od_annotation_counts(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION resync_od_annotation_counts(UUID) TO service_role;
