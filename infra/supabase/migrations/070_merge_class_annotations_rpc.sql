-- Migration: Add RPC function for merging class annotations
-- Description: Efficient bulk update that bypasses REST API limitations
-- Date: 2026-01-27

-- Function to merge annotations from one class to another
-- Returns the number of annotations moved
CREATE OR REPLACE FUNCTION merge_class_annotations(
    p_source_class_id UUID,
    p_target_class_id UUID
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    moved_count INTEGER;
BEGIN
    -- Update all annotations from source to target class
    UPDATE od_annotations
    SET class_id = p_target_class_id
    WHERE class_id = p_source_class_id;

    -- Get the count of affected rows
    GET DIAGNOSTICS moved_count = ROW_COUNT;

    RETURN moved_count;
END;
$$;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION merge_class_annotations(UUID, UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION merge_class_annotations(UUID, UUID) TO service_role;
