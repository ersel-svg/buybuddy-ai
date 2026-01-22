-- Migration: Bulk Add Products to Dataset RPC
-- This function efficiently adds filtered products to a dataset using server-side SQL
-- Handles 10K+ products without timeout or memory issues

-- ============================================
-- RPC Function: bulk_add_filtered_products_to_dataset
-- ============================================
CREATE OR REPLACE FUNCTION bulk_add_filtered_products_to_dataset(
    p_dataset_id UUID,
    p_filters JSONB DEFAULT '{}'::JSONB
) RETURNS JSONB AS $$
DECLARE
    v_inserted_count INTEGER := 0;
    v_skipped_count INTEGER := 0;
    v_total_matching INTEGER := 0;
    v_start_time TIMESTAMPTZ := NOW();
BEGIN
    -- Count total matching products first
    SELECT COUNT(*) INTO v_total_matching
    FROM products p
    WHERE
        -- Search filter (barcode, product_name, brand_name)
        (
            p_filters->>'search' IS NULL 
            OR p_filters->>'search' = ''
            OR p.barcode ILIKE '%' || (p_filters->>'search') || '%'
            OR p.product_name ILIKE '%' || (p_filters->>'search') || '%'
            OR p.brand_name ILIKE '%' || (p_filters->>'search') || '%'
        )
        -- Status filter (array)
        AND (
            p_filters->'status' IS NULL 
            OR jsonb_array_length(p_filters->'status') = 0
            OR p.status = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'status')))
        )
        -- Category filter (array)
        AND (
            p_filters->'category' IS NULL 
            OR jsonb_array_length(p_filters->'category') = 0
            OR p.category = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'category')))
        )
        -- Brand filter (array)
        AND (
            p_filters->'brand' IS NULL 
            OR jsonb_array_length(p_filters->'brand') = 0
            OR p.brand_name = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'brand')))
        )
        -- Sub-brand filter (array)
        AND (
            p_filters->'sub_brand' IS NULL 
            OR jsonb_array_length(p_filters->'sub_brand') = 0
            OR p.sub_brand = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'sub_brand')))
        )
        -- Product name filter (array)
        AND (
            p_filters->'product_name' IS NULL 
            OR jsonb_array_length(p_filters->'product_name') = 0
            OR p.product_name = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'product_name')))
        )
        -- Variant/Flavor filter (array)
        AND (
            p_filters->'variant_flavor' IS NULL 
            OR jsonb_array_length(p_filters->'variant_flavor') = 0
            OR p.variant_flavor = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'variant_flavor')))
        )
        -- Container type filter (array)
        AND (
            p_filters->'container_type' IS NULL 
            OR jsonb_array_length(p_filters->'container_type') = 0
            OR p.container_type = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'container_type')))
        )
        -- Net quantity filter (array)
        AND (
            p_filters->'net_quantity' IS NULL 
            OR jsonb_array_length(p_filters->'net_quantity') = 0
            OR p.net_quantity = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'net_quantity')))
        )
        -- Pack type filter (array)
        AND (
            p_filters->'pack_type' IS NULL 
            OR jsonb_array_length(p_filters->'pack_type') = 0
            OR p.pack_type = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'pack_type')))
        )
        -- Manufacturer country filter (array)
        AND (
            p_filters->'manufacturer_country' IS NULL 
            OR jsonb_array_length(p_filters->'manufacturer_country') = 0
            OR p.manufacturer_country = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'manufacturer_country')))
        )
        -- Boolean filters
        AND (
            (p_filters->>'has_video') IS NULL
            OR (p_filters->>'has_video' = 'true' AND p.video_url IS NOT NULL)
            OR (p_filters->>'has_video' = 'false' AND p.video_url IS NULL)
        )
        AND (
            (p_filters->>'has_image') IS NULL
            OR (p_filters->>'has_image' = 'true' AND p.primary_image_url IS NOT NULL)
            OR (p_filters->>'has_image' = 'false' AND p.primary_image_url IS NULL)
        )
        -- Range filters
        AND (
            (p_filters->>'frame_count_min') IS NULL
            OR p.frame_count >= (p_filters->>'frame_count_min')::INTEGER
        )
        AND (
            (p_filters->>'frame_count_max') IS NULL
            OR p.frame_count <= (p_filters->>'frame_count_max')::INTEGER
        );

    -- Insert products that don't already exist in the dataset
    WITH inserted AS (
        INSERT INTO dataset_products (dataset_id, product_id)
        SELECT p_dataset_id, p.id
        FROM products p
        WHERE
            -- Same filters as above
            (
                p_filters->>'search' IS NULL 
                OR p_filters->>'search' = ''
                OR p.barcode ILIKE '%' || (p_filters->>'search') || '%'
                OR p.product_name ILIKE '%' || (p_filters->>'search') || '%'
                OR p.brand_name ILIKE '%' || (p_filters->>'search') || '%'
            )
            AND (
                p_filters->'status' IS NULL 
                OR jsonb_array_length(p_filters->'status') = 0
                OR p.status = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'status')))
            )
            AND (
                p_filters->'category' IS NULL 
                OR jsonb_array_length(p_filters->'category') = 0
                OR p.category = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'category')))
            )
            AND (
                p_filters->'brand' IS NULL 
                OR jsonb_array_length(p_filters->'brand') = 0
                OR p.brand_name = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'brand')))
            )
            AND (
                p_filters->'sub_brand' IS NULL 
                OR jsonb_array_length(p_filters->'sub_brand') = 0
                OR p.sub_brand = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'sub_brand')))
            )
            AND (
                p_filters->'product_name' IS NULL 
                OR jsonb_array_length(p_filters->'product_name') = 0
                OR p.product_name = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'product_name')))
            )
            AND (
                p_filters->'variant_flavor' IS NULL 
                OR jsonb_array_length(p_filters->'variant_flavor') = 0
                OR p.variant_flavor = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'variant_flavor')))
            )
            AND (
                p_filters->'container_type' IS NULL 
                OR jsonb_array_length(p_filters->'container_type') = 0
                OR p.container_type = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'container_type')))
            )
            AND (
                p_filters->'net_quantity' IS NULL 
                OR jsonb_array_length(p_filters->'net_quantity') = 0
                OR p.net_quantity = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'net_quantity')))
            )
            AND (
                p_filters->'pack_type' IS NULL 
                OR jsonb_array_length(p_filters->'pack_type') = 0
                OR p.pack_type = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'pack_type')))
            )
            AND (
                p_filters->'manufacturer_country' IS NULL 
                OR jsonb_array_length(p_filters->'manufacturer_country') = 0
                OR p.manufacturer_country = ANY(ARRAY(SELECT jsonb_array_elements_text(p_filters->'manufacturer_country')))
            )
            AND (
                (p_filters->>'has_video') IS NULL
                OR (p_filters->>'has_video' = 'true' AND p.video_url IS NOT NULL)
                OR (p_filters->>'has_video' = 'false' AND p.video_url IS NULL)
            )
            AND (
                (p_filters->>'has_image') IS NULL
                OR (p_filters->>'has_image' = 'true' AND p.primary_image_url IS NOT NULL)
                OR (p_filters->>'has_image' = 'false' AND p.primary_image_url IS NULL)
            )
            AND (
                (p_filters->>'frame_count_min') IS NULL
                OR p.frame_count >= (p_filters->>'frame_count_min')::INTEGER
            )
            AND (
                (p_filters->>'frame_count_max') IS NULL
                OR p.frame_count <= (p_filters->>'frame_count_max')::INTEGER
            )
        ON CONFLICT (dataset_id, product_id) DO NOTHING
        RETURNING product_id
    )
    SELECT COUNT(*) INTO v_inserted_count FROM inserted;

    -- Calculate skipped (already existed)
    v_skipped_count := v_total_matching - v_inserted_count;

    -- Update dataset product_count
    UPDATE datasets
    SET 
        product_count = (
            SELECT COUNT(*) FROM dataset_products WHERE dataset_id = p_dataset_id
        ),
        updated_at = NOW()
    WHERE id = p_dataset_id;

    -- Return result
    RETURN jsonb_build_object(
        'added_count', v_inserted_count,
        'skipped_count', v_skipped_count,
        'total_matching', v_total_matching,
        'duration_ms', EXTRACT(MILLISECONDS FROM (NOW() - v_start_time))::INTEGER
    );
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION bulk_add_filtered_products_to_dataset(UUID, JSONB) TO authenticated;
GRANT EXECUTE ON FUNCTION bulk_add_filtered_products_to_dataset(UUID, JSONB) TO service_role;

-- Add comment
COMMENT ON FUNCTION bulk_add_filtered_products_to_dataset IS 
'Efficiently adds filtered products to a dataset using server-side SQL. 
Handles 10K+ products without timeout. Returns {added_count, skipped_count, total_matching, duration_ms}';
