-- Migration: Add optimized filter options function for Products
-- This replaces 13+ separate Python queries with a single SQL aggregation
-- Supports disjunctive faceting for multi-select filters

CREATE OR REPLACE FUNCTION get_product_filter_options(
    -- Filter parameters for disjunctive faceting
    p_status TEXT[] DEFAULT NULL,
    p_category TEXT[] DEFAULT NULL,
    p_brand TEXT[] DEFAULT NULL,
    p_sub_brand TEXT[] DEFAULT NULL,
    p_product_name TEXT[] DEFAULT NULL,
    p_variant_flavor TEXT[] DEFAULT NULL,
    p_container_type TEXT[] DEFAULT NULL,
    p_net_quantity TEXT[] DEFAULT NULL,
    p_pack_type TEXT[] DEFAULT NULL,
    p_manufacturer_country TEXT[] DEFAULT NULL,
    p_claims TEXT[] DEFAULT NULL,
    -- Boolean filters
    p_has_video BOOLEAN DEFAULT NULL,
    p_has_image BOOLEAN DEFAULT NULL,
    p_has_nutrition BOOLEAN DEFAULT NULL,
    p_has_description BOOLEAN DEFAULT NULL,
    p_has_prompt BOOLEAN DEFAULT NULL,
    p_has_issues BOOLEAN DEFAULT NULL,
    -- Range filters
    p_frame_count_min INTEGER DEFAULT NULL,
    p_frame_count_max INTEGER DEFAULT NULL,
    p_visibility_score_min INTEGER DEFAULT NULL,
    p_visibility_score_max INTEGER DEFAULT NULL,
    -- Exclusion
    p_exclude_dataset_id UUID DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    excluded_ids UUID[];
    result jsonb;
BEGIN
    -- Get excluded product IDs if filtering by dataset
    IF p_exclude_dataset_id IS NOT NULL THEN
        SELECT ARRAY_AGG(product_id) INTO excluded_ids
        FROM dataset_products
        WHERE dataset_id = p_exclude_dataset_id;
    END IF;

    -- Build result with conditional aggregation
    -- Each facet excludes its own filter for disjunctive behavior
    SELECT jsonb_build_object(
        -- Status options (exclude status filter for disjunctive)
        'status', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', status, 'label', INITCAP(REPLACE(status, '_', ' ')), 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT status, COUNT(*) as cnt
                FROM products p
                WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                  AND (p_sub_brand IS NULL OR p.sub_brand = ANY(p_sub_brand))
                  AND (p_product_name IS NULL OR p.product_name = ANY(p_product_name))
                  AND (p_variant_flavor IS NULL OR p.variant_flavor = ANY(p_variant_flavor))
                  AND (p_container_type IS NULL OR p.container_type = ANY(p_container_type))
                  AND (p_net_quantity IS NULL OR p.net_quantity = ANY(p_net_quantity))
                  AND (p_pack_type IS NULL OR p.pack_configuration->>'type' = ANY(p_pack_type))
                  AND (p_manufacturer_country IS NULL OR p.manufacturer_country = ANY(p_manufacturer_country))
                  AND (p_claims IS NULL OR p.claims && p_claims)
                  AND (p_has_video IS NULL OR (p_has_video = true AND p.video_url IS NOT NULL) OR (p_has_video = false AND p.video_url IS NULL))
                  AND (p_has_image IS NULL OR (p_has_image = true AND p.primary_image_url IS NOT NULL) OR (p_has_image = false AND p.primary_image_url IS NULL))
                  AND (p_has_nutrition IS NULL OR (p_has_nutrition = true AND p.nutrition_facts IS NOT NULL AND p.nutrition_facts != '{}') OR (p_has_nutrition = false AND (p.nutrition_facts IS NULL OR p.nutrition_facts = '{}')))
                  AND (p_has_description IS NULL OR (p_has_description = true AND p.marketing_description IS NOT NULL) OR (p_has_description = false AND p.marketing_description IS NULL))
                  AND (p_has_prompt IS NULL OR (p_has_prompt = true AND p.grounding_prompt IS NOT NULL) OR (p_has_prompt = false AND p.grounding_prompt IS NULL))
                  AND (p_has_issues IS NULL OR (p_has_issues = true AND p.issues_detected IS NOT NULL AND array_length(p.issues_detected, 1) > 0) OR (p_has_issues = false AND (p.issues_detected IS NULL OR array_length(p.issues_detected, 1) IS NULL)))
                  AND (p_frame_count_min IS NULL OR p.frame_count >= p_frame_count_min)
                  AND (p_frame_count_max IS NULL OR p.frame_count <= p_frame_count_max)
                  AND (p_visibility_score_min IS NULL OR p.visibility_score >= p_visibility_score_min)
                  AND (p_visibility_score_max IS NULL OR p.visibility_score <= p_visibility_score_max)
                GROUP BY status
            ) t
        ),

        -- Category options (exclude category filter)
        'category', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', category, 'label', category, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT category, COUNT(*) as cnt
                FROM products p
                WHERE category IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                  AND (p_sub_brand IS NULL OR p.sub_brand = ANY(p_sub_brand))
                  AND (p_has_video IS NULL OR (p_has_video = true AND p.video_url IS NOT NULL) OR (p_has_video = false AND p.video_url IS NULL))
                  AND (p_has_image IS NULL OR (p_has_image = true AND p.primary_image_url IS NOT NULL) OR (p_has_image = false AND p.primary_image_url IS NULL))
                GROUP BY category
            ) t
        ),

        -- Brand options (exclude brand filter)
        'brand', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', brand_name, 'label', brand_name, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT brand_name, COUNT(*) as cnt
                FROM products p
                WHERE brand_name IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_has_video IS NULL OR (p_has_video = true AND p.video_url IS NOT NULL) OR (p_has_video = false AND p.video_url IS NULL))
                  AND (p_has_image IS NULL OR (p_has_image = true AND p.primary_image_url IS NOT NULL) OR (p_has_image = false AND p.primary_image_url IS NULL))
                GROUP BY brand_name
            ) t
        ),

        -- Sub Brand options
        'subBrand', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', sub_brand, 'label', sub_brand, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT sub_brand, COUNT(*) as cnt
                FROM products p
                WHERE sub_brand IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY sub_brand
            ) t
        ),

        -- Product Name options
        'productName', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', product_name, 'label', product_name, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT product_name, COUNT(*) as cnt
                FROM products p
                WHERE product_name IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY product_name
            ) t
        ),

        -- Flavor/Variant options
        'flavor', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', variant_flavor, 'label', variant_flavor, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT variant_flavor, COUNT(*) as cnt
                FROM products p
                WHERE variant_flavor IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY variant_flavor
            ) t
        ),

        -- Container Type options
        'container', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', container_type, 'label', container_type, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT container_type, COUNT(*) as cnt
                FROM products p
                WHERE container_type IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY container_type
            ) t
        ),

        -- Net Quantity options
        'netQuantity', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', net_quantity, 'label', net_quantity, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT net_quantity, COUNT(*) as cnt
                FROM products p
                WHERE net_quantity IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY net_quantity
            ) t
        ),

        -- Pack Type options (from JSONB)
        'packType', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', pack_type, 'label', INITCAP(REPLACE(pack_type, '_', ' ')), 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT pack_configuration->>'type' as pack_type, COUNT(*) as cnt
                FROM products p
                WHERE pack_configuration->>'type' IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY pack_configuration->>'type'
            ) t
        ),

        -- Country options
        'country', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', manufacturer_country, 'label', manufacturer_country, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT manufacturer_country, COUNT(*) as cnt
                FROM products p
                WHERE manufacturer_country IS NOT NULL
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY manufacturer_country
            ) t
        ),

        -- Claims options (from array field)
        'claims', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', claim, 'label', claim, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT UNNEST(claims) as claim, COUNT(*) as cnt
                FROM products p
                WHERE claims IS NOT NULL AND array_length(claims, 1) > 0
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY UNNEST(claims)
            ) t
        ),

        -- Issue Types options (from array field)
        'issueTypes', (
            SELECT COALESCE(jsonb_agg(
                jsonb_build_object('value', issue, 'label', issue, 'count', cnt)
                ORDER BY cnt DESC
            ), '[]'::jsonb)
            FROM (
                SELECT UNNEST(issues_detected) as issue, COUNT(*) as cnt
                FROM products p
                WHERE issues_detected IS NOT NULL AND array_length(issues_detected, 1) > 0
                  AND (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
                  AND (p_status IS NULL OR p.status = ANY(p_status))
                  AND (p_category IS NULL OR p.category = ANY(p_category))
                  AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
                GROUP BY UNNEST(issues_detected)
            ) t
        ),

        -- Boolean counters (with all filters applied)
        'hasVideo', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE video_url IS NOT NULL),
                'falseCount', COUNT(*) FILTER (WHERE video_url IS NULL)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'hasImage', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE primary_image_url IS NOT NULL),
                'falseCount', COUNT(*) FILTER (WHERE primary_image_url IS NULL)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'hasNutrition', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE nutrition_facts IS NOT NULL AND nutrition_facts != '{}'),
                'falseCount', COUNT(*) FILTER (WHERE nutrition_facts IS NULL OR nutrition_facts = '{}')
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'hasDescription', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE marketing_description IS NOT NULL),
                'falseCount', COUNT(*) FILTER (WHERE marketing_description IS NULL)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'hasPrompt', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE grounding_prompt IS NOT NULL),
                'falseCount', COUNT(*) FILTER (WHERE grounding_prompt IS NULL)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'hasIssues', (
            SELECT jsonb_build_object(
                'trueCount', COUNT(*) FILTER (WHERE issues_detected IS NOT NULL AND array_length(issues_detected, 1) > 0),
                'falseCount', COUNT(*) FILTER (WHERE issues_detected IS NULL OR array_length(issues_detected, 1) IS NULL)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),

        -- Range values
        'frameCount', (
            SELECT jsonb_build_object(
                'min', COALESCE(MIN(frame_count), 0),
                'max', COALESCE(MAX(frame_count), 100)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),
        'visibilityScore', (
            SELECT jsonb_build_object(
                'min', COALESCE(MIN(visibility_score), 0),
                'max', COALESCE(MAX(visibility_score), 100)
            )
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
        ),

        -- Total count
        'totalProducts', (
            SELECT COUNT(*)
            FROM products p
            WHERE (excluded_ids IS NULL OR NOT (p.id = ANY(excluded_ids)))
              AND (p_status IS NULL OR p.status = ANY(p_status))
              AND (p_category IS NULL OR p.category = ANY(p_category))
              AND (p_brand IS NULL OR p.brand_name = ANY(p_brand))
              AND (p_sub_brand IS NULL OR p.sub_brand = ANY(p_sub_brand))
              AND (p_product_name IS NULL OR p.product_name = ANY(p_product_name))
              AND (p_variant_flavor IS NULL OR p.variant_flavor = ANY(p_variant_flavor))
              AND (p_container_type IS NULL OR p.container_type = ANY(p_container_type))
              AND (p_net_quantity IS NULL OR p.net_quantity = ANY(p_net_quantity))
              AND (p_pack_type IS NULL OR p.pack_configuration->>'type' = ANY(p_pack_type))
              AND (p_manufacturer_country IS NULL OR p.manufacturer_country = ANY(p_manufacturer_country))
              AND (p_claims IS NULL OR p.claims && p_claims)
              AND (p_has_video IS NULL OR (p_has_video = true AND p.video_url IS NOT NULL) OR (p_has_video = false AND p.video_url IS NULL))
              AND (p_has_image IS NULL OR (p_has_image = true AND p.primary_image_url IS NOT NULL) OR (p_has_image = false AND p.primary_image_url IS NULL))
              AND (p_has_nutrition IS NULL OR (p_has_nutrition = true AND p.nutrition_facts IS NOT NULL AND p.nutrition_facts != '{}') OR (p_has_nutrition = false AND (p.nutrition_facts IS NULL OR p.nutrition_facts = '{}')))
              AND (p_has_description IS NULL OR (p_has_description = true AND p.marketing_description IS NOT NULL) OR (p_has_description = false AND p.marketing_description IS NULL))
              AND (p_has_prompt IS NULL OR (p_has_prompt = true AND p.grounding_prompt IS NOT NULL) OR (p_has_prompt = false AND p.grounding_prompt IS NULL))
              AND (p_has_issues IS NULL OR (p_has_issues = true AND p.issues_detected IS NOT NULL AND array_length(p.issues_detected, 1) > 0) OR (p_has_issues = false AND (p.issues_detected IS NULL OR array_length(p.issues_detected, 1) IS NULL)))
              AND (p_frame_count_min IS NULL OR p.frame_count >= p_frame_count_min)
              AND (p_frame_count_max IS NULL OR p.frame_count <= p_frame_count_max)
              AND (p_visibility_score_min IS NULL OR p.visibility_score >= p_visibility_score_min)
              AND (p_visibility_score_max IS NULL OR p.visibility_score <= p_visibility_score_max)
        )
    ) INTO result;

    RETURN result;
END;
$$;

-- Add comment for documentation
COMMENT ON FUNCTION get_product_filter_options IS
'Returns aggregated filter options with counts for the Products filter drawer.
Supports disjunctive faceting - each facet excludes its own filter for multi-select behavior.
Replaces 13+ separate Python queries with a single optimized SQL aggregation.';
