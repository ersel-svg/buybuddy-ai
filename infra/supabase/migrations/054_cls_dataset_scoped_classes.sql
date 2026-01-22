-- Migration: CLS Dataset-Scoped Classes
-- Make cls_classes dataset-specific instead of global

-- 1. Add dataset_id column (nullable for migration)
ALTER TABLE cls_classes ADD COLUMN IF NOT EXISTS dataset_id UUID REFERENCES cls_datasets(id) ON DELETE CASCADE;

-- 2. Create index
CREATE INDEX IF NOT EXISTS idx_cls_classes_dataset ON cls_classes(dataset_id);

-- 3. Drop old global unique constraint on name
ALTER TABLE cls_classes DROP CONSTRAINT IF EXISTS cls_classes_name_key;

-- 4. Add new unique constraint (dataset + name)
-- This allows same class name in different datasets
ALTER TABLE cls_classes ADD CONSTRAINT cls_classes_dataset_name_unique UNIQUE (dataset_id, name);

-- 5. Update RPC function to filter by dataset
CREATE OR REPLACE FUNCTION get_cls_class_distribution(p_dataset_id UUID)
RETURNS JSON AS $$
BEGIN
    RETURN (
        SELECT json_agg(t)
        FROM (
            SELECT 
                c.id,
                c.name,
                c.display_name,
                c.color,
                COUNT(l.id) as image_count
            FROM cls_classes c
            LEFT JOIN cls_labels l ON l.class_id = c.id AND l.dataset_id = p_dataset_id
            WHERE c.is_active = true AND c.dataset_id = p_dataset_id
            GROUP BY c.id, c.name, c.display_name, c.color
            ORDER BY image_count DESC
        ) t
    );
END;
$$ LANGUAGE plpgsql;

-- Comment
COMMENT ON COLUMN cls_classes.dataset_id IS 'Dataset this class belongs to. Classes are now dataset-specific.';
