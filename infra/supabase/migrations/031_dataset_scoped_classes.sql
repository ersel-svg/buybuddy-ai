-- Buybuddy AI Platform - Object Detection Module
-- Migration 031: Dataset-Scoped Classes
--
-- This migration changes od_classes from global to dataset-scoped:
-- - Each class now belongs to a specific dataset
-- - Classes with same name can exist in different datasets
-- - "Template" classes can be copied to new datasets

-- ============================================
-- STEP 1: Add dataset_id column to od_classes
-- ============================================

-- Add dataset_id column (nullable first for migration)
ALTER TABLE od_classes ADD COLUMN IF NOT EXISTS dataset_id UUID REFERENCES od_datasets(id) ON DELETE CASCADE;

-- ============================================
-- STEP 2: Create index for dataset_id
-- ============================================
CREATE INDEX IF NOT EXISTS idx_od_classes_dataset ON od_classes(dataset_id);

-- ============================================
-- STEP 3: Update unique constraint
-- Class names must be unique within a dataset, not globally
-- ============================================

-- First, drop the existing unique constraint on name
ALTER TABLE od_classes DROP CONSTRAINT IF EXISTS od_classes_name_key;

-- Add new unique constraint for (dataset_id, name)
-- Note: This allows NULL dataset_id classes (templates) to have same names
ALTER TABLE od_classes ADD CONSTRAINT od_classes_dataset_name_unique UNIQUE (dataset_id, name);

-- ============================================
-- STEP 4: Migrate existing global classes to each dataset
-- For each existing dataset, copy all current classes
-- ============================================

-- First, identify existing datasets
DO $$
DECLARE
    dataset_rec RECORD;
    class_rec RECORD;
    new_class_id UUID;
BEGIN
    -- For each existing dataset
    FOR dataset_rec IN SELECT id FROM od_datasets LOOP
        -- Copy each class that doesn't have a dataset_id yet
        FOR class_rec IN
            SELECT id, name, display_name, description, color, category, parent_class_id, aliases, annotation_count, is_active, is_system
            FROM od_classes
            WHERE dataset_id IS NULL
        LOOP
            -- Create a copy for this dataset
            INSERT INTO od_classes (
                dataset_id, name, display_name, description, color, category,
                aliases, annotation_count, is_active, is_system
            ) VALUES (
                dataset_rec.id,
                class_rec.name,
                class_rec.display_name,
                class_rec.description,
                class_rec.color,
                class_rec.category,
                class_rec.aliases,
                0,  -- Reset annotation count for new dataset-specific class
                class_rec.is_active,
                class_rec.is_system
            )
            ON CONFLICT (dataset_id, name) DO NOTHING
            RETURNING id INTO new_class_id;

            -- Update annotations in this dataset to point to the new class
            IF new_class_id IS NOT NULL THEN
                UPDATE od_annotations
                SET class_id = new_class_id
                WHERE dataset_id = dataset_rec.id AND class_id = class_rec.id;
            END IF;
        END LOOP;
    END LOOP;
END $$;

-- ============================================
-- STEP 5: Keep global classes as templates (with NULL dataset_id)
-- Don't delete them - they serve as templates for new datasets
-- ============================================

-- Mark global classes as templates
UPDATE od_classes SET is_system = true WHERE dataset_id IS NULL;

-- ============================================
-- STEP 6: Update annotation counts for dataset-specific classes
-- ============================================
UPDATE od_classes c
SET annotation_count = (
    SELECT COUNT(*) FROM od_annotations a WHERE a.class_id = c.id
)
WHERE c.dataset_id IS NOT NULL;

-- ============================================
-- STEP 7: Add comment explaining the new structure
-- ============================================
COMMENT ON COLUMN od_classes.dataset_id IS 'Dataset this class belongs to. NULL means it is a template class for new datasets.';
