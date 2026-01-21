-- Migration: Automatic image_count sync for od_datasets
-- This trigger automatically updates od_datasets.image_count when images are added/removed

-- Function to update dataset image count
CREATE OR REPLACE FUNCTION update_dataset_image_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE od_datasets
        SET image_count = (
            SELECT COUNT(*) FROM od_dataset_images WHERE dataset_id = NEW.dataset_id
        )
        WHERE id = NEW.dataset_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE od_datasets
        SET image_count = (
            SELECT COUNT(*) FROM od_dataset_images WHERE dataset_id = OLD.dataset_id
        )
        WHERE id = OLD.dataset_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS trigger_update_dataset_image_count ON od_dataset_images;

-- Create trigger on od_dataset_images table
CREATE TRIGGER trigger_update_dataset_image_count
AFTER INSERT OR DELETE ON od_dataset_images
FOR EACH ROW
EXECUTE FUNCTION update_dataset_image_count();

-- Also add ON DELETE CASCADE if not already present
-- This ensures that when an od_image is deleted, related od_dataset_images are also deleted
-- Check first if the constraint exists
DO $$
BEGIN
    -- Drop existing foreign key if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'od_dataset_images_image_id_fkey'
        AND table_name = 'od_dataset_images'
    ) THEN
        ALTER TABLE od_dataset_images DROP CONSTRAINT od_dataset_images_image_id_fkey;
    END IF;

    -- Add the foreign key with CASCADE
    ALTER TABLE od_dataset_images
    ADD CONSTRAINT od_dataset_images_image_id_fkey
    FOREIGN KEY (image_id) REFERENCES od_images(id) ON DELETE CASCADE;
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'Could not modify foreign key constraint: %', SQLERRM;
END $$;

-- Sync all existing dataset counts (one-time fix)
UPDATE od_datasets d
SET image_count = (
    SELECT COUNT(*) FROM od_dataset_images di WHERE di.dataset_id = d.id
);
