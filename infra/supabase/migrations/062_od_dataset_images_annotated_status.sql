-- Migration: Add 'annotated' status to od_dataset_images
-- This status indicates that an image has AI-generated annotations but hasn't been reviewed yet
-- Flow: pending -> annotated (after AI bulk annotate) -> completed (after user review/done)

-- Drop and recreate the constraint with new status
ALTER TABLE od_dataset_images
DROP CONSTRAINT IF EXISTS od_dataset_images_status_check;

ALTER TABLE od_dataset_images
ADD CONSTRAINT od_dataset_images_status_check
CHECK (status IN ('pending', 'annotating', 'annotated', 'completed', 'skipped'));

-- Also update the od_images table constraint if it exists
ALTER TABLE od_images
DROP CONSTRAINT IF EXISTS od_images_status_check;

ALTER TABLE od_images
ADD CONSTRAINT od_images_status_check
CHECK (status IN ('pending', 'annotating', 'annotated', 'completed', 'skipped'));

-- Update existing images that have annotations but are still pending to 'annotated'
UPDATE od_dataset_images
SET status = 'annotated'
WHERE annotation_count > 0 AND status = 'pending';

-- Add comment explaining the status flow
COMMENT ON COLUMN od_dataset_images.status IS 'Image annotation status: pending (no annotations), annotated (has AI annotations, not reviewed), completed (reviewed/done), skipped (excluded)';
