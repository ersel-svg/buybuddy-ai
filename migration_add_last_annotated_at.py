#!/usr/bin/env python3
"""Migration to add last_annotated_at column to od_dataset_images table."""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment
load_dotenv("apps/api/.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    exit(1)

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# SQL to add last_annotated_at column
sql = """
-- Add last_annotated_at column to od_dataset_images for tracking recently annotated images
ALTER TABLE od_dataset_images
ADD COLUMN IF NOT EXISTS last_annotated_at TIMESTAMPTZ DEFAULT NULL;

-- Create index for efficient sorting by last_annotated_at
CREATE INDEX IF NOT EXISTS idx_od_dataset_images_last_annotated_at
ON od_dataset_images (dataset_id, last_annotated_at DESC NULLS LAST);

-- Optionally: Backfill existing images with their latest annotation timestamp
-- This updates last_annotated_at based on the most recent annotation for each image
UPDATE od_dataset_images di
SET last_annotated_at = (
    SELECT MAX(a.created_at)
    FROM od_annotations a
    WHERE a.dataset_id = di.dataset_id AND a.image_id = di.image_id
)
WHERE di.annotation_count > 0;
"""

try:
    print("Applying migration: add last_annotated_at to od_dataset_images...")
    result = supabase.rpc("exec_sql", {"sql": sql}).execute()
    print("✓ Migration applied successfully!")
    print("  - Added last_annotated_at column")
    print("  - Created index for sorting")
    print("  - Backfilled existing data")
except Exception as e:
    print(f"✗ Migration failed: {e}")
    print("\nNote: You may need to apply this migration manually via Supabase dashboard SQL editor:")
    print(sql)
