#!/usr/bin/env python3
"""Apply migration to add SOTA job types to Supabase."""

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

# SQL to update job type constraint
sql = """
-- Drop existing constraint
ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_type_check;

-- Add new constraint with all job types including new SOTA types
ALTER TABLE jobs ADD CONSTRAINT jobs_type_check CHECK (type IN (
    -- Existing Runpod job types
    'video_processing',
    'augmentation',
    'training',
    'embedding_extraction',
    'matching',
    'roboflow_import',
    'od_annotation',
    'od_training',
    'cls_annotation',
    'cls_training',
    'buybuddy_sync',
    -- Existing local background job types
    'local_bulk_add_to_dataset',
    'local_bulk_remove_from_dataset',
    'local_bulk_update_status',
    'local_bulk_delete_images',
    'local_export_dataset',
    'local_bulk_update_products',
    'local_recalculate_counts',
    'local_data_cleanup',
    -- NEW: SOTA product operation types
    'local_bulk_delete_products',
    'local_bulk_product_matcher'
));
"""

try:
    print("Applying migration...")
    result = supabase.rpc("exec_sql", {"sql": sql}).execute()
    print("✓ Migration applied successfully!")
except Exception as e:
    print(f"✗ Migration failed: {e}")
    print("\nNote: You may need to apply this migration manually via Supabase dashboard SQL editor:")
    print(sql)
