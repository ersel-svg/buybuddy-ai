#!/usr/bin/env python3
"""
Quick script to recalculate dataset counts.
Run from the api directory: python fix_counts.py
"""
import sys
sys.path.insert(0, "src")

from services.supabase import supabase_service
from services.roboflow_streaming import recalculate_dataset_counts

DATASET_ID = "0b06e620-bb9c-4086-9a08-14bffa0877d5"

print(f"Recalculating counts for dataset: {DATASET_ID}")
print("-" * 50)

# Get current counts
current = supabase_service.client.table("od_datasets").select("image_count, annotation_count").eq("id", DATASET_ID).single().execute()
if current.data:
    print(f"Current counts: {current.data}")

# Recalculate
print("Recalculating... (this may take a while for 8000+ images)")
counts = recalculate_dataset_counts(DATASET_ID)
print(f"New counts: {counts}")
print("-" * 50)
print("Done!")
