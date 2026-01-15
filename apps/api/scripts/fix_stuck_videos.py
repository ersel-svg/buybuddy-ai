#!/usr/bin/env python3
"""
Script to fix videos stuck in 'processing' status.

This script finds videos that have status='processing' but their
corresponding product has already been processed (status != 'processing').
It updates the video status to match the product status.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.supabase import supabase_service as db


async def fix_stuck_videos():
    """Find and fix videos stuck in processing status."""
    print("=" * 60)
    print("Fixing stuck videos")
    print("=" * 60)

    # 1. Get all videos with status='processing' or 'pending'
    processing_videos = (
        db.client.table("videos")
        .select("*")
        .in_("status", ["processing", "pending"])
        .execute()
    )

    if not processing_videos.data:
        print("\nNo videos stuck in 'processing' or 'pending' status.")
        return

    print(f"\nFound {len(processing_videos.data)} videos in 'processing' or 'pending' status")

    # 2. Get all products to match with videos
    products = (
        db.client.table("products")
        .select("id, video_id, barcode, status")
        .execute()
    )

    # Create lookup by video_id and barcode
    product_by_video_id = {p["video_id"]: p for p in products.data if p.get("video_id")}
    product_by_barcode = {p["barcode"]: p for p in products.data if p.get("barcode")}

    fixed_count = 0
    skipped_count = 0

    for video in processing_videos.data:
        video_id = video["id"]
        barcode = video.get("barcode")

        # Try to find matching product
        product = product_by_video_id.get(video_id) or product_by_barcode.get(barcode)

        if not product:
            print(f"  Video {video_id} ({barcode}): No matching product found - SKIPPED")
            skipped_count += 1
            continue

        product_status = product["status"]
        product_id = product["id"]

        video_status = video["status"]

        # Determine what video status should be
        if product_status in ("needs_matching", "completed", "ready"):
            new_video_status = "completed"
        elif product_status == "failed":
            new_video_status = "failed"
        elif product_status == "processing":
            print(f"  Video {video_id} ({barcode}): Product still processing - SKIPPED")
            skipped_count += 1
            continue
        elif product_status == "pending" and video_status == "pending":
            # Both pending - nothing to fix
            skipped_count += 1
            continue
        else:
            print(f"  Video {video_id} ({barcode}): Unknown product status '{product_status}' - SKIPPED")
            skipped_count += 1
            continue

        # Skip if video already has correct status
        if video_status == new_video_status:
            skipped_count += 1
            continue

        # Update video status
        db.client.table("videos").update({
            "status": new_video_status,
            "product_id": product_id,
        }).eq("id", video_id).execute()

        print(f"  Video {video_id} ({barcode}): Updated to '{new_video_status}' (product: {product_id})")
        fixed_count += 1

    print("\n" + "=" * 60)
    print(f"Summary: Fixed {fixed_count} videos, Skipped {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(fix_stuck_videos())
