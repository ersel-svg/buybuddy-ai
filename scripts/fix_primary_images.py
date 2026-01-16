#!/usr/bin/env python3
"""
Script to update all products' primary_image_url from middle frame to first frame.
"""

import os
import re
from supabase import create_client

# Load from .env
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://fdcnfqxtgzqwkidwibmn.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_KEY:
    # Try to read from .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", "apps", "api", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("SUPABASE_KEY="):
                    SUPABASE_KEY = line.split("=", 1)[1].strip().strip('"')
                    break

def main():
    if not SUPABASE_KEY:
        print("ERROR: SUPABASE_KEY not found")
        return
    
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Get all products with primary_image_url
    result = client.table("products").select("id, primary_image_url, frames_path").not_.is_("primary_image_url", "null").execute()
    
    products = result.data or []
    print(f"Found {len(products)} products with primary_image_url")
    
    updated = 0
    skipped = 0
    
    for p in products:
        old_url = p.get("primary_image_url", "")
        frames_path = p.get("frames_path", "")
        
        if not old_url:
            skipped += 1
            continue
        
        # Check if already frame_0000
        if "frame_0000.png" in old_url:
            print(f"  [SKIP] {p['id'][:8]}... already uses frame_0000")
            skipped += 1
            continue
        
        # Replace frame_XXXX.png with frame_0000.png
        new_url = re.sub(r'frame_\d{4}\.png', 'frame_0000.png', old_url)
        
        if new_url == old_url:
            print(f"  [SKIP] {p['id'][:8]}... no frame pattern found")
            skipped += 1
            continue
        
        print(f"  [UPDATE] {p['id'][:8]}...")
        print(f"    OLD: ...{old_url[-40:]}")
        print(f"    NEW: ...{new_url[-40:]}")
        
        # Update in database
        client.table("products").update({
            "primary_image_url": new_url
        }).eq("id", p["id"]).execute()
        
        updated += 1
    
    print(f"\nDone! Updated: {updated}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
