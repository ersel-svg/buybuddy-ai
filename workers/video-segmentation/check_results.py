#!/usr/bin/env python3
"""Check test results in Supabase."""

from supabase import create_client
import os

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
client = create_client(url, key)

print("=" * 70)
print("TEST SONUCLARI - SUPABASE")
print("=" * 70)

# Check product_images uploaded during test
print("\n[1] En son yuklenen product_images kayitlari:")
result = client.table("product_images").select("id, product_id, image_url, frame_index, image_type, created_at").order("created_at", desc=True).limit(10).execute()
for f in result.data:
    img_url = f.get('image_url', '')
    print(f"  Frame {f.get('frame_index')} ({f.get('image_type')}): ...{img_url[-50:]}")

# Check storage
print("\n[2] Storage'daki klasorler (frames bucket):")
try:
    files = client.storage.from_("frames").list("", {"limit": 10, "sortBy": {"column": "created_at", "order": "desc"}})
    for f in files:
        print(f"  {f.get('name')} - {f.get('created_at', 'N/A')}")
except Exception as e:
    print(f"  Error: {e}")

# Get the test product IDs from recent images
print("\n[3] Son islenen urunler:")
unique_products = set()
for f in result.data:
    pid = f.get('product_id')
    if pid:
        unique_products.add(pid)

for pid in list(unique_products)[:3]:
    prod = client.table("products").select("id, barcode, status, frame_count, primary_image_url").eq("id", pid).execute()
    if prod.data:
        p = prod.data[0]
        print(f"\n  Product ID: {pid[:20]}...")
        print(f"    Barcode: {p.get('barcode')}")
        print(f"    Status: {p.get('status')}")
        print(f"    Frame Count: {p.get('frame_count')}")

        img = p.get('primary_image_url')
        if img:
            print(f"    Primary Image: {img}")

# Show direct URLs for viewing
print("\n" + "=" * 70)
print("GORUNTULEME URL'leri (tarayicida ac):")
print("=" * 70)
for f in result.data[:5]:
    img_url = f.get('image_url', '')
    if img_url:
        print(f"\n{img_url}")
