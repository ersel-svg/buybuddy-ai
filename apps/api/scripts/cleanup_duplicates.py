#!/usr/bin/env python3
"""
Duplicate Annotation Cleanup Script (Image-based)

Image bazlı çalışır - her image için ayrı ayrı duplicate temizler.
Bu sayede Supabase timeout sorununu tamamen önler.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supabase import create_client, Client
from config import settings


def get_client() -> Client:
    url = settings.supabase_url
    if not url.endswith("/"):
        url += "/"
    return create_client(url, settings.supabase_service_role_key)


def get_all_image_ids(client: Client, dataset_id: str) -> list[str]:
    """Dataset'teki tüm image ID'lerini çek."""
    all_ids = []
    batch_size = 1000
    offset = 0

    while True:
        try:
            result = (
                client.table("od_dataset_images")
                .select("image_id")
                .eq("dataset_id", dataset_id)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
        except Exception as e:
            print(f"  Hata (image list): {e}")
            time.sleep(2)
            continue

        if not result.data:
            break

        all_ids.extend([r["image_id"] for r in result.data])
        offset += batch_size

    return all_ids


def get_annotations_for_image(client: Client, dataset_id: str, image_id: str) -> list[dict]:
    """Bir image için annotation'ları çek."""
    try:
        result = (
            client.table("od_annotations")
            .select("id, class_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at")
            .eq("dataset_id", dataset_id)
            .eq("image_id", image_id)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"    Hata (annotations): {e}")
        return []


def find_duplicates(annotations: list[dict]) -> list[str]:
    """Duplicate ID'lerini bul (en eskiyi tut)."""
    groups = defaultdict(list)

    for ann in annotations:
        key = (
            ann["class_id"],
            ann["bbox_x"],
            ann["bbox_y"],
            ann["bbox_width"],
            ann["bbox_height"],
        )
        groups[key].append(ann)

    ids_to_delete = []
    for group in groups.values():
        if len(group) > 1:
            sorted_group = sorted(group, key=lambda x: x["created_at"] or "")
            ids_to_delete.extend([a["id"] for a in sorted_group[1:]])

    return ids_to_delete


def delete_ids(client: Client, ids: list[str]) -> int:
    """ID'leri sil."""
    if not ids:
        return 0

    try:
        result = client.table("od_annotations").delete().in_("id", ids).execute()
        return len(result.data) if result.data else len(ids)
    except Exception as e:
        print(f"    Silme hatası: {e}")
        # Tek tek dene
        deleted = 0
        for id_ in ids:
            try:
                client.table("od_annotations").delete().eq("id", id_).execute()
                deleted += 1
            except:
                pass
        return deleted


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset-id", type=str, help="Sadece bu dataset'i işle")
    args = parser.parse_args()

    print("=" * 60)
    print("DUPLICATE CLEANUP (Image Bazlı)")
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    client = get_client()

    # Dataset'leri al
    if args.dataset_id:
        result = client.table("od_datasets").select("id, name").eq("id", args.dataset_id).execute()
        datasets = result.data or []
    else:
        result = client.table("od_datasets").select("id, name").execute()
        datasets = result.data or []

    print(f"\n{len(datasets)} dataset işlenecek.\n")

    grand_total_deleted = 0
    grand_total_duplicates = 0

    for ds in datasets:
        ds_id = ds["id"]
        ds_name = ds["name"]

        print(f"\n{'='*60}")
        print(f"📁 DATASET: {ds_name}")
        print("=" * 60)

        # Image ID'lerini al
        print("Image listesi çekiliyor...", end=" ")
        image_ids = get_all_image_ids(client, ds_id)
        print(f"{len(image_ids):,} image")

        if not image_ids:
            print("  (boş dataset, atlanıyor)")
            continue

        dataset_duplicates = 0
        dataset_deleted = 0

        for i, img_id in enumerate(image_ids):
            annotations = get_annotations_for_image(client, ds_id, img_id)
            ids_to_delete = find_duplicates(annotations)

            if ids_to_delete:
                dataset_duplicates += len(ids_to_delete)

                if not args.dry_run:
                    deleted = delete_ids(client, ids_to_delete)
                    dataset_deleted += deleted

            # İlerleme göster (her 100 image'da)
            if (i + 1) % 100 == 0:
                if args.dry_run:
                    print(f"  [{i+1:,}/{len(image_ids):,}] {dataset_duplicates:,} duplicate bulundu")
                else:
                    print(f"  [{i+1:,}/{len(image_ids):,}] {dataset_deleted:,} silindi")

        # Dataset özeti
        print(f"\n  Sonuç: {dataset_duplicates:,} duplicate", end="")
        if not args.dry_run:
            print(f", {dataset_deleted:,} silindi")
        else:
            print(" (dry-run)")

        grand_total_duplicates += dataset_duplicates
        grand_total_deleted += dataset_deleted

    # Genel özet
    print("\n" + "=" * 60)
    print("GENEL ÖZET")
    print("=" * 60)
    print(f"Toplam duplicate: {grand_total_duplicates:,}")
    if not args.dry_run:
        print(f"Toplam silinen: {grand_total_deleted:,}")
    else:
        print("[DRY RUN - hiçbir şey silinmedi]")

    if grand_total_deleted > 0:
        print("\n⚠️  Sayıları güncellemek için SQL'de STEP 3'ü çalıştırın!")


if __name__ == "__main__":
    main()
