"""
Handler for bulk adding products to a dataset.

This handler processes the operation in batches with progress tracking,
handling large product lists and filter-based selection efficiently.
"""

from typing import Callable

from services.supabase import supabase_service
from ..base import BaseJobHandler, JobProgress
from ..registry import job_registry
from ..utils import chunks, calculate_progress


@job_registry.register
class BulkAddProductsToDatasetHandler(BaseJobHandler):
    """
    Handler for bulk adding products to a dataset.

    Config:
        dataset_id: str - Target dataset ID
        product_ids: list[str] - Optional, specific product IDs to add
        filters: dict - Optional, filter criteria for products
            - search: str
            - status: str (comma-separated)
            - category: str (comma-separated)
            - brand: str (comma-separated)
            - sub_brand: str (comma-separated)
            - product_name: str (comma-separated)
            - variant_flavor: str (comma-separated)
            - container_type: str (comma-separated)
            - net_quantity: str (comma-separated)
            - pack_type: str (comma-separated)
            - manufacturer_country: str (comma-separated)
            - has_video: bool
            - has_image: bool
            - frame_count_min: int
            - frame_count_max: int

    Result:
        added: int - Number of products added
        skipped: int - Number of products already in dataset
        total: int - Total matching products
        errors: list[str] - Any errors encountered
    """

    job_type = "local_bulk_add_products_to_dataset"
    BATCH_SIZE = 100
    PAGE_SIZE = 1000  # Supabase pagination for fetching products
    CHECK_BATCH_SIZE = 200  # Smaller batch for .in_() queries to avoid Supabase limits

    def validate_config(self, config: dict) -> str | None:
        if not config.get("dataset_id"):
            return "dataset_id is required"
        if not config.get("product_ids") and not config.get("filters"):
            return "Either product_ids or filters must be provided"
        return None

    async def execute(
        self,
        job_id: str,
        config: dict,
        update_progress: Callable[[JobProgress], None],
    ) -> dict:
        dataset_id = config["dataset_id"]
        product_ids = config.get("product_ids")
        filters = config.get("filters", {})

        # Update initial progress
        update_progress(JobProgress(
            progress=0,
            current_step="Initializing...",
            processed=0,
            total=0,
        ))

        # Verify dataset exists
        dataset = supabase_service.client.table("datasets")\
            .select("id, name")\
            .eq("id", dataset_id)\
            .execute()

        if not dataset.data or len(dataset.data) == 0:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Get all matching product IDs (with pagination if using filters)
        update_progress(JobProgress(
            progress=5,
            current_step="Collecting product IDs...",
            processed=0,
            total=0,
        ))

        # Get product IDs either from direct list or filters
        if product_ids:
            all_product_ids = product_ids
        else:
            all_product_ids = self._get_filtered_product_ids(filters)

        total = len(all_product_ids)

        if total == 0:
            return {
                "added": 0,
                "skipped": 0,
                "total": 0,
                "message": "No products match the criteria",
            }

        # Get existing products in dataset
        update_progress(JobProgress(
            progress=10,
            current_step="Checking existing products...",
            processed=0,
            total=total,
        ))

        existing_ids = self._get_existing_product_ids(dataset_id, all_product_ids)

        # Filter out existing products
        new_product_ids = [pid for pid in all_product_ids if pid not in existing_ids]
        skipped = len(existing_ids)

        if not new_product_ids:
            return {
                "added": 0,
                "skipped": skipped,
                "total": total,
                "message": f"All {skipped} products already in dataset",
            }

        # Process in batches
        added = 0
        errors = []

        for batch_num, batch in enumerate(chunks(new_product_ids, self.BATCH_SIZE)):
            # Insert batch
            try:
                records = [
                    {
                        "dataset_id": dataset_id,
                        "product_id": pid,
                    }
                    for pid in batch
                ]
                supabase_service.client.table("dataset_products")\
                    .insert(records)\
                    .execute()
                added += len(batch)
            except Exception as e:
                error_msg = f"Batch {batch_num + 1}: {str(e)}"
                errors.append(error_msg)

            # Update progress (10-90% range for processing)
            processed = min((batch_num + 1) * self.BATCH_SIZE, len(new_product_ids))
            progress = 10 + calculate_progress(processed, len(new_product_ids)) * 0.80

            update_progress(JobProgress(
                progress=int(progress),
                current_step=f"Adding products... ({processed}/{len(new_product_ids)})",
                processed=processed + skipped,
                total=total,
                errors=errors[-5:] if errors else [],
            ))

        # Update dataset product count
        update_progress(JobProgress(
            progress=95,
            current_step="Updating dataset counts...",
            processed=total,
            total=total,
        ))

        self._update_dataset_product_count(dataset_id)

        return {
            "added": added,
            "skipped": skipped,
            "total": total,
            "errors": errors[:10] if errors else [],
            "message": f"Added {added} products, skipped {skipped} (already in dataset)",
        }

    def _get_filtered_product_ids(self, filters: dict) -> list[str]:
        """Get all product IDs matching filters with pagination."""
        all_ids = []
        offset = 0

        while True:
            query = supabase_service.client.table("products").select("id")

            # Apply filters - only apply if values are non-empty
            if filters.get("search"):
                search = str(filters["search"]).strip()
                if search:
                    query = query.or_(
                        f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
                    )

            if filters.get("status"):
                status_list = self._parse_filter_value(filters["status"])
                if status_list and len(status_list) > 0:
                    query = query.in_("status", status_list)

            if filters.get("category"):
                cat_list = self._parse_filter_value(filters["category"])
                if cat_list and len(cat_list) > 0:
                    query = query.in_("category", cat_list)

            if filters.get("brand"):
                brand_list = self._parse_filter_value(filters["brand"])
                if brand_list and len(brand_list) > 0:
                    query = query.in_("brand_name", brand_list)

            if filters.get("sub_brand"):
                sub_list = self._parse_filter_value(filters["sub_brand"])
                if sub_list and len(sub_list) > 0:
                    query = query.in_("sub_brand", sub_list)

            if filters.get("product_name"):
                pn_list = self._parse_filter_value(filters["product_name"])
                if pn_list and len(pn_list) > 0:
                    query = query.in_("product_name", pn_list)

            if filters.get("variant_flavor"):
                vf_list = self._parse_filter_value(filters["variant_flavor"])
                if vf_list and len(vf_list) > 0:
                    query = query.in_("variant_flavor", vf_list)

            if filters.get("container_type"):
                ct_list = self._parse_filter_value(filters["container_type"])
                if ct_list and len(ct_list) > 0:
                    query = query.in_("container_type", ct_list)

            if filters.get("net_quantity"):
                nq_list = self._parse_filter_value(filters["net_quantity"])
                if nq_list and len(nq_list) > 0:
                    query = query.in_("net_quantity", nq_list)

            if filters.get("pack_type"):
                pt_list = self._parse_filter_value(filters["pack_type"])
                if pt_list and len(pt_list) > 0:
                    query = query.in_("pack_type", pt_list)

            if filters.get("manufacturer_country"):
                mc_list = self._parse_filter_value(filters["manufacturer_country"])
                if mc_list and len(mc_list) > 0:
                    query = query.in_("manufacturer_country", mc_list)

            # Boolean filters
            if filters.get("has_video") is True:
                query = query.not_.is_("video_url", "null")
            elif filters.get("has_video") is False:
                query = query.is_("video_url", "null")

            if filters.get("has_image") is True:
                query = query.not_.is_("primary_image_url", "null")
            elif filters.get("has_image") is False:
                query = query.is_("primary_image_url", "null")

            # Range filters
            if filters.get("frame_count_min") is not None:
                query = query.gte("frame_count", filters["frame_count_min"])
            if filters.get("frame_count_max") is not None:
                query = query.lte("frame_count", filters["frame_count_max"])

            result = query.range(offset, offset + self.PAGE_SIZE - 1).execute()

            if not result.data:
                break

            all_ids.extend(p["id"] for p in result.data)

            if len(result.data) < self.PAGE_SIZE:
                break

            offset += self.PAGE_SIZE

        return all_ids

    def _parse_filter_value(self, value) -> list[str]:
        """Parse filter value which can be string (comma-separated) or list."""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        elif isinstance(value, list):
            return value
        return []

    def _get_existing_product_ids(self, dataset_id: str, product_ids: list[str]) -> set[str]:
        """Get existing product IDs in dataset (checking only the requested IDs)."""
        existing_ids = set()

        if not product_ids:
            return existing_ids

        # Check in smaller batches to avoid Supabase .in_() query limits
        for batch in chunks(product_ids, self.CHECK_BATCH_SIZE):
            result = supabase_service.client.table("dataset_products")\
                .select("product_id")\
                .eq("dataset_id", dataset_id)\
                .in_("product_id", batch)\
                .execute()
            existing_ids.update(r["product_id"] for r in result.data or [])

        return existing_ids

    def _update_dataset_product_count(self, dataset_id: str) -> None:
        """Update dataset product count."""
        count_result = supabase_service.client.table("dataset_products")\
            .select("product_id", count="exact")\
            .eq("dataset_id", dataset_id)\
            .execute()

        supabase_service.client.table("datasets").update({
            "product_count": count_result.count or 0,
        }).eq("id", dataset_id).execute()
