"""Supabase client wrapper for database operations."""

import asyncio
import logging
from functools import lru_cache
from typing import Optional, Any, AsyncGenerator
from datetime import datetime
import json

from supabase import create_client, Client

from config import settings

logger = logging.getLogger(__name__)


@lru_cache
def get_supabase_client() -> Client:
    """Get cached Supabase client (uses service role key to bypass RLS)."""
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise ValueError("Supabase URL and service role key must be configured")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


class SupabaseService:
    """Service class for Supabase operations."""

    def __init__(self) -> None:
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Lazy load Supabase client."""
        if self._client is None:
            self._client = get_supabase_client()
        return self._client

    # =========================================
    # Products
    # =========================================

    async def get_products(
        self,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        # Sorting
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "desc",
        # Comma-separated list filters
        status: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        sub_brand: Optional[str] = None,
        product_name: Optional[str] = None,
        variant_flavor: Optional[str] = None,
        container_type: Optional[str] = None,
        net_quantity: Optional[str] = None,
        pack_type: Optional[str] = None,
        manufacturer_country: Optional[str] = None,
        claims: Optional[str] = None,
        # Boolean filters
        has_video: Optional[bool] = None,
        has_image: Optional[bool] = None,
        has_nutrition: Optional[bool] = None,
        has_description: Optional[bool] = None,
        has_prompt: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        # Range filters
        frame_count_min: Optional[int] = None,
        frame_count_max: Optional[int] = None,
        visibility_score_min: Optional[int] = None,
        visibility_score_max: Optional[int] = None,
        # Exclusion filters
        exclude_dataset_id: Optional[str] = None,
        include_frame_counts: bool = False,
    ) -> dict[str, Any]:
        """Get products with pagination and comprehensive filters."""
        # If excluding products from a dataset, first get those product IDs
        excluded_product_ids: set[str] = set()
        if exclude_dataset_id:
            dp_response = (
                self.client.table("dataset_products")
                .select("product_id")
                .eq("dataset_id", exclude_dataset_id)
                .execute()
            )
            excluded_product_ids = {row["product_id"] for row in dp_response.data}

        query = self.client.table("products").select("*", count="exact")

        # Helper to parse comma-separated values
        def parse_csv(value: Optional[str]) -> list[str]:
            if not value:
                return []
            return [v.strip() for v in value.split(",") if v.strip()]

        # Search filter
        if search:
            query = query.or_(
                f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
            )

        # Exclude products from dataset (if specified)
        if excluded_product_ids:
            # PostgREST supports not_.in_ for negation
            query = query.not_.in_("id", list(excluded_product_ids))

        # List filters (comma-separated -> in_ query)
        status_list = parse_csv(status)
        if status_list:
            query = query.in_("status", status_list)

        category_list = parse_csv(category)
        if category_list:
            query = query.in_("category", category_list)

        brand_list = parse_csv(brand)
        if brand_list:
            query = query.in_("brand_name", brand_list)

        sub_brand_list = parse_csv(sub_brand)
        if sub_brand_list:
            query = query.in_("sub_brand", sub_brand_list)

        product_name_list = parse_csv(product_name)
        if product_name_list:
            query = query.in_("product_name", product_name_list)

        variant_list = parse_csv(variant_flavor)
        if variant_list:
            query = query.in_("variant_flavor", variant_list)

        container_list = parse_csv(container_type)
        if container_list:
            query = query.in_("container_type", container_list)

        net_quantity_list = parse_csv(net_quantity)
        if net_quantity_list:
            query = query.in_("net_quantity", net_quantity_list)

        country_list = parse_csv(manufacturer_country)
        if country_list:
            query = query.in_("manufacturer_country", country_list)

        # Pack type filter (JSONB)
        pack_type_list = parse_csv(pack_type)
        if pack_type_list:
            if len(pack_type_list) == 1:
                query = query.eq("pack_configuration->>type", pack_type_list[0])
            else:
                # For multiple pack types, use or_ with individual conditions
                conditions = ",".join([f"pack_configuration->>type.eq.{pt}" for pt in pack_type_list])
                query = query.or_(conditions)

        # Boolean filters
        if has_video is True:
            query = query.not_.is_("video_url", "null")
        elif has_video is False:
            query = query.is_("video_url", "null")

        if has_image is True:
            query = query.not_.is_("primary_image_url", "null")
        elif has_image is False:
            query = query.is_("primary_image_url", "null")

        if has_description is True:
            query = query.not_.is_("marketing_description", "null")
        elif has_description is False:
            query = query.is_("marketing_description", "null")

        if has_prompt is True:
            query = query.not_.is_("grounding_prompt", "null")
        elif has_prompt is False:
            query = query.is_("grounding_prompt", "null")

        # JSONB filters - applied BEFORE pagination for correct filtering
        # Claims filter (JSONB array) - check if claims array contains any of the selected values
        claims_list = parse_csv(claims)
        if claims_list:
            # Use overlaps operator (&&) to check if arrays share any elements
            # PostgREST: use ov (overlaps) for array containment
            query = query.overlaps("claims", claims_list)

        # Nutrition filter (JSONB object) - check if nutrition_facts is non-empty
        if has_nutrition is True:
            # Has nutrition = nutrition_facts is not null AND not empty object
            query = query.not_.is_("nutrition_facts", "null")
            query = query.neq("nutrition_facts", "{}")
        elif has_nutrition is False:
            # No nutrition = nutrition_facts is null OR empty object
            query = query.or_("nutrition_facts.is.null,nutrition_facts.eq.{}")

        # Issues filter (JSONB array) - check if issues_detected has items
        if has_issues is True:
            # Has issues = array is not null AND not empty
            query = query.not_.is_("issues_detected", "null")
            query = query.neq("issues_detected", "[]")
        elif has_issues is False:
            # No issues = array is null OR empty
            query = query.or_("issues_detected.is.null,issues_detected.eq.[]")

        # Range filters
        if frame_count_min is not None:
            query = query.gte("frame_count", frame_count_min)
        if frame_count_max is not None:
            query = query.lte("frame_count", frame_count_max)
        if visibility_score_min is not None:
            query = query.gte("visibility_score", visibility_score_min)
        if visibility_score_max is not None:
            query = query.lte("visibility_score", visibility_score_max)

        # Sorting - validate and apply
        valid_sort_columns = {
            "barcode", "brand_name", "sub_brand", "product_name", "variant_flavor",
            "category", "container_type", "net_quantity", "manufacturer_country",
            "status", "created_at", "updated_at", "frame_count", "visibility_score"
        }
        sort_column = sort_by if sort_by in valid_sort_columns else "created_at"
        sort_desc = sort_order != "asc"  # Default to desc

        # Pagination - applied AFTER all filters for correct total count
        start = (page - 1) * limit
        end = start + limit - 1
        query = query.range(start, end).order(sort_column, desc=sort_desc)

        response = query.execute()
        items = response.data

        # Optionally include frame counts for each product
        if include_frame_counts and items:
            product_ids = [p["id"] for p in items]
            # Build legacy frame_count map for fallback
            legacy_counts = {p["id"]: p.get("frame_count", 0) for p in items}
            frame_counts_map = await self.get_batch_frame_counts(
                product_ids,
                check_storage=False,  # Disable slow storage checks for list view
                legacy_frame_counts=legacy_counts
            )
            for product in items:
                product["frame_counts"] = frame_counts_map.get(
                    product["id"],
                    {"synthetic": 0, "real": 0, "augmented": 0}
                )

        return {
            "items": items,
            "total": response.count or 0,
            "page": page,
            "limit": limit,
        }

    async def get_products_cursor(
        self,
        filters: Optional[dict[str, Any]] = None,
        after_id: Optional[str] = None,
        limit: int = 500,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Get products using cursor-based pagination for streaming exports.

        Uses ID-based cursoring which is more efficient than offset pagination
        for large datasets. Returns (products, has_more).

        Args:
            filters: Dict of filter parameters (status, category, brand, etc.)
            after_id: Last product ID from previous batch (cursor)
            limit: Number of products to fetch

        Returns:
            Tuple of (products list, has_more flag)
        """
        query = self.client.table("products").select("*")

        # Default to empty dict if filters is None
        if filters is None:
            filters = {}

        # Apply filters
        if filters.get("search"):
            search = filters["search"]
            query = query.or_(
                f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
            )

        if filters.get("status"):
            status_list = filters["status"] if isinstance(filters["status"], list) else [s.strip() for s in filters["status"].split(",")]
            query = query.in_("status", status_list)

        if filters.get("category"):
            cat_list = filters["category"] if isinstance(filters["category"], list) else [c.strip() for c in filters["category"].split(",")]
            query = query.in_("category", cat_list)

        if filters.get("brand"):
            brand_list = filters["brand"] if isinstance(filters["brand"], list) else [b.strip() for b in filters["brand"].split(",")]
            query = query.in_("brand_name", brand_list)

        if filters.get("sub_brand"):
            sub_list = filters["sub_brand"] if isinstance(filters["sub_brand"], list) else [s.strip() for s in filters["sub_brand"].split(",")]
            query = query.in_("sub_brand", sub_list)

        if filters.get("product_name"):
            pn_list = filters["product_name"] if isinstance(filters["product_name"], list) else [p.strip() for p in filters["product_name"].split(",")]
            query = query.in_("product_name", pn_list)

        if filters.get("variant_flavor"):
            vf_list = filters["variant_flavor"] if isinstance(filters["variant_flavor"], list) else [v.strip() for v in filters["variant_flavor"].split(",")]
            query = query.in_("variant_flavor", vf_list)

        if filters.get("container_type"):
            ct_list = filters["container_type"] if isinstance(filters["container_type"], list) else [c.strip() for c in filters["container_type"].split(",")]
            query = query.in_("container_type", ct_list)

        if filters.get("manufacturer_country"):
            mc_list = filters["manufacturer_country"] if isinstance(filters["manufacturer_country"], list) else [m.strip() for m in filters["manufacturer_country"].split(",")]
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

        if filters.get("has_nutrition") is True:
            query = query.not_.is_("nutrition_facts", "null")
            query = query.neq("nutrition_facts", "{}")
        elif filters.get("has_nutrition") is False:
            query = query.or_("nutrition_facts.is.null,nutrition_facts.eq.{}")

        if filters.get("has_description") is True:
            query = query.not_.is_("marketing_description", "null")
        elif filters.get("has_description") is False:
            query = query.is_("marketing_description", "null")

        if filters.get("has_prompt") is True:
            query = query.not_.is_("grounding_prompt", "null")
        elif filters.get("has_prompt") is False:
            query = query.is_("grounding_prompt", "null")

        if filters.get("has_issues") is True:
            query = query.not_.is_("issues_detected", "null")
            query = query.neq("issues_detected", "[]")
        elif filters.get("has_issues") is False:
            query = query.or_("issues_detected.is.null,issues_detected.eq.[]")

        # Range filters
        if filters.get("frame_count_min") is not None:
            query = query.gte("frame_count", filters["frame_count_min"])
        if filters.get("frame_count_max") is not None:
            query = query.lte("frame_count", filters["frame_count_max"])
        if filters.get("visibility_score_min") is not None:
            query = query.gte("visibility_score", filters["visibility_score_min"])
        if filters.get("visibility_score_max") is not None:
            query = query.lte("visibility_score", filters["visibility_score_max"])

        # Cursor pagination - fetch after the given ID
        if after_id:
            query = query.gt("id", after_id)

        # Order by ID for consistent pagination and fetch one extra to check has_more
        query = query.order("id").limit(limit + 1)

        response = query.execute()
        products = response.data or []

        # Check if there are more results
        has_more = len(products) > limit
        if has_more:
            products = products[:limit]

        return products, has_more

    async def get_products_by_ids(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get products by specific IDs.

        Used for exporting specific selected products.
        """
        if not product_ids:
            return []

        response = (
            self.client.table("products")
            .select("*")
            .in_("id", product_ids)
            .execute()
        )
        return response.data or []

    async def get_products_by_barcodes(
        self,
        barcodes: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get products by barcode values.

        Used for bulk update matching.
        Uses SQL IN clause for efficient querying.
        """
        if not barcodes:
            return []

        # Clean barcodes (strip whitespace, filter empty)
        clean_barcodes = [b.strip() for b in barcodes if b and b.strip()]
        if not clean_barcodes:
            return []

        # Use IN clause for efficient SQL filtering
        response = (
            self.client.table("products")
            .select("*")
            .in_("barcode", clean_barcodes)
            .execute()
        )
        return response.data or []

    async def get_identifiers_for_products(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get all identifiers for multiple products.

        Returns list of identifier records with product_id, identifier_type, identifier_value.
        """
        if not product_ids:
            return []

        response = (
            self.client.table("product_identifiers")
            .select("product_id, identifier_type, identifier_value")
            .in_("product_id", product_ids)
            .execute()
        )
        return response.data or []

    async def update_product_fields(
        self,
        product_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update product fields without version checking.

        Used for bulk updates where optimistic locking is not needed.
        """
        response = (
            self.client.table("products")
            .update({
                **fields,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("id", product_id)
            .execute()
        )

        if not response.data:
            raise ValueError(f"Product {product_id} not found")

        return response.data[0]

    async def upsert_product_identifiers(
        self,
        product_id: str,
        identifiers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Upsert product identifiers.

        Args:
            product_id: Product UUID
            identifiers: Dict of identifier_type -> identifier_value
                         e.g., {"sku": "ABC123", "upc": "012345678901"}
        """
        if not identifiers:
            return []

        results = []
        for id_type, id_value in identifiers.items():
            if not id_value:
                continue

            # Check if identifier already exists
            existing = (
                self.client.table("product_identifiers")
                .select("id")
                .eq("product_id", product_id)
                .eq("identifier_type", id_type)
                .execute()
            )

            if existing.data:
                # Update existing
                response = (
                    self.client.table("product_identifiers")
                    .update({
                        "identifier_value": id_value,
                        "updated_at": datetime.utcnow().isoformat(),
                    })
                    .eq("id", existing.data[0]["id"])
                    .execute()
                )
            else:
                # Insert new
                response = (
                    self.client.table("product_identifiers")
                    .insert({
                        "product_id": product_id,
                        "identifier_type": id_type,
                        "identifier_value": id_value,
                        "is_primary": False,
                    })
                    .execute()
                )

            if response.data:
                results.append(response.data[0])

        return results

    async def get_batch_frame_counts(
        self,
        product_ids: list[str],
        check_storage: bool = True,
        legacy_frame_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, dict[str, int]]:
        """Get frame counts for multiple products in one query.

        Args:
            product_ids: List of product UUIDs
            check_storage: If True, check storage for augmented files when DB count is 0
            legacy_frame_counts: Optional dict of product_id -> frame_count from products table
                                 Used as fallback for synthetic count when DB has no records
        """
        if not product_ids:
            return {}

        try:
            response = (
                self.client.table("product_images")
                .select("product_id, image_type")
                .in_("product_id", product_ids)
                .execute()
            )

            # Initialize counts for all products
            counts_map: dict[str, dict[str, int]] = {
                pid: {"synthetic": 0, "real": 0, "augmented": 0}
                for pid in product_ids
            }

            # Count by product and type
            for row in response.data:
                pid = row.get("product_id")
                img_type = row.get("image_type")
                if pid in counts_map and img_type in counts_map[pid]:
                    counts_map[pid][img_type] += 1

            # Fallback for legacy data - always use legacy frame_count for synthetic
            if legacy_frame_counts:
                for pid in product_ids:
                    if counts_map[pid]["synthetic"] == 0:
                        legacy_count = legacy_frame_counts.get(pid, 0)
                        if legacy_count > 0:
                            counts_map[pid]["synthetic"] = legacy_count

            # Optional: Check storage for augmented files (slow, disabled for list view)
            if check_storage:
                for pid in product_ids:
                    if counts_map[pid]["augmented"] == 0:
                        storage_count = self._count_augmented_files_in_storage(pid)
                        if storage_count > 0:
                            counts_map[pid]["augmented"] = storage_count

            return counts_map
        except Exception:
            return {pid: {"synthetic": 0, "real": 0, "augmented": 0} for pid in product_ids}

    def _count_augmented_files_in_storage(self, product_id: str) -> int:
        """Count augmented files in storage for a product."""
        try:
            files = self.client.storage.from_("frames").list(f"{product_id}/augmented")
            return len([
                f for f in files
                if f.get("name") and f.get("name").endswith(('.jpg', '.png', '.jpeg', '.webp'))
            ])
        except Exception:
            return 0

    async def get_product(self, product_id: str) -> Optional[dict[str, Any]]:
        """Get single product by ID."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("products")
            .select("*")
            .eq("id", product_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_product(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new product."""
        response = self.client.table("products").insert(data).execute()
        return response.data[0]

    async def get_product_by_barcode(self, barcode: str) -> Optional[dict[str, Any]]:
        """Get product by barcode."""
        response = (
            self.client.table("products")
            .select("*")
            .eq("barcode", barcode)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def get_product_identifiers(self, product_id: str) -> list[dict[str, Any]]:
        """
        Get all identifiers for a product from product_identifiers table.

        Returns list of dicts with:
            - identifier_type: barcode, upc, ean, short_code, sku, custom
            - identifier_value: the actual value
            - is_primary: whether this is the primary identifier
        """
        response = (
            self.client.table("product_identifiers")
            .select("identifier_type, identifier_value, is_primary")
            .eq("product_id", product_id)
            .order("is_primary", desc=True)
            .execute()
        )
        return response.data or []

    async def get_all_product_identifier_values(self, product_id: str) -> list[str]:
        """
        Get all identifier values for a product (just the values, not types).

        Also includes the legacy products.barcode field for backward compatibility.
        """
        # Get from product_identifiers table
        identifiers = await self.get_product_identifiers(product_id)
        values = [i["identifier_value"] for i in identifiers if i.get("identifier_value")]

        # Also get legacy barcode from products table
        product = await self.get_product(product_id)
        if product and product.get("barcode"):
            legacy_barcode = product["barcode"]
            if legacy_barcode not in values:
                values.append(legacy_barcode)

        return values

    async def get_products_by_identifier_value(
        self, identifier_value: str
    ) -> list[dict[str, Any]]:
        """
        Find products that have the given identifier value.

        Searches both:
        1. product_identifiers table (all identifier types)
        2. products.barcode field (legacy, for backward compatibility)

        Returns list of products with their details.
        """
        product_ids = set()

        # Search in product_identifiers table
        id_response = (
            self.client.table("product_identifiers")
            .select("product_id")
            .eq("identifier_value", identifier_value)
            .execute()
        )
        for row in id_response.data or []:
            product_ids.add(row["product_id"])

        # Search in legacy products.barcode field
        legacy_response = (
            self.client.table("products")
            .select("id")
            .eq("barcode", identifier_value)
            .execute()
        )
        for row in legacy_response.data or []:
            product_ids.add(row["id"])

        if not product_ids:
            return []

        # Fetch full product details
        products_response = (
            self.client.table("products")
            .select("id, barcode, brand_name, product_name, primary_image_url")
            .in_("id", list(product_ids))
            .execute()
        )
        return products_response.data or []

    async def get_or_create_product(self, data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """
        Get existing product by barcode or create new one.
        Returns (product, created) tuple where created is True if a new product was created.
        """
        barcode = data.get("barcode")
        if barcode:
            existing = await self.get_product_by_barcode(barcode)
            if existing:
                # Update the existing product with new data (video_url, video_id, status)
                update_data = {}
                if data.get("video_url") and not existing.get("video_url"):
                    update_data["video_url"] = data["video_url"]
                if data.get("video_id") and not existing.get("video_id"):
                    update_data["video_id"] = data["video_id"]
                if data.get("status"):
                    update_data["status"] = data["status"]

                if update_data:
                    updated = await self.update_product(existing["id"], update_data)
                    return updated, False
                return existing, False

        # Create new product
        product = await self.create_product(data)
        return product, True

    async def update_product(
        self, product_id: str, data: dict[str, Any], expected_version: Optional[int] = None
    ) -> dict[str, Any]:
        """Update product with optimistic locking."""
        query = self.client.table("products").update({
            **data,
            "updated_at": datetime.utcnow().isoformat(),
            "version": (expected_version or 0) + 1,
        }).eq("id", product_id)

        if expected_version is not None:
            query = query.eq("version", expected_version)

        response = query.execute()

        if not response.data:
            raise ValueError("Product was modified by another user")

        return response.data[0]

    async def delete_product(self, product_id: str) -> None:
        """Delete product."""
        self.client.table("products").delete().eq("id", product_id).execute()

    async def delete_products(self, product_ids: list[str]) -> int:
        """Bulk delete products."""
        response = (
            self.client.table("products")
            .delete()
            .in_("id", product_ids)
            .execute()
        )
        return len(response.data)

    async def delete_product_cascade(self, product_id: str) -> dict[str, Any]:
        """
        Delete a product and ALL related data:
        - Product frames from product_images table
        - Storage files from frames/{product_id}/ folder
        - Dataset references from dataset_products table
        - Product identifiers from product_identifiers table
        - The product itself

        Returns counts of deleted items.
        """
        result = {
            "frames_deleted": 0,
            "files_deleted": 0,
            "dataset_refs_deleted": 0,
            "identifiers_deleted": 0,
            "product_deleted": False,
        }

        # 1. Delete all frame records from database
        frames_response = (
            self.client.table("product_images")
            .delete()
            .eq("product_id", product_id)
            .execute()
        )
        result["frames_deleted"] = len(frames_response.data) if frames_response.data else 0

        # 2. Delete storage files (frames folder)
        result["files_deleted"] = await self.delete_folder("frames", product_id)

        # 3. Delete dataset references
        dataset_response = (
            self.client.table("dataset_products")
            .delete()
            .eq("product_id", product_id)
            .execute()
        )
        result["dataset_refs_deleted"] = len(dataset_response.data) if dataset_response.data else 0

        # 4. Delete product identifiers
        identifiers_response = (
            self.client.table("product_identifiers")
            .delete()
            .eq("product_id", product_id)
            .execute()
        )
        result["identifiers_deleted"] = len(identifiers_response.data) if identifiers_response.data else 0

        # 5. Finally delete the product itself
        self.client.table("products").delete().eq("id", product_id).execute()
        result["product_deleted"] = True

        return result

    async def delete_products_cascade(self, product_ids: list[str]) -> dict[str, Any]:
        """
        Bulk delete products and ALL related data with concurrent storage operations.

        Uses asyncio.gather with semaphore to parallelize storage deletions
        while respecting rate limits. Database operations are batched.

        Returns aggregated counts of deleted items.
        """
        result = {
            "frames_deleted": 0,
            "files_deleted": 0,
            "dataset_refs_deleted": 0,
            "identifiers_deleted": 0,
            "products_deleted": 0,
        }

        if not product_ids:
            return result

        # Invalidate caches when products are deleted
        try:
            from services.cache import invalidate_all_product_caches
            invalidate_all_product_caches()
        except ImportError:
            pass

        # 1. Delete all frame records from database (single batch operation)
        frames_response = (
            self.client.table("product_images")
            .delete()
            .in_("product_id", product_ids)
            .execute()
        )
        result["frames_deleted"] = len(frames_response.data) if frames_response.data else 0

        # 2. Delete storage files concurrently
        if settings.use_concurrent_storage_delete:
            max_concurrent = settings.storage_delete_max_concurrent
            semaphore = asyncio.Semaphore(max_concurrent)

            async def bounded_delete(product_id: str) -> int:
                """Delete with semaphore to limit concurrency."""
                async with semaphore:
                    try:
                        return await self.delete_folder("frames", product_id)
                    except Exception as e:
                        logger.error(f"Failed to delete storage for {product_id}: {e}")
                        return 0

            # Execute all deletes concurrently with bounded concurrency
            delete_tasks = [bounded_delete(pid) for pid in product_ids]
            delete_results = await asyncio.gather(*delete_tasks, return_exceptions=True)

            for res in delete_results:
                if isinstance(res, int):
                    result["files_deleted"] += res
                elif isinstance(res, Exception):
                    logger.error(f"Storage delete exception: {res}")
        else:
            # Sequential fallback
            for product_id in product_ids:
                result["files_deleted"] += await self.delete_folder("frames", product_id)

        # 3. Delete dataset references (single batch)
        dataset_response = (
            self.client.table("dataset_products")
            .delete()
            .in_("product_id", product_ids)
            .execute()
        )
        result["dataset_refs_deleted"] = len(dataset_response.data) if dataset_response.data else 0

        # 4. Delete product identifiers (single batch)
        identifiers_response = (
            self.client.table("product_identifiers")
            .delete()
            .in_("product_id", product_ids)
            .execute()
        )
        result["identifiers_deleted"] = len(identifiers_response.data) if identifiers_response.data else 0

        # 5. Finally delete the products (single batch)
        products_response = (
            self.client.table("products")
            .delete()
            .in_("id", product_ids)
            .execute()
        )
        result["products_deleted"] = len(products_response.data) if products_response.data else 0

        logger.info(
            f"Cascade deleted {result['products_deleted']} products, "
            f"{result['frames_deleted']} frames, {result['files_deleted']} storage files"
        )

        return result

    async def get_product_categories(self) -> list[str]:
        """Get distinct product categories."""
        response = (
            self.client.table("products")
            .select("category")
            .not_.is_("category", "null")
            .execute()
        )
        categories = set(item["category"] for item in response.data if item["category"])
        return sorted(list(categories))

    async def get_product_filter_options(
        self,
        # Current filter selections (for cascading filters)
        status: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        sub_brand: Optional[str] = None,
        product_name: Optional[str] = None,
        variant_flavor: Optional[str] = None,
        container_type: Optional[str] = None,
        net_quantity: Optional[str] = None,
        pack_type: Optional[str] = None,
        manufacturer_country: Optional[str] = None,
        claims_filter: Optional[str] = None,
        has_video: Optional[bool] = None,
        has_image: Optional[bool] = None,
        has_nutrition: Optional[bool] = None,
        has_description: Optional[bool] = None,
        has_prompt: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        frame_count_min: Optional[int] = None,
        frame_count_max: Optional[int] = None,
        visibility_score_min: Optional[int] = None,
        visibility_score_max: Optional[int] = None,
        exclude_dataset_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get unique values for filterable fields using disjunctive faceting.

        Uses optimized PostgreSQL RPC function when available (feature flag),
        with caching and fallback to Python implementation.

        Performance: RPC version ~200ms vs Python version ~3-5s for 10k products.
        """
        # Helper to parse comma-separated values
        def parse_csv(value: Optional[str]) -> Optional[list[str]]:
            if not value:
                return None
            return [v.strip() for v in value.split(",") if v.strip()] or None

        # Parse filter values
        filter_params = {
            "status": status,
            "category": category,
            "brand": brand,
            "sub_brand": sub_brand,
            "product_name": product_name,
            "variant_flavor": variant_flavor,
            "container_type": container_type,
            "net_quantity": net_quantity,
            "pack_type": pack_type,
            "manufacturer_country": manufacturer_country,
            "claims_filter": claims_filter,
            "has_video": has_video,
            "has_image": has_image,
            "has_nutrition": has_nutrition,
            "has_description": has_description,
            "has_prompt": has_prompt,
            "has_issues": has_issues,
            "frame_count_min": frame_count_min,
            "frame_count_max": frame_count_max,
            "visibility_score_min": visibility_score_min,
            "visibility_score_max": visibility_score_max,
            "exclude_dataset_id": exclude_dataset_id,
        }

        # Try cache first
        if settings.use_filter_options_cache:
            try:
                from services.cache import get_filter_options_cached, set_filter_options_cached
                cached = get_filter_options_cached(**filter_params)
                if cached is not None:
                    logger.debug("Filter options cache hit")
                    return cached
            except ImportError:
                pass

        # Try RPC function (much faster - single SQL query)
        if settings.use_rpc_filter_options:
            try:
                result = await self._get_product_filter_options_rpc(
                    status=parse_csv(status),
                    category=parse_csv(category),
                    brand=parse_csv(brand),
                    sub_brand=parse_csv(sub_brand),
                    product_name=parse_csv(product_name),
                    variant_flavor=parse_csv(variant_flavor),
                    container_type=parse_csv(container_type),
                    net_quantity=parse_csv(net_quantity),
                    pack_type=parse_csv(pack_type),
                    manufacturer_country=parse_csv(manufacturer_country),
                    claims=parse_csv(claims_filter),
                    has_video=has_video,
                    has_image=has_image,
                    has_nutrition=has_nutrition,
                    has_description=has_description,
                    has_prompt=has_prompt,
                    has_issues=has_issues,
                    frame_count_min=frame_count_min,
                    frame_count_max=frame_count_max,
                    visibility_score_min=visibility_score_min,
                    visibility_score_max=visibility_score_max,
                    exclude_dataset_id=exclude_dataset_id,
                )

                # Cache the result
                if settings.use_filter_options_cache:
                    try:
                        from services.cache import set_filter_options_cached
                        set_filter_options_cached(result, **filter_params)
                    except ImportError:
                        pass

                return result

            except Exception as e:
                logger.warning(f"RPC get_product_filter_options failed, falling back to Python: {e}")

        # Fallback to Python implementation
        result = await self._get_product_filter_options_fallback(
            status=status,
            category=category,
            brand=brand,
            sub_brand=sub_brand,
            product_name=product_name,
            variant_flavor=variant_flavor,
            container_type=container_type,
            net_quantity=net_quantity,
            pack_type=pack_type,
            manufacturer_country=manufacturer_country,
            claims_filter=claims_filter,
            has_video=has_video,
            has_image=has_image,
            has_nutrition=has_nutrition,
            has_description=has_description,
            has_prompt=has_prompt,
            has_issues=has_issues,
            frame_count_min=frame_count_min,
            frame_count_max=frame_count_max,
            visibility_score_min=visibility_score_min,
            visibility_score_max=visibility_score_max,
            exclude_dataset_id=exclude_dataset_id,
        )

        # Cache the result
        if settings.use_filter_options_cache:
            try:
                from services.cache import set_filter_options_cached
                set_filter_options_cached(result, **filter_params)
            except ImportError:
                pass

        return result

    async def _get_product_filter_options_rpc(
        self,
        status: Optional[list[str]] = None,
        category: Optional[list[str]] = None,
        brand: Optional[list[str]] = None,
        sub_brand: Optional[list[str]] = None,
        product_name: Optional[list[str]] = None,
        variant_flavor: Optional[list[str]] = None,
        container_type: Optional[list[str]] = None,
        net_quantity: Optional[list[str]] = None,
        pack_type: Optional[list[str]] = None,
        manufacturer_country: Optional[list[str]] = None,
        claims: Optional[list[str]] = None,
        has_video: Optional[bool] = None,
        has_image: Optional[bool] = None,
        has_nutrition: Optional[bool] = None,
        has_description: Optional[bool] = None,
        has_prompt: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        frame_count_min: Optional[int] = None,
        frame_count_max: Optional[int] = None,
        visibility_score_min: Optional[int] = None,
        visibility_score_max: Optional[int] = None,
        exclude_dataset_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get filter options using optimized PostgreSQL RPC function.

        This executes a single SQL query with aggregation instead of 13+ separate queries.
        """
        rpc_params = {
            "p_status": status,
            "p_category": category,
            "p_brand": brand,
            "p_sub_brand": sub_brand,
            "p_product_name": product_name,
            "p_variant_flavor": variant_flavor,
            "p_container_type": container_type,
            "p_net_quantity": net_quantity,
            "p_pack_type": pack_type,
            "p_manufacturer_country": manufacturer_country,
            "p_claims": claims,
            "p_has_video": has_video,
            "p_has_image": has_image,
            "p_has_nutrition": has_nutrition,
            "p_has_description": has_description,
            "p_has_prompt": has_prompt,
            "p_has_issues": has_issues,
            "p_frame_count_min": frame_count_min,
            "p_frame_count_max": frame_count_max,
            "p_visibility_score_min": visibility_score_min,
            "p_visibility_score_max": visibility_score_max,
            "p_exclude_dataset_id": exclude_dataset_id,
        }

        # Remove None values for cleaner RPC call
        rpc_params = {k: v for k, v in rpc_params.items() if v is not None}

        result = self.client.rpc("get_product_filter_options", rpc_params).execute()

        if result.data:
            return result.data

        # Return empty structure if no data
        return self._empty_filter_options()

    def _empty_filter_options(self) -> dict[str, Any]:
        """Return empty filter options structure."""
        return {
            "status": [],
            "category": [],
            "brand": [],
            "subBrand": [],
            "productName": [],
            "flavor": [],
            "container": [],
            "netQuantity": [],
            "packType": [],
            "country": [],
            "claims": [],
            "issueTypes": [],
            "hasVideo": {"trueCount": 0, "falseCount": 0},
            "hasImage": {"trueCount": 0, "falseCount": 0},
            "hasNutrition": {"trueCount": 0, "falseCount": 0},
            "hasDescription": {"trueCount": 0, "falseCount": 0},
            "hasPrompt": {"trueCount": 0, "falseCount": 0},
            "hasIssues": {"trueCount": 0, "falseCount": 0},
            "frameCount": {"min": 0, "max": 100},
            "visibilityScore": {"min": 0, "max": 100},
            "totalProducts": 0,
        }

    async def _get_product_filter_options_fallback(
        self,
        # Current filter selections (for cascading filters)
        status: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        sub_brand: Optional[str] = None,
        product_name: Optional[str] = None,
        variant_flavor: Optional[str] = None,
        container_type: Optional[str] = None,
        net_quantity: Optional[str] = None,
        pack_type: Optional[str] = None,
        manufacturer_country: Optional[str] = None,
        claims_filter: Optional[str] = None,
        has_video: Optional[bool] = None,
        has_image: Optional[bool] = None,
        has_nutrition: Optional[bool] = None,
        has_description: Optional[bool] = None,
        has_prompt: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        frame_count_min: Optional[int] = None,
        frame_count_max: Optional[int] = None,
        visibility_score_min: Optional[int] = None,
        visibility_score_max: Optional[int] = None,
        exclude_dataset_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fallback Python implementation for filter options.

        Uses disjunctive faceting with 13+ separate queries.
        Slower than RPC but works without database migration.
        """
        # Helper to parse comma-separated values
        def parse_csv(value: Optional[str]) -> list[str]:
            if not value:
                return []
            return [v.strip() for v in value.split(",") if v.strip()]

        # Parse all filter values upfront
        status_list = parse_csv(status)
        category_list = parse_csv(category)
        brand_list = parse_csv(brand)
        sub_brand_list = parse_csv(sub_brand)
        product_name_list = parse_csv(product_name)
        variant_list = parse_csv(variant_flavor)
        container_list = parse_csv(container_type)
        net_quantity_list = parse_csv(net_quantity)
        pack_type_list = parse_csv(pack_type)
        country_list = parse_csv(manufacturer_country)
        claims_list = parse_csv(claims_filter)

        # Get excluded product IDs if filtering by dataset
        excluded_ids: set[str] = set()
        if exclude_dataset_id:
            dp_response = (
                self.client.table("dataset_products")
                .select("product_id")
                .eq("dataset_id", exclude_dataset_id)
                .execute()
            )
            excluded_ids = {row["product_id"] for row in dp_response.data}

        def build_query(exclude_section: Optional[str] = None):
            """Build query with all filters except specified section."""
            query = self.client.table("products").select(
                "id, status, category, brand_name, sub_brand, product_name, "
                "variant_flavor, container_type, net_quantity, pack_configuration, "
                "manufacturer_country, claims, issues_detected, video_url, "
                "primary_image_url, nutrition_facts, marketing_description, "
                "grounding_prompt, frame_count, visibility_score"
            )

            if exclude_section != "status" and status_list:
                query = query.in_("status", status_list)
            if exclude_section != "category" and category_list:
                query = query.in_("category", category_list)
            if exclude_section != "brand" and brand_list:
                query = query.in_("brand_name", brand_list)
            if exclude_section != "subBrand" and sub_brand_list:
                query = query.in_("sub_brand", sub_brand_list)
            if exclude_section != "productName" and product_name_list:
                query = query.in_("product_name", product_name_list)
            if exclude_section != "flavor" and variant_list:
                query = query.in_("variant_flavor", variant_list)
            if exclude_section != "container" and container_list:
                query = query.in_("container_type", container_list)
            if exclude_section != "netQuantity" and net_quantity_list:
                query = query.in_("net_quantity", net_quantity_list)
            if exclude_section != "country" and country_list:
                query = query.in_("manufacturer_country", country_list)

            if exclude_section != "packType" and pack_type_list:
                if len(pack_type_list) == 1:
                    query = query.eq("pack_configuration->>type", pack_type_list[0])
                else:
                    conditions = ",".join([f"pack_configuration->>type.eq.{pt}" for pt in pack_type_list])
                    query = query.or_(conditions)

            if has_video is True:
                query = query.not_.is_("video_url", "null")
            elif has_video is False:
                query = query.is_("video_url", "null")

            if has_image is True:
                query = query.not_.is_("primary_image_url", "null")
            elif has_image is False:
                query = query.is_("primary_image_url", "null")

            if has_description is True:
                query = query.not_.is_("marketing_description", "null")
            elif has_description is False:
                query = query.is_("marketing_description", "null")

            if has_prompt is True:
                query = query.not_.is_("grounding_prompt", "null")
            elif has_prompt is False:
                query = query.is_("grounding_prompt", "null")

            if frame_count_min is not None:
                query = query.gte("frame_count", frame_count_min)
            if frame_count_max is not None:
                query = query.lte("frame_count", frame_count_max)
            if visibility_score_min is not None:
                query = query.gte("visibility_score", visibility_score_min)
            if visibility_score_max is not None:
                query = query.lte("visibility_score", visibility_score_max)

            return query

        def apply_client_side_filters(items: list[dict], exclude_section: Optional[str] = None) -> list[dict]:
            result = items
            if excluded_ids:
                result = [item for item in result if item["id"] not in excluded_ids]
            if exclude_section != "claims" and claims_list:
                result = [item for item in result if item.get("claims") and any(c in item["claims"] for c in claims_list)]
            if has_nutrition is True:
                result = [item for item in result if item.get("nutrition_facts") and len(item.get("nutrition_facts", {})) > 0]
            elif has_nutrition is False:
                result = [item for item in result if not item.get("nutrition_facts") or len(item.get("nutrition_facts", {})) == 0]
            if exclude_section != "issueTypes":
                if has_issues is True:
                    result = [item for item in result if item.get("issues_detected") and len(item.get("issues_detected", [])) > 0]
                elif has_issues is False:
                    result = [item for item in result if not item.get("issues_detected") or len(item.get("issues_detected", [])) == 0]
            return result

        def extract_unique(items: list[dict], key: str) -> list[dict]:
            counts: dict[str, int] = {}
            for item in items:
                value = item.get(key)
                if value:
                    counts[value] = counts.get(value, 0) + 1
            return [{"value": k, "label": k, "count": v} for k, v in sorted(counts.items())]

        def extract_nested(items: list[dict], key: str, nested_key: str) -> list[dict]:
            counts: dict[str, int] = {}
            for item in items:
                obj = item.get(key) or {}
                value = obj.get(nested_key) if isinstance(obj, dict) else None
                if value:
                    counts[value] = counts.get(value, 0) + 1
            return [{"value": k, "label": k, "count": v} for k, v in sorted(counts.items())]

        def extract_array(items: list[dict], key: str) -> list[dict]:
            counts: dict[str, int] = {}
            for item in items:
                values = item.get(key) or []
                for value in values:
                    if value:
                        counts[value] = counts.get(value, 0) + 1
            return [{"value": k, "label": k, "count": v} for k, v in sorted(counts.items())]

        def count_boolean(items: list[dict], predicate) -> dict:
            true_count = sum(1 for item in items if predicate(item))
            return {"trueCount": true_count, "falseCount": len(items) - true_count}

        def calc_range(items: list[dict], key: str) -> dict:
            values = [item.get(key) for item in items if item.get(key) is not None]
            if not values:
                return {"min": 0, "max": 100}
            return {"min": min(values), "max": max(values)}

        disjunctive_sections = [
            ("status", "status"), ("category", "category"), ("brand", "brand_name"),
            ("subBrand", "sub_brand"), ("productName", "product_name"), ("flavor", "variant_flavor"),
            ("container", "container_type"), ("netQuantity", "net_quantity"),
            ("packType", None), ("country", "manufacturer_country"),
            ("claims", None), ("issueTypes", None),
        ]

        section_results: dict[str, list[dict]] = {}
        for section_id, _ in disjunctive_sections:
            query = build_query(exclude_section=section_id)
            response = query.execute()
            items = apply_client_side_filters(response.data, exclude_section=section_id)
            section_results[section_id] = items

        all_filters_query = build_query(exclude_section=None)
        all_filters_response = all_filters_query.execute()
        all_filtered_items = apply_client_side_filters(all_filters_response.data, exclude_section=None)

        return {
            "status": extract_unique(section_results["status"], "status"),
            "category": extract_unique(section_results["category"], "category"),
            "brand": extract_unique(section_results["brand"], "brand_name"),
            "subBrand": extract_unique(section_results["subBrand"], "sub_brand"),
            "productName": extract_unique(section_results["productName"], "product_name"),
            "flavor": extract_unique(section_results["flavor"], "variant_flavor"),
            "container": extract_unique(section_results["container"], "container_type"),
            "netQuantity": extract_unique(section_results["netQuantity"], "net_quantity"),
            "packType": extract_nested(section_results["packType"], "pack_configuration", "type"),
            "country": extract_unique(section_results["country"], "manufacturer_country"),
            "claims": extract_array(section_results["claims"], "claims"),
            "issueTypes": extract_array(section_results["issueTypes"], "issues_detected"),
            "hasVideo": count_boolean(all_filtered_items, lambda x: bool(x.get("video_url"))),
            "hasImage": count_boolean(all_filtered_items, lambda x: bool(x.get("primary_image_url"))),
            "hasNutrition": count_boolean(all_filtered_items, lambda x: bool(x.get("nutrition_facts") and len(x.get("nutrition_facts", {})) > 0)),
            "hasDescription": count_boolean(all_filtered_items, lambda x: bool(x.get("marketing_description"))),
            "hasPrompt": count_boolean(all_filtered_items, lambda x: bool(x.get("grounding_prompt"))),
            "hasIssues": count_boolean(all_filtered_items, lambda x: bool(x.get("issues_detected") and len(x.get("issues_detected", [])) > 0)),
            "frameCount": calc_range(all_filtered_items, "frame_count"),
            "visibilityScore": calc_range(all_filtered_items, "visibility_score"),
            "totalProducts": len(all_filtered_items),
        }

    # =========================================
    # Product Identifiers
    # =========================================

    async def get_product_identifiers(self, product_id: str) -> list[dict[str, Any]]:
        """Get all identifiers for a product."""
        response = (
            self.client.table("product_identifiers")
            .select("*")
            .eq("product_id", product_id)
            .order("is_primary", desc=True)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data

    async def add_product_identifier(
        self, product_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Add identifier to a product."""
        response = self.client.table("product_identifiers").insert({
            "product_id": product_id,
            "identifier_type": data["identifier_type"],
            "identifier_value": data["identifier_value"],
            "custom_label": data.get("custom_label"),
            "is_primary": data.get("is_primary", False),
        }).execute()
        return response.data[0]

    async def update_product_identifier(
        self, identifier_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update a product identifier."""
        response = (
            self.client.table("product_identifiers")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", identifier_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_product_identifier(self, identifier_id: str) -> None:
        """Delete a product identifier."""
        self.client.table("product_identifiers").delete().eq("id", identifier_id).execute()

    async def replace_product_identifiers(
        self, product_id: str, identifiers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Replace all identifiers for a product."""
        # Delete existing identifiers
        self.client.table("product_identifiers").delete().eq("product_id", product_id).execute()

        # Insert new identifiers
        if identifiers:
            records = [{
                "product_id": product_id,
                "identifier_type": i["identifier_type"],
                "identifier_value": i["identifier_value"],
                "custom_label": i.get("custom_label"),
                "is_primary": i.get("is_primary", False),
            } for i in identifiers]

            response = self.client.table("product_identifiers").insert(records).execute()
            return response.data
        return []

    async def set_primary_identifier(
        self, product_id: str, identifier_id: str
    ) -> Optional[dict[str, Any]]:
        """Set an identifier as primary (trigger handles unsetting others)."""
        response = (
            self.client.table("product_identifiers")
            .update({"is_primary": True, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", identifier_id)
            .eq("product_id", product_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def search_by_identifier(self, value: str) -> list[dict[str, Any]]:
        """Search products by any identifier value."""
        response = (
            self.client.table("product_identifiers")
            .select("product_id, identifier_type, identifier_value")
            .ilike("identifier_value", f"%{value}%")
            .execute()
        )
        return response.data

    # =========================================
    # Datasets
    # =========================================

    async def get_datasets(self) -> list[dict[str, Any]]:
        """Get all datasets."""
        response = (
            self.client.table("datasets")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def get_dataset(
        self,
        dataset_id: str,
        page: int = 1,
        limit: int = 100,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "desc",
        # Comma-separated list filters
        status: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        sub_brand: Optional[str] = None,
        product_name: Optional[str] = None,
        variant_flavor: Optional[str] = None,
        container_type: Optional[str] = None,
        net_quantity: Optional[str] = None,
        pack_type: Optional[str] = None,
        manufacturer_country: Optional[str] = None,
        claims: Optional[str] = None,
        # Boolean filters
        has_video: Optional[bool] = None,
        has_image: Optional[bool] = None,
        has_nutrition: Optional[bool] = None,
        has_description: Optional[bool] = None,
        has_prompt: Optional[bool] = None,
        has_issues: Optional[bool] = None,
        # Range filters
        frame_count_min: Optional[int] = None,
        frame_count_max: Optional[int] = None,
        visibility_score_min: Optional[int] = None,
        visibility_score_max: Optional[int] = None,
        include_frame_counts: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Get dataset with filtered products and frame counts."""
        # Get dataset - Note: Using limit(1) instead of single() due to supabase-py bug
        dataset_response = (
            self.client.table("datasets")
            .select("*")
            .eq("id", dataset_id)
            .limit(1)
            .execute()
        )

        if not dataset_response.data:
            return None

        dataset = dataset_response.data[0]

        # Helper to parse comma-separated values
        def parse_csv(value: Optional[str]) -> list[str]:
            if not value:
                return []
            return [v.strip() for v in value.split(",") if v.strip()]

        # First, get all product IDs in this dataset
        product_ids_response = (
            self.client.table("dataset_products")
            .select("product_id")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        dataset_product_ids = [item["product_id"] for item in product_ids_response.data]

        if not dataset_product_ids:
            dataset["products"] = []
            dataset["products_total"] = 0
            dataset["total_synthetic"] = 0
            dataset["total_real"] = 0
            dataset["total_augmented"] = 0
            return dataset

        # Build query for products with filters
        query = self.client.table("products").select("*", count="exact")

        # Filter by products in this dataset
        query = query.in_("id", dataset_product_ids)

        # Search filter
        if search:
            query = query.or_(
                f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
            )

        # List filters
        status_list = parse_csv(status)
        if status_list:
            query = query.in_("status", status_list)

        category_list = parse_csv(category)
        if category_list:
            query = query.in_("category", category_list)

        brand_list = parse_csv(brand)
        if brand_list:
            query = query.in_("brand_name", brand_list)

        sub_brand_list = parse_csv(sub_brand)
        if sub_brand_list:
            query = query.in_("sub_brand", sub_brand_list)

        product_name_list = parse_csv(product_name)
        if product_name_list:
            query = query.in_("product_name", product_name_list)

        variant_list = parse_csv(variant_flavor)
        if variant_list:
            query = query.in_("variant_flavor", variant_list)

        container_list = parse_csv(container_type)
        if container_list:
            query = query.in_("container_type", container_list)

        net_quantity_list = parse_csv(net_quantity)
        if net_quantity_list:
            query = query.in_("net_quantity", net_quantity_list)

        manufacturer_country_list = parse_csv(manufacturer_country)
        if manufacturer_country_list:
            query = query.in_("manufacturer_country", manufacturer_country_list)

        # Boolean filters
        if has_video is not None:
            if has_video:
                query = query.not_.is_("video_url", "null")
            else:
                query = query.is_("video_url", "null")

        if has_image is not None:
            if has_image:
                query = query.not_.is_("primary_image_url", "null")
            else:
                query = query.is_("primary_image_url", "null")

        if has_nutrition is not None:
            if has_nutrition:
                query = query.not_.is_("nutrition_facts", "null")
            else:
                query = query.is_("nutrition_facts", "null")

        if has_description is not None:
            if has_description:
                query = query.not_.is_("marketing_description", "null")
            else:
                query = query.is_("marketing_description", "null")

        if has_prompt is not None:
            if has_prompt:
                query = query.not_.is_("grounding_prompt", "null")
            else:
                query = query.is_("grounding_prompt", "null")

        if has_issues is not None:
            if has_issues:
                query = query.not_.is_("issues_detected", "null")
            else:
                query = query.is_("issues_detected", "null")

        # Range filters
        if frame_count_min is not None:
            query = query.gte("frame_count", frame_count_min)
        if frame_count_max is not None:
            query = query.lte("frame_count", frame_count_max)

        if visibility_score_min is not None:
            query = query.gte("visibility_score", visibility_score_min)
        if visibility_score_max is not None:
            query = query.lte("visibility_score", visibility_score_max)

        # Sorting
        if sort_by:
            is_desc = sort_order == "desc"
            query = query.order(sort_by, desc=is_desc)
        else:
            query = query.order("created_at", desc=True)

        # Pagination
        offset = (page - 1) * limit
        query = query.range(offset, offset + limit - 1)

        # Execute query
        products_response = query.execute()

        products = products_response.data or []
        products_total = products_response.count or 0

        # Add frame counts if requested - use batch query for efficiency
        if include_frame_counts and products:
            product_ids = [p["id"] for p in products]
            legacy_counts = {p["id"]: p.get("frame_count", 0) for p in products}
            frame_counts_map = await self.get_batch_frame_counts(
                product_ids,
                check_storage=False,  # Disable slow storage checks for list view
                legacy_frame_counts=legacy_counts
            )
            for product in products:
                counts = frame_counts_map.get(
                    product["id"],
                    {"synthetic": 0, "real": 0, "augmented": 0}
                )
                product["frame_counts"] = counts
                product["total_frames"] = sum(counts.values())

        dataset["products"] = products
        dataset["products_total"] = products_total

        # Calculate dataset totals using a single aggregate query
        totals = await self._get_dataset_frame_totals(dataset_product_ids)
        dataset["total_synthetic"] = totals.get("synthetic", 0)
        dataset["total_real"] = totals.get("real", 0)
        dataset["total_augmented"] = totals.get("augmented", 0)

        return dataset

    async def _get_dataset_frame_totals(self, product_ids: list[str]) -> dict[str, int]:
        """Get aggregated frame counts for all products in a dataset using a single query.

        This is much more efficient than calling get_product_frame_counts for each product.
        """
        if not product_ids:
            return {"synthetic": 0, "real": 0, "augmented": 0}

        try:
            # Single query to get all frame counts grouped by type
            response = (
                self.client.table("product_images")
                .select("image_type")
                .in_("product_id", product_ids)
                .execute()
            )

            totals = {"synthetic": 0, "real": 0, "augmented": 0}
            for row in response.data:
                img_type = row.get("image_type")
                if img_type in totals:
                    totals[img_type] += 1

            # Fallback: If no synthetic frames in DB, sum from products table frame_count
            if totals["synthetic"] == 0:
                products_response = (
                    self.client.table("products")
                    .select("frame_count")
                    .in_("id", product_ids)
                    .execute()
                )
                totals["synthetic"] = sum(
                    p.get("frame_count", 0) or 0 for p in products_response.data
                )

            return totals
        except Exception:
            return {"synthetic": 0, "real": 0, "augmented": 0}

    async def get_dataset_filter_options(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """Get available filter options for products in a dataset."""
        # Check if dataset exists - Note: Using limit(1) instead of single() due to supabase-py bug
        dataset_response = (
            self.client.table("datasets")
            .select("id")
            .eq("id", dataset_id)
            .limit(1)
            .execute()
        )

        if not dataset_response.data:
            return None

        # Get all product IDs in this dataset
        product_ids_response = (
            self.client.table("dataset_products")
            .select("product_id")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        dataset_product_ids = [item["product_id"] for item in product_ids_response.data]

        if not dataset_product_ids:
            # Return empty options
            return {
                "status": [],
                "category": [],
                "brand": [],
                "subBrand": [],
                "productName": [],
                "flavor": [],
                "container": [],
                "netQuantity": [],
                "packType": [],
                "country": [],
                "claims": [],
                "issueTypes": [],
                "frameCount": {"min": 0, "max": 0},
                "visibilityScore": {"min": 0, "max": 100},
                "hasVideo": {"trueCount": 0, "falseCount": 0},
                "hasImage": {"trueCount": 0, "falseCount": 0},
                "hasNutrition": {"trueCount": 0, "falseCount": 0},
                "hasDescription": {"trueCount": 0, "falseCount": 0},
                "hasPrompt": {"trueCount": 0, "falseCount": 0},
                "hasIssues": {"trueCount": 0, "falseCount": 0},
            }

        # Get only the fields needed for filter options (not SELECT *)
        filter_fields = (
            "status,category,brand_name,sub_brand,product_name,variant_flavor,"
            "container_type,net_quantity,manufacturer_country,pack_configuration,"
            "claims,issues_detected,frame_count,visibility_score,"
            "video_url,primary_image_url,nutrition_facts,marketing_description,grounding_prompt"
        )
        products_response = (
            self.client.table("products")
            .select(filter_fields)
            .in_("id", dataset_product_ids)
            .execute()
        )
        products = products_response.data or []

        # Helper to count values
        def count_values(field: str) -> list[dict]:
            counts: dict[str, int] = {}
            for p in products:
                value = p.get(field)
                if value:
                    counts[value] = counts.get(value, 0) + 1
            return [{"value": k, "label": k, "count": v} for k, v in sorted(counts.items())]

        # Helper for boolean counts
        def count_boolean(field: str, check_not_null: bool = True) -> dict:
            true_count = 0
            false_count = 0
            for p in products:
                value = p.get(field)
                if check_not_null:
                    if value is not None and value != "" and value != []:
                        true_count += 1
                    else:
                        false_count += 1
                else:
                    if value:
                        true_count += 1
                    else:
                        false_count += 1
            return {"trueCount": true_count, "falseCount": false_count}

        # Helper for array fields
        def count_array_values(field: str) -> list[dict]:
            counts: dict[str, int] = {}
            for p in products:
                values = p.get(field) or []
                for value in values:
                    if value:
                        counts[value] = counts.get(value, 0) + 1
            return [{"value": k, "label": k, "count": v} for k, v in sorted(counts.items())]

        # Get frame count range
        frame_counts = [p.get("frame_count", 0) for p in products]
        visibility_scores = [p.get("visibility_score", 0) for p in products if p.get("visibility_score") is not None]

        # Count pack types from JSONB field
        pack_type_counts: dict[str, int] = {}
        for p in products:
            pack_config = p.get("pack_configuration")
            if pack_config and isinstance(pack_config, dict):
                pack_type = pack_config.get("type")
                if pack_type:
                    pack_type_counts[pack_type] = pack_type_counts.get(pack_type, 0) + 1

        return {
            "status": count_values("status"),
            "category": count_values("category"),
            "brand": count_values("brand_name"),
            "subBrand": count_values("sub_brand"),
            "productName": count_values("product_name"),
            "flavor": count_values("variant_flavor"),
            "container": count_values("container_type"),
            "netQuantity": count_values("net_quantity"),
            "packType": [{"value": k, "label": k, "count": v} for k, v in sorted(pack_type_counts.items())],
            "country": count_values("manufacturer_country"),
            "claims": count_array_values("claims"),
            "issueTypes": count_array_values("issues_detected"),
            "frameCount": {
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
            },
            "visibilityScore": {
                "min": min(visibility_scores) if visibility_scores else 0,
                "max": max(visibility_scores) if visibility_scores else 100,
            },
            "hasVideo": count_boolean("video_url"),
            "hasImage": count_boolean("primary_image_url"),
            "hasNutrition": count_boolean("nutrition_facts"),
            "hasDescription": count_boolean("marketing_description"),
            "hasPrompt": count_boolean("grounding_prompt"),
            "hasIssues": count_boolean("issues_detected"),
        }

    async def create_dataset(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new dataset."""
        response = self.client.table("datasets").insert({
            "name": data["name"],
            "description": data.get("description"),
            "product_count": 0,
        }).execute()
        return response.data[0]

    async def update_dataset(
        self, dataset_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update dataset."""
        response = (
            self.client.table("datasets")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", dataset_id)
            .execute()
        )
        return response.data[0]

    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete dataset and its product associations."""
        # Delete associations first
        self.client.table("dataset_products").delete().eq("dataset_id", dataset_id).execute()
        # Delete dataset
        self.client.table("datasets").delete().eq("id", dataset_id).execute()

    async def add_products_to_dataset(
        self, dataset_id: str, product_ids: list[str]
    ) -> int:
        """
        Add products to dataset using batched inserts to avoid timeouts.

        For large operations (1000+ products), this splits inserts into
        smaller batches to stay within Supabase timeout limits.
        """
        if not product_ids:
            return 0

        batch_size = settings.dataset_insert_batch_size if settings.use_batched_dataset_insert else len(product_ids)

        # Get existing products to avoid duplicates (also in batches for large lists)
        existing_ids: set[str] = set()
        for i in range(0, len(product_ids), 1000):
            batch = product_ids[i:i + 1000]
            existing = (
                self.client.table("dataset_products")
                .select("product_id")
                .eq("dataset_id", dataset_id)
                .in_("product_id", batch)
                .execute()
            )
            existing_ids.update(item["product_id"] for item in existing.data)

        # Filter to only new products
        new_ids = [pid for pid in product_ids if pid not in existing_ids]

        if not new_ids:
            return 0

        # Insert in batches
        total_inserted = 0
        errors = []

        for i in range(0, len(new_ids), batch_size):
            batch = new_ids[i:i + batch_size]
            records = [{"dataset_id": dataset_id, "product_id": pid} for pid in batch]

            try:
                self.client.table("dataset_products").insert(records).execute()
                total_inserted += len(batch)
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch)} products")
            except Exception as e:
                logger.error(f"Batch insert failed at offset {i}: {e}")
                errors.append({"offset": i, "error": str(e)})
                # Continue with remaining batches instead of failing completely

        # Update product count once at the end
        if total_inserted > 0:
            await self._update_dataset_product_count(dataset_id)

        if errors:
            logger.warning(f"Dataset insert completed with {len(errors)} batch errors")

        return total_inserted

    async def add_filtered_products_to_dataset(
        self, dataset_id: str, filters: dict
    ) -> dict:
        """Add all products matching filters to dataset using server-side RPC.

        This is used for "Select All Filtered" feature where user wants to add
        all products matching current filter criteria across all pages.
        
        Uses a PostgreSQL RPC function for efficient bulk insert that handles
        10K+ products without timeout or memory issues.
        
        Returns:
            dict with keys: added_count, skipped_count, total_matching, duration_ms
        """
        # Convert filters to the format expected by the RPC function
        rpc_filters = {}
        
        # Search filter
        if filters.get("search"):
            rpc_filters["search"] = filters["search"]
        
        # Array filters - ensure they are lists
        array_filter_keys = [
            "status", "category", "brand", "sub_brand", "product_name",
            "variant_flavor", "container_type", "net_quantity", 
            "pack_type", "manufacturer_country"
        ]
        
        for key in array_filter_keys:
            if filters.get(key):
                value = filters[key]
                if isinstance(value, str):
                    rpc_filters[key] = value.split(",")
                elif isinstance(value, list):
                    rpc_filters[key] = value
        
        # Boolean filters
        if filters.get("has_video") is not None:
            rpc_filters["has_video"] = str(filters["has_video"]).lower()
        if filters.get("has_image") is not None:
            rpc_filters["has_image"] = str(filters["has_image"]).lower()
        
        # Range filters
        if filters.get("frame_count_min") is not None:
            rpc_filters["frame_count_min"] = filters["frame_count_min"]
        if filters.get("frame_count_max") is not None:
            rpc_filters["frame_count_max"] = filters["frame_count_max"]
        
        try:
            # Call the RPC function
            response = self.client.rpc(
                "bulk_add_filtered_products_to_dataset",
                {
                    "p_dataset_id": dataset_id,
                    "p_filters": rpc_filters
                }
            ).execute()
            
            if response.data:
                result = response.data
                logger.info(
                    f"Bulk add to dataset {dataset_id}: "
                    f"added={result.get('added_count', 0)}, "
                    f"skipped={result.get('skipped_count', 0)}, "
                    f"total={result.get('total_matching', 0)}, "
                    f"duration={result.get('duration_ms', 0)}ms"
                )
                return result
            
            return {"added_count": 0, "skipped_count": 0, "total_matching": 0}
            
        except Exception as e:
            logger.error(f"RPC bulk_add_filtered_products_to_dataset failed: {e}")
            # Fallback to pagination-based approach if RPC fails
            logger.info("Falling back to pagination-based bulk add...")
            return await self._add_filtered_products_to_dataset_fallback(dataset_id, filters)
    
    async def _add_filtered_products_to_dataset_fallback(
        self, dataset_id: str, filters: dict
    ) -> dict:
        """Fallback method using pagination when RPC is not available."""
        all_product_ids = []
        page_size = 1000
        offset = 0
        
        while True:
            # Build query with pagination
            query = self.client.table("products").select("id")
            
            # Apply filters
            if filters.get("search"):
                search = filters["search"]
                query = query.or_(
                    f"barcode.ilike.%{search}%,product_name.ilike.%{search}%,brand_name.ilike.%{search}%"
                )

            if filters.get("status"):
                status_list = filters["status"] if isinstance(filters["status"], list) else filters["status"].split(",")
                query = query.in_("status", status_list)

            if filters.get("category"):
                cat_list = filters["category"] if isinstance(filters["category"], list) else filters["category"].split(",")
                query = query.in_("category", cat_list)

            if filters.get("brand"):
                brand_list = filters["brand"] if isinstance(filters["brand"], list) else filters["brand"].split(",")
                query = query.in_("brand_name", brand_list)

            if filters.get("sub_brand"):
                sub_list = filters["sub_brand"] if isinstance(filters["sub_brand"], list) else filters["sub_brand"].split(",")
                query = query.in_("sub_brand", sub_list)

            if filters.get("product_name"):
                pn_list = filters["product_name"] if isinstance(filters["product_name"], list) else filters["product_name"].split(",")
                query = query.in_("product_name", pn_list)

            if filters.get("variant_flavor"):
                vf_list = filters["variant_flavor"] if isinstance(filters["variant_flavor"], list) else filters["variant_flavor"].split(",")
                query = query.in_("variant_flavor", vf_list)

            if filters.get("container_type"):
                ct_list = filters["container_type"] if isinstance(filters["container_type"], list) else filters["container_type"].split(",")
                query = query.in_("container_type", ct_list)

            if filters.get("net_quantity"):
                nq_list = filters["net_quantity"] if isinstance(filters["net_quantity"], list) else filters["net_quantity"].split(",")
                query = query.in_("net_quantity", nq_list)
            
            if filters.get("pack_type"):
                pt_list = filters["pack_type"] if isinstance(filters["pack_type"], list) else filters["pack_type"].split(",")
                query = query.in_("pack_type", pt_list)

            if filters.get("manufacturer_country"):
                mc_list = filters["manufacturer_country"] if isinstance(filters["manufacturer_country"], list) else filters["manufacturer_country"].split(",")
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

            # Apply pagination
            query = query.range(offset, offset + page_size - 1)
            
            response = query.execute()
            
            if not response.data:
                break
            
            all_product_ids.extend([item["id"] for item in response.data])
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
            logger.debug(f"Fetched {len(all_product_ids)} product IDs so far...")

        if not all_product_ids:
            return {"added_count": 0, "skipped_count": 0, "total_matching": 0}

        total_matching = len(all_product_ids)
        
        # Use existing method to add products (handles duplicates and batching)
        added_count = await self.add_products_to_dataset(dataset_id, all_product_ids)
        
        return {
            "added_count": added_count,
            "skipped_count": total_matching - added_count,
            "total_matching": total_matching
        }

    async def remove_product_from_dataset(
        self, dataset_id: str, product_id: str
    ) -> None:
        """Remove product from dataset."""
        self.client.table("dataset_products").delete().eq(
            "dataset_id", dataset_id
        ).eq("product_id", product_id).execute()

        await self._update_dataset_product_count(dataset_id)

    async def _update_dataset_product_count(self, dataset_id: str) -> None:
        """Update the product count for a dataset."""
        count_response = (
            self.client.table("dataset_products")
            .select("*", count="exact")
            .eq("dataset_id", dataset_id)
            .execute()
        )
        count = count_response.count or 0

        self.client.table("datasets").update({"product_count": count}).eq(
            "id", dataset_id
        ).execute()

    # =========================================
    # Jobs
    # =========================================

    async def get_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get jobs, optionally filtered by type and status."""
        query = self.client.table("jobs").select("*").order("created_at", desc=True)

        if job_type:
            query = query.eq("type", job_type)
        if status:
            query = query.eq("status", status)

        response = query.limit(limit).execute()
        return response.data

    async def get_active_jobs_count(self, job_type: Optional[str] = None) -> int:
        """Get count of active (running/queued/pending) jobs."""
        query = (
            self.client.table("jobs")
            .select("id", count="exact")
            .in_("status", ["running", "queued", "pending"])
        )
        if job_type:
            query = query.eq("type", job_type)
        response = query.execute()
        return response.count or 0

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get single job."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("jobs").select("*").eq("id", job_id).limit(1).execute()
        )
        return response.data[0] if response.data else None

    async def get_job_by_runpod_id(self, runpod_job_id: str) -> Optional[dict[str, Any]]:
        """Get job by Runpod job ID (for webhook lookups)."""
        response = (
            self.client.table("jobs")
            .select("*")
            .eq("runpod_job_id", runpod_job_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new job."""
        response = self.client.table("jobs").insert({
            "type": data["type"],
            "status": "pending",
            "progress": 0,
            "config": data.get("config", {}),
        }).execute()
        return response.data[0]

    async def update_job(self, job_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update job status."""
        response = (
            self.client.table("jobs")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", job_id)
            .execute()
        )
        return response.data[0]

    # =========================================
    # Training Jobs
    # =========================================

    async def get_training_jobs(self) -> list[dict[str, Any]]:
        """Get training jobs with dataset info."""
        response = (
            self.client.table("training_jobs")
            .select("*, datasets(name)")
            .order("created_at", desc=True)
            .execute()
        )

        # Flatten dataset name
        for job in response.data:
            if job.get("datasets"):
                job["dataset_name"] = job["datasets"]["name"]
                del job["datasets"]

        return response.data

    async def create_training_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create training job."""
        # Create base job
        job = await self.create_job({"type": "training", "config": data})

        # Create training-specific record
        training_job = {
            "id": job["id"],
            "dataset_id": data["dataset_id"],
            "epochs": data.get("epochs", 30),
            "batch_size": data.get("batch_size", 32),
            "learning_rate": data.get("learning_rate", 0.0001),
        }
        self.client.table("training_jobs").insert(training_job).execute()

        return {**job, **training_job}

    # =========================================
    # Videos
    # =========================================

    async def get_videos(self, limit: int = 10000) -> list[dict[str, Any]]:
        """Get all videos using pagination (Supabase has 1000 row limit per request)."""
        all_videos = []
        page_size = 1000
        offset = 0

        while offset < limit:
            response = (
                self.client.table("videos")
                .select("*")
                .order("created_at", desc=True)
                .range(offset, offset + page_size - 1)
                .execute()
            )

            if not response.data:
                break

            all_videos.extend(response.data)
            offset += page_size

            # If we got less than page_size, we've reached the end
            if len(response.data) < page_size:
                break

        return all_videos[:limit]

    async def sync_videos_from_buybuddy(self, videos: list[dict[str, Any]]) -> int:
        """Sync videos from Buybuddy API."""
        if not videos:
            return 0

        # Process in batches to avoid Supabase query size limits
        BATCH_SIZE = 100
        video_urls = [v["video_url"] for v in videos if v.get("video_url")]

        # Get existing video_urls in batches
        existing_urls = set()
        for i in range(0, len(video_urls), BATCH_SIZE):
            batch_urls = video_urls[i:i + BATCH_SIZE]
            existing = (
                self.client.table("videos")
                .select("video_url")
                .in_("video_url", batch_urls)
                .execute()
            )
            existing_urls.update(
                item["video_url"] for item in existing.data if item.get("video_url")
            )

        # Insert new videos (filter by video_url to avoid duplicates)
        new_videos = []
        for v in videos:
            if v.get("video_url") and v["video_url"] not in existing_urls:
                # Remove video_id as it doesn't exist in table schema
                video_data = {k: v[k] for k in v if k != "video_id"}
                new_videos.append(video_data)

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(new_videos), BATCH_SIZE):
            batch = new_videos[i:i + BATCH_SIZE]
            self.client.table("videos").insert(batch).execute()
            total_inserted += len(batch)

        return total_inserted

    # =========================================
    # Model Artifacts
    # =========================================

    async def get_models(self) -> list[dict[str, Any]]:
        """Get all model artifacts."""
        response = (
            self.client.table("model_artifacts")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def activate_model(self, model_id: str) -> dict[str, Any]:
        """Activate a model and deactivate others."""
        # Deactivate all models
        self.client.table("model_artifacts").update({"is_active": False}).execute()

        # Activate selected model
        response = (
            self.client.table("model_artifacts")
            .update({"is_active": True})
            .eq("id", model_id)
            .execute()
        )
        return response.data[0]

    # =========================================
    # Embedding Indexes
    # =========================================

    async def get_embedding_indexes(self) -> list[dict[str, Any]]:
        """Get all embedding indexes."""
        response = (
            self.client.table("embedding_indexes")
            .select("*, model_artifacts(name)")
            .order("created_at", desc=True)
            .execute()
        )

        # Flatten model name
        for idx in response.data:
            if idx.get("model_artifacts"):
                idx["model_name"] = idx["model_artifacts"]["name"]
                del idx["model_artifacts"]

        return response.data

    async def create_embedding_index(
        self, name: str, model_id: str
    ) -> dict[str, Any]:
        """Create new embedding index."""
        response = self.client.table("embedding_indexes").insert({
            "name": name,
            "model_artifact_id": model_id,
            "vector_count": 0,
            "index_path": f"/indexes/{name.lower().replace(' ', '_')}.faiss",
        }).execute()
        return response.data[0]

    # =========================================
    # Product Images (unified: synthetic, real, augmented)
    # =========================================

    async def get_product_frames(
        self, product_id: str, image_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Get frames/images for a product.

        Args:
            product_id: Product UUID
            image_type: Optional filter - 'synthetic', 'real', or 'augmented'
        """
        try:
            query = (
                self.client.table("product_images")
                .select("*")
                .eq("product_id", product_id)
            )
            if image_type:
                query = query.eq("image_type", image_type)

            response = query.order("frame_index", desc=False).order("created_at", desc=False).execute()
            return response.data
        except Exception:
            return []

    async def get_product_frame_counts(self, product_id: str, check_storage: bool = True) -> dict[str, int]:
        """Get frame counts by type for a product.

        Args:
            product_id: Product UUID
            check_storage: If True, check storage for augmented files when DB count is 0
        """
        try:
            response = (
                self.client.table("product_images")
                .select("image_type")
                .eq("product_id", product_id)
                .execute()
            )
            counts = {"synthetic": 0, "real": 0, "augmented": 0}
            for row in response.data:
                img_type = row.get("image_type")
                if img_type in counts:
                    counts[img_type] += 1

            # Check storage for augmented files if DB shows 0 (handles legacy data)
            if check_storage and counts["augmented"] == 0:
                storage_count = self._count_augmented_files_in_storage(product_id)
                if storage_count > 0:
                    counts["augmented"] = storage_count

            return counts
        except Exception:
            return {"synthetic": 0, "real": 0, "augmented": 0}

    async def sync_augmented_from_storage(self, product_id: str) -> dict[str, Any]:
        """
        Sync augmented images from storage to database.

        Scans the storage for augmented files and registers them in the database
        if they don't already exist.

        Returns sync statistics.
        """
        try:
            # Check existing records
            existing_response = (
                self.client.table("product_images")
                .select("image_path")
                .eq("product_id", product_id)
                .eq("image_type", "augmented")
                .execute()
            )
            existing_paths = {r.get("image_path") for r in existing_response.data}

            # List files in storage
            files = self.client.storage.from_("frames").list(f"{product_id}/augmented")
            image_files = [
                f for f in files
                if f.get("name") and f.get("name").endswith(('.jpg', '.png', '.jpeg', '.webp'))
            ]

            # Prepare records for files not in database
            new_records = []
            for f in image_files:
                file_name = f.get("name")
                storage_path = f"{product_id}/augmented/{file_name}"

                if storage_path not in existing_paths:
                    # Determine source from filename
                    if "light" in file_name:
                        source = "aug_syn_light"
                    elif "heavy" in file_name:
                        source = "aug_syn_heavy"
                    elif "_aug_" in file_name:
                        source = "aug_real"
                    else:
                        source = "augmented"

                    image_url = f"{self.client.supabase_url}/storage/v1/object/public/frames/{storage_path}"
                    new_records.append({
                        "product_id": product_id,
                        "image_type": "augmented",
                        "source": source,
                        "image_path": storage_path,
                        "image_url": image_url,
                    })

            # Insert new records
            inserted = 0
            if new_records:
                response = self.client.table("product_images").insert(new_records).execute()
                inserted = len(response.data)

            return {
                "product_id": product_id,
                "storage_files": len(image_files),
                "existing_records": len(existing_paths),
                "new_records_added": inserted,
            }
        except Exception as e:
            return {
                "product_id": product_id,
                "error": str(e),
            }

    async def add_product_frames(
        self,
        product_id: str,
        frames: list[dict],
    ) -> int:
        """
        Add frames to a product.

        Args:
            product_id: Product UUID
            frames: List of dicts with keys: image_type, source, image_path, image_url, frame_index (optional)
        """
        if not frames:
            return 0

        records = []
        for frame in frames:
            records.append({
                "product_id": product_id,
                "image_type": frame.get("image_type", "synthetic"),
                "source": frame.get("source", "video_frame"),
                "image_path": frame.get("image_path"),
                "image_url": frame.get("image_url"),
                "frame_index": frame.get("frame_index"),
            })

        response = self.client.table("product_images").insert(records).execute()
        return len(response.data)

    async def delete_product_frames(
        self, product_id: str, frame_ids: list[str]
    ) -> int:
        """Delete specific frames from a product."""
        if not frame_ids:
            return 0

        response = (
            self.client.table("product_images")
            .delete()
            .eq("product_id", product_id)
            .in_("id", frame_ids)
            .execute()
        )
        return len(response.data) if response.data else 0

    # Backward compatible methods for real images
    async def get_real_images(self, product_id: str) -> list[dict[str, Any]]:
        """Get real images for a product (backward compatible)."""
        return await self.get_product_frames(product_id, image_type="real")

    async def get_real_image_count(self, product_id: str) -> int:
        """Get count of real images for a product."""
        counts = await self.get_product_frame_counts(product_id)
        return counts.get("real", 0)

    async def add_real_images(
        self, product_id: str, image_urls: list[str]
    ) -> int:
        """Add real images to a product (backward compatible)."""
        if not image_urls:
            return 0

        frames = [
            {
                "image_type": "real",
                "source": "matching",
                "image_url": url,
                "image_path": url,  # For real images, path is the URL
            }
            for url in image_urls
        ]
        return await self.add_product_frames(product_id, frames)

    async def remove_real_images(
        self, product_id: str, image_ids: list[str]
    ) -> None:
        """Remove real images from a product (backward compatible)."""
        await self.delete_product_frames(product_id, image_ids)

    # =========================================
    # Storage
    # =========================================

    async def upload_file(
        self, bucket: str, path: str, file_data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload file to Supabase Storage."""
        self.client.storage.from_(bucket).upload(
            path, file_data, {"content-type": content_type}
        )
        return self.get_public_url(bucket, path)

    def get_public_url(self, bucket: str, path: str) -> str:
        """Get public URL for a file."""
        return f"{settings.supabase_url}/storage/v1/object/public/{bucket}/{path}"

    async def delete_file(self, bucket: str, path: str) -> None:
        """Delete file from storage."""
        self.client.storage.from_(bucket).remove([path])

    async def delete_folder(self, bucket: str, folder_path: str) -> int:
        """Delete all files in a storage folder."""
        try:
            # List all files in the folder
            files = self.client.storage.from_(bucket).list(folder_path)
            if not files:
                return 0

            # Build full paths and delete
            paths = [f"{folder_path}/{f['name']}" for f in files]
            if paths:
                self.client.storage.from_(bucket).remove(paths)
            return len(paths)
        except Exception as e:
            print(f"[Supabase] Error deleting folder {folder_path}: {e}")
            return 0

    async def cleanup_product_for_reprocess(self, product_id: str) -> dict[str, Any]:
        """
        Clean up a product's synthetic frames for reprocessing.
        Returns counts of deleted items.
        """
        result = {"frames_deleted": 0, "files_deleted": 0}

        # 1. Delete synthetic frame records from database
        response = (
            self.client.table("product_images")
            .delete()
            .eq("product_id", product_id)
            .eq("image_type", "synthetic")
            .execute()
        )
        result["frames_deleted"] = len(response.data) if response.data else 0

        # 2. Delete storage files
        result["files_deleted"] = await self.delete_folder("frames", product_id)

        # 3. Reset product frame-related fields
        self.client.table("products").update({
            "status": "processing",
            "frame_count": 0,
            "frames_path": None,
            "primary_image_url": None,
        }).eq("id", product_id).execute()

        return result

    # =========================================
    # Cutout Images
    # =========================================

    async def get_cutouts(
        self,
        page: int = 1,
        limit: int = 50,
        has_embedding: Optional[bool] = None,
        is_matched: Optional[bool] = None,
        predicted_upc: Optional[str] = None,
        # NEW: Visual picker filters
        matched_category: Optional[str] = None,
        matched_brand: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get cutout images with pagination and filters."""
        # Determine if we need to JOIN with products table
        needs_product_join = matched_category or matched_brand or search
        
        if needs_product_join:
            # Use JOIN query for product filters
            select_fields = "*, products!left(id, product_name, brand_name, category)"
            query = self.client.table("cutout_images").select(select_fields, count="exact")
        else:
            query = self.client.table("cutout_images").select("*", count="exact")

        if has_embedding is not None:
            query = query.eq("has_embedding", has_embedding)
        if is_matched is True:
            query = query.not_.is_("matched_product_id", "null")
        elif is_matched is False:
            query = query.is_("matched_product_id", "null")
        if predicted_upc:
            query = query.eq("predicted_upc", predicted_upc)
        
        # Date range filters
        if date_from:
            query = query.gte("synced_at", f"{date_from}T00:00:00")
        if date_to:
            query = query.lte("synced_at", f"{date_to}T23:59:59")
        
        # Product-based filters (requires matched cutouts)
        if matched_category:
            query = query.not_.is_("matched_product_id", "null")
            query = query.eq("products.category", matched_category)
        
        if matched_brand:
            query = query.not_.is_("matched_product_id", "null")
            query = query.eq("products.brand_name", matched_brand)
        
        if search:
            search_term = f"%{search}%"
            # Search in predicted_upc or matched product name
            query = query.or_(
                f"predicted_upc.ilike.{search_term},"
                f"products.product_name.ilike.{search_term},"
                f"products.brand_name.ilike.{search_term}"
            )

        # Pagination
        start = (page - 1) * limit
        end = start + limit - 1
        query = query.range(start, end).order("synced_at", desc=True)

        response = query.execute()
        
        # Extract cutout data, flatten product info if joined
        items = []
        for row in response.data or []:
            item = {k: v for k, v in row.items() if k != "products"}
            if "products" in row and row["products"]:
                item["matched_product"] = row["products"]
            items.append(item)
        
        return {
            "items": items,
            "total": response.count or 0,
            "page": page,
            "limit": limit,
        }

    async def get_cutout(self, cutout_id: str) -> Optional[dict[str, Any]]:
        """Get single cutout by ID."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("cutout_images")
            .select("*")
            .eq("id", cutout_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def get_cutout_by_external_id(self, external_id: int) -> Optional[dict[str, Any]]:
        """Get cutout by BuyBuddy external ID."""
        response = (
            self.client.table("cutout_images")
            .select("*")
            .eq("external_id", external_id)
            .maybe_single()
            .execute()
        )
        return response.data

    async def sync_cutouts(self, cutouts: list[dict[str, Any]]) -> dict[str, int]:
        """
        Sync cutout images from BuyBuddy API.

        Args:
            cutouts: List of cutout dicts with external_id, image_url, predicted_upc

        Returns:
            Dict with synced_count and skipped_count
        """
        if not cutouts:
            return {"synced_count": 0, "skipped_count": 0}

        # Get existing external IDs
        external_ids = [c["external_id"] for c in cutouts]
        existing = (
            self.client.table("cutout_images")
            .select("external_id")
            .in_("external_id", external_ids)
            .execute()
        )
        existing_ids = {item["external_id"] for item in existing.data}

        # Filter new cutouts
        new_cutouts = [
            {
                "external_id": c["external_id"],
                "image_url": c["image_url"],
                "predicted_upc": c.get("predicted_upc"),
            }
            for c in cutouts
            if c["external_id"] not in existing_ids
        ]

        if new_cutouts:
            self.client.table("cutout_images").insert(new_cutouts).execute()

        return {
            "synced_count": len(new_cutouts),
            "skipped_count": len(cutouts) - len(new_cutouts),
        }

    async def update_cutout(
        self, cutout_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update cutout image."""
        response = (
            self.client.table("cutout_images")
            .update(data)
            .eq("id", cutout_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def update_cutouts_bulk(
        self, cutout_ids: list[str], data: dict[str, Any]
    ) -> int:
        """Bulk update cutouts."""
        if not cutout_ids:
            return 0
        response = (
            self.client.table("cutout_images")
            .update(data)
            .in_("id", cutout_ids)
            .execute()
        )
        return len(response.data) if response.data else 0

    async def match_cutout_to_product(
        self,
        cutout_id: str,
        product_id: str,
        similarity: Optional[float] = None,
        matched_by: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Mark a cutout as matched to a product."""
        data = {
            "matched_product_id": product_id,
            "matched_at": datetime.utcnow().isoformat(),
        }
        if similarity is not None:
            data["match_similarity"] = similarity
        if matched_by:
            data["matched_by"] = matched_by

        return await self.update_cutout(cutout_id, data)

    async def get_cutout_stats(self) -> dict[str, int]:
        """Get cutout statistics."""
        # Total count
        total_resp = (
            self.client.table("cutout_images")
            .select("*", count="exact")
            .execute()
        )
        total = total_resp.count or 0

        # With embedding
        with_emb_resp = (
            self.client.table("cutout_images")
            .select("*", count="exact")
            .eq("has_embedding", True)
            .execute()
        )
        with_embedding = with_emb_resp.count or 0

        # Matched
        matched_resp = (
            self.client.table("cutout_images")
            .select("*", count="exact")
            .not_.is_("matched_product_id", "null")
            .execute()
        )
        matched = matched_resp.count or 0

        return {
            "total": total,
            "with_embedding": with_embedding,
            "without_embedding": total - with_embedding,
            "matched": matched,
            "unmatched": total - matched,
        }

    # =========================================
    # Embedding Models
    # =========================================

    async def get_embedding_models(self) -> list[dict[str, Any]]:
        """Get all embedding models."""
        response = (
            self.client.table("embedding_models")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def get_embedding_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get single embedding model."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("embedding_models")
            .select("*")
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def get_active_embedding_model(self) -> Optional[dict[str, Any]]:
        """Get the active matching model."""
        response = (
            self.client.table("embedding_models")
            .select("*")
            .eq("is_matching_active", True)
            .maybe_single()
            .execute()
        )
        return response.data

    async def create_embedding_model(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create new embedding model."""
        response = (
            self.client.table("embedding_models")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_embedding_model(
        self, model_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update embedding model."""
        response = (
            self.client.table("embedding_models")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", model_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def activate_embedding_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Activate a model for matching (deactivates others)."""
        # Deactivate all currently active models
        self.client.table("embedding_models").update(
            {"is_matching_active": False}
        ).eq("is_matching_active", True).execute()

        # Activate selected
        response = (
            self.client.table("embedding_models")
            .update({
                "is_matching_active": True,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("id", model_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_embedding_model(self, model_id: str) -> bool:
        """Delete embedding model."""
        self.client.table("embedding_models").delete().eq("id", model_id).execute()
        return True

    # =========================================
    # Embedding Jobs
    # =========================================

    async def get_embedding_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get embedding jobs."""
        query = (
            self.client.table("embedding_jobs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if status:
            query = query.eq("status", status)
        response = query.execute()
        return response.data

    async def get_embedding_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get single embedding job."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("embedding_jobs")
            .select("*")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_embedding_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create embedding job."""
        response = (
            self.client.table("embedding_jobs")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_embedding_job(
        self, job_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update embedding job."""
        response = (
            self.client.table("embedding_jobs")
            .update(data)
            .eq("id", job_id)
            .execute()
        )
        return response.data[0] if response.data else None

    # =========================================
    # Embedding Exports
    # =========================================

    async def get_embedding_exports(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get embedding exports."""
        response = (
            self.client.table("embedding_exports")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data

    async def get_embedding_export(self, export_id: str) -> Optional[dict[str, Any]]:
        """Get single export."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("embedding_exports")
            .select("*")
            .eq("id", export_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_embedding_export(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create export record."""
        response = (
            self.client.table("embedding_exports")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_embedding_export(
        self, export_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update export record."""
        response = (
            self.client.table("embedding_exports")
            .update(data)
            .eq("id", export_id)
            .execute()
        )
        return response.data[0] if response.data else None

    # =========================================
    # Embedding Collections (Metadata)
    # =========================================

    async def get_embedding_collections(
        self,
        collection_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all embedding collection metadata."""
        query = (
            self.client.table("embedding_collections")
            .select("*")
            .order("created_at", desc=True)
        )
        if collection_type:
            query = query.eq("collection_type", collection_type)
        response = query.execute()
        return response.data

    async def get_embedding_collection(self, collection_id: str) -> Optional[dict[str, Any]]:
        """Get single embedding collection by ID."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("embedding_collections")
            .select("*")
            .eq("id", collection_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def get_embedding_collection_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Get embedding collection by name."""
        response = (
            self.client.table("embedding_collections")
            .select("*")
            .eq("name", name)
            .maybe_single()
            .execute()
        )
        return response.data

    async def create_embedding_collection(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create embedding collection metadata."""
        response = (
            self.client.table("embedding_collections")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_embedding_collection(
        self, collection_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update embedding collection metadata."""
        response = (
            self.client.table("embedding_collections")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", collection_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_embedding_collection(self, collection_id: str) -> bool:
        """Delete embedding collection metadata."""
        self.client.table("embedding_collections").delete().eq("id", collection_id).execute()
        return True

    # =========================================
    # Matched Products (for Training)
    # =========================================

    async def get_matched_products(
        self,
        page: int = 1,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Get products that have at least one matched cutout.
        Used for training extraction tab.
        """
        # Get distinct product IDs that have matches
        matched_response = (
            self.client.table("cutout_images")
            .select("matched_product_id")
            .not_.is_("matched_product_id", "null")
            .execute()
        )

        # Get unique product IDs
        matched_product_ids = list(set(
            item["matched_product_id"]
            for item in matched_response.data
            if item.get("matched_product_id")
        ))

        if not matched_product_ids:
            return {
                "items": [],
                "total": 0,
                "page": page,
                "limit": limit,
            }

        # Pagination
        start = (page - 1) * limit
        end = start + limit - 1

        # Get products with their match counts
        products_response = (
            self.client.table("products")
            .select("*", count="exact")
            .in_("id", matched_product_ids)
            .range(start, end)
            .order("created_at", desc=True)
            .execute()
        )

        items = products_response.data

        # Add match counts for each product
        for product in items:
            count_response = (
                self.client.table("cutout_images")
                .select("*", count="exact")
                .eq("matched_product_id", product["id"])
                .execute()
            )
            product["matched_cutout_count"] = count_response.count or 0

        return {
            "items": items,
            "total": len(matched_product_ids),
            "page": page,
            "limit": limit,
        }

    async def get_matched_products_count(self) -> int:
        """Get count of products that have at least one matched cutout."""
        response = (
            self.client.table("cutout_images")
            .select("matched_product_id")
            .not_.is_("matched_product_id", "null")
            .execute()
        )
        unique_ids = set(
            item["matched_product_id"]
            for item in response.data
            if item.get("matched_product_id")
        )
        return len(unique_ids)

    async def get_matched_cutouts_for_product(
        self,
        product_id: str,
    ) -> list[dict[str, Any]]:
        """Get all cutouts matched to a specific product."""
        response = (
            self.client.table("cutout_images")
            .select("*")
            .eq("matched_product_id", product_id)
            .order("matched_at", desc=True)
            .execute()
        )
        return response.data

    async def get_matched_cutouts_for_products(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get all cutouts matched to multiple products (batch query)."""
        if not product_ids:
            return []

        # Batch the query to avoid URL length limits
        # Each UUID is ~36 chars, plus encoding overhead - limit to 50 IDs per batch
        batch_size = 50
        all_cutouts = []

        for i in range(0, len(product_ids), batch_size):
            batch_ids = product_ids[i:i + batch_size]
            response = (
                self.client.table("cutout_images")
                .select("*")
                .in_("matched_product_id", batch_ids)
                .order("matched_at", desc=True)
                .execute()
            )
            all_cutouts.extend(response.data)

        return all_cutouts

    async def get_product_images_by_types(
        self,
        product_id: str,
        image_types: list[str],
        frame_selection: str = "first",
        frame_interval: int = 5,
        max_frames: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get product images filtered by type with frame selection logic.

        Args:
            product_id: Product UUID
            image_types: List of types ('synthetic', 'real', 'augmented')
            frame_selection: 'first', 'all', 'key_frames', 'interval'
            frame_interval: For interval selection, pick every N frames
            max_frames: Maximum frames to return per type
        """
        all_images = []

        for img_type in image_types:
            response = (
                self.client.table("product_images")
                .select("*")
                .eq("product_id", product_id)
                .eq("image_type", img_type)
                .order("frame_index", desc=False)
                .execute()
            )
            images = response.data

            if not images:
                continue

            # Apply frame selection
            if frame_selection == "first":
                selected = images[:1]
            elif frame_selection == "all":
                selected = images[:max_frames]
            elif frame_selection == "key_frames":
                # Pick 0°, 90°, 180°, 270° angles (4 frames)
                frame_count = len(images)
                step = max(1, frame_count // 4)
                indices = [0] + [i * step for i in range(1, 4) if i * step < frame_count]
                selected = [images[i] for i in indices if i < len(images)]
                selected = selected[:max_frames]
            elif frame_selection == "interval":
                selected = images[::frame_interval][:max_frames]
            else:
                selected = images[:1]

            all_images.extend(selected)

        return all_images

    async def get_products_for_extraction(
        self,
        source_type: str,
        source_product_ids: Optional[list[str]] = None,
        source_dataset_id: Optional[str] = None,
        source_filter: Optional[dict] = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        Get products based on source configuration for extraction jobs.

        Args:
            source_type: 'all', 'selected', 'dataset', 'matched', 'filter', 'new'
            source_product_ids: For 'selected' source_type
            source_dataset_id: For 'dataset' source_type
            source_filter: For 'filter' source_type (JSONB filter)
            limit: Max products to return
        """
        if source_type == "all":
            response = (
                self.client.table("products")
                .select("*")
                .eq("status", "complete")
                .limit(limit)
                .execute()
            )
            return response.data

        elif source_type == "selected" and source_product_ids:
            response = (
                self.client.table("products")
                .select("*")
                .in_("id", source_product_ids)
                .execute()
            )
            return response.data

        elif source_type == "dataset" and source_dataset_id:
            # Get product IDs from dataset
            dp_response = (
                self.client.table("dataset_products")
                .select("product_id")
                .eq("dataset_id", source_dataset_id)
                .execute()
            )
            product_ids = [item["product_id"] for item in dp_response.data]
            if not product_ids:
                return []

            response = (
                self.client.table("products")
                .select("*")
                .in_("id", product_ids)
                .execute()
            )
            return response.data

        elif source_type == "matched":
            # Get products with matched cutouts
            matched_response = (
                self.client.table("cutout_images")
                .select("matched_product_id")
                .not_.is_("matched_product_id", "null")
                .execute()
            )
            product_ids = list(set(
                item["matched_product_id"]
                for item in matched_response.data
                if item.get("matched_product_id")
            ))
            if not product_ids:
                return []

            response = (
                self.client.table("products")
                .select("*")
                .in_("id", product_ids)
                .execute()
            )
            return response.data

        elif source_type == "filter" and source_filter:
            # Build query from filter
            query = self.client.table("products").select("*")

            if source_filter.get("status"):
                query = query.eq("status", source_filter["status"])
            if source_filter.get("category"):
                query = query.in_("category", source_filter["category"])
            if source_filter.get("brand"):
                query = query.in_("brand_name", source_filter["brand"])
            if source_filter.get("has_video") is True:
                query = query.not_.is_("video_url", "null")
            if source_filter.get("min_frame_count"):
                query = query.gte("frame_count", source_filter["min_frame_count"])

            response = query.limit(limit).execute()
            return response.data

        elif source_type == "new":
            # Products without embeddings (no qdrant_point_id or has_embedding)
            # This requires checking which products don't have embeddings yet
            response = (
                self.client.table("products")
                .select("*")
                .eq("status", "complete")
                .limit(limit)
                .execute()
            )
            # Note: actual filtering for "new" would need to check Qdrant
            return response.data

        return []

    # =========================================
    # Cutout Sync State
    # =========================================

    SYNC_STATE_ID = "00000000-0000-0000-0000-000000000001"

    async def get_cutout_sync_state(self) -> Optional[dict[str, Any]]:
        """Get the current cutout sync state."""
        response = (
            self.client.table("cutout_sync_state")
            .select("*")
            .eq("id", self.SYNC_STATE_ID)
            .maybe_single()
            .execute()
        )
        return response.data

    async def update_sync_state_for_new(
        self,
        max_external_id: int,
        synced_count: int,
    ) -> None:
        """Update sync state after 'sync new' operation."""
        current = await self.get_cutout_sync_state()

        update_data = {
            "max_synced_external_id": max(
                current.get("max_synced_external_id") or 0,
                max_external_id
            ) if current else max_external_id,
            "total_synced": (current.get("total_synced") or 0) + synced_count if current else synced_count,
            "last_sync_new_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Set min if not set
        if not current or not current.get("min_synced_external_id"):
            update_data["min_synced_external_id"] = max_external_id

        self.client.table("cutout_sync_state").upsert({
            "id": self.SYNC_STATE_ID,
            **update_data,
        }).execute()

    async def update_sync_state_for_backfill(
        self,
        min_external_id: int,
        synced_count: int,
        backfill_completed: bool = False,
        last_page: Optional[int] = None,
    ) -> None:
        """Update sync state after 'backfill' operation."""
        current = await self.get_cutout_sync_state()

        update_data = {
            "min_synced_external_id": min(
                current.get("min_synced_external_id") or min_external_id,
                min_external_id
            ) if current else min_external_id,
            "total_synced": (current.get("total_synced") or 0) + synced_count if current else synced_count,
            "backfill_completed": backfill_completed,
            "last_backfill_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Save last backfill page if provided
        if last_page is not None:
            update_data["last_backfill_page"] = last_page

        # Set max if not set
        if not current or not current.get("max_synced_external_id"):
            update_data["max_synced_external_id"] = min_external_id

        self.client.table("cutout_sync_state").upsert({
            "id": self.SYNC_STATE_ID,
            **update_data,
        }).execute()

    async def get_synced_external_id_range(self) -> tuple[Optional[int], Optional[int]]:
        """Get min and max synced external IDs."""
        state = await self.get_cutout_sync_state()
        if not state:
            return None, None
        return state.get("min_synced_external_id"), state.get("max_synced_external_id")

    async def update_last_backfill_page(self, page: int) -> None:
        """Update only the last_backfill_page field."""
        self.client.table("cutout_sync_state").update({
            "last_backfill_page": page,
            "last_backfill_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", self.SYNC_STATE_ID).execute()

    async def sync_cutouts_incremental(
        self,
        cutouts: list[dict[str, Any]],
        mode: str = "new",  # "new" or "backfill"
    ) -> dict[str, Any]:
        """
        Sync cutouts with incremental logic.

        For 'new' mode: Only sync cutouts with external_id > max_synced_external_id
        For 'backfill' mode: Only sync cutouts with external_id < min_synced_external_id

        Returns dict with synced_count, skipped_count, stopped_early, and boundary IDs.
        """
        if not cutouts:
            return {
                "synced_count": 0,
                "skipped_count": 0,
                "stopped_early": False,
                "new_min_external_id": None,
                "new_max_external_id": None,
            }

        min_synced, max_synced = await self.get_synced_external_id_range()

        # Filter based on mode
        if mode == "new" and max_synced is not None:
            # Only keep cutouts newer than max_synced
            filtered_cutouts = [c for c in cutouts if c["external_id"] > max_synced]
            stopped_early = len(filtered_cutouts) < len(cutouts)
        elif mode == "backfill" and min_synced is not None:
            # Only keep cutouts older than min_synced
            filtered_cutouts = [c for c in cutouts if c["external_id"] < min_synced]
            stopped_early = len(filtered_cutouts) < len(cutouts)
        else:
            # First sync or no state - sync all
            filtered_cutouts = cutouts
            stopped_early = False

        if not filtered_cutouts:
            return {
                "synced_count": 0,
                "skipped_count": len(cutouts),
                "stopped_early": True,
                "new_min_external_id": None,
                "new_max_external_id": None,
            }

        # Check for existing in DB (duplike protection)
        external_ids = [c["external_id"] for c in filtered_cutouts]
        existing = (
            self.client.table("cutout_images")
            .select("external_id")
            .in_("external_id", external_ids)
            .execute()
        )
        existing_ids = {item["external_id"] for item in existing.data}

        # Filter new cutouts
        new_cutouts = [
            {
                "external_id": c["external_id"],
                "image_url": c["image_url"],
                "predicted_upc": c.get("predicted_upc"),
            }
            for c in filtered_cutouts
            if c["external_id"] not in existing_ids
        ]

        if new_cutouts:
            self.client.table("cutout_images").insert(new_cutouts).execute()

        # Calculate new boundaries
        new_external_ids = [c["external_id"] for c in new_cutouts]
        new_min = min(new_external_ids) if new_external_ids else None
        new_max = max(new_external_ids) if new_external_ids else None

        # Update sync state
        if new_cutouts:
            if mode == "new":
                await self.update_sync_state_for_new(new_max, len(new_cutouts))
            else:
                await self.update_sync_state_for_backfill(new_min, len(new_cutouts))

        return {
            "synced_count": len(new_cutouts),
            "skipped_count": len(cutouts) - len(new_cutouts),
            "stopped_early": stopped_early,
            "new_min_external_id": new_min,
            "new_max_external_id": new_max,
        }

    # =========================================
    # Training Runs
    # =========================================

    async def get_training_runs(
        self,
        status: Optional[str] = None,
        base_model_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get training runs with pagination and filters."""
        query = self.client.table("training_runs").select("*", count="exact")

        if status:
            query = query.eq("status", status)
        if base_model_type:
            query = query.eq("base_model_type", base_model_type)

        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        response = query.execute()

        return {
            "items": response.data,
            "total": response.count or 0,
        }

    async def get_training_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get single training run."""
        # Note: Using limit(1) instead of single() due to supabase-py bug with bytes parsing
        response = (
            self.client.table("training_runs")
            .select("*")
            .eq("id", run_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_training_run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create training run."""
        response = (
            self.client.table("training_runs")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_training_run(
        self, run_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update training run."""
        response = (
            self.client.table("training_runs")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", run_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_training_run(
        self, run_id: str, force: bool = False
    ) -> dict[str, Any]:
        """
        Delete training run with all checkpoints and storage files.

        Args:
            run_id: Training run ID
            force: If True, also delete linked trained_models. If False, block if models exist.

        Returns:
            {"success": True} or {"success": False, "error": str, "linked_models": [...]}
        """
        # Check for linked trained_models (no cascade delete on this table)
        linked_models_response = (
            self.client.table("trained_models")
            .select("id, name")
            .eq("training_run_id", run_id)
            .execute()
        )
        linked_models = linked_models_response.data

        if linked_models and not force:
            return {
                "success": False,
                "error": "Training run has registered models. Use force=true to delete them as well.",
                "linked_models": linked_models,
            }

        # If force and has linked models, delete them first
        if linked_models and force:
            for model in linked_models:
                self.client.table("trained_models").delete().eq("id", model["id"]).execute()
                print(f"Deleted linked trained_model: {model['name']}")

        # Get all checkpoints to delete storage files
        checkpoints = await self.get_training_checkpoints(run_id)

        # Delete checkpoint files from storage
        for checkpoint in checkpoints:
            if checkpoint.get("checkpoint_url"):
                try:
                    url = checkpoint["checkpoint_url"]
                    if "/checkpoints/" in url:
                        storage_path = url.split("/checkpoints/")[-1]
                        self.client.storage.from_("checkpoints").remove([storage_path])
                        print(f"Deleted checkpoint from storage: {storage_path}")
                except Exception as e:
                    print(f"Warning: Failed to delete checkpoint file: {e}")

        # Delete the entire training folder from storage (catches any leftover files)
        try:
            folder_path = f"training/{run_id}"
            files = self.client.storage.from_("checkpoints").list(folder_path)
            if files:
                file_paths = [f"{folder_path}/{f['name']}" for f in files]
                self.client.storage.from_("checkpoints").remove(file_paths)
                print(f"Deleted {len(file_paths)} files from storage folder")
        except Exception as e:
            print(f"Warning: Failed to delete storage folder: {e}")

        # Delete from database (cascades to checkpoints and metrics_history)
        self.client.table("training_runs").delete().eq("id", run_id).execute()
        return {"success": True}

    # =========================================
    # Training Checkpoints
    # =========================================

    async def get_training_checkpoints(
        self,
        run_id: str,
        is_best: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Get checkpoints for a training run."""
        query = (
            self.client.table("training_checkpoints")
            .select("*")
            .eq("training_run_id", run_id)
            .order("epoch", desc=True)
        )
        if is_best is not None:
            query = query.eq("is_best", is_best)
        response = query.execute()
        return response.data

    async def get_training_checkpoint(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Get single checkpoint."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("training_checkpoints")
            .select("*")
            .eq("id", checkpoint_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_training_checkpoint(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create training checkpoint."""
        response = (
            self.client.table("training_checkpoints")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_training_checkpoint(
        self,
        checkpoint_id: str,
        data: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Update training checkpoint."""
        response = (
            self.client.table("training_checkpoints")
            .update(data)
            .eq("id", checkpoint_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_training_checkpoint(
        self, checkpoint_id: str, delete_storage: bool = True
    ) -> bool:
        """Delete training checkpoint and optionally the file from storage."""
        # First get the checkpoint to find the storage URL
        checkpoint = await self.get_training_checkpoint(checkpoint_id)
        if not checkpoint:
            return False

        # Delete from storage if requested and URL exists
        if delete_storage and checkpoint.get("checkpoint_url"):
            try:
                url = checkpoint["checkpoint_url"]
                # Extract storage path from URL
                # URL format: https://xxx.supabase.co/storage/v1/object/public/checkpoints/training/run_id/file.pth
                if "/checkpoints/" in url:
                    storage_path = url.split("/checkpoints/")[-1]
                    self.client.storage.from_("checkpoints").remove([storage_path])
                    print(f"Deleted checkpoint file from storage: {storage_path}")
            except Exception as e:
                print(f"Warning: Failed to delete checkpoint file from storage: {e}")
                # Continue with database deletion even if storage deletion fails

        # Delete from database
        self.client.table("training_checkpoints").delete().eq(
            "id", checkpoint_id
        ).execute()
        return True

    # ===========================================
    # Training Metrics History
    # ===========================================

    async def get_training_metrics_history(
        self,
        run_id: str,
        training_type: str = "embedding",
    ) -> list[dict[str, Any]]:
        """Get training metrics history for a run (per-epoch metrics for charts)."""
        response = (
            self.client.table("training_metrics_history")
            .select("*")
            .eq("training_run_id", run_id)
            .eq("training_type", training_type)
            .order("epoch", desc=False)
            .execute()
        )
        return response.data

    async def create_training_metrics_history(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a training metrics history entry."""
        response = (
            self.client.table("training_metrics_history")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def upsert_training_metrics_history(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Upsert a training metrics history entry (update if exists)."""
        response = (
            self.client.table("training_metrics_history")
            .upsert(data, on_conflict="training_run_id,epoch")
            .execute()
        )
        return response.data[0]

    # =========================================
    # Trained Models
    # =========================================

    async def get_trained_models(
        self,
        is_active: Optional[bool] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get trained models."""
        query = (
            self.client.table("trained_models")
            .select("*, training_run:training_runs(name, base_model_type)")
            .order("created_at", desc=True)
            .limit(limit)
        )
        if is_active is not None:
            query = query.eq("is_active", is_active)
        response = query.execute()
        return response.data

    async def get_trained_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get single trained model."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("trained_models")
            .select("*, training_run:training_runs(*), checkpoint:training_checkpoints(*)")
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_trained_model(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create trained model."""
        response = (
            self.client.table("trained_models")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_trained_model(
        self, model_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update trained model."""
        response = (
            self.client.table("trained_models")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", model_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_trained_model(self, model_id: str) -> bool:
        """Delete trained model."""
        self.client.table("trained_models").delete().eq("id", model_id).execute()
        return True

    async def activate_trained_model(self, model_id: str) -> Optional[dict[str, Any]]:
        """Activate a trained model (deactivates others)."""
        # Deactivate all
        self.client.table("trained_models").update(
            {"is_active": False}
        ).execute()

        # Activate selected
        response = (
            self.client.table("trained_models")
            .update({
                "is_active": True,
                "updated_at": datetime.utcnow().isoformat(),
            })
            .eq("id", model_id)
            .execute()
        )
        return response.data[0] if response.data else None

    # =========================================
    # Training Configs
    # =========================================

    async def get_training_configs(
        self,
        base_model_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get training configs."""
        query = (
            self.client.table("training_configs")
            .select("*")
            .order("created_at", desc=True)
        )
        if base_model_type:
            query = query.eq("base_model_type", base_model_type)
        response = query.execute()
        return response.data

    async def get_training_config(self, config_id: str) -> Optional[dict[str, Any]]:
        """Get single training config."""
        # Note: Using limit(1) instead of single() due to supabase-py bug
        response = (
            self.client.table("training_configs")
            .select("*")
            .eq("id", config_id)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    async def create_training_config(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create training config."""
        response = (
            self.client.table("training_configs")
            .insert(data)
            .execute()
        )
        return response.data[0]

    async def update_training_config(
        self, config_id: str, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Update training config."""
        response = (
            self.client.table("training_configs")
            .update({**data, "updated_at": datetime.utcnow().isoformat()})
            .eq("id", config_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def delete_training_config(self, config_id: str) -> bool:
        """Delete training config (if not default)."""
        # Check if it's a default config
        config = await self.get_training_config(config_id)
        if config and config.get("is_default"):
            return False
        self.client.table("training_configs").delete().eq("id", config_id).execute()
        return True

    # =========================================
    # Model Evaluations
    # =========================================

    async def get_model_evaluations(
        self,
        trained_model_id: str,
    ) -> list[dict[str, Any]]:
        """Get evaluations for a trained model."""
        response = (
            self.client.table("model_evaluations")
            .select("*")
            .eq("trained_model_id", trained_model_id)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def create_model_evaluation(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create model evaluation."""
        response = (
            self.client.table("model_evaluations")
            .insert(data)
            .execute()
        )
        return response.data[0]

    # =========================================
    # Products for Training
    # =========================================

    async def get_products_for_training(
        self,
        min_frames: int = 1,
        dataset_id: Optional[str] = None,
        product_ids: Optional[list[str]] = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        Get products for training.

        Returns products with frames, optionally filtered by dataset or specific IDs.
        Includes: id, barcode, brand_name, category, container_type, sub_brand,
                  manufacturer_country, variant_flavor, net_quantity, frames_path, frame_count, product_name

        Uses pagination to handle Supabase's 1000 row limit.
        """
        select_fields = "id,barcode,brand_name,category,container_type,sub_brand,manufacturer_country,variant_flavor,net_quantity,frames_path,frame_count,product_name,custom_fields"

        # If filtering by dataset, get those IDs first
        dataset_product_ids = None
        if dataset_id:
            dp_response = (
                self.client.table("dataset_products")
                .select("product_id")
                .eq("dataset_id", dataset_id)
                .execute()
            )
            dataset_product_ids = [item["product_id"] for item in dp_response.data]
            if not dataset_product_ids:
                return []

        # Pagination parameters
        page_size = 1000  # Supabase max per request
        all_products = []
        offset = 0

        while len(all_products) < limit:
            query = self.client.table("products").select(select_fields)

            # Filter by frame count
            query = query.gte("frame_count", min_frames)

            # Filter by dataset if provided
            if dataset_product_ids:
                query = query.in_("id", dataset_product_ids)

            # Filter by specific IDs if provided
            if product_ids:
                query = query.in_("id", product_ids)

            query = query.range(offset, offset + page_size - 1)
            response = query.execute()

            if not response.data:
                break

            all_products.extend(response.data)
            offset += page_size

            # If we got fewer than page_size, we've reached the end
            if len(response.data) < page_size:
                break

        return all_products[:limit]

    async def get_matched_products_for_training(
        self,
        min_frames: int = 1,
        min_matches: int = 1,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """
        Get products that have matched cutouts for training.

        These are products with real-world matches - good for training
        models that need to generalize to store images.
        """
        # Get product IDs with matches
        matches_response = (
            self.client.table("cutout_images")
            .select("matched_product_id")
            .not_.is_("matched_product_id", "null")
            .execute()
        )

        # Count matches per product
        product_match_counts: dict[str, int] = {}
        for row in matches_response.data:
            pid = row["matched_product_id"]
            product_match_counts[pid] = product_match_counts.get(pid, 0) + 1

        # Filter by minimum matches
        matched_product_ids = [
            pid for pid, count in product_match_counts.items()
            if count >= min_matches
        ]

        if not matched_product_ids:
            return []

        # Get product details
        return await self.get_products_for_training(
            min_frames=min_frames,
            product_ids=matched_product_ids,
            limit=limit,
        )


# Singleton instance
supabase_service = SupabaseService()
